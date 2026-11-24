import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from proxy_function_gap.trainer.data import sha256_file


def _hamming1(left, right):
    return len(left) == len(right) and sum(a != b for a, b in zip(left, right)) == 1


def _near_existing(sequence, half_index):
    for position in (0, 1):
        key = sequence[:25] if position == 0 else sequence[25:]
        for other in half_index.get((position, key), []):
            if _hamming1(sequence, other):
                return True
    return False


def _add_half_index(sequence, half_index):
    half_index[(0, sequence[:25])].append(sequence)
    half_index[(1, sequence[25:])].append(sequence)


def _add_third_index(sequence, third_index):
    third_index[(0, sequence[:17])].append(sequence)
    third_index[(1, sequence[17:34])].append(sequence)
    third_index[(2, sequence[34:])].append(sequence)


def _hamming2_against(sequence, third_index):
    matches = set()
    for position, key in ((0, sequence[:17]), (1, sequence[17:34]), (2, sequence[34:])):
        for other in third_index.get((position, key), []):
            if sum(a != b for a, b in zip(sequence, other)) == 2:
                matches.add(other)
    return matches


def _write_check(directory, name, errors, **extra):
    report = {"pass": not errors, "errors": errors[:20], **extra}
    (directory / name).write_text(json.dumps(report, indent=2) + "\n")
    return report


def validate_castillo_pools(first, second, context, sample_role_paths, count=16, pool_size=64):
    first = Path(first)
    second = Path(second)
    pool_errors = []
    sample_errors = []
    build_errors = []
    sequences = []
    sample_sequences = set()
    sample_hashes = {}
    for path in sample_role_paths:
        path = Path(path)
        sample_hashes[str(path)] = sha256_file(path)
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample_sequences.add(row["utr50"])
    manifest = json.loads((first / "pool_manifest.json").read_text())
    second_manifest = json.loads((second / "pool_manifest.json").read_text())
    if manifest != second_manifest:
        build_errors.append("manifest_content")
    if sample_hashes != manifest["sample_role_hashes"]:
        sample_errors.append("sample_role_contract")
    expected_contract = {
        "schema_version": "1.1",
        "context": context,
        "selection_policy": "strict_h1",
        "h2_policy": "predeclared_sensitivity_diagnostic_only",
        "exact_sample_overlap_excluded": True,
        "hamming1_excluded": True,
        "pool_count": count,
        "pool_size": pool_size,
        "selected_rows": count * pool_size,
        "outer_metrics_computed": False,
        "labels_returned_to_learner": False,
    }
    for key, expected in expected_contract.items():
        if manifest.get(key) != expected:
            build_errors.append(f"manifest_{key}")
    if sha256_file(manifest["raw_file"]) != manifest["raw_file_sha256"]:
        build_errors.append("raw_file_hash")
    if sha256_file(manifest["train_reference"]) != manifest["train_reference_sha256"]:
        build_errors.append("train_reference_hash")
    for index in range(count):
        pool_id = f"castillo_{context}_{index:03d}"
        for directory in (first, second):
            path = directory / "learner" / f"{pool_id}.jsonl.gz"
            with gzip.open(path, "rt") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if len(rows) != pool_size or any("measured_rl" in row for row in rows) or any(row["pool_id"] != pool_id for row in rows):
                pool_errors.append(f"learner_contract:{pool_id}")
            pool_sequences = [row["utr50"] for row in rows]
            if len(set(pool_sequences)) != len(pool_sequences):
                pool_errors.append(f"exact_duplicate:{pool_id}")
            expected_pool_hash = hashlib.sha256("\n".join(pool_sequences).encode()).hexdigest()
            if manifest["pool_hashes"].get(pool_id) != expected_pool_hash or any(row["pool_hash"] != expected_pool_hash for row in rows):
                build_errors.append(f"pool_hash:{pool_id}")
            if abs(sum(row["q_value"] for row in rows) - 1) > 1e-12:
                pool_errors.append(f"q_sum:{pool_id}")
        learner_path = first / "learner" / f"{pool_id}.jsonl.gz"
        rebuild_path = second / "learner" / f"{pool_id}.jsonl.gz"
        if sha256_file(learner_path) != sha256_file(rebuild_path):
            build_errors.append(f"double_build:{pool_id}")
        with gzip.open(learner_path, "rt") as handle:
            sequences.extend(json.loads(line)["utr50"] for line in handle if line.strip())
    if len(set(sequences)) != len(sequences):
        pool_errors.append("global_exact_duplicate")
    for relative in ("evaluator/labels.tsv.gz", "pool_manifest.json"):
        if sha256_file(first / relative) != sha256_file(second / relative):
            build_errors.append(f"double_build_metadata:{relative}")
    sample_half_index = defaultdict(list)
    sample_third_index = defaultdict(list)
    for sequence in sample_sequences:
        _add_half_index(sequence, sample_half_index)
        _add_third_index(sequence, sample_third_index)
    half_index = defaultdict(list)
    pool_third_index = defaultdict(list)
    within_pool_h2 = 0
    cross_sample_h2 = 0
    for sequence in sequences:
        if sequence in sample_sequences:
            sample_errors.append("exact_sample_overlap")
        if _near_existing(sequence, sample_half_index):
            sample_errors.append("cross_study_hamming1")
        if _near_existing(sequence, half_index):
            pool_errors.append("hamming1_leakage")
        within_pool_h2 += len(_hamming2_against(sequence, pool_third_index))
        cross_sample_h2 += len(_hamming2_against(sequence, sample_third_index))
        _add_half_index(sequence, half_index)
        _add_third_index(sequence, pool_third_index)
    common = {"context": context, "pools": count, "sequences": len(sequences), "labels_returned_to_learner": False}
    checks = {
        "check1_pool_isolation": _write_check(first, "check1_pool_isolation.json", pool_errors, **common, checks=["pool_exact", "cross_pool_exact", "pool_and_cross_pool_hamming1"]),
        "check2_sample_isolation": _write_check(first, "check2_sample_isolation.json", sample_errors, **common, checks=["sample_train_dev_calibration_exact", "sample_train_dev_calibration_hamming1"]),
        "check3_double_build": _write_check(first, "check3_double_build.json", build_errors, **common, checks=["learner_hashes", "label_hashes", "manifest_content_and_hash", "raw_sample_train_hashes", "pool_hashes"]),
    }
    errors = pool_errors + sample_errors + build_errors
    report = {
        "pass": not errors,
        **common,
        "independent_checks": checks,
        "h2_sensitivity_diagnostic": {
            "within_and_cross_pool_pairs": within_pool_h2,
            "cross_sample_pairs": cross_sample_h2,
            "distance": 2,
            "used_for_selection": False,
            "used_for_pass_gate": False,
            "status": "predeclared_sensitivity_diagnostic_only",
        },
        "errors": errors[:20],
        "evidence_boundary": "retrospective_protocol_sensitivity_not_new_blind_test",
    }
    (first / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--sample-roles", nargs="+", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--pool-size", type=int, default=64)
    args = parser.parse_args()
    report = validate_castillo_pools(args.first, args.second, args.context, args.sample_roles, count=args.count, pool_size=args.pool_size)
    print(json.dumps(report))
    if not report["pass"]:
        raise ValueError("Castillo pool validation failed")


if __name__ == "__main__":
    main()
