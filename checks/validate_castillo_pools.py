import argparse
import csv
import gzip
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


def validate_castillo_pools(first, second, context, sample_role_paths, count=16, pool_size=64):
    first = Path(first)
    second = Path(second)
    errors = []
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
    if sample_hashes != manifest["sample_role_hashes"]:
        errors.append("sample_role_contract")
    for index in range(count):
        pool_id = f"castillo_{context}_{index:03d}"
        for directory in (first, second):
            path = directory / "learner" / f"{pool_id}.jsonl.gz"
            with gzip.open(path, "rt") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if len(rows) != pool_size or any("measured_rl" in row for row in rows) or any(row["pool_id"] != pool_id for row in rows):
                errors.append(f"learner_contract:{pool_id}")
            pool_sequences = [row["utr50"] for row in rows]
            if len(set(pool_sequences)) != len(pool_sequences):
                errors.append(f"exact_duplicate:{pool_id}")
            if abs(sum(row["q_value"] for row in rows) - 1) > 1e-12:
                errors.append(f"q_sum:{pool_id}")
        if sha256_file(first / "learner" / f"{pool_id}.jsonl.gz") != sha256_file(second / "learner" / f"{pool_id}.jsonl.gz"):
            errors.append(f"double_build:{pool_id}")
        with gzip.open(first / "learner" / f"{pool_id}.jsonl.gz", "rt") as handle:
            sequences.extend(json.loads(line)["utr50"] for line in handle if line.strip())
    if len(set(sequences)) != len(sequences):
        errors.append("global_exact_duplicate")
    if sha256_file(first / "evaluator/labels.tsv.gz") != sha256_file(second / "evaluator/labels.tsv.gz") or sha256_file(first / "pool_manifest.json") != sha256_file(second / "pool_manifest.json"):
        errors.append("double_build_metadata")
    sample_half_index = defaultdict(list)
    for sequence in sample_sequences:
        _add_half_index(sequence, sample_half_index)
    half_index = defaultdict(list)
    for sequence in sequences:
        if sequence in sample_sequences:
            errors.append("exact_sample_overlap")
        if _near_existing(sequence, sample_half_index):
            errors.append("cross_study_hamming1")
        if _near_existing(sequence, half_index):
            errors.append("hamming1_leakage")
        _add_half_index(sequence, half_index)
    report = {"pass": not errors, "context": context, "pools": count, "sequences": len(sequences), "checks": ["double_build_hashes", "learner_no_labels_and_q", "pool_and_global_exact_uniqueness", "cross_pool_hamming1", "cross_study_hamming1"], "errors": errors[:20], "labels_returned_to_learner": False}
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
