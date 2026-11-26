from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from backbone import HalfIndex, digest, encode_codes, make_pools, save_json, valid_sequence


def target_keys(path):
    result = set()
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("UTR") or row["utr"]
            if valid_sequence(value):
                result.add(value)
    return result


def eligible(assignment, sequence):
    if assignment["model_lineage"] != "original_evolution" or not valid_sequence(sequence["normalized_sequence"]):
        return False
    if not sequence["measured_rl"] or not np.isfinite(float(sequence["measured_rl"])):
        return False
    if assignment["library_type"] == "designed_library":
        return assignment["E1_pool_eligible"] == "true" and assignment["evaluation_context_id"] == "eGFP|unmod|designed_single_assay"
    return assignment["library_type"] == "random_library" and assignment["role"] in {"train", "development", "historical_measured_state_outer"} and assignment["evaluation_context_id"] != "eGFP|unmod|varying_single_assay"


def component_split(unit):
    value = int.from_bytes(hashlib.sha256(f"tcell_multicontext_backbone_v1|{unit}".encode()).digest(), "big")
    return "dev" if value % 10 == 0 else "train"


def build_pack(path, rows_by_context, context_names):
    codes = []
    labels = []
    pools = []
    pool_contexts = []
    selection_pools = []
    selection_contexts = []
    row_offset = 0
    role_counts = {}
    for context_id, context in enumerate(context_names):
        rows = rows_by_context[context]
        sequences = [row[0] for row in rows]
        codes.append(encode_codes(sequences))
        labels.append(np.asarray([row[1] for row in rows], dtype=np.float64))
        local_pools = make_pools(sequences, context, rows[0][2])
        global_pools = np.where(local_pools >= 0, local_pools + row_offset, -1)
        pools.append(global_pools)
        pool_contexts.extend([context_id] * len(local_pools))
        complete = global_pools[np.all(global_pools >= 0, axis=1)]
        if len(complete):
            selection_pools.append(complete)
            selection_contexts.extend([context_id] * len(complete))
        role_counts[context] = len(rows)
        row_offset += len(rows)
    np.savez(path, codes=np.concatenate(codes), labels=np.concatenate(labels), pools=np.concatenate(pools), pool_contexts=np.asarray(pool_contexts, dtype=np.int64), selection_pools=np.concatenate(selection_pools), selection_pool_contexts=np.asarray(selection_contexts, dtype=np.int64), context_names=np.asarray(context_names))
    return role_counts


def build_original(config, output):
    manifest_path = Path(config["sample"]["original_data_manifest"])
    manifest = json.loads(manifest_path.read_text())
    pack_path = Path(manifest["pack"]["path"])
    if digest(pack_path) != manifest["pack"]["sha256"]:
        raise ValueError("original pack hash mismatch")
    with np.load(pack_path, allow_pickle=False) as pack:
        original_counts = {role: int(pack[f"x_{role}"].shape[0]) for role in ("train", "dev")}
    result = {}
    for role in ("train", "dev"):
        source = manifest["role_sources"][role]
        path = Path(source["path"])
        if digest(path) != source["sha256"]:
            raise ValueError("original role source hash mismatch")
        contexts = ["eGFP|unmod|1", "eGFP|unmod|designed_single_assay"]
        grouped = {context: [] for context in contexts}
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["context"] not in grouped:
                    raise ValueError("unexpected original context")
                if not valid_sequence(row["utr50"]):
                    raise ValueError("invalid original sequence")
                grouped[row["context"]].append((row["utr50"], float(row["measured_rl"]), role))
        counts = build_pack(output / f"original_{role}.npz", grouped, contexts)
        if sum(counts.values()) != original_counts[role]:
            raise ValueError("original pack/TSV count mismatch")
        result[role] = counts
    return {"sha256": digest(manifest_path), "pack_sha256": manifest["pack"]["sha256"], "role_counts": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    out = Path(config["artifact_root"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "configs").mkdir()
    (out / "data").mkdir()
    shutil.copyfile(config_path, out / "configs" / "backbone_frozen.json")
    config_hash = digest(config_path)
    workspace = Path(__file__).resolve().parent
    code_files = sorted(list(workspace.glob("*.py")) + list((workspace / "tests").glob("*.py")))
    save_json(out / "verification" / "prefreeze.json", {"analysis_id": config["analysis_id"], "config_sha256": config_hash, "code_sha256": {str(path.relative_to(workspace)): digest(path) for path in code_files}, "target_mrl_read": False, "target_evaluation_count": 0})

    target_index = HalfIndex()
    target_counts = {}
    target_hashes = {}
    for context, path in config["target_if_and_only_if_dual_gate_pass"]["raw"].items():
        target_hashes[context] = digest(path)
        if target_hashes[context] != config["target_if_and_only_if_dual_gate_pass"]["raw_sha256"][context]:
            raise ValueError("target raw hash mismatch")
        keys = target_keys(path)
        target_counts[context] = len(keys)
        for sequence in keys:
            target_index.add(sequence)
    if target_counts != {"defined_tcell_r1": 349633, "defined_tcell_r2": 352724}:
        raise ValueError("target key count mismatch")

    source_manifest_path = Path(config["sample"]["source_manifest"])
    source_manifest = json.loads(source_manifest_path.read_text())
    expected_raw = {item["name"]: item["sha256"] for item in source_manifest["files"]}
    raw_root = Path(config["sample"]["sequence_index"]).parents[2] / "raw" / "GSE114002"
    raw_hashes = {}
    for name, expected in expected_raw.items():
        path = raw_root / name
        if path.is_file():
            raw_hashes[name] = digest(path)
            if raw_hashes[name] != expected:
                raise ValueError(f"Sample raw hash mismatch: {name}")

    sequence_path = Path(config["sample"]["sequence_index"])
    assignment_path = Path(config["sample"]["role_assignments"])
    excluded_units = set()
    eligible_counts = Counter()
    for pass_name in ("target_exclusion", "pack_build"):
        print(json.dumps({"stage": pass_name, "status": "started"}), flush=True)
        grouped = {role: {context: [] for context in config["sample"]["contexts"]} for role in ("train", "dev")}
        with gzip.open(sequence_path, "rt", newline="") as sequence_handle, gzip.open(assignment_path, "rt", newline="") as assignment_handle:
            sequences = csv.DictReader(sequence_handle, delimiter="\t")
            assignments = (row for row in csv.DictReader(assignment_handle, delimiter="\t") if row["model_lineage"] == "original_evolution")
            for sequence, assignment in zip(sequences, assignments):
                if sequence["raw_row_id"] != assignment["raw_row_id"]:
                    raise ValueError("Sample sequence/assignment order mismatch")
                if not eligible(assignment, sequence):
                    continue
                context = assignment["evaluation_context_id"]
                if context not in config["sample"]["contexts"]:
                    raise ValueError(f"unrecognized eligible context: {context}")
                unit = assignment["allocation_unit_id"]
                if pass_name == "target_exclusion":
                    eligible_counts[context] += 1
                    exact, h1 = target_index.match(sequence["normalized_sequence"])
                    if exact or h1:
                        excluded_units.add(unit)
                elif unit not in excluded_units:
                    role = component_split(unit)
                    grouped[role][context].append((sequence["normalized_sequence"], float(sequence["measured_rl"]), role))
            if next(sequences, None) is not None or next(assignments, None) is not None:
                raise ValueError("Sample input length mismatch")
        if pass_name == "pack_build":
            role_counts = {}
            for role in ("train", "dev"):
                role_counts[role] = build_pack(out / "data" / f"extended_{role}.npz", grouped[role], config["sample"]["contexts"])
            print(json.dumps({"stage": "pack_written", "role": role, "counts": role_counts[role]}), flush=True)
            retained_sequences = sorted(row[0] for role in grouped.values() for rows in role.values() for row in rows)
            np.save(out / "data" / "extended_train_dev_sequences.npy", np.asarray(retained_sequences, dtype="S50"))

    original = build_original(config, out / "data")
    report = {
        "status": "PASS", "analysis_id": config["analysis_id"], "config_sha256": config_hash,
        "source_manifest_sha256": digest(source_manifest_path), "sample_sequence_index_sha256": digest(sequence_path),
        "sample_role_assignments_sha256": digest(assignment_path), "sample_raw_sha256": raw_hashes,
        "target_raw_sha256": target_hashes, "target_sequence_key_counts": target_counts,
        "target_label_columns_accessed": False, "target_mrl_read": False, "target_evaluation_count": 0,
        "eligible_before_target_exclusion": dict(eligible_counts), "target_excluded_allocation_components": len(excluded_units),
        "extended_role_counts": role_counts, "original": original,
        "split_rule": config["sample"]["split"], "exact_h1_component_split": True,
        "files": {str(path.relative_to(out)): digest(path) for path in sorted((out / "data").glob("*")) if path.is_file()},
    }
    save_json(out / "prepare_sample_manifest.json", report)
    save_json(out / "verification" / "round1_sample_prepare.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
