import argparse
import csv
import datetime
import json
from collections import Counter
from pathlib import Path

from .common import finite_float, load_config, sha256_file, sha256_text
from .fresh_proxy_common import intersections, open_deterministic_tsv_gz, read_tsv, write_json, write_row


LABELED_FIELDS = [
    "row_id", "sequence_id", "utr50", "measured_rl", "role", "allocation_unit_id", "context",
    "source_file", "source_file_sha256", "raw_row_sha256", "sequence_hash", "normalization_version",
    "near_group_id", "trajectory_id", "replicate_id", "replicate_count", "replicate_summary_rule", "split_version",
]
QUERY_FIELDS = [field for field in LABELED_FIELDS if field != "measured_rl"]


def source_views(config, purpose):
    root = Path(config["parent_split_v1_dir"])
    if purpose == "labels":
        return [
            root / "learner/train_features_labels.tsv.gz",
            root / "learner/dev_features_labels.tsv.gz",
            root / "learner/calibration_features_labels.tsv.gz",
            root / "evaluator/query_label_vault.tsv.gz",
        ]
    return [
        root / "learner/train_features_labels.tsv.gz",
        root / "learner/dev_features_labels.tsv.gz",
        root / "learner/calibration_features_labels.tsv.gz",
        root / "learner/query_unlabeled.tsv.gz",
    ]


def build(config, build_dir):
    build_dir.mkdir(parents=True, exist_ok=False)
    (build_dir / "learner").mkdir()
    assignments = {}
    role_counts = Counter()
    role_sequences = {role: set() for role in config["proof_roles"]}
    role_groups = {role: set() for role in config["proof_roles"]}
    role_trajectories = {role: set() for role in config["proof_roles"]}
    sequence_contexts = set()
    for row in read_tsv(Path(config["parent_split_dir"]) / "role_assignments_v2.tsv.gz"):
        if row["model_lineage"] != config["model_lineage_for_roles"] or row["evaluation_context_id"] not in config["allowed_contexts"]:
            continue
        role = row["role"]
        if role not in config["proof_roles"]:
            continue
        raw_id = row["raw_row_id"]
        assignments[raw_id] = row
        role_counts[role] += 1
        role_sequences[role].add(row["sequence_id"])
        role_groups[role].add(row["near_group_id"])
        if row["trajectory_id"]:
            role_trajectories[role].add(row["trajectory_id"])
        key = (row["sequence_id"], row["evaluation_context_id"])
        if key in sequence_contexts:
            raise ValueError(f"Duplicate sequence/context requiring pre-registered aggregation: {key}")
        sequence_contexts.add(key)
    manifest = json.loads(Path(config["input_manifest"]).read_text())
    source_hashes = {item["name"]: item["sha256"] for item in manifest["files"]}
    role_writers = {}
    handles = []
    found = Counter()
    missing_rl = 0
    access_log = []
    try:
        for role in config["pack_roles"]:
            fields = QUERY_FIELDS if role == "query" else LABELED_FIELDS
            handle, writer = open_deterministic_tsv_gz(build_dir / "learner" / f"{role}{'_unlabeled' if role == 'query' else ''}.tsv.gz", fields)
            handles.append(handle)
            role_writers[role] = writer
        for purpose in ["labels", "query"]:
            for source in source_views(config, purpose):
                access_log.append({"path": str(source), "purpose": purpose, "outer_labels": False})
                for source_row in read_tsv(source):
                    raw_id = source_row["raw_row_id"]
                    if raw_id not in assignments:
                        continue
                    assignment = assignments[raw_id]
                    role = assignment["role"]
                    if purpose == "labels" and role not in {"train", "dev", "calibration"}:
                        continue
                    if purpose == "query" and role != "query":
                        continue
                    normalized = source_row["normalized_sequence"]
                    if len(normalized) != 50 or source_row["normalization_version"] != config["normalization_version"]:
                        raise ValueError("Unexpected sequence or normalization version")
                    if sha256_text(normalized) != assignment["sequence_id"].replace("sample2019_", ""):
                        raise ValueError("Sequence hash mismatch")
                    row = {
                        "row_id": raw_id,
                        "sequence_id": assignment["sequence_id"],
                        "utr50": normalized,
                        "role": role,
                        "allocation_unit_id": assignment["allocation_unit_id"],
                        "context": assignment["evaluation_context_id"],
                        "source_file": assignment["source_file"],
                        "source_file_sha256": source_hashes[assignment["source_file"]],
                        "raw_row_sha256": assignment["raw_row_sha256"],
                        "sequence_hash": sha256_text(normalized),
                        "normalization_version": config["normalization_version"],
                        "near_group_id": assignment["near_group_id"],
                        "trajectory_id": assignment["trajectory_id"],
                        "replicate_id": assignment["replicate_id"],
                        "replicate_count": 1,
                        "replicate_summary_rule": config["replicate_summary_rule"],
                        "split_version": config["parent_split_version"],
                    }
                    if role != "query":
                        measured = finite_float(source_row["measured_rl"])
                        if measured is None:
                            missing_rl += 1
                            continue
                        row["measured_rl"] = measured
                    write_row(role_writers[role], row)
                    found[(purpose, role)] += 1
    finally:
        for handle in handles:
            handle.close()
    for role in ["train", "dev", "calibration"]:
        if found[("labels", role)] + missing_rl != role_counts[role]:
            raise ValueError(f"Label coverage mismatch for {role}: {found[('labels', role)]} + {missing_rl} != {role_counts[role]}")
    if found[("query", "query")] != role_counts["query"]:
        raise ValueError("Query coverage mismatch")
    sequence_overlap = intersections(role_sequences)
    group_overlap = intersections(role_groups)
    trajectory_overlap = intersections(role_trajectories)
    if any(item["count"] for item in sequence_overlap + group_overlap + trajectory_overlap):
        raise ValueError("Role overlap detected")
    file_hashes = {}
    for path in sorted((build_dir / "learner").iterdir()):
        if path.is_file():
            file_hashes[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    counts = {
        "rows_by_role": dict(role_counts),
        "unique_sequences_by_role": {role: len(role_sequences[role]) for role in config["proof_roles"]},
        "near_components_by_role": {role: len(role_groups[role]) for role in config["proof_roles"]},
        "trajectories_by_role": {role: len(role_trajectories[role]) for role in config["proof_roles"]},
        "excluded_missing_rl": missing_rl,
        "sequence_context_rows": len(sequence_contexts),
        "outer_labels_packaged": 0,
    }
    write_json(build_dir / "role_counts.json", counts)
    write_json(build_dir / "source_access_log.json", access_log)
    pack = {
        "schema_version": "1.0",
        "pack_version": config["pack_version"],
        "normalization_version": config["normalization_version"],
        "split_version": config["parent_split_version"],
        "split_manifest_sha256": sha256_file(Path(config["parent_split_dir"]) / "split_manifest_v2.json"),
        "data_manifest_sha256": sha256_file(Path(config["input_manifest"])),
        "build_dir": str(build_dir),
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "role_counts": counts,
        "files": file_hashes,
        "source_access_log": str(build_dir / "source_access_log.json"),
        "train_statistics": config["train_statistics"],
        "replicate_summary_rule": config["replicate_summary_rule"],
        "allowed_contexts": config["allowed_contexts"],
        "outer_learner_package_created": False,
        "outer_labels_read_or_packaged": False,
        "outer_metrics_computed": False,
        "proxy_training_performed": False,
    }
    write_json(build_dir / "pack_manifest.json", pack)
    print(json.dumps({"build_dir": str(build_dir), "rows_by_role": dict(role_counts), "unique_sequences_by_role": counts["unique_sequences_by_role"], "outer_labels_read_or_packaged": False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    for name in config["build_names"]:
        build(config, root / name)


if __name__ == "__main__":
    main()
