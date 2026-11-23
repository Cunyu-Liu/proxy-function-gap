import argparse
import json
from collections import Counter
from pathlib import Path

from .common import finite_float, load_config, sha256_file, sha256_text
from .fresh_proxy_common import intersections, read_tsv, write_json


def selected_assignments(config):
    result = {}
    for row in read_tsv(Path(config["parent_split_dir"]) / "role_assignments_v2.tsv.gz"):
        if row["model_lineage"] == config["model_lineage_for_roles"] and row["evaluation_context_id"] in config["allowed_contexts"] and row["role"] in config["proof_roles"]:
            result[row["raw_row_id"]] = row
    return result


def role_sets(config, field):
    result = {role: set() for role in config["proof_roles"]}
    for row in selected_assignments(config).values():
        if row[field]:
            result[row["role"]].add(row[field])
    return result


def round1(config, root):
    build = root / config["build_names"][0]
    assignments = selected_assignments(config)
    manifest = json.loads(Path(config["input_manifest"]).read_text())
    raw_hashes = {item["name"]: item["sha256"] for item in manifest["files"]}
    for name, expected in raw_hashes.items():
        raw_path = Path(config["input_manifest"]).parent / "raw/GSE114002" / name
        if sha256_file(raw_path) != expected:
            raise ValueError(f"Raw file hash changed: {name}")
    errors = []
    counts = Counter()
    sequence_contexts = set()
    for role in config["pack_roles"]:
        path = build / "learner" / f"{role}{'_unlabeled' if role == 'query' else ''}.tsv.gz"
        for row in read_tsv(path):
            counts[role] += 1
            raw_id = row["row_id"]
            if raw_id not in assignments:
                errors.append("unexpected_source_row")
                continue
            source = assignments[raw_id]
            if row["role"] != role or source["role"] != role:
                errors.append("role_mismatch")
            for pack_field, source_field in [("sequence_id", "sequence_id"), ("allocation_unit_id", "allocation_unit_id"), ("context", "evaluation_context_id"), ("source_file", "source_file"), ("raw_row_sha256", "raw_row_sha256"), ("near_group_id", "near_group_id"), ("trajectory_id", "trajectory_id")]:
                if row[pack_field] != source[source_field]:
                    errors.append("source_field_mismatch:" + pack_field)
            if row["utr50"] != row["utr50"][:50] or len(row["utr50"]) != 50:
                errors.append("utr50_length")
            if row["sequence_hash"] != sha256_text(row["utr50"]) or row["sequence_id"] != "sample2019_" + row["sequence_hash"]:
                errors.append("sequence_hash")
            if row["normalization_version"] != config["normalization_version"]:
                errors.append("normalization_version")
            if row["source_file_sha256"] != raw_hashes[row["source_file"]]:
                errors.append("source_file_hash")
            if row["context"] not in config["allowed_contexts"]:
                errors.append("context_forbidden")
            if role == "query":
                if "measured_rl" in row:
                    errors.append("query_label_leak")
            elif finite_float(row["measured_rl"]) is None:
                errors.append("missing_labeled_mrl")
            key = (row["sequence_id"], row["context"])
            if key in sequence_contexts:
                errors.append("duplicate_sequence_context")
            sequence_contexts.add(key)
    expected = Counter(row["role"] for row in assignments.values() if row["role"] in config["pack_roles"])
    if counts != expected:
        errors.append({"role_counts": dict(counts), "expected": dict(expected)})
    result = {
        "round": 1,
        "rows_by_role": dict(counts),
        "sequence_context_rows": len(sequence_contexts),
        "errors": errors,
        "allowed_contexts": config["allowed_contexts"],
        "train_statistics": config["train_statistics"],
        "pass": not errors,
    }
    write_json(build / "verification_round1_source.json", result)
    if not result["pass"]:
        raise ValueError("Fresh proxy source/context verification failed")
    return result


def round2(config, root):
    build = root / config["build_names"][0]
    sequence_overlap = intersections(role_sets(config, "sequence_id"))
    group_overlap = intersections(role_sets(config, "near_group_id"))
    trajectory_overlap = intersections(role_sets(config, "trajectory_id"))
    negative = []
    sequences = role_sets(config, "sequence_id")
    groups = role_sets(config, "near_group_id")
    trajectories = role_sets(config, "trajectory_id")
    train_sequence = next(iter(sequences["train"]))
    sequences["historical_measured_state_outer"].add(train_sequence)
    detected = any(item["count"] for item in intersections(sequences))
    negative.append({"test": "deliberate_sequence_outer_injection_detected", "pass": detected})
    train_group = next(iter(groups["train"]))
    groups["historical_measured_state_outer"].add(train_group)
    detected = any(item["count"] for item in intersections(groups))
    negative.append({"test": "deliberate_near_component_outer_injection_detected", "pass": detected})
    if trajectories["train"]:
        train_trajectory = next(iter(trajectories["train"]))
        trajectory_test_name = "deliberate_trajectory_outer_injection_detected"
    else:
        train_trajectory = "synthetic_trajectory_for_negative_test"
        trajectories["train"].add(train_trajectory)
        trajectory_test_name = "synthetic_trajectory_outer_injection_detected_when_no_real_trajectories"
    trajectories["historical_measured_state_outer"].add(train_trajectory)
    detected = any(item["count"] for item in intersections(trajectories))
    negative.append({"test": trajectory_test_name, "pass": detected})
    query_path = build / "learner/query_unlabeled.tsv.gz"
    with __import__("gzip").open(query_path, "rt") as handle:
        header = handle.readline().strip().split("\t")
    negative.append({"test": "query_has_no_label_columns", "pass": not any(column in header for column in ["measured_rl", "measured_status", "predicted_mrl", "predicted_status"])})
    learner_files = [path.name for path in (build / "learner").iterdir()]
    negative.append({"test": "no_outer_learner_file", "pass": not any("outer" in name for name in learner_files)})
    access = json.loads((build / "source_access_log.json").read_text())
    negative.append({"test": "no_outer_source_access", "pass": all("outer_label_vault" not in item["path"] and "outer_features" not in item["path"] for item in access)})
    pack = json.loads((build / "pack_manifest.json").read_text())
    negative.append({"test": "train_statistics_deferred", "pass": pack["train_statistics"] == "DEFERRED_TO_TRAINER"})
    negative.append({"test": "outer_labels_not_read_or_packaged", "pass": not pack["outer_labels_read_or_packaged"] and not pack["outer_learner_package_created"]})
    result = {
        "round": 2,
        "sequence_intersections": sequence_overlap,
        "near_component_intersections": group_overlap,
        "trajectory_intersections": trajectory_overlap,
        "negative_tests": negative,
        "trajectory_limitation": "All step rows are control_nonexact in split-v2; train/dev/calibration/query/outer trajectory sets are empty, so intersection is vacuously empty and the injection test is synthetic.",
        "official_seen_control_quarantine_roles_in_pack": [],
        "pass": not any(item["count"] for item in sequence_overlap + group_overlap + trajectory_overlap) and all(test["pass"] for test in negative),
    }
    write_json(build / "verification_round2_leakage.json", result)
    if not result["pass"]:
        raise ValueError("Fresh proxy leakage verification failed")
    return result


def normalized_manifest(path, build_dir):
    value = json.loads(path.read_text())
    value.pop("created_utc", None)
    value.pop("build_dir", None)
    value["source_access_log"] = "<BUILD_DIR>/source_access_log.json"
    return value


def round3(config, root):
    first = root / config["build_names"][0]
    second = root / config["build_names"][1]
    comparisons = []
    for name in ["train.tsv.gz", "dev.tsv.gz", "calibration.tsv.gz", "query_unlabeled.tsv.gz"]:
        a = first / "learner" / name
        b = second / "learner" / name
        comparisons.append({"file": name, "sha256_build_a": sha256_file(a), "sha256_build_b": sha256_file(b), "match": sha256_file(a) == sha256_file(b)})
    for name in ["role_counts.json", "source_access_log.json"]:
        same = json.loads((first / name).read_text()) == json.loads((second / name).read_text())
        comparisons.append({"file": name, "match": same})
    manifests_equal = normalized_manifest(first / "pack_manifest.json", first) == normalized_manifest(second / "pack_manifest.json", second)
    comparisons.append({"file": "pack_manifest_semantic", "match": manifests_equal})
    split_hash = sha256_file(Path(config["parent_split_dir"]) / "split_manifest_v2.json")
    result = {
        "round": 3,
        "comparisons": comparisons,
        "split_manifest_sha256": split_hash,
        "data_manifest_sha256": sha256_file(Path(config["input_manifest"])),
        "pass": all(item["match"] for item in comparisons),
    }
    write_json(first / "verification_round3_double_build.json", result)
    if not result["pass"]:
        raise ValueError("Fresh proxy double-build verification failed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["output_root"])
    first = round1(config, root)
    second = round2(config, root)
    third = round3(config, root)
    summary = {
        "status": "FRESH_PROXY_DATA_PACKS_VERIFIED_NO_OUTER_LABELS",
        "pack_version": config["pack_version"],
        "rounds": {"source_context": first["pass"], "leakage_negative": second["pass"], "double_build_hashes": third["pass"]},
        "rows_by_role": first["rows_by_role"],
        "outer_labels_read_or_packaged": False,
        "outer_learner_package_created": False,
        "outer_metrics_computed": False,
        "proxy_training_performed": False,
    }
    write_json(root / "verification_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
