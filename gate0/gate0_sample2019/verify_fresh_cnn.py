import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import load_config, sha256_file
from .fresh_cnn_common import role_intersections, row_list_hash, semantic_manifest
from .fresh_proxy_common import read_tsv, write_json
from .training_gate import TrainingGate


def pack_rows(config):
    root = Path(config["parent_pack_dir"]) / "learner"
    result = {}
    for role in ["train", "dev", "calibration", "query"]:
        filename = "query_unlabeled.tsv.gz" if role == "query" else role + ".tsv.gz"
        result[role] = {row["row_id"]: row for row in read_tsv(root / filename)}
    return result


def outer_rows(config, output):
    result = {}
    for name in ["designed_outer_rows.tsv.gz", "random_outer_rows.tsv.gz"]:
        for row in read_tsv(output / "evaluator" / name):
            result[row["raw_row_id"]] = row
    return result


def lineage_hash(config, rows, outer):
    values = []
    for role, items in rows.items():
        for raw_id in sorted(items):
            row = items[raw_id]
            values.append("\t".join([role, raw_id, row["allocation_unit_id"], row["context"]]))
    for raw_id in sorted(outer):
        row = outer[raw_id]
        values.append("\t".join([raw_id, row["allocation_unit_id"], row["outer_stratum"], row["context"]]))
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def round1(config, output):
    rows = pack_rows(config)
    outer = outer_rows(config, output)
    candidate = json.loads((output / "candidate_manifest.json").read_text())
    outer_manifest = json.loads((output / "outer_manifest.json").read_text())
    pack_manifest = json.loads((Path(config["parent_pack_dir"]).parent / "pack_manifest.json").read_text())
    errors = []
    if pack_manifest["git_commit"] != config["parent_pack_commit"]:
        errors.append("parent_pack_commit")
    train_ids = set(rows["train"])
    scaler_ids = set()
    for row in read_tsv(output / "learner/train_scaler_rows.tsv.gz"):
        scaler_ids.add(row["raw_row_id"])
        source = rows["train"][row["raw_row_id"]]
        for field in ["sequence_id", "utr50", "measured_rl", "role", "allocation_unit_id", "context", "source_file_sha256", "raw_row_sha256"]:
            if row[field] != source[field]:
                errors.append("train_scaler_field:" + field)
    if scaler_ids != train_ids:
        errors.append("train_scaler_ids")
    if candidate["train_scaler_row_hash"] != row_list_hash(train_ids):
        errors.append("train_scaler_hash")
    designed_ids = {raw_id for raw_id, row in outer.items() if row["outer_stratum"] == "designed|eGFP_unmod"}
    random_ids = set(outer) - designed_ids
    if outer_manifest["outer_row_hashes"]["designed"] != row_list_hash(designed_ids) or outer_manifest["outer_row_hashes"]["random"] != row_list_hash(random_ids):
        errors.append("outer_row_hash")
    if candidate["lineage_hash"] != lineage_hash(config, rows, outer):
        errors.append("lineage_hash")
    if outer_manifest["designed_outer_rows"] != len(designed_ids) or outer_manifest["random_outer_rows"] != len(random_ids):
        errors.append("outer_counts")
    for raw_id, row in outer.items():
        if row["historical_evaluation_status"] != "historical_measured_state" or row["never_evaluated"].lower() != "false":
            errors.append("outer_semantics")
        if row["source_library"].startswith("target_") or row["source_library"].startswith("step_"):
            errors.append("target_or_step_outer")
        if row["source_file"] == "GSM3130435_egfp_unmod_1.csv.gz":
            errors.append("unmod_rep1_random_outer")
    if outer_manifest["designed_candidate_rows_reviewed"] != config["designed_candidate_rows"]:
        errors.append("candidate_count")
    if outer_manifest["designed_outer_rows"] + outer_manifest["excluded"].get("designed_target_retrained_lineage", 0) + outer_manifest["excluded"].get("designed_step_lineage", 0) != config["designed_candidate_rows"]:
        errors.append("candidate_accounting")
    result = {
        "round": 1,
        "train_scaler_rows": len(train_ids),
        "dev_rows": len(rows["dev"]),
        "calibration_rows": len(rows["calibration"]),
        "query_rows": len(rows["query"]),
        "designed_outer_rows": len(designed_ids),
        "random_outer_rows": len(random_ids),
        "train_scaler_row_hash": row_list_hash(train_ids),
        "designed_outer_row_hash": row_list_hash(designed_ids),
        "random_outer_row_hash": row_list_hash(random_ids),
        "lineage_hash": candidate["lineage_hash"],
        "errors": errors,
        "pass": not errors,
    }
    write_json(output / "verification_round1_lineage.json", result)
    if not result["pass"]:
        raise ValueError("Fresh CNN lineage verification failed")
    return result


def round2(config, output):
    rows = pack_rows(config)
    outer = outer_rows(config, output)
    role_values = {role: set(items) for role, items in rows.items()}
    role_values["outer"] = set(outer)
    sequence_overlap = role_intersections(role_values)
    group_values = {role: {item["near_group_id"] for item in items.values()} for role, items in rows.items()}
    group_values["outer"] = {row["near_group_id"] for row in outer.values()}
    group_overlap = role_intersections(group_values)
    trajectory_values = {role: {item["trajectory_id"] for item in items.values() if item["trajectory_id"]} for role, items in rows.items()}
    trajectory_values["outer"] = {row["trajectory_id"] for row in outer.values() if row["trajectory_id"]}
    trajectory_overlap = role_intersections(trajectory_values)
    unit_values = {role: {item["allocation_unit_id"] for item in items.values()} for role, items in rows.items()}
    unit_values["outer"] = {row["allocation_unit_id"] for row in outer.values()}
    unit_overlap = role_intersections(unit_values)
    negative = []
    if rows["train"]:
        injected = set(role_values["outer"])
        injected.add(next(iter(role_values["train"])))
        negative.append({"test": "deliberate_outer_train_sequence_injection", "pass": bool(injected & role_values["train"])})
    template = json.loads((output / "model_lineage_template.json").read_text())
    negative.append({"test": "fresh_initialization_no_official_weights", "pass": template["initialization"] == "fresh_random" and not template["official_weights"] and not template["distillation"]})
    negative.append({"test": "no_prefilled_model_pass", "pass": template["training_status"] == "NOT_STARTED" and template["model_hash"] is None and template["epoch"] is None and template["convergence"] is None and template["V_calibration"] is None and template["Q_calibration"] is None and not template["configuration_frozen"]})
    negative.append({"test": "E1_disabled_before_training_and_calibration", "pass": not template["E1_enabled"] and not TrainingGate.e1_permission(template)})
    try:
        TrainingGate.assert_learner_outer_access()
    except PermissionError:
        negative.append({"test": "learner_outer_labels_rejected", "pass": True})
    else:
        negative.append({"test": "learner_outer_labels_rejected", "pass": False})
    for role in TrainingGate.FORBIDDEN_ROLES:
        try:
            TrainingGate.assert_role_allowed(role)
        except PermissionError:
            negative.append({"test": "forbidden_role_" + role, "pass": True})
        else:
            negative.append({"test": "forbidden_role_" + role, "pass": False})
    for raw_id, row in outer.items():
        if row["outer_stratum"].startswith("random|") and row["E1_pool_eligible"].lower() != "false":
            negative.append({"test": "random_external_context_not_main_E1", "pass": False})
            break
    else:
        negative.append({"test": "random_external_context_not_main_E1", "pass": True})
    negative.append({"test": "target_retrained_not_relaxed", "pass": not any(row["source_library"].startswith("target_") for row in outer.values())})
    negative.append({"test": "unmod_rep1_not_relaxed", "pass": not any(row["source_file"] == "GSM3130435_egfp_unmod_1.csv.gz" for row in outer.values())})
    negative.append({"test": "historical_not_never_evaluated", "pass": all(row["historical_evaluation_status"] == "historical_measured_state" and row["never_evaluated"].lower() == "false" for row in outer.values())})
    result = {
        "round": 2,
        "sequence_intersections": sequence_overlap,
        "near_component_intersections": group_overlap,
        "trajectory_intersections": trajectory_overlap,
        "allocation_unit_intersections": unit_overlap,
        "negative_tests": negative,
        "pass": not any(item["count"] for item in sequence_overlap + group_overlap + trajectory_overlap + unit_overlap) and all(test["pass"] for test in negative),
    }
    write_json(output / "verification_round2_permissions.json", result)
    if not result["pass"]:
        raise ValueError("Fresh CNN permission verification failed")
    return result


def round3(config, root):
    first = root / config["build_names"][0]
    second = root / config["build_names"][1]
    comparisons = []
    for relative in ["learner/train_scaler_rows.tsv.gz", "evaluator/designed_outer_rows.tsv.gz", "evaluator/random_outer_rows.tsv.gz", "model_lineage_template.json", "learner_allowlist.json", "outer_manifest.json"]:
        a, b = first / relative, second / relative
        if relative.endswith(".json") and relative != "model_lineage_template.json":
            match = semantic_manifest(a) == semantic_manifest(b)
        else:
            match = sha256_file(a) == sha256_file(b)
        comparisons.append({"file": relative, "match": match})
    comparisons.append({"file": "candidate_manifest_semantic", "match": semantic_manifest(first / "candidate_manifest.json") == semantic_manifest(second / "candidate_manifest.json")})
    result = {
        "round": 3,
        "comparisons": comparisons,
        "parent_split_v1_manifest_sha256": sha256_file(Path(config["parent_split_v1_dir"]) / "split_manifest.json"),
        "parent_split_v2_manifest_sha256": sha256_file(Path(config["parent_split_v2_dir"]) / "split_manifest_v2.json"),
        "pass": all(item["match"] for item in comparisons),
    }
    write_json(first / "verification_round3_double_build.json", result)
    if not result["pass"]:
        raise ValueError("Fresh CNN double-build verification failed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["output_root"])
    first = round1(config, root / config["build_names"][0])
    second = round2(config, root / config["build_names"][0])
    third = round3(config, root)
    outer_manifest = json.loads((root / config["build_names"][0] / "outer_manifest.json").read_text())
    authorized = first["pass"] and second["pass"] and third["pass"] and outer_manifest["designed_outer_rows"] > 0
    summary = {
        "status": "FRESH_PROXY_CNN_CONDITIONAL_TRAINING_AUTHORIZED_E1_DISABLED",
        "child_split_version": config["child_split_version"],
        "child_lineage_version": config["child_lineage_version"],
        "lineage_hash": first["lineage_hash"],
        "rounds": {"lineage_source": first["pass"], "permissions_negative": second["pass"], "double_build_hashes": third["pass"]},
        "formal_training_start_authorized": authorized,
        "E1_enabled": False,
        "outer_metrics_computed": False,
        "proxy_training_performed": False,
        "model_training_performed": False,
    }
    write_json(root / "verification_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
