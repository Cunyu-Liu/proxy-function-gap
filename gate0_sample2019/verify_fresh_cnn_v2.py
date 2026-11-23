import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import load_config, sha256_file
from .fresh_cnn_common import role_intersections, row_list_hash, semantic_manifest
from .fresh_proxy_common import read_tsv, write_json
from .npy_io import decode_float64, decode_unicode, read_npz
from .training_gate import TrainingGate


def pack_rows(config):
    result = {}
    for role in ["train", "dev", "calibration", "query"]:
        filename = "query_unlabeled.tsv.gz" if role == "query" else role + ".tsv.gz"
        result[role] = {row["row_id"]: row for row in read_tsv(Path(config["parent_pack_dir"]) / "learner" / filename)}
    return result


def outer_rows(config, output):
    result = {}
    for name in ["designed_outer_rows.tsv.gz", "random_outer_rows.tsv.gz"]:
        for row in read_tsv(output / "evaluator" / name):
            result[row["raw_row_id"]] = row
    return result


def lineage_hash(rows, outer):
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
    errors = []
    if outer_manifest["fresh_main_random_unmod_rep1_rows"] == 0:
        errors.append("unmod_rep1_not_restored")
    if candidate["lineage_hash"] != lineage_hash(rows, outer):
        errors.append("lineage_hash")
    train_ids = set(rows["train"])
    scaler_ids = set()
    for row in read_tsv(output / "learner/train_scaler_rows.tsv.gz"):
        scaler_ids.add(row["raw_row_id"])
        source = rows["train"][row["raw_row_id"]]
        for field in ["sequence_id", "utr50", "measured_rl", "role", "allocation_unit_id", "context", "source_file_sha256", "raw_row_sha256"]:
            if row[field] != source[field]:
                errors.append("train_scaler_field:" + field)
    if scaler_ids != train_ids or candidate["train_scaler_row_hash"] != row_list_hash(train_ids):
        errors.append("train_scaler_hash")
    designed = {raw_id for raw_id, row in outer.items() if row["outer_stratum"] == "designed|eGFP_unmod"}
    random = set(outer) - designed
    if outer_manifest["outer_row_hashes"]["designed"] != row_list_hash(designed) or outer_manifest["outer_row_hashes"]["random"] != row_list_hash(random):
        errors.append("outer_row_hashes")
    main_random = {raw_id for raw_id, row in outer.items() if row["outer_stratum"] == "random_main|eGFP_unmod_rep1"}
    if outer_manifest["outer_row_hashes"]["fresh_main_random_unmod_rep1"] != row_list_hash(main_random):
        errors.append("main_random_hash")
    if outer_manifest["designed_outer_rows"] + outer_manifest["excluded"].get("designed_target_retrained_lineage", 0) + outer_manifest["excluded"].get("designed_step_lineage", 0) != config["designed_candidate_rows"]:
        errors.append("designed_candidate_accounting")
    for raw_id, row in outer.items():
        if row["source_library"].startswith(("target_", "step_")):
            errors.append("target_or_step_outer")
        if row["historical_evaluation_status"] != "historical_measured_state" or row["never_evaluated"].lower() != "false":
            errors.append("outer_semantics")
    result = {
        "round": 1,
        "train_scaler_rows": len(train_ids),
        "dev_rows": len(rows["dev"]),
        "calibration_rows": len(rows["calibration"]),
        "query_rows": len(rows["query"]),
        "designed_outer_rows": len(designed),
        "random_outer_rows": len(random),
        "fresh_main_random_unmod_rep1_rows": len(main_random),
        "train_scaler_row_hash": row_list_hash(train_ids),
        "designed_outer_row_hash": row_list_hash(designed),
        "random_outer_row_hash": row_list_hash(random),
        "fresh_main_random_unmod_rep1_hash": row_list_hash(main_random),
        "lineage_hash": candidate["lineage_hash"],
        "errors": errors,
        "pass": not errors,
    }
    write_json(output / "verification_round1_lineage.json", result)
    if not result["pass"]:
        raise ValueError("Fresh CNN v2 lineage verification failed")
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
    unit_values = {role: {item["allocation_unit_id"] for item in items.values()} for role, items in rows.items()}
    unit_values["outer"] = {row["allocation_unit_id"] for row in outer.values()}
    unit_overlap = role_intersections(unit_values)
    trajectory_values = {role: {item["trajectory_id"] for item in items.values() if item["trajectory_id"]} for role, items in rows.items()}
    trajectory_values["outer"] = {row["trajectory_id"] for row in outer.values() if row["trajectory_id"]}
    trajectory_overlap = role_intersections(trajectory_values)
    random_designed = [item for item in role_intersections({"random": {raw_id for raw_id, row in outer.items() if row["outer_stratum"].startswith("random_")}, "designed": {raw_id for raw_id, row in outer.items() if row["outer_stratum"].startswith("designed|")}})]
    random_designed_near = [item for item in role_intersections({"random": {row["near_group_id"] for row in outer.values() if row["outer_stratum"].startswith("random_")}, "designed": {row["near_group_id"] for row in outer.values() if row["outer_stratum"].startswith("designed|")}})]
    random_designed_units = [item for item in role_intersections({"random": {row["allocation_unit_id"] for row in outer.values() if row["outer_stratum"].startswith("random_")}, "designed": {row["allocation_unit_id"] for row in outer.values() if row["outer_stratum"].startswith("designed|")}})]
    random_designed_trajectories = [item for item in role_intersections({"random": {row["trajectory_id"] for row in outer.values() if row["outer_stratum"].startswith("random_") and row["trajectory_id"]}, "designed": {row["trajectory_id"] for row in outer.values() if row["outer_stratum"].startswith("designed|")}})]
    negative = []
    main_random = {raw_id for raw_id, row in outer.items() if row["outer_stratum"] == "random_main|eGFP_unmod_rep1"}
    negative.append({"test": "unmod_rep1_fresh_random_outer_restored", "pass": bool(main_random)})
    negative.append({"test": "original_development_not_inherited_as_fresh_exclusion", "pass": bool(main_random)})
    negative.append({"test": "target_retrained_still_excluded", "pass": not any(row["source_library"].startswith("target_") for row in outer.values())})
    negative.append({"test": "random_designed_strata_disjoint", "pass": not any(item["count"] for item in random_designed + random_designed_near + random_designed_units + random_designed_trajectories)})
    negative.append({"test": "random_main_and_designed_main_E1_allowed", "pass": all(row["E1_pool_eligible"].lower() == "true" for row in outer.values() if row["outer_stratum"] == "random_main|eGFP_unmod_rep1" or row["outer_stratum"] == "designed|eGFP_unmod")})
    negative.append({"test": "random_external_not_main_E1", "pass": all(row["E1_pool_eligible"].lower() == "false" for row in outer.values() if row["outer_stratum"].startswith("random_external|"))})
    template = json.loads((output / "model_lineage_template.json").read_text())
    negative.append({"test": "fresh_no_official_weights_or_distillation", "pass": template["initialization"] == "fresh_random" and not template["official_weights"] and not template["distillation"]})
    negative.append({"test": "no_prefilled_model_pass", "pass": template["training_status"] == "NOT_STARTED" and template["model_hash"] is None and template["epoch"] is None and template["convergence"] is None and template["V_calibration"] is None and template["Q_calibration"] is None and not template["configuration_frozen"]})
    negative.append({"test": "E1_disabled", "pass": not template["E1_enabled"] and not TrainingGate.e1_permission(template)})
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
    negative.append({"test": "historical_not_never_evaluated", "pass": all(row["historical_evaluation_status"] == "historical_measured_state" and row["never_evaluated"].lower() == "false" for row in outer.values())})
    result = {
        "round": 2,
        "sequence_intersections": sequence_overlap,
        "near_component_intersections": group_overlap,
        "allocation_unit_intersections": unit_overlap,
        "trajectory_intersections": trajectory_overlap,
        "random_designed_intersections": random_designed,
        "random_designed_near_intersections": random_designed_near,
        "random_designed_allocation_unit_intersections": random_designed_units,
        "random_designed_trajectory_intersections": random_designed_trajectories,
        "negative_tests": negative,
        "pass": not any(item["count"] for item in sequence_overlap + group_overlap + unit_overlap + trajectory_overlap + random_designed + random_designed_near + random_designed_units + random_designed_trajectories) and all(test["pass"] for test in negative),
    }
    write_json(output / "verification_round2_permissions.json", result)
    if not result["pass"]:
        raise ValueError("Fresh CNN v2 permission verification failed")
    return result


def compare_npz_tsv(npz_path, tsv_path, role):
    arrays = read_npz(npz_path)
    rows = list(read_tsv(tsv_path))
    errors = []
    if "synthetic" not in arrays or arrays["synthetic"][2] != b"\x00":
        errors.append("synthetic")
    ids = decode_unicode(*arrays["row_id"])
    expected_ids = [row["raw_row_id"] for row in rows]
    if ids != expected_ids:
        errors.append("row_id_order")
    if len(ids) != len(rows) or set(ids) != set(expected_ids):
        errors.append("row_id_set_or_count")
    if role == "query" and "measured_rl" in arrays:
        errors.append("query_label")
    if role != "query" and "measured_rl" not in arrays:
        errors.append("missing_label")
    if "utr50_onehot" not in arrays or arrays["utr50_onehot"][1] != (len(rows), 50, 4):
        errors.append("onehot_shape")
    for field in ["sequence_id", "utr50", "allocation_unit_id", "context", "source_file_sha256", "raw_row_sha256", "normalization_version", "split_version"]:
        if field not in arrays:
            errors.append("missing_field:" + field)
    for index, row in enumerate(rows[:25]):
        if decode_unicode(*arrays["utr50"])[index] != row["utr50"]:
            errors.append("utr50_string")
        if role != "query" and decode_float64(arrays["measured_rl"][2])[index] != float(row["measured_rl"]):
            errors.append("measured_rl")
        sequence = row["utr50"]
        payload = arrays["utr50_onehot"][2]
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        for position, base in enumerate(sequence):
            vector = payload[(index * 50 + position) * 4:(index * 50 + position + 1) * 4]
            expected = [0, 0, 0, 0]
            expected[mapping[base]] = 1
            if list(vector) != expected:
                errors.append("onehot_content")
                break
    return errors


def round3(config, root):
    first = root / config["build_names"][0]
    second = root / config["build_names"][1]
    comparisons = []
    sources = {
        "train": (first / "production/train.npz", Path(config["parent_pack_dir"]) / "learner/train.tsv.gz", "train"),
        "train_scaler": (first / "production/train_scaler.npz", first / "learner/train_scaler_rows.tsv.gz", "train"),
        "dev": (first / "production/dev.npz", Path(config["parent_pack_dir"]) / "learner/dev.tsv.gz", "dev"),
        "calibration": (first / "production/calibration.npz", Path(config["parent_pack_dir"]) / "learner/calibration.tsv.gz", "calibration"),
        "query": (first / "production/query_unlabeled.npz", Path(config["parent_pack_dir"]) / "learner/query_unlabeled.tsv.gz", "query"),
        "designed": (first / "production/designed_outer.npz", first / "evaluator/designed_outer_rows.tsv.gz", "historical_measured_state_outer"),
        "random": (first / "production/random_outer.npz", first / "evaluator/random_outer_rows.tsv.gz", "historical_measured_state_outer"),
    }
    for role, (npz, tsv, kind) in sources.items():
        errors = compare_npz_tsv(npz, tsv, kind)
        comparisons.append({"role": role, "errors": errors, "double_build_hash_match": sha256_file(npz) == sha256_file(second / npz.relative_to(first))})
    manifest_a = json.loads((first / "data_manifest.json").read_text())
    manifest_b = json.loads((second / "data_manifest.json").read_text())
    for role, item in manifest_a["production_files"].items():
        if sha256_file(first / "production" / Path(item["path"]).name) != item["sha256"]:
            comparisons.append({"role": role, "errors": ["manifest_file_hash"]})
        other = manifest_b["production_files"][role]
        if item["sha256"] != other["sha256"]:
            comparisons.append({"role": role, "errors": ["double_build_manifest_hash"]})
    semantic_a = json.loads(json.dumps(manifest_a).replace(str(first), "<BUILD_DIR>"))
    semantic_b = json.loads(json.dumps(manifest_b).replace(str(second), "<BUILD_DIR>"))
    comparisons.append({"file": "data_manifest_semantic", "match": semantic_a == semantic_b})
    result = {
        "round": 3,
        "comparisons": comparisons,
        "pass": all(not item.get("errors") and item.get("double_build_hash_match", item.get("match", True)) for item in comparisons),
    }
    write_json(first / "verification_round3_production.json", result)
    if not result["pass"]:
        raise ValueError("Fresh CNN v2 production verification failed")
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
    outer = json.loads((root / config["build_names"][0] / "outer_manifest.json").read_text())
    authorized = first["pass"] and second["pass"] and third["pass"] and outer["designed_outer_rows"] > 0 and outer["fresh_main_random_unmod_rep1_rows"] > 0
    summary = {
        "status": "FRESH_PROXY_CNN_V2_CONDITIONAL_TRAINING_AUTHORIZED_E1_DISABLED",
        "child_split_version": config["child_split_version"],
        "child_lineage_version": config["child_lineage_version"],
        "lineage_hash": first["lineage_hash"],
        "rounds": {"lineage_source": first["pass"], "permissions_negative": second["pass"], "production_schema_hashes": third["pass"]},
        "formal_training_start_authorized": authorized,
        "E1_enabled": False,
        "outer_metrics_computed": False,
        "proxy_training_performed": False,
        "model_training_performed": False,
        "synthetic": False,
    }
    write_json(root / "verification_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
