import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_config, sha256_file
from .lineage_protocol import ModelSpecificEvaluator


def rows(path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def round1(config, output):
    models = config["model_lineages"]
    counts = {model: Counter() for model in models}
    group_roles = {model: defaultdict(set) for model in models}
    trajectory_roles = {model: defaultdict(set) for model in models}
    unit_roles = {model: defaultdict(set) for model in models}
    ids = {model: set() for model in models}
    duplicate_ids = Counter()
    violations = []
    context_counts = {model: Counter() for model in models}
    for row in rows(output / "role_assignments_v2.tsv.gz"):
        model = row["model_lineage"]
        role = row["role"]
        counts[model][role] += 1
        group_roles[model][row["near_group_id"]].add(role)
        unit_roles[model][row["allocation_unit_id"]].add(role)
        if row["trajectory_id"]:
            trajectory_roles[model][row["trajectory_id"]].add(role)
        if row["raw_row_id"] in ids[model]:
            duplicate_ids[model] += 1
        ids[model].add(row["raw_row_id"])
        context_counts[model][(row["evaluation_context_id"], role)] += 1
        library = row["source_library"]
        designed_false_control = library.startswith(("target_", "step_")) and row["designed_flag"] == "False"
        if designed_false_control and role != "control_nonexact":
            violations.append("target_or_step_designed_false_outside_control")
        if role == "historical_measured_state_outer":
            if model == "retrained_evolution":
                violations.append("retrained_outer")
            if row["historical_evaluation_status"] != "historical_measured_state" or row["never_evaluated"] != "false":
                violations.append("outer_semantics")
            if row["source_file"] == "GSM3130435_egfp_unmod_1.csv.gz":
                violations.append("original_unmod_rep1_random_outer")
            if row["official_model_role"] != "not_applicable":
                violations.append("official_seen_outer")
        if row["E1_pool_eligible"] == "true":
            if model != "original_evolution" or role != "historical_measured_state_outer" or row["designed_flag"] != "True" or library.startswith("step_") or row["evaluation_context_id"] != config["original_e1_context"]:
                violations.append("invalid_E1_pool")
        if row["Paper1_historical_replay_eligible"] == "true":
            if model != "original_evolution" or role != "historical_measured_state_outer" or row["designed_flag"] != "True" or not library.startswith("step_"):
                violations.append("invalid_Paper1_pool")
        if row["evaluation_context_id"] != config["original_e1_context"] and row["E1_pool_eligible"] == "true":
            violations.append("context_mixing")
        if model == "retrained_evolution" and (row["E1_pool_eligible"] == "true" or row["Paper1_historical_replay_eligible"] == "true"):
            violations.append("retrained_eligible_pool")
        if "scaler_40k" in row["exposure_flags"] and role == "historical_measured_state_outer":
            violations.append("scaler_outer")
    for model in models:
        for name, groups in [("near", group_roles[model]), ("trajectory", trajectory_roles[model]), ("unit", unit_roles[model])]:
            if any(len(roles) > 1 for roles in groups.values()):
                violations.append(name + "_crosses_roles")
    negative = []
    evaluator = ModelSpecificEvaluator(config)
    negative.append({"test": "retrained_E1_permission", "pass": not evaluator.outer_permission("retrained_evolution", config["original_e1_context"])})
    negative.append({"test": "original_rep2_E1_permission", "pass": not evaluator.outer_permission("original_evolution", "eGFP|unmod|2")})
    negative.append({"test": "original_step_not_E1", "pass": not evaluator.outer_permission("original_evolution", config["original_e1_context"], is_step=True)})
    try:
        evaluator.evaluate_outer()
    except PermissionError:
        negative.append({"test": "outer_metrics_disabled", "pass": True})
    else:
        negative.append({"test": "outer_metrics_disabled", "pass": False})
    original_manifest = json.loads((output / "original_evolution_role_manifest.json").read_text())
    retrained_manifest = json.loads((output / "retrained_evolution_role_manifest.json").read_text())
    result = {
        "round": 1,
        "role_counts": {model: dict(counts[model]) for model in models},
        "duplicate_raw_row_ids_per_model": dict(duplicate_ids),
        "violations": sorted(set(violations)),
        "negative_tests": negative,
        "original_E1_authorized_after_verification": original_manifest["E1_authorized_after_verification"],
        "retrained_E1_authorized_after_verification": retrained_manifest["E1_authorized_after_verification"],
        "pass": not violations and all(test["pass"] for test in negative) and original_manifest["E1_authorized_after_verification"] and not retrained_manifest["E1_authorized_after_verification"] and not any(duplicate_ids.values()),
    }
    result["context_counts"] = {model: {f"{context}|{role}": count for (context, role), count in sorted(context_counts[model].items())} for model in models}
    write_json(output / "verification_v2_round1_lineage.json", result)
    if not result["pass"]:
        raise ValueError("V2 lineage verification failed")
    return result


def round2(config, output):
    manifest = json.loads((output / "split_manifest_v2.json").read_text())
    counts = {model: Counter() for model in config["model_lineages"]}
    e1_count = 0
    original_outer_count = 0
    control_violations = []
    for row in rows(output / "role_assignments_v2.tsv.gz"):
        model = row["model_lineage"]
        if row["E1_pool_eligible"] == "true":
            e1_count += 1
        if row["role"] == "historical_measured_state_outer":
            original_outer_count += 1
    original_outer = 0
    for row in rows(output / "evaluator/original_evolution_historical_outer_label_vault.tsv.gz"):
        original_outer += 1
        if row["historical_evaluation_status"] != "historical_measured_state" or row["never_evaluated"] != "false":
            control_violations.append("outer_vault_semantics")
    original_e1 = 0
    for row in rows(output / "evaluator/original_evolution_e1_outer_label_vault.tsv.gz"):
        original_e1 += 1
        if row["evaluation_context_id"] != config["original_e1_context"] or row["source_library"].startswith("step_"):
            control_violations.append("E1_vault_scope")
    retrained_outer = sum(1 for _ in rows(output / "evaluator/retrained_evolution_outer_label_vault.tsv.gz"))
    if retrained_outer:
        control_violations.append("retrained_outer_vault_nonempty")
    if original_e1 != e1_count or original_outer != original_outer_count:
        control_violations.append("outer_vault_row_count")
    context_summary = list(csv.DictReader(open(output / "replicate_context_summary.tsv"), delimiter="\t"))
    expected_context = Counter()
    for row in rows(output / "role_assignments_v2.tsv.gz"):
        expected_context[(row["model_lineage"], row["evaluation_context_id"], row["role"])] += 1
    for record in context_summary:
        key = (record["model_lineage"], record["evaluation_context_id"], record["role"])
        if int(record["rows"]) != expected_context[key]:
            control_violations.append("replicate_context_count")
        if record["E1_context"].lower() == "true" and record["evaluation_context_id"] != config["original_e1_context"]:
            control_violations.append("replicate_context_mixing")
    e1_manifest = json.loads((output / "e1_pool_manifest.json").read_text())
    if not e1_manifest["original_evolution"]["E1_authorized_after_verification"] or e1_manifest["retrained_evolution"]["E1_authorized_after_verification"]:
        control_violations.append("E1_authorization_manifest")
    result = {
        "round": 2,
        "original_E1_vault_rows": original_e1,
        "original_historical_outer_vault_rows": original_outer,
        "retrained_outer_vault_rows": retrained_outer,
        "violations": sorted(set(control_violations)),
        "outer_metrics_computed": False,
        "outer_labels_returned": False,
        "pass": not control_violations,
    }
    write_json(output / "verification_v2_round2_permissions.json", result)
    if not result["pass"]:
        raise ValueError("V2 permission/context verification failed")
    return result


def round3(config, output):
    parent = Path(config["parent_split_dir"])
    manifest = json.loads((output / "split_manifest_v2.json").read_text())
    digest = hashlib.sha256()
    mismatches = []
    counts = Counter()
    parent_iters = {model: rows(parent / "role_assignments.tsv.gz") for model in config["model_lineages"]}
    for row in rows(output / "role_assignments_v2.tsv.gz"):
            model = row["model_lineage"]
            parent_row = next(parent_iters[model])
            counts[model] += 1
            for key in ["raw_row_id", "sequence_id", "near_group_id", "allocation_unit_id"]:
                if row[key] != parent_row[key] and len(mismatches) < 20:
                    mismatches.append({"model": model, "field": key})
            if row["raw_row_sha256"] != parent_row["raw_row_sha256"]:
                mismatches.append({"model": model, "field": "raw_row_sha256"})
            if row["v1_role"] != parent_row["role"]:
                mismatches.append({"model": model, "field": "v1_role"})
            canonical = "\t".join([model, row["raw_row_id"], row["allocation_unit_id"], row["role"], row["evaluation_context_id"], row["E1_pool_eligible"]]) + "\n"
            digest.update(canonical.encode())
    for model in config["model_lineages"]:
        if next(parent_iters[model], None) is not None:
            mismatches.append({"model": model, "field": "parent_extra_rows"})
    if digest.hexdigest() != manifest["split_hash"]:
        mismatches.append({"field": "split_hash"})
    if sha256_file(parent / "split_manifest.json") != manifest["parent_split_manifest_sha256"]:
        mismatches.append({"field": "parent_immutability"})
    if sha256_file(Path(config["input_manifest"])) != manifest["data_manifest_sha256"]:
        mismatches.append({"field": "data_manifest_hash"})
    if sha256_file(Path(config["split_ready_checksums"])) != manifest["split_ready_checksums_sha256"]:
        mismatches.append({"field": "split_ready_checksums"})
    for name, expected in manifest["file_hashes"].items():
        if sha256_file(output / name) != expected:
            mismatches.append({"field": "file_hash", "name": name})
    unit_counts = Counter()
    for row in rows(output / "model_specific_allocation_units.tsv.gz"):
        unit_counts[row["model_lineage"]] += 1
    control_counts = Counter()
    for row in rows(output / "control_roles.tsv.gz"):
        control_counts[row["model_lineage"]] += 1
    if any(control_counts[model] != 16528 for model in config["model_lineages"]):
        mismatches.append({"field": "control_trigger_count", "counts": dict(control_counts)})
    result = {
        "round": 3,
        "assignment_rows_per_model": dict(counts),
        "assignment_rows_total": sum(counts.values()),
        "allocation_units_per_model": dict(unit_counts),
        "control_trigger_rows_per_model": dict(control_counts),
        "split_hash": digest.hexdigest(),
        "mismatches": mismatches,
        "pass": counts == Counter({model: 2753885 for model in config["model_lineages"]}) and unit_counts == Counter({model: 776514 for model in config["model_lineages"]}) and not mismatches,
    }
    write_json(output / "verification_v2_round3_rows.json", result)
    if not result["pass"]:
        raise ValueError("V2 row/hash verification failed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["output_dir"])
    for name in ["verification_v2_round1_lineage.json", "verification_v2_round2_permissions.json", "verification_v2_round3_rows.json", "verification_summary_v2.json", "e1_authorization.json"]:
        if (output / name).exists():
            raise FileExistsError(output / name)
    first = round1(config, output)
    second = round2(config, output)
    third = round3(config, output)
    original_authorized = first["original_E1_authorized_after_verification"] and second["pass"] and third["pass"]
    original_manifest = json.loads((output / "original_evolution_role_manifest.json").read_text())
    paper1_authorized = original_authorized and original_manifest["Paper1_historical_replay_eligible_rows"] > 0
    retrained_authorized = False
    authorization = {
        "split_version": config["split_version"],
        "original_evolution": {
            "E1_authorized": original_authorized,
            "scope": "Fixed-pool eGFP unmod designed single-assay, original_evolution, historical measured-state outer only. Step is excluded from E1 and only allowed for complete-trajectory Paper1 historical replay.",
            "never_evaluated": False,
            "Paper1_historical_replay_authorized": paper1_authorized,
            "Paper1_scope": "No eligible complete step trajectories remain after training/scaler/control exclusions; do not claim a trajectory outer." if not paper1_authorized else "Complete-trajectory historical measured-state replay only",
        },
        "retrained_evolution": {
            "E1_authorized": retrained_authorized,
            "scope": "NO-GO: the 2695 retraining sequence list and linked lineage are unresolved. Random, target/step designed=True and all possible outer remain development/quarantine.",
        },
        "outer_metrics_computed": False,
        "outer_labels_returned": False,
        "E1_evaluation_performed": False,
    }
    write_json(output / "e1_authorization.json", authorization)
    summary = {
        "status": "MODEL_SPECIFIC_SPLIT_V2_VERIFIED_NO_OUTER_METRICS",
        "split_version": config["split_version"],
        "split_hash": third["split_hash"],
        "rounds": {"lineage_negative": first["pass"], "permissions_context": second["pass"], "row_conservation_hashes": third["pass"]},
        "original_E1_authorized": original_authorized,
        "original_Paper1_historical_replay_authorized": paper1_authorized,
        "retrained_E1_authorized": retrained_authorized,
        "authorization": authorization,
        "outer_metrics_computed": False,
        "outer_labels_returned": False,
        "proxy_training_performed": False,
    }
    write_json(output / "verification_summary_v2.json", summary)
    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "split_artifact_checksums.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(output / "split_artifact_checksums.json", files)
    print(json.dumps({"status": summary["status"], "rounds": summary["rounds"], "original_E1_authorized": original_authorized, "retrained_E1_authorized": retrained_authorized}))


if __name__ == "__main__":
    main()
