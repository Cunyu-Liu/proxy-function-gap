import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_config, make_raw_row_id, open_tsv_gz, sha256_file, sha256_text, write_row


ASSIGNMENT_FIELDS = [
    "model_lineage", "raw_row_id", "raw_row_sha256", "sequence_id", "near_group_id", "allocation_unit_id",
    "v1_role", "role", "pool_class", "evaluation_context_id", "pool_id", "source_file", "library_type",
    "source_library", "designed_flag", "trajectory_id", "round", "replicate_id", "official_model_role",
    "exposure_flags", "historical_evaluation_status", "never_evaluated", "E1_pool_eligible",
    "Paper1_historical_replay_eligible", "split_version",
]
VAULT_FIELDS = [
    "model_lineage", "raw_row_id", "sequence_id", "allocation_unit_id", "role", "evaluation_context_id",
    "source_library", "trajectory_id", "round", "measured_rl", "measured_status", "predicted_mrl",
    "predicted_status", "historical_evaluation_status", "never_evaluated",
]
UNIT_FIELDS = [
    "model_lineage", "allocation_unit_id", "v1_role", "role", "row_count", "source_libraries",
    "library_types", "trajectory_ids", "exposure_flags", "E1_pool_eligible_rows",
    "Paper1_historical_replay_eligible_rows",
]


def rows(path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def context_for(row):
    if row["library_type"] == "designed_library":
        return "eGFP|unmod|designed_single_assay"
    if row["library_type"] == "varying_length_library":
        return "eGFP|unmod|varying_single_assay"
    cds = "mCherry" if "mcherry" in row["source_file"] else "eGFP"
    chemistry = "m1pseudo" if "m1pseudo" in row["source_file"] else "pseudo" if "pseudo" in row["source_file"] else "unmod"
    replicate = row["source_file"].split("_")[-1].split(".")[0]
    return f"{cds}|{chemistry}|{replicate}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    parent = Path(config["parent_split_dir"])
    source = Path(config["split_ready_dir"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluator").mkdir()
    parent_validation = json.loads((parent / "final_validation.json").read_text())
    if not parent_validation["pass"]:
        raise ValueError("Parent v1 validation failed")
    parent_manifest_hash = sha256_file(parent / "split_manifest.json")
    scaler_ids = {make_raw_row_id("GSM3130435", index) for index in range(1, config["scaler_rows"] + 1)}
    parent_strata = {row["allocation_unit_id"]: row["stratum"] for row in rows(parent / "allocation_units.tsv.gz")}
    units = {}
    for assignment, sequence in zip(rows(parent / "role_assignments.tsv.gz"), rows(source / "sequence_index.tsv.gz")):
        if assignment["raw_row_id"] != sequence["raw_row_id"]:
            raise ValueError("Parent/source row mismatch")
        unit_id = assignment["allocation_unit_id"]
        if unit_id not in units:
            units[unit_id] = {
                "v1_role": assignment["role"],
                "stratum": parent_strata[unit_id],
                "source_libraries": set(),
                "library_types": set(),
                "trajectory_ids": set(),
                "official_roles": set(),
                "control": False,
                "scaler": False,
                "random": False,
                "unmod_rep1": False,
                "target_true": False,
                "step_true": False,
                "varying": False,
            }
        unit = units[unit_id]
        library = sequence["source_library"]
        designed = sequence["designed_flag"] == "True"
        unit["source_libraries"].add(library)
        unit["library_types"].add(sequence["library_type"])
        if sequence["trajectory_id"]:
            unit["trajectory_ids"].add(sequence["trajectory_id"])
        unit["official_roles"].add(assignment["official_model_role"])
        if library.startswith(("target_", "step_")) and sequence["designed_flag"] == "False":
            unit["control"] = True
        if assignment["raw_row_id"] in scaler_ids:
            unit["scaler"] = True
        if sequence["library_type"] == "random_library":
            unit["random"] = True
        if sequence["source_file"] == "GSM3130435_egfp_unmod_1.csv.gz":
            unit["unmod_rep1"] = True
        if library.startswith("target_") and designed:
            unit["target_true"] = True
        if library.startswith("step_") and designed:
            unit["step_true"] = True
        if sequence["library_type"] == "varying_length_library":
            unit["varying"] = True
    unit_roles = {"original_evolution": {}, "retrained_evolution": {}}
    for unit_id, unit in units.items():
        flags = []
        if "official_boundary_ambiguous" in unit["official_roles"]:
            flags.append("official_boundary")
        if "official_test_seen" in unit["official_roles"]:
            flags.append("official_test")
        if "official_train_seen" in unit["official_roles"]:
            flags.append("official_train")
        if unit["control"]:
            flags.append("target_or_step_designed_false")
        if unit["scaler"]:
            flags.append("scaler_40k")
        if unit["random"]:
            flags.append("random_retrain_2695_unknown")
        if unit["target_true"]:
            flags.append("target_designed_true")
        if unit["step_true"]:
            flags.append("step_designed_true")
        if unit["varying"]:
            flags.append("unsupported_variable_context")
        unit["exposure_flags"] = flags
        official = unit["official_roles"]
        if unit["control"]:
            original = retrained = "control_nonexact"
        elif "official_boundary_ambiguous" in official:
            original = retrained = "official_boundary_ambiguous"
        elif "official_test_seen" in official:
            original = retrained = "official_seen_test"
        elif unit["scaler"]:
            original = retrained = "scaler_seen_quarantine"
        elif "official_train_seen" in official:
            original = retrained = "train"
        else:
            original = unit["v1_role"]
            retrained = unit["v1_role"]
            if unit["varying"]:
                original = "quarantine_unsupported_context"
            elif unit["v1_role"] == "outer":
                if unit["random"] and unit["unmod_rep1"]:
                    original = "development"
                elif unit["library_types"] == {"designed_library"}:
                    original = "historical_measured_state_outer"
                elif unit["random"] and not unit["unmod_rep1"]:
                    original = "historical_measured_state_outer"
                else:
                    original = "development"
            if unit["random"] or unit["target_true"] or unit["step_true"] or unit["v1_role"] == "outer":
                retrained = "development" if unit["v1_role"] != "outer" else "quarantine_retrain_lineage_unknown"
        unit_roles["original_evolution"][unit_id] = original
        unit_roles["retrained_evolution"][unit_id] = retrained
    role_counts = {model: Counter() for model in config["model_lineages"]}
    strata_counts = {model: defaultdict(Counter) for model in config["model_lineages"]}
    context_counts = {model: Counter() for model in config["model_lineages"]}
    e1_counts = Counter()
    paper1_counts = Counter()
    control_counts = Counter()
    split_hash = hashlib.sha256()
    writers = {}
    handles = []
    try:
        for key, filename, fields in [
            ("assignments", "role_assignments_v2.tsv.gz", ASSIGNMENT_FIELDS),
            ("original_e1", "evaluator/original_evolution_e1_outer_label_vault.tsv.gz", VAULT_FIELDS),
            ("original_outer", "evaluator/original_evolution_historical_outer_label_vault.tsv.gz", VAULT_FIELDS),
            ("retrained_outer", "evaluator/retrained_evolution_outer_label_vault.tsv.gz", VAULT_FIELDS),
            ("controls", "control_roles.tsv.gz", ["model_lineage", "raw_row_id", "sequence_id", "allocation_unit_id", "source_library", "designed_flag", "role", "control_reason"]),
        ]:
            handle, writer = open_tsv_gz(output / filename, fields)
            handles.append(handle)
            writers[key] = writer
        endpoints = rows(source / "endpoint_coverage.tsv.gz")
        for assignment, sequence in zip(rows(parent / "role_assignments.tsv.gz"), rows(source / "sequence_index.tsv.gz")):
            endpoint = next(endpoints)
            if assignment["raw_row_id"] != sequence["raw_row_id"] or endpoint["raw_row_id"] != sequence["raw_row_id"]:
                raise ValueError("Row mismatch")
            unit_id = assignment["allocation_unit_id"]
            unit = units[unit_id]
            context = context_for(sequence)
            for model in config["model_lineages"]:
                role = unit_roles[model][unit_id]
                is_step = sequence["source_library"].startswith("step_")
                historical = role == "historical_measured_state_outer"
                e1_eligible = model == "original_evolution" and historical and not is_step and context == config["original_e1_context"] and sequence["designed_flag"] == "True"
                paper1_eligible = model == "original_evolution" and historical and is_step and sequence["designed_flag"] == "True"
                pool_class = "control_nonexact" if role == "control_nonexact" else "design_pool" if sequence["library_type"] == "designed_library" else "random_context"
                pool_id = "|".join([model, context, unit["stratum"], role])
                exposure = list(unit["exposure_flags"])
                if model == "retrained_evolution" and (unit["random"] or unit["target_true"] or unit["step_true"]):
                    exposure.append("retrain_2695_list_unknown")
                row = {
                    "model_lineage": model,
                    "raw_row_id": assignment["raw_row_id"],
                    "raw_row_sha256": assignment["raw_row_sha256"],
                    "sequence_id": assignment["sequence_id"],
                    "near_group_id": assignment["near_group_id"],
                    "allocation_unit_id": unit_id,
                    "v1_role": assignment["role"],
                    "role": role,
                    "pool_class": pool_class,
                    "evaluation_context_id": context,
                    "pool_id": pool_id,
                    "source_file": sequence["source_file"],
                    "library_type": sequence["library_type"],
                    "source_library": sequence["source_library"],
                    "designed_flag": sequence["designed_flag"],
                    "trajectory_id": sequence["trajectory_id"],
                    "round": sequence["round"],
                    "replicate_id": sequence["replicate_id"],
                    "official_model_role": assignment["official_model_role"],
                    "exposure_flags": exposure,
                    "historical_evaluation_status": "historical_measured_state" if historical else "not_applicable",
                    "never_evaluated": False,
                    "E1_pool_eligible": e1_eligible,
                    "Paper1_historical_replay_eligible": paper1_eligible,
                    "split_version": config["split_version"],
                }
                write_row(writers["assignments"], row)
                canonical = "\t".join([model, assignment["raw_row_id"], unit_id, role, context, str(e1_eligible).lower()]) + "\n"
                split_hash.update(canonical.encode())
                role_counts[model][role] += 1
                strata_counts[model][unit["stratum"]][role] += 1
                context_counts[model][(context, role)] += 1
                e1_counts[model] += int(e1_eligible)
                paper1_counts[model] += int(paper1_eligible)
                if role == "control_nonexact" and sequence["designed_flag"] == "False" and sequence["source_library"].startswith(("target_", "step_")):
                    control_counts[model] += 1
                    write_row(writers["controls"], {
                        "model_lineage": model,
                        "raw_row_id": assignment["raw_row_id"],
                        "sequence_id": assignment["sequence_id"],
                        "allocation_unit_id": unit_id,
                        "source_library": sequence["source_library"],
                        "designed_flag": sequence["designed_flag"],
                        "role": role,
                        "control_reason": "target_or_step_designed_false",
                    })
                if model == "original_evolution" and historical:
                    write_row(writers["original_outer"], {
                        "model_lineage": model,
                        "raw_row_id": assignment["raw_row_id"],
                        "sequence_id": assignment["sequence_id"],
                        "allocation_unit_id": unit_id,
                        "role": role,
                        "evaluation_context_id": context,
                        "source_library": sequence["source_library"],
                        "trajectory_id": sequence["trajectory_id"],
                        "round": sequence["round"],
                        "measured_rl": endpoint["measured_rl"],
                        "measured_status": endpoint["measured_status"],
                        "predicted_mrl": endpoint["predicted_mrl"],
                        "predicted_status": endpoint["predicted_status"],
                        "historical_evaluation_status": "historical_measured_state",
                        "never_evaluated": False,
                    })
                    if e1_eligible:
                        write_row(writers["original_e1"], {
                            "model_lineage": model,
                            "raw_row_id": assignment["raw_row_id"],
                            "sequence_id": assignment["sequence_id"],
                            "allocation_unit_id": unit_id,
                            "role": role,
                            "evaluation_context_id": context,
                            "source_library": sequence["source_library"],
                            "trajectory_id": sequence["trajectory_id"],
                            "round": sequence["round"],
                            "measured_rl": endpoint["measured_rl"],
                            "measured_status": endpoint["measured_status"],
                            "predicted_mrl": endpoint["predicted_mrl"],
                            "predicted_status": endpoint["predicted_status"],
                            "historical_evaluation_status": "historical_measured_state",
                            "never_evaluated": False,
                        })
        if next(endpoints, None) is not None:
            raise ValueError("Extra endpoint rows")
    finally:
        for handle in handles:
            handle.close()
    unit_writers = {}
    unit_handle, unit_writer = open_tsv_gz(output / "model_specific_allocation_units.tsv.gz", UNIT_FIELDS)
    try:
        unit_row_counts = defaultdict(Counter)
        unit_e1_counts = defaultdict(Counter)
        unit_paper1_counts = defaultdict(Counter)
        for row in rows(output / "role_assignments_v2.tsv.gz"):
            key = (row["model_lineage"], row["allocation_unit_id"])
            unit_row_counts[key]["rows"] += 1
            unit_e1_counts[key]["rows"] += int(row["E1_pool_eligible"] == "true")
            unit_paper1_counts[key]["rows"] += int(row["Paper1_historical_replay_eligible"] == "true")
        for model in config["model_lineages"]:
            for unit_id, unit in units.items():
                role = unit_roles[model][unit_id]
                key = (model, unit_id)
                write_row(unit_writer, {
                    "model_lineage": model,
                    "allocation_unit_id": unit_id,
                    "v1_role": unit["v1_role"],
                    "role": role,
                    "row_count": unit_row_counts[key]["rows"],
                    "source_libraries": sorted(unit["source_libraries"]),
                    "library_types": sorted(unit["library_types"]),
                    "trajectory_ids": sorted(unit["trajectory_ids"]),
                    "exposure_flags": unit["exposure_flags"],
                    "E1_pool_eligible_rows": unit_e1_counts[key]["rows"],
                    "Paper1_historical_replay_eligible_rows": unit_paper1_counts[key]["rows"],
                })
    finally:
        unit_handle.close()
    evidence = json.loads(Path(__file__).parent.parent.joinpath("configs/lineage_evidence_v2.json").read_text())
    model_manifests = {}
    for model in config["model_lineages"]:
        original = model == "original_evolution"
        authorized = original and e1_counts[model] > 0
        manifest = {
            "schema_version": "2.0",
            "model_lineage": model,
            "split_version": config["split_version"],
            "role_counts": dict(role_counts[model]),
            "role_counts_by_stratum": {key: dict(value) for key, value in sorted(strata_counts[model].items())},
            "historical_outer_rows": role_counts[model].get("historical_measured_state_outer", 0),
            "E1_pool_eligible_rows": e1_counts[model],
            "Paper1_historical_replay_eligible_rows": paper1_counts[model],
            "control_trigger_rows": control_counts[model],
            "E1_authorized_after_verification": authorized,
            "E1_scope": "original_evolution fixed-pool, eGFP unmod designed single-assay, historical measured-state only" if original else "none; retraining 2695 lineage is incomplete",
            "never_evaluated_outer": False,
            "outer_metrics_computed": False,
            "outer_label_vault": str(output / ("evaluator/original_evolution_historical_outer_label_vault.tsv.gz" if original else "evaluator/retrained_evolution_outer_label_vault.tsv.gz")),
            "lineage_evidence": evidence[model],
            "limitations": [
                "Outer is historical measured-state, not never-evaluated.",
                "Rep2, modified RNA and mCherry are outside the fixed eGFP unmod E1 context.",
                "Missing retraining lineage disables retrained outer." if not original else "Unmodified rep1 random has no original fixed-pool E1 outer because the older evolution-model training subset is not fully recoverable.",
            ],
        }
        path = output / f"{model}_role_manifest.json"
        write_json(path, manifest)
        model_manifests[model] = str(path)
    context_records = []
    for model in config["model_lineages"]:
        for (context, role), count in sorted(context_counts[model].items()):
            context_records.append({
                "model_lineage": model,
                "evaluation_context_id": context,
                "role": role,
                "rows": count,
                "E1_context": context == config["original_e1_context"],
                "replicate_summary_rule": evidence["replicate_summary"],
            })
    with open(output / "replicate_context_summary.tsv", "w", newline="") as handle:
        fields = ["model_lineage", "evaluation_context_id", "role", "rows", "E1_context", "replicate_summary_rule"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(context_records)
    e1_manifest = {
        "schema_version": "2.0",
        "split_version": config["split_version"],
        "original_evolution": {
            "E1_authorized_after_verification": e1_counts["original_evolution"] > 0,
            "eligible_rows": e1_counts["original_evolution"],
            "scope": "Fixed-pool eGFP unmod designed single-assay only; step is Paper1 historical replay, not E1.",
            "outer_vault": str(output / "evaluator/original_evolution_e1_outer_label_vault.tsv.gz"),
        },
        "retrained_evolution": {
            "E1_authorized_after_verification": False,
            "eligible_rows": 0,
            "scope": "NO-GO until the 2695 retraining sequence list and all linked components are resolved.",
            "outer_vault": str(output / "evaluator/retrained_evolution_outer_label_vault.tsv.gz"),
        },
        "outer_metrics_computed": False,
        "E1_enabled": False,
    }
    write_json(output / "e1_pool_manifest.json", e1_manifest)
    manifest = {
        "schema_version": "2.0",
        "split_version": config["split_version"],
        "parent_split_version": config["parent_split_version"],
        "parent_split_manifest_sha256": parent_manifest_hash,
        "split_hash": split_hash.hexdigest(),
        "split_hash_method": "SHA256 of canonical model/raw_row_id/allocation_unit_id/role/context/E1_pool_eligible lines",
        "data_manifest_sha256": sha256_file(Path(config["input_manifest"])),
        "split_ready_checksums_sha256": sha256_file(Path(config["split_ready_checksums"])),
        "config_sha256": sha256_file(Path(args.config)),
        "lineage_evidence_sha256": sha256_file(Path(__file__).parent.parent / "configs/lineage_evidence_v2.json"),
        "rows_per_model": 2753885,
        "assignment_rows_total": 5507770,
        "allocation_units_per_model": len(units),
        "role_counts": {model: dict(role_counts[model]) for model in config["model_lineages"]},
        "model_manifests": model_manifests,
        "e1_pool_manifest": str(output / "e1_pool_manifest.json"),
        "replicate_context_summary": str(output / "replicate_context_summary.tsv"),
        "control_trigger_rows_per_model": dict(control_counts),
        "scaler_rows": config["scaler_rows"],
        "retraining_2695_list_status": config["retraining_sequence_list_status"],
        "outer_metrics_computed": False,
        "E1_enabled": False,
        "file_hashes": {
            name: sha256_file(output / name)
            for name in [
                "role_assignments_v2.tsv.gz", "model_specific_allocation_units.tsv.gz", "control_roles.tsv.gz",
                "original_evolution_role_manifest.json", "retrained_evolution_role_manifest.json",
                "e1_pool_manifest.json", "replicate_context_summary.tsv",
                "evaluator/original_evolution_e1_outer_label_vault.tsv.gz",
                "evaluator/original_evolution_historical_outer_label_vault.tsv.gz",
                "evaluator/retrained_evolution_outer_label_vault.tsv.gz",
            ]
        },
    }
    write_json(output / "split_manifest_v2.json", manifest)
    print(json.dumps({"split_version": config["split_version"], "split_hash": manifest["split_hash"], "role_counts": manifest["role_counts"], "E1_counts": dict(e1_counts), "Paper1_counts": dict(paper1_counts), "control_counts": dict(control_counts), "output": str(output)}))


if __name__ == "__main__":
    main()
