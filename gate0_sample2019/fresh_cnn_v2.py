import argparse
import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import load_config, sha256_file, sha256_text
from .fresh_cnn_common import role_intersections, row_list_hash
from .fresh_proxy_common import open_deterministic_tsv_gz, read_tsv, write_json, write_row
from .npy_io import bool_array, float64_array, one_hot, unicode_array, write_npz


OUTER_FIELDS = [
    "raw_row_id", "sequence_id", "utr50", "measured_rl", "role", "allocation_unit_id", "near_group_id",
    "trajectory_id", "source_library", "context", "outer_stratum", "E1_pool_eligible",
    "historical_evaluation_status", "never_evaluated", "source_file", "source_file_sha256", "raw_row_sha256",
    "child_split_version",
]
TRAIN_FIELDS = ["raw_row_id", "sequence_id", "utr50", "measured_rl", "role", "allocation_unit_id", "context", "source_file_sha256", "raw_row_sha256"]


def build(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "learner").mkdir()
    (output / "evaluator").mkdir()
    (output / "production").mkdir()
    pack = Path(config["parent_pack_dir"])
    pack_manifest = json.loads((pack.parent / "pack_manifest.json").read_text())
    if pack_manifest["git_commit"] != config["parent_pack_commit"]:
        raise ValueError("Parent pack commit mismatch")
    role_rows = {}
    for role in ["train", "dev", "calibration", "query"]:
        filename = "query_unlabeled.tsv.gz" if role == "query" else role + ".tsv.gz"
        role_rows[role] = {row["row_id"]: row for row in read_tsv(pack / "learner" / filename)}
    v1 = {row["raw_row_id"]: row for row in read_tsv(Path(config["parent_split_v1_dir"]) / "role_assignments.tsv.gz")}
    original = {row["raw_row_id"]: row for row in read_tsv(Path(config["parent_split_v2_dir"]) / "role_assignments_v2.tsv.gz") if row["model_lineage"] == "original_evolution"}
    pack_sequences = {role: set(rows) for role, rows in role_rows.items()}
    pack_near = {row["near_group_id"] for rows in role_rows.values() for row in rows.values()}
    pack_units = {row["allocation_unit_id"] for rows in role_rows.values() for row in rows.values()}
    pack_trajectories = {row["trajectory_id"] for rows in role_rows.values() for row in rows.values() if row["trajectory_id"]}
    designed = {}
    excluded = Counter()
    for raw_id, row in original.items():
        if row["E1_pool_eligible"] != "true":
            continue
        if row["source_library"].startswith("target_"):
            excluded["designed_target_retrained_lineage"] += 1
            continue
        if row["source_library"].startswith("step_"):
            excluded["designed_step_lineage"] += 1
            continue
        designed[raw_id] = row
    if len(designed) + excluded["designed_target_retrained_lineage"] + excluded["designed_step_lineage"] != config["designed_candidate_rows"]:
        raise ValueError("Designed candidate accounting failed")
    random = {}
    for raw_id, row in v1.items():
        if row["role"] == "outer" and row["library_type"] == "random_library":
            random[raw_id] = original[raw_id]
    def outer_stratum(row):
        if row["source_file"] == "GSM3130435_egfp_unmod_1.csv.gz":
            return "random_main|eGFP_unmod_rep1"
        return "random_external|" + row["evaluation_context_id"]
    designed_near = {row["near_group_id"] for row in designed.values()}
    designed_units = {row["allocation_unit_id"] for row in designed.values()}
    designed_trajectories = {row["trajectory_id"] for row in designed.values() if row["trajectory_id"]}
    outer = {}
    for raw_id, row in random.items():
        near = row["near_group_id"]
        unit = row["allocation_unit_id"]
        trajectory = row["trajectory_id"]
        if near in pack_near or unit in pack_units or trajectory in pack_trajectories:
            excluded["random_pack_overlap"] += 1
            continue
        if near in designed_near or unit in designed_units or trajectory in designed_trajectories:
            excluded["random_designed_overlap"] += 1
            continue
        if row["source_file"] == "GSM3130435_egfp_unmod_1.csv.gz" and row["role"] in {"official_seen_test", "official_boundary_ambiguous", "scaler_seen_quarantine", "control_nonexact", "quarantine_unsupported_context"}:
            excluded["random_official_control_quarantine"] += 1
            continue
        outer[raw_id] = {
            **row,
            "outer_stratum": outer_stratum(row),
            "E1_pool_eligible": row["source_file"] == "GSM3130435_egfp_unmod_1.csv.gz",
            "historical_evaluation_status": "historical_measured_state",
            "never_evaluated": False,
        }
    for raw_id, row in designed.items():
        if row["near_group_id"] in pack_near or row["allocation_unit_id"] in pack_units or row["trajectory_id"] in pack_trajectories:
            excluded["designed_pack_overlap"] += 1
            continue
        outer[raw_id] = {
            **row,
            "outer_stratum": "designed|eGFP_unmod",
            "E1_pool_eligible": True,
            "historical_evaluation_status": "historical_measured_state",
            "never_evaluated": False,
        }
    labels = {}
    for row in read_tsv(Path(config["parent_split_v1_dir"]) / "evaluator/outer_label_vault.tsv.gz"):
        if row["raw_row_id"] in outer:
            labels[row["raw_row_id"]] = row
    if set(labels) != set(outer):
        raise ValueError("Outer label coverage mismatch")
    source_manifest = json.loads(Path(config["input_manifest"]).read_text())
    source_hashes = {item["name"]: item["sha256"] for item in source_manifest["files"]}
    handles = []
    writers = {}
    try:
        for name, fields in [("learner/train_scaler_rows.tsv.gz", TRAIN_FIELDS), ("evaluator/designed_outer_rows.tsv.gz", OUTER_FIELDS), ("evaluator/random_outer_rows.tsv.gz", OUTER_FIELDS)]:
            handle, writer = open_deterministic_tsv_gz(output / name, fields)
            handles.append(handle)
            writers[name] = writer
        for raw_id in sorted(role_rows["train"]):
            row = role_rows["train"][raw_id]
            write_row(writers["learner/train_scaler_rows.tsv.gz"], {
                "raw_row_id": raw_id,
                "sequence_id": row["sequence_id"],
                "utr50": row["utr50"],
                "measured_rl": row["measured_rl"],
                "role": "train",
                "allocation_unit_id": row["allocation_unit_id"],
                "context": row["context"],
                "source_file_sha256": row["source_file_sha256"],
                "raw_row_sha256": row["raw_row_sha256"],
            })
        for raw_id in sorted(outer):
            assignment = dict(outer[raw_id])
            label = labels[raw_id]
            assignment["utr50"] = label["normalized_sequence"]
            assignment["context"] = assignment["evaluation_context_id"]
            assignment["source_file_sha256"] = source_hashes[assignment["source_file"]]
            assignment["measured_rl"] = label["measured_rl"]
            outer[raw_id] = assignment
            target = "evaluator/designed_outer_rows.tsv.gz" if assignment["outer_stratum"].startswith("designed|") else "evaluator/random_outer_rows.tsv.gz"
            write_row(writers[target], {
                "raw_row_id": raw_id,
                "sequence_id": assignment["sequence_id"],
                "utr50": label["normalized_sequence"],
                "measured_rl": label["measured_rl"],
                "role": "historical_measured_state_outer",
                "allocation_unit_id": assignment["allocation_unit_id"],
                "near_group_id": assignment["near_group_id"],
                "trajectory_id": assignment["trajectory_id"],
                "source_library": assignment["source_library"],
                "context": assignment["evaluation_context_id"],
                "outer_stratum": assignment["outer_stratum"],
                "E1_pool_eligible": assignment["E1_pool_eligible"],
                "historical_evaluation_status": "historical_measured_state",
                "never_evaluated": False,
                "source_file": assignment["source_file"],
                "source_file_sha256": source_hashes[assignment["source_file"]],
                "raw_row_sha256": assignment["raw_row_sha256"],
                "child_split_version": config["child_split_version"],
            })
    finally:
        for handle in handles:
            handle.close()
    lineage_values = []
    for role, rows in role_rows.items():
        for raw_id in sorted(rows):
            row = rows[raw_id]
            lineage_values.append("\t".join([role, raw_id, row["allocation_unit_id"], row["context"]]))
    for raw_id in sorted(outer):
        row = outer[raw_id]
        lineage_values.append("\t".join([raw_id, row["allocation_unit_id"], row["outer_stratum"], row["evaluation_context_id"]]))
    lineage_hash = hashlib.sha256("\n".join(lineage_values).encode()).hexdigest()
    model_template = {
        "child_lineage_version": config["child_lineage_version"],
        "lineage_hash": lineage_hash,
        "initialization": config["fresh_initialization"],
        "official_weights": False,
        "distillation": False,
        "training_status": "NOT_STARTED",
        "model_hash": None,
        "epoch": None,
        "convergence": None,
        "V_calibration": None,
        "Q_calibration": None,
        "configuration_frozen": False,
        "E1_enabled": False,
        "template_rule": "Do not write model hash, epoch, convergence, calibration or pass status before evidence exists.",
        "historical_evaluation_status": "historical_measured_state",
        "never_evaluated": False,
        "independence_rule": "Original/retrained manifests are unchanged. Fresh has no official weights and independently re-reviews random outer, including unmodified rep1.",
    }
    write_json(output / "model_lineage_template.json", model_template)
    learner_allowlist = {
        "child_split_version": config["child_split_version"],
        "feature_label_sources": {
            role: {
                "path": str(pack / "learner" / ("query_unlabeled.tsv.gz" if role == "query" else role + ".tsv.gz")),
                "sha256": sha256_file(pack / "learner" / ("query_unlabeled.tsv.gz" if role == "query" else role + ".tsv.gz")),
                "query_unlabeled": role == "query",
            }
            for role in role_rows
        },
        "train_scaler_rows": {"path": str(output / "learner/train_scaler_rows.tsv.gz"), "sha256": sha256_file(output / "learner/train_scaler_rows.tsv.gz"), "hash": row_list_hash(role_rows["train"])},
        "outer_access": False,
        "official_seen_control_quarantine_access": False,
        "E1_enabled": False,
    }
    write_json(output / "learner_allowlist.json", learner_allowlist)
    outer_counts = Counter(row["outer_stratum"] for row in outer.values())
    final_random_ids = {raw_id for raw_id, row in outer.items() if row["outer_stratum"].startswith("random_")}
    outer_manifest = {
        "child_split_version": config["child_split_version"],
        "child_lineage_version": config["child_lineage_version"],
        "lineage_hash": lineage_hash,
        "designed_candidate_rows_reviewed": config["designed_candidate_rows"],
        "designed_outer_rows": len(designed),
        "random_outer_rows": len(final_random_ids),
        "fresh_main_random_unmod_rep1_rows": outer_counts.get("random_main|eGFP_unmod_rep1", 0),
        "outer_strata": dict(outer_counts),
        "outer_row_hashes": {
            "designed": row_list_hash(designed),
            "random": row_list_hash(final_random_ids),
            "fresh_main_random_unmod_rep1": row_list_hash({raw_id for raw_id, row in outer.items() if row["outer_stratum"] == "random_main|eGFP_unmod_rep1"}),
        },
        "excluded": dict(excluded),
        "designed_vault": str(output / "evaluator/designed_outer_rows.tsv.gz"),
        "random_vault": str(output / "evaluator/random_outer_rows.tsv.gz"),
        "historical_evaluation_status": "historical_measured_state",
        "never_evaluated": False,
        "E1_enabled": False,
        "outer_metrics_enabled": False,
        "independence_rule": "Fresh independently re-reviews v1 random outer; original development and retrained restrictions are not inherited as blanket random exclusions.",
    }
    write_json(output / "outer_manifest.json", outer_manifest)
    production_files = {}
    for role, rows in role_rows.items():
        arrays = arrays_for_rows(config, rows, role)
        path = output / "production" / f"{role}{'_unlabeled' if role == 'query' else ''}.npz"
        write_npz(path, arrays)
        production_files[role] = {"path": str(path), "sha256": sha256_file(path)}
    write_npz(output / "production/train_scaler.npz", arrays_for_rows(config, role_rows["train"], "train"))
    production_files["train_scaler"] = {"path": str(output / "production/train_scaler.npz"), "sha256": sha256_file(output / "production/train_scaler.npz")}
    for role, name in [("designed", "designed_outer"), ("random", "random_outer")]:
        rows = {raw_id: outer[raw_id] for raw_id in outer if outer[raw_id]["outer_stratum"].startswith(role + "|") or (role == "random" and outer[raw_id]["outer_stratum"].startswith("random_"))}
        path = output / "production" / f"{name}.npz"
        write_npz(path, arrays_for_rows(config, rows, "historical_measured_state_outer", outer=True))
        production_files[role] = {"path": str(path), "sha256": sha256_file(path)}
    data_manifest = {
        "schema_version": "1.0",
        "pack_version": config["child_split_version"],
        "normalization_version": config["normalization_version"],
        "synthetic": False,
        "train_statistics": config["train_statistics"],
        "canonical_order": "NPZ row_id order is byte-for-byte identical to source TSV row order",
        "source_tsv": {
            "train": str(pack / "learner/train.tsv.gz"),
            "train_scaler": str(output / "learner/train_scaler_rows.tsv.gz"),
            "dev": str(pack / "learner/dev.tsv.gz"),
            "calibration": str(pack / "learner/calibration.tsv.gz"),
            "query": str(pack / "learner/query_unlabeled.tsv.gz"),
            "designed_outer": str(output / "evaluator/designed_outer_rows.tsv.gz"),
            "random_outer": str(output / "evaluator/random_outer_rows.tsv.gz"),
        },
        "source_tsv_hashes": {
            "train": sha256_file(pack / "learner/train.tsv.gz"),
            "train_scaler": sha256_file(output / "learner/train_scaler_rows.tsv.gz"),
            "dev": sha256_file(pack / "learner/dev.tsv.gz"),
            "calibration": sha256_file(pack / "learner/calibration.tsv.gz"),
            "query": sha256_file(pack / "learner/query_unlabeled.tsv.gz"),
            "designed_outer": sha256_file(output / "evaluator/designed_outer_rows.tsv.gz"),
            "random_outer": sha256_file(output / "evaluator/random_outer_rows.tsv.gz"),
        },
        "canonical_row_id_hashes": {
            role: hashlib.sha256(("\n".join(rows) + ("\n" if rows else "")).encode()).hexdigest()
            for role, rows in role_rows.items()
        } | {
            "designed_outer": hashlib.sha256(("\n".join(sorted(designed)) + "\n").encode()).hexdigest(),
            "random_outer": hashlib.sha256(("\n".join(sorted(random)) + "\n").encode()).hexdigest(),
        },
        "production_files": production_files,
        "pack_manifest_sha256": sha256_file(pack.parent / "pack_manifest.json"),
        "split_v1_manifest_sha256": sha256_file(Path(config["parent_split_v1_dir"]) / "split_manifest.json"),
        "split_v2_manifest_sha256": sha256_file(Path(config["parent_split_v2_dir"]) / "split_manifest_v2.json"),
        "learner_allowlist_sha256": sha256_file(output / "learner_allowlist.json"),
        "source_file_hashes": source_hashes,
        "outer_learner_package_created": False,
        "outer_labels_access": "Evaluator only",
        "E1_enabled": False,
    }
    write_json(output / "data_manifest.json", data_manifest)
    candidate = {
        "schema_version": "2.0",
        "child_split_version": config["child_split_version"],
        "child_lineage_version": config["child_lineage_version"],
        "lineage_hash": lineage_hash,
        "parent_pack_commit": config["parent_pack_commit"],
        "parent_pack_manifest_sha256": sha256_file(pack.parent / "pack_manifest.json"),
        "parent_split_v1_manifest_sha256": sha256_file(Path(config["parent_split_v1_dir"]) / "split_manifest.json"),
        "parent_split_v2_manifest_sha256": sha256_file(Path(config["parent_split_v2_dir"]) / "split_manifest_v2.json"),
        "data_manifest_sha256": sha256_file(Path(config["input_manifest"])),
        "build_dir": str(output),
        "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "train_scaler_rows": len(role_rows["train"]),
        "train_scaler_row_hash": row_list_hash(role_rows["train"]),
        "dev_rows": len(role_rows["dev"]),
        "calibration_rows": len(role_rows["calibration"]),
        "query_rows": len(role_rows["query"]),
        "outer_manifest": str(output / "outer_manifest.json"),
        "model_lineage_template": str(output / "model_lineage_template.json"),
        "learner_allowlist": str(output / "learner_allowlist.json"),
        "production_data_manifest": str(output / "data_manifest.json"),
        "E1_enabled": False,
        "formal_training_start_authorized_pending_verification": True,
        "formal_training_start_authorized": False,
        "official_weights": False,
        "distillation": False,
        "outer_metrics_computed": False,
        "synthetic": False,
    }
    write_json(output / "candidate_manifest.json", candidate)
    print(json.dumps({"build_dir": str(output), "train_scaler_rows": len(role_rows["train"]), "designed_outer_rows": len(designed), "random_outer_rows": len(final_random_ids), "fresh_main_random_unmod_rep1_rows": outer_manifest["fresh_main_random_unmod_rep1_rows"], "excluded": dict(excluded), "lineage_hash": lineage_hash}))


def arrays_for_rows(config, rows, role, outer=False):
    ordered = list(rows.values())
    sequences = [row["utr50"] if "utr50" in row else row["normalized_sequence"] for row in ordered]
    arrays = {
        "row_id": unicode_array([row["raw_row_id"] if "raw_row_id" in row else row["row_id"] for row in ordered]),
        "sequence_id": unicode_array([row["sequence_id"] for row in ordered]),
        "utr50": unicode_array(sequences),
        "utr50_onehot": one_hot(sequences),
        "role": unicode_array([role for _ in ordered]),
        "allocation_unit_id": unicode_array([row["allocation_unit_id"] for row in ordered]),
        "context": unicode_array([row["context"] if "context" in row else row["evaluation_context_id"] for row in ordered]),
        "source_file_sha256": unicode_array([row["source_file_sha256"] for row in ordered]),
        "raw_row_sha256": unicode_array([row["raw_row_sha256"] for row in ordered]),
        "sequence_hash": unicode_array([sha256_text(sequence) for sequence in sequences]),
        "normalization_version": unicode_array([config["normalization_version"] for _ in ordered]),
        "split_version": unicode_array([config["child_split_version"] for _ in ordered]),
        "synthetic": bool_array(False),
    }
    if role != "query":
        arrays["measured_rl"] = float64_array([row["measured_rl"] for row in ordered])
    for field in ["near_group_id", "trajectory_id", "replicate_id", "outer_stratum", "E1_pool_eligible", "historical_evaluation_status", "measured_status"]:
        if field in ordered[0] if ordered else False:
            arrays[field] = unicode_array([str(row[field]) for row in ordered])
    return arrays


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
