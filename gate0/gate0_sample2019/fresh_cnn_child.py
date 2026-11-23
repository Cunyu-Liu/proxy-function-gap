import argparse
import datetime
import json
from collections import Counter
from pathlib import Path

from .common import load_config, sha256_file
from .fresh_cnn_common import canonical_hash, role_intersections, row_list_hash
from .fresh_proxy_common import open_deterministic_tsv_gz, read_tsv, write_json, write_row


OUTER_FIELDS = [
    "raw_row_id", "sequence_id", "utr50", "measured_rl", "role", "allocation_unit_id", "near_group_id",
    "trajectory_id", "source_library", "context", "outer_stratum", "E1_pool_eligible", "historical_evaluation_status",
    "never_evaluated", "source_file", "source_file_sha256", "raw_row_sha256", "child_split_version",
]
TRAIN_FIELDS = ["raw_row_id", "sequence_id", "utr50", "measured_rl", "role", "allocation_unit_id", "context", "source_file_sha256", "raw_row_sha256"]


def build(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "learner").mkdir()
    (output / "evaluator").mkdir()
    pack = Path(config["parent_pack_dir"])
    pack_manifest = json.loads((pack.parent / "pack_manifest.json").read_text())
    if pack_manifest["git_commit"] != config["parent_pack_commit"]:
        raise ValueError("Fresh pack commit mismatch")
    role_rows = {role: {} for role in ["train", "dev", "calibration", "query"]}
    for role in role_rows:
        filename = "query_unlabeled.tsv.gz" if role == "query" else role + ".tsv.gz"
        for row in read_tsv(pack / "learner" / filename):
            role_rows[role][row["row_id"]] = row
    v2 = {}
    retrained = {}
    for row in read_tsv(Path(config["parent_split_v2_dir"]) / "role_assignments_v2.tsv.gz"):
        if row["model_lineage"] == "original_evolution":
            v2[row["raw_row_id"]] = row
        elif row["model_lineage"] == "retrained_evolution":
            retrained[row["raw_row_id"]] = row
    v1 = {}
    for row in read_tsv(Path(config["parent_split_v1_dir"]) / "role_assignments.tsv.gz"):
        v1[row["raw_row_id"]] = row
    random_candidates = {}
    designed_candidates = {}
    excluded = Counter()
    for raw_id, row in v1.items():
        if row["role"] != "outer" or row["library_type"] != "random_library":
            continue
        original = v2[raw_id]
        if original["role"] != "historical_measured_state_outer":
            excluded["random_blocked_by_original"] += 1
            continue
        if original["source_file"] == "GSM3130435_egfp_unmod_1.csv.gz":
            excluded["random_unmod_rep1"] += 1
            continue
        random_candidates[raw_id] = original
    for raw_id, row in v2.items():
        if row["E1_pool_eligible"] != "true":
            continue
        if row["source_library"].startswith("target_"):
            excluded["designed_target_retrained_lineage"] += 1
            continue
        if row["source_library"].startswith("step_"):
            excluded["designed_step_lineage"] += 1
            continue
        if v1[raw_id]["role"] != "outer":
            raise ValueError("Designed candidate is not in v1 outer")
        designed_candidates[raw_id] = row
    if len(designed_candidates) + excluded["designed_target_retrained_lineage"] + excluded["designed_step_lineage"] != config["designed_candidate_rows"]:
        raise ValueError("Designed candidate accounting mismatch")
    outer = {}
    for raw_id, row in random_candidates.items():
        outer[raw_id] = {
            **row,
            "outer_stratum": "random|" + row["evaluation_context_id"],
            "E1_pool_eligible": False,
            "historical_evaluation_status": "historical_measured_state",
            "never_evaluated": False,
        }
    for raw_id, row in designed_candidates.items():
        outer[raw_id] = {
            **row,
            "outer_stratum": "designed|eGFP_unmod",
            "E1_pool_eligible": True,
            "historical_evaluation_status": "historical_measured_state",
            "never_evaluated": False,
        }
    outer_label_vault = Path(config["parent_split_v1_dir"]) / "evaluator/outer_label_vault.tsv.gz"
    label_rows = {}
    for row in read_tsv(outer_label_vault):
        if row["raw_row_id"] in outer:
            label_rows[row["raw_row_id"]] = row
    if set(label_rows) != set(outer):
        raise ValueError("Outer label coverage mismatch")
    manifest = json.loads(Path(config["input_manifest"]).read_text())
    source_hashes = {item["name"]: item["sha256"] for item in manifest["files"]}
    writers = {}
    handles = []
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
            assignment = outer[raw_id]
            label = label_rows[raw_id]
            target = "evaluator/designed_outer_rows.tsv.gz" if assignment["E1_pool_eligible"] else "evaluator/random_outer_rows.tsv.gz"
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
    role_values = {role: set(rows) for role, rows in role_rows.items()}
    role_values["outer"] = set(outer)
    if any(item["count"] for item in role_intersections(role_values)):
        raise ValueError("Pack/outer sequence overlap")
    outer_counts = Counter(row["outer_stratum"] for row in outer.values())
    lineage_hash_input = []
    for role, rows in role_rows.items():
        for raw_id in sorted(rows):
            assignment = rows[raw_id]
            lineage_hash_input.append("\t".join([role, raw_id, assignment["allocation_unit_id"], assignment["context"]]))
    for raw_id in sorted(outer):
        row = outer[raw_id]
        lineage_hash_input.append("\t".join([raw_id, row["allocation_unit_id"], row["outer_stratum"], row["evaluation_context_id"]]))
    lineage_hash = __import__("hashlib").sha256("\n".join(lineage_hash_input).encode()).hexdigest()
    model_template = {
        "child_lineage_version": config["child_lineage_version"],
        "lineage_hash": lineage_hash,
        "initialization": config["fresh_initialization"],
        "official_weights": config["official_weights"],
        "distillation": config["distillation"],
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
    outer_manifest = {
        "child_split_version": config["child_split_version"],
        "child_lineage_version": config["child_lineage_version"],
        "lineage_hash": lineage_hash,
        "designed_candidate_rows_reviewed": config["designed_candidate_rows"],
        "designed_outer_rows": len(designed_candidates),
        "random_outer_rows": len(random_candidates),
        "outer_strata": dict(outer_counts),
        "outer_row_hashes": {
            "designed": row_list_hash(designed_candidates),
            "random": row_list_hash(random_candidates),
        },
        "excluded": dict(excluded),
        "designed_vault": str(output / "evaluator/designed_outer_rows.tsv.gz"),
        "random_vault": str(output / "evaluator/random_outer_rows.tsv.gz"),
        "historical_evaluation_status": "historical_measured_state",
        "never_evaluated": False,
        "E1_enabled": False,
        "outer_metrics_enabled": False,
    }
    write_json(output / "outer_manifest.json", outer_manifest)
    candidate = {
        "schema_version": "1.0",
        "child_split_version": config["child_split_version"],
        "child_lineage_version": config["child_lineage_version"],
        "lineage_hash": lineage_hash,
        "parent_pack_commit": config["parent_pack_commit"],
        "parent_pack_manifest_sha256": sha256_file(pack / "pack_manifest.json"),
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
        "E1_enabled": False,
        "training_start_authorized_pending_verification": True,
        "formal_training_start_authorized": False,
        "official_weights": False,
        "distillation": False,
        "outer_metrics_computed": False,
    }
    write_json(output / "candidate_manifest.json", candidate)
    print(json.dumps({"build_dir": str(output), "train_scaler_rows": len(role_rows["train"]), "designed_outer_rows": len(designed_candidates), "random_outer_rows": len(random_candidates), "excluded": dict(excluded), "lineage_hash": lineage_hash}))


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
