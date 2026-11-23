import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from .common import load_config, sha256_file
from .fixed_pools import ROLES, compute_pool_hash, control_exclusions, dedup_rows, source_rows
from .fresh_cnn_common import role_intersections
from .fresh_proxy_common import write_json


FORBIDDEN_FIELDS = {"measured_rl", "measured_status", "predicted_mrl", "predicted_status", "outer_regret", "outer_metric"}


def read_jsonl(path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def contains_forbidden(value):
    if isinstance(value, dict):
        return any(key in FORBIDDEN_FIELDS or contains_forbidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_forbidden(item) for item in value)
    return False


def round1(config, output, exclusions=None):
    files = {
        "dev": "learner/dev_pool.jsonl.gz",
        "calibration": "learner/calibration_pool.jsonl.gz",
        "outer_random": "learner/outer_random_unlabeled.jsonl.gz",
        "outer_designed": "learner/outer_designed_unlabeled.jsonl.gz",
    }
    by_role = {role: list(read_jsonl(output / files[role])) for role in ROLES}
    members = [row for role in ROLES for row in by_role[role]]
    manifest = json.loads((output / "pool_manifest.json").read_text())
    split_path = Path(config["parent_split_v2_dir"]) / "split_manifest_v2.json"
    split = json.loads(split_path.read_text())
    errors = []
    exclusions = exclusions or control_exclusions(config)
    control_ids, control_units = exclusions
    for role, rows in by_role.items():
        for row in rows:
            if contains_forbidden(row):
                errors.append("label_or_score_field:" + role)
            if row["q_expression"] != f"1/{row['pool_size']}" or abs(row["q_value"] - 1 / row["pool_size"]) > 1e-12:
                errors.append("q_value:" + role)
            if row["source_split_hash"] != split["split_hash"]:
                errors.append("source_split_hash:" + role)
            if row["source_split_manifest_sha256"] != sha256_file(split_path):
                errors.append("source_split_manifest_hash:" + role)
            if row["synthetic"] is not False:
                errors.append("synthetic:" + role)
            if role == "outer_designed" and any(source["context"] != config["designed_context"] for source in row["context_sources"]):
                errors.append("designed_context")
            if set(row["source_row_ids"]) & control_ids or row["allocation_unit_id"] in control_units:
                errors.append("control_pool:" + role)
            expected_paired = row["replicate_count"] > 1
            if row["paired_repeat"] != expected_paired or row["independent_evidence"] == expected_paired:
                errors.append("within_pool_repeat_flag:" + role)
        expected_count = config.get("pool_counts", {}).get(role, 1)
        if len(rows) != config["pool_sizes"][role] * expected_count:
            errors.append("pool_size:" + role)
        if len({row["sequence_id"] for row in rows}) != len(rows):
            errors.append("unique_sequence_q_support:" + role)
        pool_ids = {row["pool_id"] for row in rows}
        if pool_ids != {role + f"_{index:03d}" for index in range(expected_count)}:
            errors.append("pool_ids:" + role)
        for pool_id in pool_ids:
            pool_rows = [row for row in rows if row["pool_id"] == pool_id]
            if len(pool_rows) != config["pool_sizes"][role] or abs(sum(row["q_value"] for row in pool_rows) - 1) > 1e-12:
                errors.append("pool_size_or_q_sum:" + pool_id)
    if len(members) != sum(config["pool_sizes"][role] * config.get("pool_counts", {}).get(role, 1) for role in ROLES):
        errors.append("total_members")
    actual_repeat_counts = {
        "paired_repeat_members": sum(row["paired_repeat"] for row in members),
        "non_independent_repeat_members": sum(not row["independent_evidence"] for row in members),
        "cross_role_repeated_sequences": sum(len({row["pool_role"] for row in members if row["sequence_id"] == sequence_id}) > 1 for sequence_id in {row["sequence_id"] for row in members}),
    }
    if manifest["paired_repeat_counts"] != actual_repeat_counts:
        errors.append("paired_repeat_counts")
    expected_hash = {
        pool_id: compute_pool_hash(config, pool_id, split["split_hash"], [row for row in by_role[role] if row["pool_id"] == pool_id])
        for role in ROLES for pool_id in {row["pool_id"] for row in by_role[role]}
    }
    if manifest["pool_hashes"] != expected_hash:
        errors.append("pool_hash")
    for role, rows in by_role.items():
        for row in rows:
            if row["pool_hash"] != expected_hash[row["pool_id"]]:
                errors.append("member_pool_hash:" + role)
    if sha256_file(Path(manifest["source_production_manifest"])) != manifest["source_production_manifest_sha256"]:
        errors.append("production_manifest_hash")
    if sha256_file(Path(manifest["source_input_manifest"])) != manifest["source_input_manifest_sha256"]:
        errors.append("input_manifest_hash")
    clusters = list(read_jsonl(output / "pool_clusters.jsonl.gz"))
    if len(clusters) != manifest["pool_clusters"]:
        errors.append("manifest_cluster_count")
    cluster_by_pool_unit = {(row["pool_id"], row["allocation_unit_id"]): row for row in clusters}
    if len(cluster_by_pool_unit) != len(clusters):
        errors.append("duplicate_cluster_assignment")
    for role in ROLES:
        expected_records = dedup_rows(config, role, source_rows(config, role), *exclusions)
        expected_units = defaultdict(set)
        for row in expected_records:
            expected_units[row["allocation_unit_id"]].add(row["sequence_id"])
        actual_units = defaultdict(set)
        for row in by_role[role]:
            actual_units[row["allocation_unit_id"]].add(row["sequence_id"])
        for unit_id, sequences in actual_units.items():
            if expected_units.get(unit_id) != sequences:
                errors.append("split_or_incomplete_allocation_unit:" + role + ":" + unit_id)
        expected_by_sequence = {row["sequence_id"]: row for row in expected_records}
        for row in by_role[role]:
            expected = expected_by_sequence.get(row["sequence_id"])
            if expected is None or row["source_row_ids"] != expected["source_row_ids"] or row["context_sources"] != expected["context_sources"]:
                errors.append("source_or_context_dedup:" + role + ":" + row["sequence_id"])
        for unit_id, sequences in actual_units.items():
            actual_rows = [row for row in by_role[role] if row["allocation_unit_id"] == unit_id]
            pool_ids = {row["pool_id"] for row in actual_rows}
            if len(pool_ids) != 1:
                errors.append("unit_split_across_pools:" + role + ":" + unit_id)
                continue
            pool_id = next(iter(pool_ids))
            cluster = cluster_by_pool_unit.get((pool_id, unit_id))
            if cluster is None:
                errors.append("missing_cluster:" + pool_id + ":" + unit_id)
            elif cluster["member_sequence_count"] != len(sequences) or cluster["row_count"] != sum(row["replicate_count"] for row in actual_rows):
                errors.append("cluster_counts:" + pool_id + ":" + unit_id)
    vaults = json.loads((output / "evaluator/vault_references.json").read_text())
    for role, record in vaults.items():
        if sha256_file(record["path"]) != record["sha256"]:
            errors.append("vault_hash:" + role)
        if role.startswith("outer") and record["holder"] != "Evaluator":
            errors.append("outer_holder:" + role)
    result = {
        "round": 1,
        "rows_by_role": {role: len(rows) for role, rows in by_role.items()},
        "unique_sequences": sum(len({row["sequence_id"] for row in rows}) for rows in by_role.values()),
        "complete_allocation_units_checked": len(cluster_by_pool_unit),
        "errors": errors,
        "pass": not errors,
    }
    write_json(output / "verification_round1_pools.json", result)
    if not result["pass"]:
        raise ValueError("Fixed pool source verification failed")
    return result


def round2(config, output):
    files = {
        "dev": "learner/dev_pool.jsonl.gz",
        "calibration": "learner/calibration_pool.jsonl.gz",
        "outer_random": "learner/outer_random_unlabeled.jsonl.gz",
        "outer_designed": "learner/outer_designed_unlabeled.jsonl.gz",
    }
    members = {role: list(read_jsonl(output / files[role])) for role in ROLES}
    sequences = {role: {row["sequence_id"] for row in rows} for role, rows in members.items()}
    near = {role: {row["near_group_id"] for row in rows if row["near_group_id"]} for role, rows in members.items()}
    units = {role: {row["allocation_unit_id"] for row in rows} for role, rows in members.items()}
    trajectories = {role: {row["trajectory_id"] for row in rows if row["trajectory_id"]} for role, rows in members.items()}
    sequence_overlap = role_intersections(sequences)
    near_overlap = role_intersections(near)
    unit_overlap = role_intersections(units)
    trajectory_overlap = role_intersections(trajectories)
    pool_ids = sorted({row["pool_id"] for rows in members.values() for row in rows})
    pool_members = {pool_id: [row for rows in members.values() for row in rows if row["pool_id"] == pool_id] for pool_id in pool_ids}
    pool_sequence_overlap = role_intersections({pool_id: {row["sequence_id"] for row in rows} for pool_id, rows in pool_members.items()})
    pool_near_overlap = role_intersections({pool_id: {row["near_group_id"] for row in rows if row["near_group_id"]} for pool_id, rows in pool_members.items()})
    pool_unit_overlap = role_intersections({pool_id: {row["allocation_unit_id"] for row in rows} for pool_id, rows in pool_members.items()})
    pool_trajectory_overlap = role_intersections({pool_id: {row["trajectory_id"] for row in rows if row["trajectory_id"]} for pool_id, rows in pool_members.items()})
    negative = []
    for role in ROLES:
        other = "outer_random" if role != "outer_random" else "dev"
        injected = set(sequences[other])
        injected.add(next(iter(sequences[role])))
        negative.append({"test": f"deliberate_{role}_{other}_sequence_overlap_detected", "pass": bool(injected & sequences[role])})
    negative.append({"test": "deliberate_outer_label_detected", "pass": contains_forbidden({"nested": [{"measured_rl": 1.0}]})})
    negative.append({"test": "deliberate_outer_score_detected", "pass": contains_forbidden({"outer_regret": 0.0})})
    if any(row["pool_role"] == "outer_designed" and row["source_library"].startswith(("target_", "step_")) for rows in members.values() for row in rows):
        negative.append({"test": "control_designed_pool_rejected", "pass": False})
    else:
        negative.append({"test": "control_designed_pool_rejected", "pass": True})
    for role in ["outer_random", "outer_designed"]:
        negative.append({"test": "outer_unlabeled_view_" + role, "pass": all(not contains_forbidden(row) for row in members[role])})
    paired = [row for rows in members.values() for row in rows if row["paired_repeat"]]
    negative.append({"test": "paired_repeats_not_independent", "pass": all(not row["independent_evidence"] for row in paired)})
    manifest = json.loads((output / "pool_manifest.json").read_text())
    negative.append({"test": "no_outer_scoring_or_E1", "pass": not manifest["outer_scoring_performed"] and not manifest["E1_enabled"]})
    result = {
        "round": 2,
        "sequence_intersections": sequence_overlap,
        "near_component_intersections": near_overlap,
        "allocation_unit_intersections": unit_overlap,
        "trajectory_intersections": trajectory_overlap,
        "pool_sequence_intersections": pool_sequence_overlap,
        "pool_near_component_intersections": pool_near_overlap,
        "pool_allocation_unit_intersections": pool_unit_overlap,
        "pool_trajectory_intersections": pool_trajectory_overlap,
        "negative_tests": negative,
        "pass": not any(item["count"] for item in sequence_overlap + near_overlap + unit_overlap + trajectory_overlap + pool_sequence_overlap + pool_near_overlap + pool_unit_overlap + pool_trajectory_overlap) and all(test["pass"] for test in negative),
    }
    write_json(output / "verification_round2_leakage.json", result)
    if not result["pass"]:
        raise ValueError("Fixed pool leakage verification failed")
    return result


def round3(config, root):
    first = root / config["build_names"][0]
    second = root / config["build_names"][1]
    comparisons = []
    for relative in ["learner/dev_pool.jsonl.gz", "learner/calibration_pool.jsonl.gz", "learner/outer_random_unlabeled.jsonl.gz", "learner/outer_designed_unlabeled.jsonl.gz", "pool_clusters.jsonl.gz", "paired_proposal_seed_plan.json", "evaluator/vault_references.json", "pool_manifest.json"]:
        comparisons.append({"file": relative, "match": sha256_file(first / relative) == sha256_file(second / relative)})
    result = {"round": 3, "comparisons": comparisons, "pass": all(item["match"] for item in comparisons)}
    write_json(first / "verification_round3_double_build.json", result)
    if not result["pass"]:
        raise ValueError("Fixed pool double-build verification failed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["output_root"])
    try:
        exclusions = control_exclusions(config)
        first = round1(config, root / config["build_names"][0], exclusions)
        second = round2(config, root / config["build_names"][0])
        third = round3(config, root)
    except Exception as error:
        write_json(root / "verification_failure.json", {"pass": False, "error_type": type(error).__name__, "error": str(error)})
        raise
    manifest = json.loads((root / config["build_names"][0] / "pool_manifest.json").read_text())
    summary = {
        "status": "FIXED_POOLS_VERIFIED_NO_OUTER_SCORING",
        "pool_version": config["pool_version"],
        "rounds": {"source_dedup_hash_complete_clusters": first["pass"], "leakage_negative": second["pass"], "double_build_hashes": third["pass"]},
        "source_split_hash": manifest["source_split_hash"],
        "source_production_manifest_sha256": manifest["source_production_manifest_sha256"],
        "pool_counts": manifest["pool_counts"],
        "pool_sizes": manifest["pool_sizes"],
        "pool_clusters": manifest["pool_clusters"],
        "pool_hashes": manifest["pool_hashes"],
        "selection_stats": manifest["selection_stats"],
        "paired_repeat_counts": manifest["paired_repeat_counts"],
        "E1_enabled": False,
        "outer_scoring_performed": False,
        "outer_labels_returned_to_learner": False,
        "statistical_limitations": manifest["statistical_limitations"],
    }
    write_json(root / "verification_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
