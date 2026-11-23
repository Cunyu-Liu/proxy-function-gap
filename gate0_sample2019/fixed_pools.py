import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from .common import load_config, sha256_file, sha256_text
from .fresh_proxy_common import read_tsv, write_json


ROLES = ["dev", "calibration", "outer_random", "outer_designed"]
METADATA_FIELDS = {
    "raw_row_id", "row_id", "sequence_id", "context", "utr50",
    "allocation_unit_id", "near_group_id", "trajectory_id", "source_file",
    "source_file_sha256", "raw_row_sha256", "replicate_id", "outer_stratum",
    "source_library",
}
CONTEXT_SOURCE_FIELDS = [
    "context", "source_row_ids", "replicate_ids", "replicate_count",
    "source_file", "source_file_sha256", "raw_row_sha256", "outer_stratum",
    "source_library",
]


def write_jsonl(path, rows):
    raw = open(path, "xb")
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=5) as handle:
        for row in rows:
            handle.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    raw.close()


def source_rows(config, role):
    if role in {"dev", "calibration"}:
        path = Path(config["parent_pack_dir"]) / "learner" / (role + ".tsv.gz")
    else:
        filename = "random_outer_rows.tsv.gz" if role == "outer_random" else "designed_outer_rows.tsv.gz"
        path = Path(config["parent_child_dir"]) / "evaluator" / filename
    return [{key: value for key, value in row.items() if key in METADATA_FIELDS} for row in read_tsv(path)]


def control_exclusions(config):
    control_ids = set()
    control_units = set()
    path = Path(config["parent_split_v2_dir"]) / "role_assignments_v2.tsv.gz"
    for assignment in read_tsv(path):
        if assignment["pool_class"] == "control_nonexact":
            control_ids.add(assignment["raw_row_id"])
            control_units.add(assignment["allocation_unit_id"])
    return control_ids, control_units


def dedup_rows(config, role, rows, control_ids, control_units):
    groups = defaultdict(list)
    for row in rows:
        row_id = row["raw_row_id"] if "raw_row_id" in row else row["row_id"]
        if row_id in control_ids or row["allocation_unit_id"] in control_units:
            continue
        if role == "outer_designed" and row["context"] != config["designed_context"]:
            continue
        if role == "outer_random" and row["outer_stratum"].startswith("designed|"):
            continue
        if role == "outer_designed" and not row["outer_stratum"].startswith("designed|"):
            continue
        groups[(row["sequence_id"], row["context"])].append(row)
    context_records = []
    for (sequence_id, context), members in groups.items():
        members.sort(key=lambda row: row["raw_row_id"] if "raw_row_id" in row else row["row_id"])
        first = members[0]
        context_records.append({
            "sequence_id": sequence_id,
            "context": context,
            "utr50": first["utr50"],
            "allocation_unit_id": first["allocation_unit_id"],
            "near_group_id": first["near_group_id"],
            "trajectory_id": first["trajectory_id"],
            "source_file": first["source_file"],
            "source_file_sha256": first["source_file_sha256"],
            "raw_row_sha256": first["raw_row_sha256"],
            "source_row_ids": [row["raw_row_id"] if "raw_row_id" in row else row["row_id"] for row in members],
            "replicate_ids": sorted({row["replicate_id"] if "replicate_id" in row else row["source_file"] for row in members}),
            "replicate_count": len(members),
            "outer_stratum": first.get("outer_stratum", ""),
            "source_library": first.get("source_library", ""),
        })
    by_sequence = defaultdict(list)
    for row in context_records:
        by_sequence[row["sequence_id"]].append(row)
    result = []
    for sequence_id, contexts in by_sequence.items():
        contexts.sort(key=lambda row: row["context"])
        first = contexts[0]
        for other in contexts[1:]:
            for key in ["utr50", "allocation_unit_id", "near_group_id", "trajectory_id"]:
                if other[key] != first[key]:
                    raise ValueError(f"Inconsistent {key} across contexts for {sequence_id}")
        row = dict(first)
        row["context_sources"] = [{key: context[key] for key in CONTEXT_SOURCE_FIELDS} for context in contexts]
        row["source_row_ids"] = [row_id for context in contexts for row_id in context["source_row_ids"]]
        row["replicate_ids"] = sorted({replicate_id for context in contexts for replicate_id in context["replicate_ids"]})
        row["replicate_count"] = sum(context["replicate_count"] for context in contexts)
        result.append(row)
    return result


def select_complete_units(config, role, records, blocked=None):
    blocked = blocked or {key: set() for key in ["sequences", "near", "units", "trajectories"]}
    groups = defaultdict(list)
    for row in records:
        groups[row["allocation_unit_id"]].append(row)
    candidates = []
    excluded_units = 0
    for unit_id, members in groups.items():
        sequences = {row["sequence_id"] for row in members}
        near = {row["near_group_id"] for row in members if row["near_group_id"]}
        trajectories = {row["trajectory_id"] for row in members if row["trajectory_id"]}
        if sequences & blocked["sequences"] or near & blocked["near"] or unit_id in blocked["units"] or trajectories & blocked["trajectories"]:
            excluded_units += 1
            continue
        contexts = {source["context"] for row in members for source in row["context_sources"]}
        row_count = sum(row["replicate_count"] for row in members)
        order_hash = sha256_text(config["pool_version"] + "|" + role + "|" + unit_id)
        candidates.append((len(members), len(contexts), row_count, order_hash, unit_id, members))
    candidates.sort(key=lambda item: item[:5])
    target = config["pool_sizes"][role]
    reachable = [None] * (target + 1)
    reachable[0] = (-1, -1)
    for index, candidate in enumerate(candidates):
        weight = candidate[0]
        if weight > target:
            continue
        for total in range(target, weight - 1, -1):
            if reachable[total] is None and reachable[total - weight] is not None:
                reachable[total] = (total - weight, index)
    if reachable[target] is None:
        raise ValueError(f"No exact complete-allocation-unit pool of {target} unique sequences for {role}")
    selected = []
    total = target
    while total:
        previous, index = reachable[total]
        selected.extend(candidates[index][5])
        total = previous
    selected.sort(key=lambda row: sha256_text(config["pool_version"] + "|" + role + "|" + row["sequence_id"]))
    for row in selected:
        blocked["sequences"].add(row["sequence_id"])
        blocked["units"].add(row["allocation_unit_id"])
        if row["near_group_id"]:
            blocked["near"].add(row["near_group_id"])
        if row["trajectory_id"]:
            blocked["trajectories"].add(row["trajectory_id"])
    stats = {
        "eligible_allocation_units": len(candidates),
        "cross_role_blocked_allocation_units": excluded_units,
        "selected_allocation_units": len({row["allocation_unit_id"] for row in selected}),
    }
    return selected, stats


def compute_pool_hash(config, pool_id, split_hash, rows):
    identity = {
        "pool_id": pool_id,
        "pool_version": config["pool_version"],
        "source_split_hash": split_hash,
        "selection_rule": config["selection_rule"],
        "q_rule": config["q_rule"],
        "members": sorted(({
            "sequence_id": row["sequence_id"],
            "allocation_unit_id": row["allocation_unit_id"],
            "context_sources": [{"context": source["context"], "source_row_ids": source["source_row_ids"]} for source in row["context_sources"]],
        } for row in rows), key=lambda row: row["sequence_id"]),
    }
    return sha256_text(json.dumps(identity, sort_keys=True, separators=(",", ":")))


def build(config, output, exclusions=None):
    output.mkdir(parents=True, exist_ok=False)
    (output / "learner").mkdir()
    (output / "evaluator").mkdir()
    split_path = Path(config["parent_split_v2_dir"]) / "split_manifest_v2.json"
    production_path = Path(config["parent_child_dir"]) / "data_manifest.json"
    split_manifest = json.loads(split_path.read_text())
    split_hash = split_manifest["split_hash"]
    split_manifest_hash = sha256_file(split_path)
    exclusions = exclusions or control_exclusions(config)
    selection_stats = {}
    pool_hashes = {}
    all_members = []
    all_clusters = []
    blocked = {key: set() for key in ["sequences", "near", "units", "trajectories"]}
    for role in ROLES:
        records = dedup_rows(config, role, source_rows(config, role), *exclusions)
        chosen, selection_stats[role] = select_complete_units(config, role, records, blocked)
        size = config["pool_sizes"][role]
        pool_id = role + "_000"
        pool_hash = compute_pool_hash(config, pool_id, split_hash, chosen)
        pool_hashes[pool_id] = pool_hash
        for index, row in enumerate(chosen):
            member = {
                "pool_id": pool_id,
                "pool_role": role,
                "pool_index": index,
                "pool_size": size,
                "pool_hash": pool_hash,
                "cluster_id": row["allocation_unit_id"],
                "allocation_unit_id": row["allocation_unit_id"],
                "near_group_id": row["near_group_id"],
                "trajectory_id": row["trajectory_id"],
                "sequence_id": row["sequence_id"],
                "row_id": row["source_row_ids"][0],
                "source_row_ids": row["source_row_ids"],
                "utr50": row["utr50"],
                "context": row["context"],
                "context_sources": row["context_sources"],
                "outer_stratum": row["outer_stratum"],
                "source_library": row["source_library"],
                "source_file": row["source_file"],
                "source_file_sha256": row["source_file_sha256"],
                "raw_row_sha256": row["raw_row_sha256"],
                "replicate_ids": row["replicate_ids"],
                "replicate_count": row["replicate_count"],
                "paired_repeat_group_id": "paired_" + sha256_text(row["sequence_id"])[:24],
                "paired_repeat": row["replicate_count"] > 1,
                "independent_evidence": row["replicate_count"] == 1,
                "q_rule": config["q_rule"],
                "q_value": 1.0 / size,
                "q_expression": f"1/{size}",
                "source_split_hash": split_hash,
                "source_split_manifest_sha256": split_manifest_hash,
                "synthetic": False,
            }
            all_members.append(member)
        cluster_groups = defaultdict(list)
        for row in chosen:
            cluster_groups[row["allocation_unit_id"]].append(row)
        for cluster_id, members in sorted(cluster_groups.items()):
            all_clusters.append({
                "pool_id": pool_id,
                "cluster_id": cluster_id,
                "allocation_unit_id": cluster_id,
                "near_group_ids": sorted({row["near_group_id"] for row in members}),
                "trajectory_ids": sorted({row["trajectory_id"] for row in members if row["trajectory_id"]}),
                "member_sequence_count": len(members),
                "row_count": sum(row["replicate_count"] for row in members),
                "context": sorted({source["context"] for row in members for source in row["context_sources"]}),
                "independent_evidence_unit": "complete allocation unit including applicable near-component and trajectory constraints",
            })
    sequence_roles = defaultdict(set)
    for member in all_members:
        sequence_roles[member["sequence_id"]].add(member["pool_role"])
    for member in all_members:
        if len(sequence_roles[member["sequence_id"]]) > 1:
            member["paired_repeat"] = True
            member["independent_evidence"] = False
    write_jsonl(output / "learner/dev_pool.jsonl.gz", [row for row in all_members if row["pool_role"] == "dev"])
    write_jsonl(output / "learner/calibration_pool.jsonl.gz", [row for row in all_members if row["pool_role"] == "calibration"])
    write_jsonl(output / "learner/outer_random_unlabeled.jsonl.gz", [row for row in all_members if row["pool_role"] == "outer_random"])
    write_jsonl(output / "learner/outer_designed_unlabeled.jsonl.gz", [row for row in all_members if row["pool_role"] == "outer_designed"])
    write_jsonl(output / "pool_clusters.jsonl.gz", all_clusters)
    seed_plan = []
    for pool_id, pool_hash in sorted(pool_hashes.items()):
        seed_plan.append({
            "pool_id": pool_id,
            "pool_hash": pool_hash,
            "proposal_round": 0,
            "seed_namespace": sha256_text(config["pool_version"] + "|" + split_hash + "|" + pool_id),
            "proposal_seed": sha256_text(config["pool_version"] + "|" + split_hash + "|" + pool_id + "|round_0"),
            "selection_rule": "Deterministic proposal only; no draws or scores are performed in this task.",
            "paired_repeat_policy": "If a sequence appears in multiple roles, pair it by sequence_id and do not count it as independent biological evidence.",
        })
    write_json(output / "paired_proposal_seed_plan.json", seed_plan)
    vaults = {
        "dev": {"holder": "Learner", "path": str(Path(config["parent_pack_dir"]) / "learner/dev.tsv.gz"), "sha256": sha256_file(Path(config["parent_pack_dir"]) / "learner/dev.tsv.gz")},
        "calibration": {"holder": "Learner", "path": str(Path(config["parent_pack_dir"]) / "learner/calibration.tsv.gz"), "sha256": sha256_file(Path(config["parent_pack_dir"]) / "learner/calibration.tsv.gz")},
        "outer_random": {"holder": "Evaluator", "path": str(Path(config["parent_child_dir"]) / "evaluator/random_outer_rows.tsv.gz"), "sha256": sha256_file(Path(config["parent_child_dir"]) / "evaluator/random_outer_rows.tsv.gz")},
        "outer_designed": {"holder": "Evaluator", "path": str(Path(config["parent_child_dir"]) / "evaluator/designed_outer_rows.tsv.gz"), "sha256": sha256_file(Path(config["parent_child_dir"]) / "evaluator/designed_outer_rows.tsv.gz")},
    }
    write_json(output / "evaluator/vault_references.json", vaults)
    manifest = {
        "schema_version": "1.0",
        "pool_version": config["pool_version"],
        "source_split_hash": split_hash,
        "source_split_manifest_sha256": split_manifest_hash,
        "source_production_manifest": str(production_path),
        "source_production_manifest_sha256": sha256_file(production_path),
        "source_input_manifest": config["input_manifest"],
        "source_input_manifest_sha256": sha256_file(Path(config["input_manifest"])),
        "selection_rule": config["selection_rule"],
        "selection_stats": selection_stats,
        "pool_hashes": pool_hashes,
        "pool_counts": {role: 1 for role in ROLES},
        "pool_sizes": dict(config["pool_sizes"]),
        "pool_members": len(all_members),
        "pool_clusters": len(all_clusters),
        "q_rule": config["q_rule"],
        "replicate_summary_rule": config["replicate_summary_rule"],
        "paired_repeat_counts": {
            "paired_repeat_members": sum(row["paired_repeat"] for row in all_members),
            "non_independent_repeat_members": sum(not row["independent_evidence"] for row in all_members),
            "cross_role_repeated_sequences": sum(len(roles) > 1 for roles in sequence_roles.values()),
        },
        "files": {
            "dev": "learner/dev_pool.jsonl.gz",
            "calibration": "learner/calibration_pool.jsonl.gz",
            "outer_random": "learner/outer_random_unlabeled.jsonl.gz",
            "outer_designed": "learner/outer_designed_unlabeled.jsonl.gz",
            "clusters": "pool_clusters.jsonl.gz",
        },
        "outer_labels_returned_to_learner": False,
        "outer_scoring_performed": False,
        "E1_enabled": False,
        "synthetic": False,
        "statistical_limitations": [
            "One fixed pool per role; this is a protocol fixture, not an independent replication series.",
            "No variance, confidence interval or generalization claim is authorized from these pool sizes.",
            "Paired repeats, if present, are not independent biological evidence.",
            "External random contexts are outside the main E1 context.",
            "Independence-first complete-unit selection is deterministic and is not a probability sample of all source rows.",
        ],
    }
    write_json(output / "pool_manifest.json", manifest)
    print(json.dumps({"output": str(output), "pool_sizes": dict(config["pool_sizes"]), "pool_members": len(all_members), "pool_clusters": len(all_clusters), "selection_stats": selection_stats, "paired_repeat_counts": manifest["paired_repeat_counts"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    exclusions = control_exclusions(config)
    for name in config["build_names"]:
        build(config, root / name, exclusions)


if __name__ == "__main__":
    main()
