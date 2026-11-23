import argparse
import csv
import gzip
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_config, make_raw_row_id, open_tsv_gz, sha256_file, sha256_text, write_row
from .grouping import find, union


ASSIGNMENT_FIELDS = [
    "raw_row_id", "raw_row_sha256", "sequence_id", "near_group_id", "allocation_unit_id", "role",
    "library_type", "library_family", "source_library", "trajectory_id", "round", "replicate_id",
    "official_model_role", "new_blind_evaluation_allowed", "split_version",
]
UNIT_FIELDS = [
    "allocation_unit_id", "near_group_ids", "role", "stratum", "row_count", "sequence_count",
    "missing_row_count", "source_file_count", "source_files", "library_types", "source_libraries",
    "trajectory_count", "trajectory_ids", "official_test_rows", "official_train_rows",
    "official_boundary_ambiguous_rows", "new_blind_evaluation_allowed",
]
BASE_VIEW = [
    "raw_row_id", "sequence_id", "near_group_id", "allocation_unit_id", "role", "official_model_role",
    "raw_sequence", "normalized_sequence", "normalization_version", "source_file", "source_library",
    "library_type", "trajectory_id", "round", "replicate_id",
]
LABEL_COLUMNS = ["measured_rl", "measured_status", "predicted_mrl", "predicted_status"]
LABEL_VIEW = ["raw_row_id", "sequence_id", "allocation_unit_id", "official_model_role"] + LABEL_COLUMNS


def rows(path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def official_roles(config):
    model = config["official_model"]
    source = Path(config["input_manifest"]).parent / "raw/GSE114002" / model["source_file"]
    ranked = []
    with gzip.open(source, "rt", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            ranked.append((float(row[model["sort_field"]]), row_number))
    ranked.sort(key=lambda item: -item[0])
    result = {}
    counts = Counter()
    tie_counts = Counter(reads for reads, _ in ranked)
    for rank, (reads, row_number) in enumerate(ranked, 1):
        raw_row_id = make_raw_row_id("GSM3130435", row_number)
        if reads in {model["test_boundary_reads"], model["train_boundary_reads"]}:
            role = "official_boundary_ambiguous"
        elif rank <= model["test_rows"]:
            role = "official_test_seen"
        elif rank <= model["combined_rows"]:
            role = "official_train_seen"
        else:
            continue
        result[raw_row_id] = role
        counts[role] += 1
    metadata = {
        "source_file": model["source_file"],
        "sort_field": model["sort_field"],
        "sort_direction": model["sort_direction"],
        "test_rows": model["test_rows"],
        "train_rows": model["train_rows"],
        "exact_role_counts": dict(counts),
        "test_boundary_reads": model["test_boundary_reads"],
        "train_boundary_reads": model["train_boundary_reads"],
        "test_boundary_tied_rows": tie_counts[model["test_boundary_reads"]],
        "train_boundary_tied_rows": tie_counts[model["train_boundary_reads"]],
        "boundary_policy": model["boundary_policy"],
        "author_code": "training_MRL_CNN.ipynb cell 2: sort total_reads descending, take first 280000, first 20000 as test and remaining 260000 as train.",
        "limitation": "Pandas quicksort tie order is not reconstructed. Boundary-tied rows are conservatively quarantined and cannot become outer.",
    }
    return result, metadata


def stratum_for(unit):
    if unit["trajectory_ids"]:
        libraries = sorted(unit["source_libraries"])
        return "designed_step|" + ("|".join(libraries) if libraries else "unknown")
    if len(unit["library_types"]) > 1:
        return "mixed_random_designed"
    library_type = next(iter(unit["library_types"]))
    if library_type == "random_library":
        return "random|" + "|".join(sorted(unit["library_families"]))
    if library_type == "designed_library":
        return "designed|" + "|".join(sorted(unit["source_libraries"]))
    if library_type == "varying_length_library":
        return "varying_length"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    source_dir = Path(config["split_ready_dir"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    for role in ["learner", "evaluator"]:
        (output / role).mkdir()
    prior = json.loads((source_dir / "final_validation.json").read_text())
    if not prior["pass"] or prior["E1_authorized"]:
        raise ValueError("Prior split-ready artifacts are not valid or E1 was authorized")
    official, official_metadata = official_roles(config)
    parent = {}
    sizes = {}
    trajectories = defaultdict(set)
    for row in rows(source_dir / "sequence_index.tsv.gz"):
        group = row["near_group_id"]
        if group not in parent:
            parent[group] = group
            sizes[group] = 1
        if row["trajectory_id"]:
            trajectories[row["trajectory_id"]].add(group)
    for groups in trajectories.values():
        for left, right in itertools.combinations(sorted(groups), 2):
            union(parent, sizes, left, right)
    components = defaultdict(list)
    for group in parent:
        components[find(parent, group)].append(group)
    group_to_unit = {}
    unit_members = {}
    for members in components.values():
        members.sort()
        unit_id = "unit_" + sha256_text("\n".join(members))[:24]
        for group in members:
            group_to_unit[group] = unit_id
        unit_members[unit_id] = members
    group_sizes = {}
    for row in rows(source_dir / "split_ready_groups.tsv.gz"):
        group_sizes[row["near_group_id"]] = int(row["member_sequence_count"])
    units = {}
    nonexact_step_rows = 0
    for row in rows(source_dir / "sequence_index.tsv.gz"):
        unit_id = group_to_unit[row["near_group_id"]]
        if unit_id not in units:
            units[unit_id] = {
                "allocation_unit_id": unit_id,
                "near_group_ids": unit_members[unit_id],
                "row_count": 0,
                "sequence_count": 0,
                "missing_row_count": 0,
                "source_files": set(),
                "library_types": set(),
                "library_families": set(),
                "source_libraries": set(),
                "trajectory_ids": set(),
                "official_test_rows": 0,
                "official_train_rows": 0,
                "official_boundary_ambiguous_rows": 0,
            }
            for group in unit_members[unit_id]:
                units[unit_id]["sequence_count"] += group_sizes[group]
        unit = units[unit_id]
        unit["row_count"] += 1
        unit["missing_row_count"] += int(row["label_status"] == "missing")
        unit["source_files"].add(row["source_file"])
        unit["library_types"].add(row["library_type"])
        unit["library_families"].add(row["library_family"])
        unit["source_libraries"].add(row["source_library"])
        if row["trajectory_id"]:
            unit["trajectory_ids"].add(row["trajectory_id"])
        elif row["source_library"].startswith("step_"):
            nonexact_step_rows += 1
        official_role = official.get(row["raw_row_id"])
        if official_role == "official_test_seen":
            unit["official_test_rows"] += 1
        elif official_role == "official_train_seen":
            unit["official_train_rows"] += 1
        elif official_role == "official_boundary_ambiguous":
            unit["official_boundary_ambiguous_rows"] += 1
    strata = defaultdict(list)
    for unit in units.values():
        unit["stratum"] = stratum_for(unit)
        if unit["official_boundary_ambiguous_rows"]:
            unit["role"] = "official_boundary_ambiguous"
        elif unit["official_test_rows"]:
            unit["role"] = "official_seen_test"
        elif unit["official_train_rows"]:
            unit["role"] = "train"
        else:
            unit["role"] = None
            strata[unit["stratum"]].append(unit)
    limitations = []
    weights = config["role_weights"]
    roles = config["roles"]
    for stratum, members in sorted(strata.items()):
        members.sort(key=lambda unit: sha256_text(config["split_version"] + "|" + unit["allocation_unit_id"]))
        n = len(members)
        if n < 6:
            non_outer_roles = ["train", "dev", "calibration", "query"]
            initial = non_outer_roles[:n] + ["train"] * max(0, n - len(non_outer_roles))
            limitations.append({
                "stratum": stratum,
                "available_new_allocation_units": n,
                "limitation": "Insufficient independent units for all five roles and two outer units; no outer role is fabricated.",
            })
        else:
            initial = roles + ["outer"]
        for index, role in enumerate(initial):
            members[index]["role"] = role
        for unit in members[len(initial):]:
            value = int(sha256_text(config["split_version"] + "|role|" + unit["allocation_unit_id"])[:16], 16) / float(16 ** 16)
            cumulative = 0.0
            for role in roles:
                cumulative += weights[role]
                if value <= cumulative:
                    unit["role"] = role
                    break
    role_strata = defaultdict(Counter)
    for unit in units.values():
        role_strata[unit["stratum"]][unit["role"]] += 1
    for stratum, counts in sorted(role_strata.items()):
        if counts.get("outer", 0) < config["minimum_outer_units_per_stratum"]:
            limitations.append({"stratum": stratum, "outer_allocation_units": counts.get("outer", 0), "limitation": "Fewer than two independent outer units; no outer claims are allowed for this stratum."})
    limitations.append({
        "step_rows_without_trajectory_metadata": nonexact_step_rows,
        "limitation": "Non-exact step rows without trajectory/round metadata are allocated only by their near component and are not eligible for trajectory replay.",
    })
    limitations.append(official_metadata)
    handles = []
    writers = {}
    assignment_hash = hashlib.sha256()
    assignments = {}
    role_counts = Counter()
    unit_counts = Counter()
    try:
        handle, writer = open_tsv_gz(output / "role_assignments.tsv.gz", ASSIGNMENT_FIELDS)
        handles.append(handle)
        writers["assignments"] = writer
        handle, writer = open_tsv_gz(output / "allocation_units.tsv.gz", UNIT_FIELDS)
        handles.append(handle)
        writers["units"] = writer
        for unit_id in sorted(units):
            unit = units[unit_id]
            role = unit["role"]
            unit_counts[role] += 1
            write_row(writers["units"], {
                **unit,
                "source_file_count": len(unit["source_files"]),
                "source_files": sorted(unit["source_files"]),
                "library_types": sorted(unit["library_types"]),
                "source_libraries": sorted(unit["source_libraries"]),
                "trajectory_count": len(unit["trajectory_ids"]),
                "trajectory_ids": sorted(unit["trajectory_ids"]),
                "new_blind_evaluation_allowed": role == "outer",
            })
        for row in rows(source_dir / "sequence_index.tsv.gz"):
            unit_id = group_to_unit[row["near_group_id"]]
            role = units[unit_id]["role"]
            official_role = official.get(row["raw_row_id"], "not_applicable")
            assignments[row["raw_row_id"]] = (role, unit_id, official_role)
            role_counts[role] += 1
            canonical = "\t".join([row["raw_row_id"], row["sequence_id"], row["near_group_id"], unit_id, role, official_role]) + "\n"
            assignment_hash.update(canonical.encode())
            write_row(writers["assignments"], {
                "raw_row_id": row["raw_row_id"],
                "raw_row_sha256": row["raw_row_sha256"],
                "sequence_id": row["sequence_id"],
                "near_group_id": row["near_group_id"],
                "allocation_unit_id": unit_id,
                "role": role,
                "library_type": row["library_type"],
                "library_family": row["library_family"],
                "source_library": row["source_library"],
                "trajectory_id": row["trajectory_id"],
                "round": row["round"],
                "replicate_id": row["replicate_id"],
                "official_model_role": official_role,
                "new_blind_evaluation_allowed": role == "outer",
                "split_version": config["split_version"],
            })
    finally:
        for handle in handles:
            handle.close()
    split_hash = assignment_hash.hexdigest()
    view_paths = {
        "train": "learner/train_features_labels.tsv.gz",
        "dev": "learner/dev_features_labels.tsv.gz",
        "calibration": "learner/calibration_features_labels.tsv.gz",
        "official_seen_test": "learner/official_seen_test_features_labels.tsv.gz",
        "query": "learner/query_unlabeled.tsv.gz",
        "query_vault": "evaluator/query_label_vault.tsv.gz",
        "outer_features": "evaluator/outer_features.tsv.gz",
        "outer_vault": "evaluator/outer_label_vault.tsv.gz",
        "ambiguous_vault": "evaluator/official_boundary_ambiguous_quarantine.tsv.gz",
    }
    view_writers = {}
    view_handles = []
    try:
        for key, relative in view_paths.items():
            fields = BASE_VIEW if key in {"query", "outer_features"} else BASE_VIEW + LABEL_COLUMNS
            handle, writer = open_tsv_gz(output / relative, fields)
            view_handles.append(handle)
            view_writers[key] = writer
        endpoints = rows(source_dir / "endpoint_coverage.tsv.gz")
        for sequence in rows(source_dir / "sequence_index.tsv.gz"):
            endpoint = next(endpoints)
            if endpoint["raw_row_id"] != sequence["raw_row_id"]:
                raise ValueError("Endpoint/sequence row mismatch")
            role, unit_id, official_role = assignments[sequence["raw_row_id"]]
            base = {
                "raw_row_id": sequence["raw_row_id"],
                "sequence_id": sequence["sequence_id"],
                "near_group_id": sequence["near_group_id"],
                "allocation_unit_id": unit_id,
                "role": role,
                "official_model_role": official_role,
                "raw_sequence": sequence["raw_sequence"],
                "normalized_sequence": sequence["normalized_sequence"],
                "normalization_version": sequence["normalization_version"],
                "source_file": sequence["source_file"],
                "source_library": sequence["source_library"],
                "library_type": sequence["library_type"],
                "trajectory_id": sequence["trajectory_id"],
                "round": sequence["round"],
                "replicate_id": sequence["replicate_id"],
            }
            labels = {key: endpoint[key] for key in LABEL_COLUMNS}
            if role in {"train", "dev", "calibration", "official_seen_test"}:
                write_row(view_writers[role], {**base, **labels})
            elif role == "query":
                write_row(view_writers["query"], base)
                write_row(view_writers["query_vault"], {**base, **labels})
            elif role == "outer":
                write_row(view_writers["outer_features"], base)
                write_row(view_writers["outer_vault"], {**base, **labels})
            elif role == "official_boundary_ambiguous":
                write_row(view_writers["ambiguous_vault"], {**base, **labels})
            else:
                raise ValueError(role)
        if next(endpoints, None) is not None:
            raise ValueError("Extra endpoint rows")
    finally:
        for handle in view_handles:
            handle.close()
    learner = {
        "schema_version": "1.0",
        "split_version": config["split_version"],
        "split_hash": split_hash,
        "feature_roles": {},
        "label_roles": {},
        "parameter_edit_roles": config["parameter_edit_roles"],
        "outer_labels_returned": False,
        "E1_enabled": False,
    }
    for role in ["train", "dev", "calibration", "official_seen_test"]:
        learner["feature_roles"][role] = {"path": str(output / view_paths[role]), "columns": BASE_VIEW}
        learner["label_roles"][role] = {"path": str(output / view_paths[role]), "columns": LABEL_VIEW, "new_blind_evaluation_allowed": False}
    learner["feature_roles"]["query"] = {"path": str(output / view_paths["query"]), "columns": BASE_VIEW}
    write_json(output / "learner_allowlist.json", learner)
    evaluator = {
        "schema_version": "1.0",
        "split_version": config["split_version"],
        "split_hash": split_hash,
        "label_vaults": {
            "query": {"path": str(output / view_paths["query_vault"]), "reveal_protocol": "selection hash + not_before + new run_id"},
            "outer": {"path": str(output / view_paths["outer_vault"]), "reveal_protocol": "disabled before E1"},
            "official_boundary_ambiguous": {"path": str(output / view_paths["ambiguous_vault"]), "reveal_protocol": "quarantined"},
        },
        "outer_features": str(output / view_paths["outer_features"]),
        "outer_labels_returned_to_learner": False,
        "outer_regret_enabled": False,
        "E1_enabled": False,
    }
    write_json(output / "evaluator_manifest.json", evaluator)
    role_counts_by_stratum = defaultdict(Counter)
    for unit in units.values():
        role_counts_by_stratum[unit["stratum"]][unit["role"]] += unit["row_count"]
    role_manifest = {
        "schema_version": "1.0",
        "split_version": config["split_version"],
        "split_hash": split_hash,
        "roles": roles + config["additional_nonblind_roles"],
        "role_counts": dict(role_counts),
        "allocation_unit_counts": dict(unit_counts),
        "role_counts_by_stratum": {key: dict(value) for key, value in sorted(role_counts_by_stratum.items())},
        "allocation_unit_counts_by_stratum": {key: dict(value) for key, value in sorted(role_strata.items())},
        "label_holders": {
            "train": "Learner",
            "dev": "Learner",
            "calibration": "Learner",
            "official_seen_test": "Learner; official seen test only, never a new blind test",
            "query": "Evaluator until selection reveal",
            "outer": "Evaluator only",
            "official_boundary_ambiguous": "Evaluator quarantine",
        },
        "parameter_edit_roles": config["parameter_edit_roles"],
        "role_views": view_paths,
        "limitations": limitations,
        "E1_enabled": False,
    }
    write_json(output / "role_manifest.json", role_manifest)
    key_files = [
        "role_assignments.tsv.gz", "allocation_units.tsv.gz", "learner_allowlist.json",
        "evaluator_manifest.json", "role_manifest.json",
    ] + [str(path) for path in view_paths.values()]
    manifest = {
        "schema_version": "1.0",
        "split_version": config["split_version"],
        "split_hash": split_hash,
        "split_hash_method": "SHA256 of canonical raw_row_id/sequence_id/near_group_id/allocation_unit_id/role/official_model_role lines in raw row order",
        "data_manifest_path": config["input_manifest"],
        "data_manifest_sha256": sha256_file(Path(config["input_manifest"])),
        "split_ready_checksums_path": config["split_ready_checksums"],
        "split_ready_checksums_sha256": sha256_file(Path(config["split_ready_checksums"])),
        "config_sha256": sha256_file(Path(args.config)),
        "assignment_rows": sum(role_counts.values()),
        "allocation_units": len(units),
        "role_counts": dict(role_counts),
        "allocation_unit_counts": dict(unit_counts),
        "official_model_evidence": official_metadata,
        "limitations": limitations,
        "role_manifest": str(output / "role_manifest.json"),
        "learner_allowlist": str(output / "learner_allowlist.json"),
        "evaluator_manifest": str(output / "evaluator_manifest.json"),
        "file_hashes": {name: sha256_file(output / name) for name in key_files},
        "outer_results_accessed": False,
        "outer_regret_computed": False,
        "proxy_training_performed": False,
        "E1_enabled": False,
    }
    write_json(output / "split_manifest.json", manifest)
    print(json.dumps({"split_version": config["split_version"], "split_hash": split_hash, "rows": sum(role_counts.values()), "allocation_units": len(units), "role_counts": dict(role_counts), "limitations": limitations, "output": str(output)}))


if __name__ == "__main__":
    main()
