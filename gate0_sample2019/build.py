import argparse
import csv
import gzip
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import (
    finite_float,
    hamming_distance,
    load_config,
    make_raw_row_id,
    normalize_sequence,
    open_tsv_gz,
    raw_row_hash,
    sha256_file,
    sha256_text,
    source_metadata,
    write_row,
)
from .grouping import find, partition_violations, union


SEQUENCE_INDEX_FIELDS = [
    "raw_row_id", "source_row_number", "source_file", "gsm", "raw_index", "raw_row_sha256",
    "raw_sequence", "raw_sequence_sha256", "normalized_sequence", "sequence_hash", "sequence_id",
    "normalization_version", "raw_format", "normalized_length", "near_group_id", "library_type",
    "source_library", "library_family", "replicate_id", "replicate_role", "replicate_group",
    "trajectory_id", "round", "cds", "chemistry", "cell", "mother", "designed_flag",
    "match_score", "source_id", "info1", "info2", "measured_rl", "label_status",
]
ENDPOINT_FIELDS = [
    "raw_row_id", "sequence_id", "near_group_id", "source_file", "source_library", "trajectory_id",
    "round", "replicate_id", "measured_rl", "measured_status", "predicted_mrl", "predicted_status",
    "endpoint_status", "missing_reason", "total_reads", "total",
]
INCLUSION_FIELDS = [
    "raw_row_id", "sequence_id", "near_group_id", "split_ready_label_eligible",
    "trajectory_replay_eligible", "inclusion_status", "reason_codes", "measured_status",
    "predicted_status", "proposed_split_unit", "freeze_status",
]
UNIQUE_FIELDS = [
    "sequence_id", "normalized_sequence", "sequence_hash", "normalized_length", "raw_sequences",
    "raw_sequence_hashes", "raw_formats", "normalization_version", "near_group_id", "near_group_size",
    "row_count", "measured_row_count", "missing_row_count", "predicted_row_count", "source_file_count",
    "source_files", "source_libraries", "trajectory_count", "trajectory_ids", "replicate_ids",
    "replicate_group_count", "cross_file_exact",
]
GROUP_FIELDS = [
    "near_group_id", "member_sequence_count", "row_count", "measured_row_count", "missing_row_count",
    "predicted_row_count", "source_file_count", "source_files", "source_libraries", "trajectory_count",
    "trajectory_ids", "replicate_ids", "replicate_group_count", "normalized_lengths", "raw_formats",
    "cross_file_group", "proposed_split_unit", "freeze_status", "outer_results_accessed",
    "independent_evidence_rule",
]
EDGE_FIELDS = ["sequence_id_1", "sequence_id_2", "normalized_length", "hamming_distance", "cross_file"]


def new_aggregate(sequence, sequence_id):
    return {
        "sequence": sequence,
        "sequence_id": sequence_id,
        "raw_sequences": set(),
        "raw_hashes": set(),
        "formats": set(),
        "lengths": set(),
        "rows": 0,
        "measured": 0,
        "missing": 0,
        "files": defaultdict(lambda: [0, 0, 0]),
        "libraries": set(),
        "trajectories": set(),
        "replicates": set(),
        "replicate_groups": set(),
    }


def trajectory_fields(row, source_library):
    if not source_library.startswith("step_") or row.get("designed") != "True":
        return "", "", False
    generation = finite_float(row.get("info2"))
    if generation is None or generation != int(generation) or not row.get("id"):
        return "", "", False
    return f"{source_library}:{row['id']}", str(int(generation)), True


def build_near_groups(aggregates):
    parent = {sequence: sequence for sequence in aggregates}
    sizes = {sequence: 1 for sequence in aggregates}
    buckets = defaultdict(list)
    for number, sequence in enumerate(aggregates, 1):
        midpoint = len(sequence) // 2
        buckets[(len(sequence), "L", sequence[:midpoint])].append(sequence)
        buckets[(len(sequence), "R", sequence[midpoint:])].append(sequence)
        if number % 200000 == 0:
            print(f"near_bucket_build={number}", flush=True)
    edges = set()
    candidate_pairs = 0
    max_bucket = 0
    for bucket in buckets.values():
        max_bucket = max(max_bucket, len(bucket))
        if len(bucket) < 2:
            continue
        if len(bucket) > 20000:
            raise RuntimeError(f"Candidate bucket too large for audited pairwise comparison: {len(bucket)}")
        for left, right in itertools.combinations(bucket, 2):
            candidate_pairs += 1
            if hamming_distance(left, right) == 1:
                edge = tuple(sorted((left, right)))
                edges.add(edge)
                union(parent, sizes, left, right)
    components = defaultdict(list)
    for sequence in aggregates:
        components[find(parent, sequence)].append(sequence)
    group_of = {}
    members_of = {}
    for members in components.values():
        members.sort(key=lambda sequence: aggregates[sequence]["sequence_id"])
        group_id = "h1_" + sha256_text("\n".join(aggregates[sequence]["sequence_id"] for sequence in members))[:24]
        for sequence in members:
            group_of[sequence] = group_id
        members_of[group_id] = members
    return group_of, members_of, edges, {"candidate_pairs": candidate_pairs, "max_bucket_size": max_bucket, "bucket_count": len(buckets)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["output_dir"])
    if not (output / "length_format_validation.json").exists():
        raise FileNotFoundError("Run inspect_formats first")
    protected = [p for p in output.iterdir() if p.name != "length_format_validation.json"]
    if protected:
        raise FileExistsError(f"Derived output is not empty: {protected}")
    manifest_path = Path(config["input_manifest"])
    manifest = json.loads(manifest_path.read_text())
    aggregates = {}
    file_sets = {item["name"]: {"raw": set(), "normalized": set()} for item in manifest["files"]}
    file_reports = []
    total_rows = 0
    total_missing = 0
    for item in manifest["files"]:
        name = item["name"]
        path = Path(item["path"])
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise ValueError(f"Raw hash changed: {name}")
        rule = config["fixed_formats"][name]
        gsm, library_type, family, replicate_id, replicate_role, cds, chemistry = source_metadata(name)
        rows = 0
        missing = 0
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                raw = row["utr"]
                normalized, raw_format = normalize_sequence(raw, rule)
                sequence_id = "sample2019_" + sha256_text(normalized)
                if normalized not in aggregates:
                    aggregates[normalized] = new_aggregate(normalized, sequence_id)
                aggregate = aggregates[normalized]
                source_library = row.get("library") or item["sample_title"]
                trajectory_id, generation, _ = trajectory_fields(row, source_library)
                measured = finite_float(row.get("rl"))
                aggregate["rows"] += 1
                aggregate["raw_sequences"].add(raw)
                aggregate["raw_hashes"].add(sha256_text(raw))
                aggregate["formats"].add(raw_format)
                aggregate["lengths"].add(len(normalized))
                aggregate["libraries"].add(source_library)
                aggregate["replicates"].add(replicate_id)
                aggregate["replicate_groups"].add(family)
                if trajectory_id:
                    aggregate["trajectories"].add(trajectory_id)
                file_stat = aggregate["files"][name]
                file_stat[0] += 1
                if measured is None:
                    aggregate["missing"] += 1
                    file_stat[2] += 1
                    missing += 1
                else:
                    aggregate["measured"] += 1
                    file_stat[1] += 1
                file_sets[name]["raw"].add(raw)
                file_sets[name]["normalized"].add(normalized)
        total_rows += rows
        total_missing += missing
        file_reports.append({"name": name, "rows": rows, "missing_rl": missing, "raw_sha256": digest, "unique_normalized": len(file_sets[name]["normalized"])})
        print(f"aggregate_file={name} rows={rows} missing={missing} unique_normalized={len(file_sets[name]['normalized'])}", flush=True)
    if total_rows != 2753885 or total_missing != 3193:
        raise ValueError(f"Unexpected row/missing totals: {total_rows}, {total_missing}")
    group_of, members_of, edges, near_stats = build_near_groups(aggregates)
    print(f"near_groups_built groups={len(members_of)} edges={len(edges)} stats={near_stats}", flush=True)
    group_rows = {}
    for group_id, members in members_of.items():
        group_rows[group_id] = {
            "near_group_id": group_id,
            "member_sequence_count": len(members),
            "row_count": 0,
            "measured_row_count": 0,
            "missing_row_count": 0,
            "predicted_row_count": 0,
            "source_files": set(),
            "source_libraries": set(),
            "trajectory_ids": set(),
            "replicate_ids": set(),
            "replicate_groups": set(),
            "normalized_lengths": set(),
            "raw_formats": set(),
        }
    for sequence, aggregate in aggregates.items():
        group = group_rows[group_of[sequence]]
        group["row_count"] += aggregate["rows"]
        group["measured_row_count"] += aggregate["measured"]
        group["missing_row_count"] += aggregate["missing"]
        group["source_files"].update(aggregate["files"])
        group["source_libraries"].update(aggregate["libraries"])
        group["trajectory_ids"].update(aggregate["trajectories"])
        group["replicate_ids"].update(aggregate["replicates"])
        group["replicate_groups"].update(aggregate["replicate_groups"])
        group["normalized_lengths"].update(aggregate["lengths"])
        group["raw_formats"].update(aggregate["formats"])
    handles = []
    writers = {}
    try:
        for key, filename, fields in [
            ("sequence_index", "sequence_index.tsv.gz", SEQUENCE_INDEX_FIELDS),
            ("endpoint", "endpoint_coverage.tsv.gz", ENDPOINT_FIELDS),
            ("inclusion", "inclusion_exclusion.tsv.gz", INCLUSION_FIELDS),
            ("unique", "unique_sequences.tsv.gz", UNIQUE_FIELDS),
            ("groups", "split_ready_groups.tsv.gz", GROUP_FIELDS),
            ("edges", "near_duplicate_edges.tsv.gz", EDGE_FIELDS),
        ]:
            handle, writer = open_tsv_gz(output / filename, fields)
            handles.append(handle)
            writers[key] = writer
        for sequence in sorted(aggregates, key=lambda value: aggregates[value]["sequence_id"]):
            aggregate = aggregates[sequence]
            group_id = group_of[sequence]
            write_row(writers["unique"], {
                "sequence_id": aggregate["sequence_id"],
                "normalized_sequence": sequence,
                "sequence_hash": sha256_text(sequence),
                "normalized_length": len(sequence),
                "raw_sequences": sorted(aggregate["raw_sequences"]),
                "raw_sequence_hashes": sorted(aggregate["raw_hashes"]),
                "raw_formats": sorted(aggregate["formats"]),
                "normalization_version": config["normalization_version"],
                "near_group_id": group_id,
                "near_group_size": len(members_of[group_id]),
                "row_count": aggregate["rows"],
                "measured_row_count": aggregate["measured"],
                "missing_row_count": aggregate["missing"],
                "predicted_row_count": 0,
                "source_file_count": len(aggregate["files"]),
                "source_files": sorted(aggregate["files"]),
                "source_libraries": sorted(aggregate["libraries"]),
                "trajectory_count": len(aggregate["trajectories"]),
                "trajectory_ids": sorted(aggregate["trajectories"]),
                "replicate_ids": sorted(aggregate["replicates"]),
                "replicate_group_count": len(aggregate["replicate_groups"]),
                "cross_file_exact": len(aggregate["files"]) > 1,
            })
        for group_id in sorted(group_rows):
            group = group_rows[group_id]
            write_row(writers["groups"], {
                **group,
                "source_file_count": len(group["source_files"]),
                "source_files": sorted(group["source_files"]),
                "source_libraries": sorted(group["source_libraries"]),
                "trajectory_count": len(group["trajectory_ids"]),
                "trajectory_ids": sorted(group["trajectory_ids"]),
                "replicate_ids": sorted(group["replicate_ids"]),
                "replicate_group_count": len(group["replicate_groups"]),
                "normalized_lengths": sorted(group["normalized_lengths"]),
                "raw_formats": sorted(group["raw_formats"]),
                "cross_file_group": len(group["source_files"]) > 1,
                "proposed_split_unit": "hamming1_component",
                "freeze_status": "PROPOSED_NOT_FROZEN",
                "outer_results_accessed": False,
                "independent_evidence_rule": "Keep the full Hamming-1 component together; experimental replicates and rows from one trajectory are not independent evidence.",
            })
        edge_file_pair_counts = Counter()
        cross_file_edges = 0
        for left, right in sorted(edges):
            left_id = aggregates[left]["sequence_id"]
            right_id = aggregates[right]["sequence_id"]
            files_left = set(aggregates[left]["files"])
            files_right = set(aggregates[right]["files"])
            cross = len(files_left | files_right) > 1
            cross_file_edges += int(cross)
            for file_left, file_right in itertools.product(files_left, files_right):
                if file_left != file_right:
                    edge_file_pair_counts[tuple(sorted((file_left, file_right)))] += 1
            write_row(writers["edges"], {
                "sequence_id_1": left_id,
                "sequence_id_2": right_id,
                "normalized_length": len(left),
                "hamming_distance": 1,
                "cross_file": cross,
            })
        for item in manifest["files"]:
            name = item["name"]
            path = Path(item["path"])
            rule = config["fixed_formats"][name]
            gsm, library_type, family, replicate_id, replicate_role, cds, chemistry = source_metadata(name)
            with gzip.open(path, "rt", newline="") as handle:
                for source_row_number, row in enumerate(csv.DictReader(handle), 1):
                    raw = row["utr"]
                    normalized, raw_format = normalize_sequence(raw, rule)
                    aggregate = aggregates[normalized]
                    sequence_id = aggregate["sequence_id"]
                    group_id = group_of[normalized]
                    raw_index = row.get("") or str(source_row_number - 1)
                    raw_row_id = make_raw_row_id(gsm, source_row_number)
                    source_library = row.get("library") or item["sample_title"]
                    trajectory_id, generation, trajectory_eligible = trajectory_fields(row, source_library)
                    measured = finite_float(row.get("rl"))
                    total = finite_float(row.get("total"))
                    missing_reason = "" if measured is not None else "missing_rl_total_zero" if total == 0 else "missing_rl"
                    measured_status = "measured" if measured is not None else "missing"
                    label_status = "measured" if measured is not None else "missing"
                    predicted_status = "not_available_raw"
                    write_row(writers["sequence_index"], {
                        "raw_row_id": raw_row_id,
                        "source_row_number": source_row_number,
                        "source_file": name,
                        "gsm": gsm,
                        "raw_index": raw_index,
                        "raw_row_sha256": raw_row_hash(row),
                        "raw_sequence": raw,
                        "raw_sequence_sha256": sha256_text(raw),
                        "normalized_sequence": normalized,
                        "sequence_hash": sha256_text(normalized),
                        "sequence_id": sequence_id,
                        "normalization_version": config["normalization_version"],
                        "raw_format": raw_format,
                        "normalized_length": len(normalized),
                        "near_group_id": group_id,
                        "library_type": library_type,
                        "source_library": source_library,
                        "library_family": family,
                        "replicate_id": replicate_id,
                        "replicate_role": replicate_role,
                        "replicate_group": family,
                        "trajectory_id": trajectory_id,
                        "round": generation,
                        "cds": cds,
                        "chemistry": chemistry,
                        "cell": "HEK293T",
                        "mother": row.get("mother", ""),
                        "designed_flag": row.get("designed", ""),
                        "match_score": row.get("match_score", ""),
                        "source_id": row.get("id", ""),
                        "info1": row.get("info1", ""),
                        "info2": row.get("info2", ""),
                        "measured_rl": measured,
                        "label_status": label_status,
                    })
                    write_row(writers["endpoint"], {
                        "raw_row_id": raw_row_id,
                        "sequence_id": sequence_id,
                        "near_group_id": group_id,
                        "source_file": name,
                        "source_library": source_library,
                        "trajectory_id": trajectory_id,
                        "round": generation,
                        "replicate_id": replicate_id,
                        "measured_rl": measured,
                        "measured_status": measured_status,
                        "predicted_mrl": None,
                        "predicted_status": predicted_status,
                        "endpoint_status": label_status,
                        "missing_reason": missing_reason,
                        "total_reads": row.get("total_reads", ""),
                        "total": row.get("total", ""),
                    })
                    eligible = measured is not None
                    write_row(writers["inclusion"], {
                        "raw_row_id": raw_row_id,
                        "sequence_id": sequence_id,
                        "near_group_id": group_id,
                        "split_ready_label_eligible": eligible,
                        "trajectory_replay_eligible": trajectory_eligible,
                        "inclusion_status": "included_label_available" if eligible else "excluded_missing_rl",
                        "reason_codes": "" if eligible else missing_reason,
                        "measured_status": measured_status,
                        "predicted_status": predicted_status,
                        "proposed_split_unit": "hamming1_component",
                        "freeze_status": "PROPOSED_NOT_FROZEN",
                    })
            print(f"row_outputs_written={name}", flush=True)
    finally:
        for handle in handles:
            handle.close()
    exact_pairs = []
    for left, right in itertools.combinations(file_sets, 2):
        exact_pairs.append({
            "left": left,
            "right": right,
            "raw_exact_intersection": len(file_sets[left]["raw"] & file_sets[right]["raw"]),
            "normalized_exact_intersection": len(file_sets[left]["normalized"] & file_sets[right]["normalized"]),
        })
    unmod_designed = next(x for x in exact_pairs if x["left"] == "GSM3130435_egfp_unmod_1.csv.gz" and x["right"] == "GSM3130443_designed_library.csv.gz")
    if unmod_designed["raw_exact_intersection"] != 4906:
        raise ValueError(f"Expected raw unmod/designed overlap 4906, got {unmod_designed}")
    overlap = {
        "normalization_version": config["normalization_version"],
        "raw_sequence_policy": "Raw sequences retained; normalized exact identity uses the validated author-aligned UTR insert.",
        "exact_file_pairs": exact_pairs,
        "unmodified_designed_reference": unmod_designed,
        "near_duplicate": {
            **near_stats,
            "metric": "Hamming distance 1, same normalized length",
            "edge_count": len(edges),
            "cross_file_edge_count": cross_file_edges,
            "file_pair_edge_counts": [{"left": left, "right": right, "edge_count": count} for (left, right), count in sorted(edge_file_pair_counts.items())],
            "candidate_guarantee": "Distance-1 sequences share an exact left or right half; every candidate is verified over the full sequence.",
        },
        "variable_length_policy": "Exact identity is computed on full variable-length sequences. Near edges are computed only between equal normalized lengths; cross-length similarity is not defined or imputed.",
        "outer_results_accessed": False,
    }
    (output / "overlap_report.json").write_text(json.dumps(overlap, indent=2) + "\n")
    recommendation = {
        "status": "PROPOSED_NOT_FROZEN",
        "split_unit": "hamming1_component",
        "rules": [
            "Assign every exact normalized sequence and its full Hamming-1 component to one partition.",
            "Keep all experimental replicates of the same normalized sequence in one partition; replicate measurements are not independent evidence.",
            "Use complete trajectories as outer units where trajectory_id is present; do not split rounds of a trajectory.",
            "Use study/library families as additional blocking variables and audit normalized exact and near overlap before any split is frozen.",
            "The 4906 raw unmodified/designed exact overlaps and all normalized overlaps must remain inside one group.",
        ],
        "outer_results_accessed": False,
        "proxy_training_performed": False,
        "E1_evaluation_performed": False,
        "freeze_requires": ["three verification rounds pass", "negative leakage tests reject deliberately invalid partitions", "main agent protocol approval"],
    }
    (output / "split_freeze_recommendation.json").write_text(json.dumps(recommendation, indent=2) + "\n")
    source_files = sorted(Path(__file__).parent.glob("*.py")) + [Path(__file__).parent.parent / "configs/gate0_v1.json"]
    code_hashes = {str(path): sha256_file(path) for path in source_files}
    derived_names = [
        "sequence_index.tsv.gz", "endpoint_coverage.tsv.gz", "inclusion_exclusion.tsv.gz",
        "unique_sequences.tsv.gz", "split_ready_groups.tsv.gz", "near_duplicate_edges.tsv.gz",
        "overlap_report.json", "split_freeze_recommendation.json", "length_format_validation.json",
    ]
    report = {
        "status": "BUILD_COMPLETE_VERIFICATION_PENDING",
        "normalization_version": config["normalization_version"],
        "config_sha256": sha256_file(Path(args.config)),
        "input_manifest_sha256": sha256_file(manifest_path),
        "code_sha256": code_hashes,
        "rows": total_rows,
        "missing_rl": total_missing,
        "unique_normalized_sequences": len(aggregates),
        "near_group_count": len(members_of),
        "near_edge_count": len(edges),
        "file_reports": file_reports,
        "derived_files": {name: {"bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)} for name in derived_names},
        "outer_results_accessed": False,
        "training_performed": False,
        "E1_evaluation_performed": False,
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "rows": total_rows, "missing_rl": total_missing, "unique_sequences": len(aggregates), "near_groups": len(members_of), "near_edges": len(edges), "output": str(output)}))


if __name__ == "__main__":
    main()
