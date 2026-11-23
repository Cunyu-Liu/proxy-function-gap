import argparse
import csv
import gzip
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
    raw_row_hash,
    sha256_file,
    sha256_text,
)
from .grouping import exact_assignment_violations, partition_violations


def rows_gz(path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def round1(config, manifest, output):
    derived_path = output / "sequence_index.tsv.gz"
    derived_rows = rows_gz(derived_path)
    mismatches = []
    raw_ids = set()
    duplicate_raw_ids = 0
    original_index_seen = set()
    original_index_duplicates = Counter()
    total = 0
    per_file = []
    for item in manifest["files"]:
        name = item["name"]
        path = Path(item["path"])
        digest = sha256_file(path)
        rule = config["fixed_formats"][name]
        count = 0
        file_mismatches = 0
        with gzip.open(path, "rt", newline="") as handle:
            for source_row_number, raw in enumerate(csv.DictReader(handle), 1):
                try:
                    derived = next(derived_rows)
                except StopIteration:
                    raise ValueError("sequence_index ended before raw input")
                raw_sequence = raw["utr"]
                normalized, raw_format = normalize_sequence(raw_sequence, rule)
                raw_index = raw.get("") or str(source_row_number - 1)
                expected = {
                    "raw_row_id": make_raw_row_id(item["sample_accession"], source_row_number),
                    "source_row_number": str(source_row_number),
                    "source_file": name,
                    "raw_row_sha256": raw_row_hash(raw),
                    "raw_sequence": raw_sequence,
                    "raw_sequence_sha256": sha256_text(raw_sequence),
                    "normalized_sequence": normalized,
                    "sequence_hash": sha256_text(normalized),
                    "sequence_id": "sample2019_" + sha256_text(normalized),
                    "normalization_version": config["normalization_version"],
                    "raw_format": raw_format,
                    "normalized_length": str(len(normalized)),
                }
                for key, value in expected.items():
                    if derived.get(key) != value:
                        file_mismatches += 1
                        if len(mismatches) < 20:
                            mismatches.append({"source_file": name, "source_row_number": source_row_number, "field": key, "expected": value, "observed": derived.get(key)})
                original_key = (name, raw_index)
                if original_key in original_index_seen:
                    original_index_duplicates[name] += 1
                else:
                    original_index_seen.add(original_key)
                raw_id = expected["raw_row_id"]
                if raw_id in raw_ids:
                    duplicate_raw_ids += 1
                raw_ids.add(raw_id)
                if not derived.get("near_group_id"):
                    file_mismatches += 1
                count += 1
                total += 1
        if count != item["data_rows"] or digest != item["sha256"]:
            file_mismatches += 1
        per_file.append({"name": name, "rows": count, "raw_sha256": digest, "mismatches": file_mismatches})
        print(f"verify_round1_file={name} rows={count} mismatches={file_mismatches}", flush=True)
    extra = sum(1 for _ in derived_rows)
    result = {
        "round": 1,
        "total_raw_rows": total,
        "duplicate_raw_row_ids": duplicate_raw_ids,
        "original_raw_index_duplicate_rows": dict(original_index_duplicates),
        "extra_derived_rows": extra,
        "mismatch_examples": mismatches,
        "files": per_file,
        "pass": total == 2753885 and duplicate_raw_ids == 0 and original_index_duplicates.get("GSM3130443_designed_library.csv.gz") == 46988 and sum(original_index_duplicates.values()) == 46988 and extra == 0 and not mismatches and all(x["mismatches"] == 0 for x in per_file),
    }
    write_json(output / "verification_round1_rows.json", result)
    if not result["pass"]:
        raise ValueError("Round 1 row conservation failed")
    return result


def round2(config, output):
    endpoints = rows_gz(output / "endpoint_coverage.tsv.gz")
    inclusions = rows_gz(output / "inclusion_exclusion.tsv.gz")
    status_counts = Counter()
    missing_reasons = Counter()
    per_file = Counter()
    trajectory_ids = set()
    trajectory_eligible = 0
    errors = []
    total = 0
    for endpoint, inclusion in itertools.zip_longest(endpoints, inclusions):
        if endpoint is None or inclusion is None:
            errors.append("endpoint/inclusion row count mismatch")
            break
        total += 1
        if endpoint["raw_row_id"] != inclusion["raw_row_id"]:
            errors.append("raw_row_id mismatch")
        measured = finite_float(endpoint["measured_rl"])
        status = endpoint["endpoint_status"]
        status_counts[status] += 1
        per_file[endpoint["source_file"]] += 1
        if endpoint["predicted_mrl"] or endpoint["predicted_status"] != "not_available_raw" or inclusion["predicted_status"] != "not_available_raw":
            errors.append("unexpected predicted MRL or status")
        if status == "missing":
            missing_reasons[endpoint["missing_reason"]] += 1
            if measured is not None or inclusion["inclusion_status"] != "excluded_missing_rl" or inclusion["split_ready_label_eligible"] != "false":
                errors.append("missing-label inclusion mismatch")
            if finite_float(endpoint["total"]) != 0:
                errors.append("missing rl with nonzero total")
        elif status == "measured":
            if measured is None or inclusion["inclusion_status"] != "included_label_available" or inclusion["split_ready_label_eligible"] != "true":
                errors.append("measured-label inclusion mismatch")
        else:
            errors.append("unknown endpoint status")
        if inclusion["trajectory_replay_eligible"] == "true":
            trajectory_eligible += 1
            trajectory_ids.add(endpoint["trajectory_id"])
            if not endpoint["round"]:
                errors.append("trajectory eligible row without round")
        elif endpoint["trajectory_id"] or endpoint["round"]:
            errors.append("unexpected trajectory metadata on ineligible row")
        if inclusion["proposed_split_unit"] != "hamming1_component" or inclusion["freeze_status"] != "PROPOSED_NOT_FROZEN":
            errors.append("unexpected split/freeze state")
        if len(errors) >= 20:
            break
    unique_rows = 0
    unique_row_sum = 0
    unique_missing_sum = 0
    unique_errors = 0
    for row in rows_gz(output / "unique_sequences.tsv.gz"):
        unique_rows += 1
        unique_row_sum += int(row["row_count"])
        unique_missing_sum += int(row["missing_row_count"])
        if row["sequence_hash"] != sha256_text(row["normalized_sequence"]):
            unique_errors += 1
        if row["sequence_id"] != "sample2019_" + row["sequence_hash"]:
            unique_errors += 1
        if int(row["normalized_length"]) != len(row["normalized_sequence"]):
            unique_errors += 1
        if int(row["predicted_row_count"]) != 0:
            unique_errors += 1
    group_rows = 0
    group_row_sum = 0
    group_missing_sum = 0
    group_errors = 0
    for row in rows_gz(output / "split_ready_groups.tsv.gz"):
        group_rows += 1
        group_row_sum += int(row["row_count"])
        group_missing_sum += int(row["missing_row_count"])
        if row["freeze_status"] != "PROPOSED_NOT_FROZEN" or row["outer_results_accessed"] != "false":
            group_errors += 1
        if int(row["predicted_row_count"]) != 0:
            group_errors += 1
    result = {
        "round": 2,
        "endpoint_rows": total,
        "status_counts": dict(status_counts),
        "missing_reasons": dict(missing_reasons),
        "per_file_rows": dict(per_file),
        "trajectory_eligible_rows": trajectory_eligible,
        "trajectory_count": len(trajectory_ids),
        "unique_sequence_rows": unique_rows,
        "unique_row_sum": unique_row_sum,
        "unique_missing_sum": unique_missing_sum,
        "unique_errors": unique_errors,
        "split_group_rows": group_rows,
        "group_row_sum": group_row_sum,
        "group_missing_sum": group_missing_sum,
        "group_errors": group_errors,
        "errors": errors,
    }
    result["pass"] = (
        total == 2753885 and status_counts.get("missing") == 3193 and trajectory_eligible == 5710
        and len(trajectory_ids) == 80 and not errors and unique_row_sum == 2753885
        and unique_missing_sum == 3193 and unique_errors == 0 and group_row_sum == 2753885
        and group_missing_sum == 3193 and group_errors == 0
    )
    write_json(output / "verification_round2_fields.json", result)
    if not result["pass"]:
        raise ValueError("Round 2 field/missing verification failed")
    return result


def round3(config, output):
    sequences = {}
    for row in rows_gz(output / "unique_sequences.tsv.gz"):
        sequences[row["sequence_id"]] = (row["normalized_sequence"], row["near_group_id"])
    exact_violations = []
    exact_row_count = 0
    replicate_stats = defaultdict(lambda: {"rows": 0, "sequence_ids": set(), "measured_rows": 0, "missing_rows": 0, "trajectory_ids": set(), "roles": set(), "groups": set()})
    for row in rows_gz(output / "sequence_index.tsv.gz"):
        exact_row_count += 1
        replicate = replicate_stats[row["replicate_id"]]
        replicate["rows"] += 1
        replicate["sequence_ids"].add(row["sequence_id"])
        replicate["measured_rows"] += int(row["label_status"] == "measured")
        replicate["missing_rows"] += int(row["label_status"] == "missing")
        replicate["roles"].add(row["replicate_role"])
        replicate["groups"].add(row["replicate_group"])
        if row["trajectory_id"]:
            replicate["trajectory_ids"].add(row["trajectory_id"])
        if sequences[row["sequence_id"]][1] != row["near_group_id"] and len(exact_violations) < 20:
            exact_violations.append({"raw_row_id": row["raw_row_id"], "sequence_id": row["sequence_id"], "derived_group": row["near_group_id"], "unique_table_group": sequences[row["sequence_id"]][1]})
    actual_violations = []
    edge_count = 0
    cross_file_edges = 0
    first_edge = None
    for edge in rows_gz(output / "near_duplicate_edges.tsv.gz"):
        edge_count += 1
        left_id = edge["sequence_id_1"]
        right_id = edge["sequence_id_2"]
        left, left_group = sequences[left_id]
        right, right_group = sequences[right_id]
        if hamming_distance(left, right) != 1 or left_group != right_group:
            if len(actual_violations) < 20:
                actual_violations.append(edge)
        if edge["cross_file"] == "true":
            cross_file_edges += 1
        if first_edge is None:
            first_edge = (left, right)
    negative = []
    synthetic_left = "A" * 50
    synthetic_right = "A" * 49 + "C"
    synthetic_violations = partition_violations([(synthetic_left, synthetic_right)], {synthetic_left: "group_A", synthetic_right: "group_B"})
    negative.append({"test": "synthetic_hamming1_cross_group", "expected": "reject", "observed": "reject" if synthetic_violations else "accept", "pass": bool(synthetic_violations)})
    if first_edge is not None:
        real_violations = partition_violations([first_edge], {first_edge[0]: "group_A", first_edge[1]: "group_B"})
        negative.append({"test": "real_hamming1_cross_group", "expected": "reject", "observed": "reject" if real_violations else "accept", "pass": bool(real_violations), "sequences": [first_edge[0], first_edge[1]]})
    else:
        negative.append({"test": "real_hamming1_cross_group", "expected": "reject", "observed": "not_run_no_edges", "pass": False})
    exact_variant = None
    for row in rows_gz(output / "unique_sequences.tsv.gz"):
        if "|" in row["raw_sequences"]:
            exact_variant = row
            break
    if exact_variant:
        variants = exact_variant["raw_sequences"].split("|")
        fake_records = [
            {"raw_row_id": "raw_format_1", "sequence_id": exact_variant["sequence_id"], "near_group_id": "group_A"},
            {"raw_row_id": "raw_format_2", "sequence_id": exact_variant["sequence_id"], "near_group_id": "group_B"},
        ]
        violations = exact_assignment_violations(fake_records)
        negative.append({"test": "raw_length_variant_exact_normalized_grouping", "expected": "reject deliberate cross-group assignment of raw 59/54 nt variants sharing one normalized sequence", "observed": "reject" if violations else "accept", "pass": bool(violations), "sequence_id": exact_variant["sequence_id"], "actual_near_group_id": exact_variant["near_group_id"], "raw_sequences": variants})
    else:
        negative.append({"test": "raw_length_variant_exact_normalized_grouping", "expected": "reject", "observed": "not_run_no_variants", "pass": False})
    replicate_example = None
    for row in rows_gz(output / "unique_sequences.tsv.gz"):
        if int(row["source_file_count"]) > 1:
            replicate_example = row
            break
    if replicate_example:
        files = replicate_example["source_files"].split("|")
        fake_records = [
            {"raw_row_id": files[0], "sequence_id": replicate_example["sequence_id"], "near_group_id": "group_A"},
            {"raw_row_id": files[1], "sequence_id": replicate_example["sequence_id"], "near_group_id": "group_B"},
        ]
        violations = exact_assignment_violations(fake_records)
        negative.append({"test": "replicate_rows_must_share_group", "expected": "reject independent replicate split", "observed": "reject" if violations else "accept", "pass": bool(violations), "sequence_id": replicate_example["sequence_id"], "source_files": files, "replicate_ids": replicate_example["replicate_ids"]})
    else:
        negative.append({"test": "replicate_rows_must_share_group", "expected": "reject", "observed": "not_run", "pass": False})
    replicate_records = []
    for replicate_id, value in sorted(replicate_stats.items()):
        replicate_records.append({
            "replicate_id": replicate_id,
            "replicate_groups": sorted(value["groups"]),
            "replicate_roles": sorted(value["roles"]),
            "rows": value["rows"],
            "unique_normalized_sequences": len(value["sequence_ids"]),
            "measured_rows": value["measured_rows"],
            "missing_rows": value["missing_rows"],
            "trajectory_count": len(value["trajectory_ids"]),
            "independent_biological_evidence": False,
        })
    write_json(output / "replicate_summary.json", {
        "replicates": replicate_records,
        "relationship_rule": "Experimental replicates of one normalized sequence remain in the same split group and are not independent biological evidence.",
        "unmodified_designed_raw_exact_overlap": 4906,
        "outer_results_accessed": False,
    })
    result = {
        "round": 3,
        "exact_assignment_row_count": exact_row_count,
        "exact_assignment_violations": exact_violations,
        "edge_count": edge_count,
        "cross_file_edge_count": cross_file_edges,
        "actual_partition_violations": actual_violations,
        "negative_tests": negative,
        "replicate_measurements_are_independent": False,
        "replicate_summary_path": str(output / "replicate_summary.json"),
        "outer_results_accessed": False,
        "pass": exact_row_count == 2753885 and not exact_violations and not actual_violations and all(test["pass"] for test in negative),
    }
    write_json(output / "negative_leakage_tests.json", result)
    write_json(output / "verification_round3_leakage.json", result)
    if not result["pass"]:
        raise ValueError("Round 3 leakage verification failed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["output_dir"])
    manifest = json.loads(Path(config["input_manifest"]).read_text())
    for name in ["verification_round1_rows.json", "verification_round2_fields.json", "verification_round3_leakage.json", "verification_summary.json"]:
        if (output / name).exists():
            raise FileExistsError(output / name)
    first = round1(config, manifest, output)
    second = round2(config, output)
    third = round3(config, output)
    summary = {
        "status": "SPLIT_READY_ARTIFACTS_VERIFIED_PROPOSED_NOT_FROZEN",
        "normalization_version": config["normalization_version"],
        "rounds": {"row_conservation": first["pass"], "fields_missing": second["pass"], "leakage_negative_tests": third["pass"]},
        "rows": first["total_raw_rows"],
        "missing_rl_excluded": second["status_counts"].get("missing"),
        "unique_normalized_sequences": second["unique_sequence_rows"],
        "split_ready_groups": second["split_group_rows"],
        "near_duplicate_edges": third["edge_count"],
        "outer_results_accessed": False,
        "proxy_training_performed": False,
        "E1_evaluation_performed": False,
        "E1_authorized": False,
        "remaining_blockers": [
            "Main agent must review and freeze the actual development/query/outer partition; this task only supplies split-ready groups.",
            "Predicted MRL remains unavailable in raw data and must be generated separately before trajectory replay or E1.",
            "Figure 3 reproduction and all scientific evaluation remain pending and were not attempted.",
        ],
    }
    write_json(output / "verification_summary.json", summary)
    names = [path.name for path in output.iterdir() if path.is_file() and path.name not in {"artifact_checksums.json", "code_provenance.json"}]
    checksums = {name: {"bytes": (output / name).stat().st_size, "sha256": sha256_file(output / name)} for name in sorted(names)}
    write_json(output / "artifact_checksums.json", checksums)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
