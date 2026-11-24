import argparse
import collections
import itertools
import math
from pathlib import Path

from .build import CDS_SHEET, POOLED_SHEET, source_path, validate_inputs
from .common import (
    canonical_json,
    distribution,
    finite_float,
    levenshtein_bit,
    levenshtein_full,
    load_config,
    normalize_rna,
    raw_row_hash,
    read_json,
    read_tsv,
    sha256_file,
    utc_now,
    write_json,
)
from .endpoints import ALL_ENDPOINTS
from .xlsx import read_table


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def raw_sources(config):
    pooled = read_table(source_path(config, "pooled_attributes"), POOLED_SHEET)[1]
    cds = read_table(source_path(config, "cds_attributes"), CDS_SHEET)[1]
    return {row["Sequence ID"]: row for row in pooled + cds}


def round1(config, output):
    inputs = validate_inputs(config)
    raw = raw_sources(config)
    sequences = read_tsv(output / "sequence_index.tsv.gz")
    coverage = read_tsv(output / "endpoint_coverage.tsv.gz")
    measurements = read_tsv(output / "endpoint_measurements.tsv.gz")
    require(measurements == [item for item in coverage if item["status"] in {"measured", "predicted"}], "Measurement/coverage long-table mismatch")
    require(len(raw) == 257 and len(sequences) == 257, "Primary row count mismatch")
    require({row["source_construct_id"] for row in sequences} == set(raw), "Construct ID mismatch")
    for derived in sequences:
        row = raw[derived["source_construct_id"]]
        require(raw_row_hash(row) == derived["raw_row_sha256"], f"Raw row hash mismatch: {derived['source_construct_id']}")
        require(str(derived["source_row"]) == str(row["__row_number__"]), "Source row mismatch")
        expected_sequence = row["RNA sequence"] if derived["source_family"] == "pooled_233" else row["HBB_sequence"]
        require(derived["raw_rna_sequence"] == expected_sequence, "Raw RNA mismatch")
        require(derived["normalized_sequence"] == normalize_rna(expected_sequence), "Normalized RNA mismatch")
    for item in coverage:
        row = raw[item["source_construct_id"]]
        column = item["source_column"]
        expected = row.get(column, "") if column else ""
        require(item["raw_value"] == expected, f"Cell mismatch: {item['source_construct_id']}:{column}")
        require(str(item["source_row"]) == str(row["__row_number__"]), "Coverage source row mismatch")
        if item["endpoint"] == "predicted_expression":
            require(item["status"] in {"predicted", "missing"}, "Predicted expression mislabeled")
    manifest = read_json(output / "data_manifest.json")
    crosscheck = read_json(output / "supplementary_crosscheck.json")
    require(crosscheck["pooled_GEO_vs_Supplementary_Data1"]["differences"] == [], "Pooled cross-source differences")
    require(crosscheck["cds_GEO_vs_Supplementary_Data6"]["differences"] == [], "CDS cross-source differences")
    require(crosscheck["expression_mapping_all_within_one_percent"] is True, "Source Data expression mapping mismatch")
    require(len(crosscheck["source_data_fig4c_vs_GEO_expression"]) == 96, "Incomplete Source Data expression crosscheck")
    require(manifest["prediction_never_relabeled_as_measurement"] is True, "Manifest prediction guard missing")
    require(set(manifest["sources"]) == set(inputs), "Manifest source set mismatch")
    return {"source_files_verified": len(inputs), "raw_rows_verified": len(raw), "coverage_cells_verified": len(coverage), "raw_row_hashes_verified": len(sequences)}


def round2(config, output):
    raw = raw_sources(config)
    sequences = read_tsv(output / "sequence_index.tsv.gz")
    coverage = read_tsv(output / "endpoint_coverage.tsv.gz")
    summaries = read_tsv(output / "distribution_summary.tsv")
    for item in sequences:
        row = raw[item["source_construct_id"]]
        sequence = normalize_rna(item["raw_rna_sequence"])
        if item["source_family"] == "pooled_233":
            standard = "GGG" + "".join(row[column] for column in ["Sequence 5 UTR", "Sequence CDS", "Sequence Constant1", "Sequence Barcode", "Sequence Constant2", "Sequence 3 UTR", "Sequence Constant3"])
            alternate = "GGG" + "".join(row[column] for column in ["Sequence 5 UTR", "Sequence CDS", "Sequence 3 UTR", "Sequence Constant1", "Sequence Barcode", "Sequence Constant2", "Sequence Constant3"])
            require((sequence == standard) != (sequence == alternate), f"Assembly not uniquely verified: {item['source_construct_id']}")
            require(int(float(row["RNA length"])) == len(sequence), "Reported RNA length mismatch")
        else:
            require(normalize_rna(row["Sequence"]) == normalize_rna(row["CDS_sequence"]), "CDS copy mismatch")
            require(normalize_rna(row["CDS_sequence"]) in sequence, "CDS not contained in HBB full sequence")
    by_key = {(item["source_construct_id"], item["endpoint"]): item for item in coverage}
    require(len(by_key) == 257 * len(ALL_ENDPOINTS), "Endpoint grid is not complete and unique")
    for item in coverage:
        present = item["raw_value"] != ""
        numeric = finite_float(item["raw_value"])
        if not present:
            require(item["status"] == "missing" and item["numeric_value"] == "", "Missing value was imputed")
        elif item["status"] == "missing":
            require(item["numeric_value"] == "", "Sentinel/missing value was imputed")
            sentinels = config.get("source_sentinels", {}).get(item["source_column"], [])
            require(numeric is not None and any(numeric == float(value) for value in sentinels), "Unexpected missing source value")
        else:
            require(numeric is not None and finite_float(item["numeric_value"]) == numeric, "Numeric conversion mismatch")
            require(item["status"] in {"measured", "predicted"}, "Invalid value status")
    for summary in summaries:
        selected = [
            finite_float(item["numeric_value"]) for item in coverage
            if item["source_family"] == summary["source_family"] and item["endpoint"] == summary["endpoint"] and item["status"] == summary["status"]
        ]
        expected = distribution(selected)
        require(int(summary["n"]) == expected["n"], "Distribution count mismatch")
        for field in ["mean", "sd", "min", "q1", "median", "q3", "max"]:
            require(math.isclose(float(summary[field]), expected[field], rel_tol=1e-12, abs_tol=1e-12), f"Distribution mismatch: {summary['endpoint']}:{field}")
    edges = read_tsv(output / "near_duplicate_edges.tsv.gz")
    expected_edges = set()
    for left, right in itertools.combinations(sequences, 2):
        if left["normalized_sequence"] == right["normalized_sequence"]:
            continue
        threshold = max(1, math.floor(config["near_duplicate"]["maximum_normalized_edit_distance"] * min(int(left["normalized_length"]), int(right["normalized_length"]))))
        distance = levenshtein_bit(left["normalized_sequence"], right["normalized_sequence"])
        if distance <= threshold:
            expected_edges.add((left["sequence_id"], right["sequence_id"], distance, threshold))
    actual_edges = {(item["left_sequence_id"], item["right_sequence_id"], int(item["edit_distance"]), int(item["threshold"])) for item in edges}
    require(actual_edges == expected_edges, "Near-duplicate edge set mismatch")
    for item in edges:
        left = next(row for row in sequences if row["sequence_id"] == item["left_sequence_id"])
        right = next(row for row in sequences if row["sequence_id"] == item["right_sequence_id"])
        require(levenshtein_full(left["normalized_sequence"], right["normalized_sequence"]) == int(item["edit_distance"]), "Independent DP distance mismatch")
    return {"sequence_assemblies_verified": len(sequences), "endpoint_grid_cells": len(by_key), "distributions_verified": len(summaries), "near_duplicate_pairs_verified": len(edges), "all_pair_distances_recomputed": len(sequences) * (len(sequences) - 1) // 2}


def round3(config, output):
    checksums = read_json(output / "delivery_checksums.json")["files"]
    for relative, item in checksums.items():
        path = output / relative
        require(path.is_file() and path.stat().st_size == item["bytes"], f"Delivery size mismatch: {relative}")
        require(sha256_file(path) == item["sha256"], f"Delivery hash mismatch: {relative}")
    sequences = read_tsv(output / "sequence_index.tsv.gz")
    roles = read_tsv(output / "role_assignments.tsv.gz")
    edges = read_tsv(output / "near_duplicate_edges.tsv.gz")
    require(len(roles) == len(sequences) == 257, "Role row mismatch")
    sequence_role = {item["sequence_id"]: item["role"] for item in roles}
    cluster_role = {}
    for item in roles:
        require(item["role"] in {"development", "candidate", "permanent_evaluation"}, "Unknown role")
        require(item["role"] == sequence_role[item["sequence_id"]], "Duplicate role assignment")
        if item["cluster_id"] in cluster_role:
            require(cluster_role[item["cluster_id"]] == item["role"], "Cluster spans roles")
        cluster_role[item["cluster_id"]] = item["role"]
    for item in sequences:
        require(sequence_role[item["sequence_id"]] == item["role"] and cluster_role[item["cluster_id"]] == item["role"], "Sequence/index role mismatch")
    for edge in edges:
        require(sequence_role[edge["left_sequence_id"]] == sequence_role[edge["right_sequence_id"]], "Near duplicate crosses roles")
    role_manifest = read_json(output / "role_manifest.json")
    require(role_manifest["frozen"] and not role_manifest["outcome_values_used_for_allocation"], "Role freeze contract failed")
    counts = {family: {role: 0 for role in role_manifest["targets"]} for family in ["pooled_233", "cds_24"]}
    for item in roles:
        counts[item["source_family"]][item["role"]] += 1
    require(counts == role_manifest["counts"], "Role counts mismatch")
    coverage = read_tsv(output / "endpoint_coverage.tsv.gz")
    require(not any(item["endpoint"] == "predicted_expression" and item["status"] == "measured" for item in coverage), "Prediction leaked into measured endpoints")
    decision = read_json(output / "gate0_decision.json")
    require(decision["sample2019_mrl_same_endpoint_GO"] is False, "MRL gate incorrectly open")
    require(decision["model_training_or_inference_authorized"] is False, "Model gate incorrectly open")
    require(decision["outer_results_accessed"] is False and decision["positive_outcome_based_data_selection"] is False, "Forbidden analysis flag")
    require(decision["pooled_measured_protein_expression_constructs"] == 0, "Pooled measured expression claim unsupported")
    overlap = read_json(output / "overlap_report.json")
    require(overlap["cross_role_full_or_near_duplicate_edges"] == 0, "Overlap report leakage")
    return {"delivery_files_verified": len(checksums), "role_assignments_verified": len(roles), "clusters_verified": len(cluster_role), "role_counts": counts, "label_semantics_verified": True, "gate_decision_verified": decision["qualification"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--round", required=True, type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["output_dir"])
    destination = output / f"verification_round{args.round}.json"
    if destination.exists():
        raise FileExistsError(destination)
    checks = {1: round1, 2: round2, 3: round3}
    started = utc_now()
    details = checks[args.round](config, output)
    report = {"dataset_id": config["dataset_id"], "round": args.round, "started_at": started, "completed_at": utc_now(), "config_sha256": sha256_file(args.config), "status": "PASS", "details": details}
    write_json(destination, report)
    if args.round == 3:
        rounds = [read_json(output / f"verification_round{number}.json") for number in [1, 2, 3]]
        require(all(item["status"] == "PASS" for item in rounds), "Not all rounds passed")
        write_json(output / "verification_summary.json", {"dataset_id": config["dataset_id"], "status": "PASS", "rounds": rounds, "model_runs": 0, "conclusion": read_json(output / "gate0_decision.json")["qualification"]})
    print(canonical_json(report))


if __name__ == "__main__":
    main()
