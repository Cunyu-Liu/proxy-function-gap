import argparse
import collections
import csv
import gzip
import hashlib
import itertools
import math
import tarfile
from pathlib import Path

from .common import (
    canonical_json,
    checksum_directory,
    distribution,
    ensure_new_directory,
    finite_float,
    levenshtein_bit,
    load_config,
    normalize_rna,
    open_tsv_gz,
    raw_row_hash,
    row_value,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
    write_row,
    write_tsv,
)
from .endpoints import ALL_ENDPOINTS, CDS_ENDPOINTS, CDS_FEATURES, POOLED_ENDPOINTS, POOLED_FEATURES
from .xlsx import iter_sheet_rows, read_table, workbook_sheets

POOLED_SHEET = "Attributes for pooled 233 seque"
CDS_SHEET = "Attributes for 24 CDS designs"
SEQUENCE_FIELDS = [
    "sequence_id", "source_construct_id", "source_family", "source_file", "source_sheet", "source_row",
    "raw_row_sha256", "human_readable_name", "group", "category", "description_5utr", "description_cds",
    "description_3utr", "type_5utr", "type_cds", "type_3utr", "raw_rna_sequence", "normalized_sequence",
    "sequence_sha256", "source_reported_length", "normalized_length", "sequence_5utr", "sequence_cds",
    "sequence_3utr", "barcode", "assembly_rule", "assembly_valid", "sequence_alphabet_valid", "length_valid",
    "replicate_relation", "cell", "reporter_context", "cluster_id", "role",
]
COVERAGE_FIELDS = [
    "sequence_id", "source_construct_id", "source_family", "endpoint", "status", "raw_value", "numeric_value",
    "error_raw_value", "error_numeric_value", "provenance", "units", "assay", "cell", "time", "construct",
    "chemistry", "replicate_n", "replicate_source_file", "source_file", "source_sheet", "source_row", "source_column", "error_source_column",
]
INCLUSION_FIELDS = [
    "sequence_id", "source_construct_id", "endpoint", "sequence_valid", "source_value_present", "numeric_value_valid",
    "included", "reason", "status",
]


def source_path(config, key):
    item = config["inputs"][key]
    return Path(config["raw_root"]) / item["relative_path"]


def validate_inputs(config):
    result = {}
    for key, item in config["inputs"].items():
        path = source_path(config, key)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise ValueError(f"SHA256 mismatch for {key}: {digest}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Size mismatch for {key}: {path}")
        result[key] = {
            **item,
            "absolute_path": str(path),
            "verified_sha256": digest,
            "verified_bytes": path.stat().st_size,
        }
    return result


def pooled_assembly(row):
    components = {
        key: row[column]
        for key, column in {
            "5utr": "Sequence 5 UTR", "cds": "Sequence CDS", "constant1": "Sequence Constant1",
            "barcode": "Sequence Barcode", "constant2": "Sequence Constant2", "3utr": "Sequence 3 UTR",
            "constant3": "Sequence Constant3",
        }.items()
    }
    orders = {
        "GGG+5UTR+CDS+C1+barcode+C2+3UTR+C3": ["5utr", "cds", "constant1", "barcode", "constant2", "3utr", "constant3"],
        "GGG+5UTR+CDS+3UTR+C1+barcode+C2+C3": ["5utr", "cds", "3utr", "constant1", "barcode", "constant2", "constant3"],
    }
    for rule, order in orders.items():
        if "GGG" + "".join(components[key] for key in order) == normalize_rna(row["RNA sequence"]):
            return rule, True
    return "no_exact_supported_assembly", False


def make_pooled(row, filename):
    sequence = normalize_rna(row["RNA sequence"])
    rule, assembly_valid = pooled_assembly(row)
    reported = int(float(row["RNA length"]))
    return {
        "sequence_id": "leppek2022_" + sha256_text(sequence),
        "source_construct_id": row["Sequence ID"], "source_family": "pooled_233", "source_file": filename,
        "source_sheet": POOLED_SHEET, "source_row": row["__row_number__"], "raw_row_sha256": raw_row_hash(row),
        "human_readable_name": row["Human-readable name"], "group": row["Group"], "category": "pooled_library",
        "description_5utr": row["Description 5 UTR"], "description_cds": row["Description CDS"],
        "description_3utr": row["Description 3 UTR"], "type_5utr": row["Type 5 UTR"],
        "type_cds": row["Type CDS"], "type_3utr": row["Type 3 UTR"], "raw_rna_sequence": row["RNA sequence"],
        "normalized_sequence": sequence, "sequence_sha256": sha256_text(sequence), "source_reported_length": reported,
        "normalized_length": len(sequence), "sequence_5utr": row["Sequence 5 UTR"], "sequence_cds": row["Sequence CDS"],
        "sequence_3utr": row["Sequence 3 UTR"], "barcode": row["Sequence Barcode"], "assembly_rule": rule,
        "assembly_valid": assembly_valid, "sequence_alphabet_valid": True, "length_valid": len(sequence) == reported,
        "replicate_relation": "pooled_summary_with_fit_errors_not_independent_sequence_replicates", "cell": "HEK293T",
        "reporter_context": row["Description CDS"],
    }


def make_cds(row, filename):
    sequence = normalize_rna(row["HBB_sequence"])
    return {
        "sequence_id": "leppek2022_" + sha256_text(sequence),
        "source_construct_id": row["Sequence ID"], "source_family": "cds_24", "source_file": filename,
        "source_sheet": CDS_SHEET, "source_row": row["__row_number__"], "raw_row_sha256": raw_row_hash(row),
        "human_readable_name": row["Title"], "group": "individual_CDS_designs", "category": row["Category"],
        "description_5utr": "HBB", "description_cds": row["Title"], "description_3utr": "HBB",
        "type_5utr": "cellular", "type_cds": row["Category"], "type_3utr": "cellular",
        "raw_rna_sequence": row["HBB_sequence"], "normalized_sequence": sequence, "sequence_sha256": sha256_text(sequence),
        "source_reported_length": "", "normalized_length": len(sequence), "sequence_5utr": "HBB_not_separately_tabulated",
        "sequence_cds": normalize_rna(row["CDS_sequence"]), "sequence_3utr": "HBB_not_separately_tabulated", "barcode": "none_individual_assay",
        "assembly_rule": "HBB_sequence_direct_full_length;CDS_sequence_matches_Sequence",
        "assembly_valid": normalize_rna(row["Sequence"]) == normalize_rna(row["CDS_sequence"]) and row["CDS_sequence"] in row["HBB_sequence"],
        "sequence_alphabet_valid": True, "length_valid": True,
        "replicate_relation": "individual_mean_and_SD_exact_n_from_Source_Data_Fig_4c", "cell": "HEK293T",
        "reporter_context": "HBB_3xHA_Nluc",
    }


def definition_for(endpoint, family):
    source = POOLED_ENDPOINTS if family == "pooled_233" else CDS_ENDPOINTS
    if endpoint in source:
        return source[endpoint], source[endpoint]["column"]
    features = POOLED_FEATURES if family == "pooled_233" else CDS_FEATURES
    if endpoint in features:
        definition = {
            "status": "predicted", "provenance": "published_computational_sequence_or_structure_feature",
            "units": "see_source_column_definition", "assay": "computational_feature_not_assay",
            "cell": "not_applicable", "time": "not_applicable", "construct": family,
            "chemistry": "not_applicable", "error_column": "",
        }
        return definition, features[endpoint]
    return None, None


def coverage_for(construct, row, replicate_row=None, replicate_source="", sentinels=None):
    sentinels = sentinels or {}
    for endpoint in ALL_ENDPOINTS:
        definition, column = definition_for(endpoint, construct["source_family"])
        replicate_column = ""
        if replicate_row is not None:
            if endpoint in {"nluc_fluc_6h", "nluc_fluc_6h_norm_to_start"}:
                replicate_column = "6h_n"
            elif endpoint in {"nluc_fluc_24h", "nluc_fluc_24h_norm_to_start"}:
                replicate_column = "24h_n"
            elif endpoint in {"cds_in_solution_degradation_coefficient", "cds_in_solution_half_life", "cds_half_life_norm_to_start"}:
                replicate_column = "half_life_n"
        raw_value = row.get(column, "") if column else ""
        value = finite_float(raw_value)
        sentinel = column in sentinels and value is not None and any(value == float(item) for item in sentinels[column])
        status = "missing" if definition is None or raw_value == "" or sentinel else definition["status"]
        if sentinel:
            value = None
        error_column = definition.get("error_column", "") if definition else ""
        error_raw = row.get(error_column, "") if error_column else ""
        item = {
            "sequence_id": construct["sequence_id"], "source_construct_id": construct["source_construct_id"],
            "source_family": construct["source_family"], "endpoint": endpoint, "status": status,
            "raw_value": raw_value, "numeric_value": "" if value is None else repr(value),
            "error_raw_value": error_raw, "error_numeric_value": finite_float(error_raw),
            "provenance": definition.get("provenance", "not_assayed_or_not_reported") if definition else "not_assayed_for_source_family",
            "units": definition.get("units", "") if definition else "", "assay": definition.get("assay", "") if definition else "",
            "cell": definition.get("cell", "") if definition else "", "time": definition.get("time", "") if definition else "",
            "construct": definition.get("construct", "") if definition else "", "chemistry": definition.get("chemistry", "") if definition else "",
            "replicate_n": replicate_row.get(replicate_column, "") if replicate_row is not None and replicate_column else "",
            "replicate_source_file": replicate_source if replicate_column else "",
            "source_file": construct["source_file"], "source_sheet": construct["source_sheet"],
            "source_row": construct["source_row"], "source_column": column or "", "error_source_column": error_column,
        }
        yield item


def inclusion_for(item, construct):
    present = item["raw_value"] != ""
    numeric_valid = item["numeric_value"] != ""
    sequence_valid = all(construct[key] for key in ["assembly_valid", "sequence_alphabet_valid", "length_valid"])
    included = sequence_valid and present and numeric_valid
    if not sequence_valid:
        reason = "sequence_validation_failure"
    elif not present:
        reason = "not_assayed_or_not_reported"
    elif not numeric_valid:
        reason = "source_sentinel_excluded" if finite_float(item["raw_value"]) is not None else "source_value_not_finite_numeric"
    else:
        reason = "included_without_imputation"
    return {
        "sequence_id": item["sequence_id"], "source_construct_id": item["source_construct_id"], "endpoint": item["endpoint"],
        "sequence_valid": sequence_valid, "source_value_present": present, "numeric_value_valid": numeric_valid,
        "included": included, "reason": reason, "status": item["status"],
    }


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def cluster_constructs(constructs, fraction):
    by_sequence = collections.defaultdict(list)
    for item in constructs:
        by_sequence[item["normalized_sequence"]].append(item["sequence_id"])
    exact_groups = [sorted(ids) for ids in by_sequence.values() if len(ids) > 1]
    union = UnionFind(item["sequence_id"] for item in constructs)
    for group in exact_groups:
        for left, right in zip(group, group[1:]):
            union.union(left, right)
    edges = []
    for left, right in itertools.combinations(constructs, 2):
        if left["normalized_sequence"] == right["normalized_sequence"]:
            continue
        threshold = max(1, math.floor(fraction * min(left["normalized_length"], right["normalized_length"])))
        distance = levenshtein_bit(left["normalized_sequence"], right["normalized_sequence"])
        if distance <= threshold:
            edges.append({
                "left_sequence_id": left["sequence_id"], "right_sequence_id": right["sequence_id"],
                "left_source_construct_id": left["source_construct_id"], "right_source_construct_id": right["source_construct_id"],
                "left_length": left["normalized_length"], "right_length": right["normalized_length"],
                "edit_distance": distance, "threshold": threshold,
                "normalized_edit_distance": distance / min(left["normalized_length"], right["normalized_length"]),
            })
            union.union(left["sequence_id"], right["sequence_id"])
    components = collections.defaultdict(list)
    for item in constructs:
        components[union.find(item["sequence_id"])].append(item)
    component_for = {}
    component_records = []
    for members in components.values():
        members.sort(key=lambda item: item["source_construct_id"])
        digest = sha256_text("\n".join(item["sequence_id"] for item in members))[:16]
        cluster_id = "leppek_cluster_" + digest
        for item in members:
            component_for[item["sequence_id"]] = cluster_id
        component_records.append({"cluster_id": cluster_id, "members": members})
    return exact_groups, edges, component_records, component_for


def assign_roles(components, seed, targets):
    counts = {family: {role: 0 for role in targets} for family in ["pooled_233", "cds_24"]}
    totals = collections.Counter()
    for component in components:
        totals.update(item["source_family"] for item in component["members"])
    assignments = {}
    ordered = sorted(
        components,
        key=lambda component: (
            -len(component["members"]),
            sha256_text(seed + "|" + component["cluster_id"]),
            component["cluster_id"],
        ),
    )
    for component in ordered:
        increments = collections.Counter(item["source_family"] for item in component["members"])
        scores = {}
        for role in targets:
            score = 0.0
            for family in counts:
                target = targets[role] * totals[family]
                after = counts[family][role] + increments[family]
                score += ((after - target) ** 2 - (counts[family][role] - target) ** 2) / max(1.0, target)
            scores[role] = score
        selected = min(targets, key=lambda role: (scores[role], sha256_text(seed + "|" + component["cluster_id"] + "|" + role), role))
        assignments[component["cluster_id"]] = selected
        for family, increment in increments.items():
            counts[family][selected] += increment
    return assignments, counts


def component_overlaps(constructs):
    result = []
    for left, right in itertools.combinations(constructs, 2):
        for component, key in [("5UTR", "sequence_5utr"), ("CDS", "sequence_cds"), ("3UTR", "sequence_3utr")]:
            left_value, right_value = left[key], right[key]
            if "not_separately_tabulated" in left_value or "not_separately_tabulated" in right_value:
                continue
            if left_value == right_value:
                result.append({
                    "left_sequence_id": left["sequence_id"], "right_sequence_id": right["sequence_id"],
                    "left_source_construct_id": left["source_construct_id"], "right_source_construct_id": right["source_construct_id"],
                    "component": component, "component_sha256": sha256_text(left_value),
                    "left_role": left["role"], "right_role": right["role"],
                })
    return result


def values_equal(left, right):
    if left == right:
        return True
    left_number, right_number = finite_float(left), finite_float(right)
    if left_number is None or right_number is None:
        return False
    return math.isclose(left_number, right_number, rel_tol=1e-10, abs_tol=1e-12)


def crosscheck_primary_copies(geo_rows, publisher_rows):
    geo = {row["Sequence ID"]: row for row in geo_rows}
    publisher = {row["Sequence ID"]: row for row in publisher_rows}
    if set(geo) != set(publisher):
        raise ValueError("GEO and publisher construct IDs differ")
    differences = []
    compared = 0
    for construct_id in sorted(geo):
        for column in geo[construct_id]:
            if column.startswith("__") or column not in publisher[construct_id]:
                continue
            compared += 1
            if not values_equal(geo[construct_id][column], publisher[construct_id][column]):
                differences.append({"sequence_id": construct_id, "column": column, "geo": geo[construct_id][column], "publisher": publisher[construct_id][column]})
    return {"constructs": len(geo), "cells_compared": compared, "differences": differences}


def field_dictionary(config, pooled_fields, cds_fields):
    measured_columns = {}
    for source in [POOLED_ENDPOINTS, CDS_ENDPOINTS]:
        for definition in source.values():
            measured_columns[definition["column"]] = (definition["status"], definition["provenance"])
            if definition.get("error_column"):
                measured_columns[definition["error_column"]] = ("measured", "uncertainty_of_" + definition["provenance"])
    rows = []
    for family, filename, fields in [
        ("pooled_233", Path(config["inputs"]["pooled_attributes"]["relative_path"]).name, pooled_fields),
        ("cds_24", Path(config["inputs"]["cds_attributes"]["relative_path"]).name, cds_fields),
    ]:
        for column in fields:
            if column.startswith("__column_"):
                continue
            if column in measured_columns:
                status, description = measured_columns[column]
            elif column in POOLED_FEATURES.values() or column in CDS_FEATURES.values():
                status, description = "predicted", "published_computational_feature_not_an_experimental_measurement"
            elif column.startswith(("MFE ", "bprna", "degscore", "dG(", "SUP ", "AUP ", "punp_", "mean bp", "MLD", "n_", "hp_")):
                status, description = "predicted", "published_computational_sequence_or_structure_feature"
            else:
                status, description = "metadata_or_sequence", "source_identifier_metadata_or_sequence_not_an_endpoint"
            rows.append({"source_family": family, "source_file": filename, "source_column": column, "classification": status, "description": description})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output = Path(config["output_dir"])
    ensure_new_directory(output)
    inputs = validate_inputs(config)
    pooled_path = source_path(config, "pooled_attributes")
    cds_path = source_path(config, "cds_attributes")
    pooled_fields, pooled_rows = read_table(pooled_path, POOLED_SHEET)
    cds_fields, cds_rows = read_table(cds_path, CDS_SHEET)
    if len(pooled_rows) != 233 or len(cds_rows) != 24:
        raise ValueError("Unexpected primary source row count")
    constructs = [make_pooled(row, pooled_path.name) for row in pooled_rows] + [make_cds(row, cds_path.name) for row in cds_rows]
    significance_path = source_path(config, "source_data")
    _, significance_rows = read_table(significance_path, "Fig_4c")
    aliases = {"1220183B1": "Nluc start", "1220184B1": "LinearDesign-beamsize50", "1220196B1": "12"}
    significance_by_name = {row["name"].casefold(): row for row in significance_rows}
    replicates = {}
    for item in constructs:
        if item["source_family"] != "cds_24":
            continue
        name = aliases.get(item["source_construct_id"], item["human_readable_name"])
        if name.casefold() not in significance_by_name:
            raise ValueError(f"Missing Source Data Fig_4c replicate mapping: {name}")
        replicates[item["source_construct_id"]] = significance_by_name[name.casefold()]
        replicate = significance_by_name[name.casefold()]
        item["replicate_relation"] = f"half_life_n={replicate['half_life_n']};expression_6h_n={replicate['6h_n']};expression_24h_n={replicate['24h_n']}"
    if len(replicates) != 24 or len(significance_rows) != 24:
        raise ValueError("Source Data Fig_4c replicate mapping is incomplete")
    publisher_pooled_rows = read_table(source_path(config, "publisher_pooled_attributes"), POOLED_SHEET)[1]
    publisher_cds_rows = read_table(source_path(config, "publisher_cds_attributes"), CDS_SHEET)[1]
    crosscheck = {
        "pooled_GEO_vs_Supplementary_Data1": crosscheck_primary_copies(pooled_rows, publisher_pooled_rows),
        "cds_GEO_vs_Supplementary_Data6": crosscheck_primary_copies(cds_rows, publisher_cds_rows),
        "source_data_fig4c_vs_GEO_expression": [],
        "supplementary_data5_vs_source_data_name_value_differences": [],
    }
    expression_pairs = {
        "Nluc_Fluc_6h_mean_norm_to_START": "6h_mean", "Nluc_Fluc_6h_sd_norm_to_START": "6h_sd",
        "Nluc_Fluc_24h_mean_norm_to_START": "24h_mean", "Nluc_Fluc_24h_sd_norm_to_START": "24h_sd",
    }
    data5_rows = read_table(source_path(config, "cds_significance"), "Sheet1")[1]
    data5_by_name = {row["name"]: row for row in data5_rows}
    for row in cds_rows:
        replicate = replicates[row["Sequence ID"]]
        for source_column, replicate_column in expression_pairs.items():
            left, right = finite_float(row[source_column]), finite_float(replicate[replicate_column])
            relative_difference = abs(left - right) / max(1.0, abs(left), abs(right))
            crosscheck["source_data_fig4c_vs_GEO_expression"].append({"sequence_id": row["Sequence ID"], "source_column": source_column, "GEO": left, "Source_Data_Fig_4c": right, "relative_difference": relative_difference})
        data5 = data5_by_name[replicate["name"]]
        for column in ["half_life_mean", "half_life_sd", "half_life_n", "6h_mean", "6h_sd", "6h_n", "24h_mean", "24h_sd", "24h_n"]:
            if not values_equal(data5[column], replicate[column]):
                crosscheck["supplementary_data5_vs_source_data_name_value_differences"].append({"name": replicate["name"], "column": column, "Supplementary_Data5": data5[column], "Source_Data_Fig_4c": replicate[column]})
    if any(section["differences"] for key, section in crosscheck.items() if key in {"pooled_GEO_vs_Supplementary_Data1", "cds_GEO_vs_Supplementary_Data6"}):
        write_json(output / "supplementary_crosscheck.json", crosscheck)
        raise ValueError("Cross-source primary value mismatch")
    crosscheck["expression_mapping_all_within_one_percent"] = all(item["relative_difference"] <= 0.01 for item in crosscheck["source_data_fig4c_vs_GEO_expression"])
    crosscheck["half_life_normalization_note"] = "GEO S5 normalized half-life is retained as published; Source Data Fig_4c reports figure-level normalized replicate summaries. They are not forced to be numerically identical; only exact n is taken from Source Data."
    crosscheck["replicate_n_authority"] = "Source Data Fig_4c names and expression values align with GEO S5; Supplementary Data 5 name-value offsets are reported, not selectively repaired."
    if not crosscheck["expression_mapping_all_within_one_percent"]:
        write_json(output / "supplementary_crosscheck.json", crosscheck)
        raise ValueError("Source Data expression mapping exceeds one percent")
    write_json(output / "supplementary_crosscheck.json", crosscheck)
    if len({item["source_construct_id"] for item in constructs}) != 257:
        raise ValueError("Source construct IDs are not unique across primary entities")
    if len({item["sequence_id"] for item in constructs}) != 257:
        raise ValueError("Full-length exact duplicates require an explicit many-to-one entity model")
    raw_rows = {item["source_construct_id"]: row for item, row in zip(constructs, pooled_rows + cds_rows)}
    coverage = [item for construct in constructs for item in coverage_for(construct, raw_rows[construct["source_construct_id"]], replicates.get(construct["source_construct_id"]), significance_path.name, config.get("source_sentinels"))]
    inclusions = [inclusion_for(item, construct) for construct in constructs for item in coverage if item["sequence_id"] == construct["sequence_id"]]
    exact_groups, edges, components, component_for = cluster_constructs(constructs, config["near_duplicate"]["maximum_normalized_edit_distance"])
    assignments, role_counts = assign_roles(components, config["role_freeze"]["seed"], config["role_freeze"]["targets"])
    for item in constructs:
        item["cluster_id"] = component_for[item["sequence_id"]]
        item["role"] = assignments[item["cluster_id"]]
    component_edges = component_overlaps(constructs)

    with gzip.open(output / "sequence_index.tsv.gz", "wt", newline="", compresslevel=5) as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=SEQUENCE_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for item in constructs:
            writer.writerow({key: row_value(item.get(key)) for key in SEQUENCE_FIELDS})
    for filename, fields, rows in [
        ("endpoint_coverage.tsv.gz", COVERAGE_FIELDS, coverage),
        ("endpoint_measurements.tsv.gz", COVERAGE_FIELDS, [item for item in coverage if item["status"] in {"measured", "predicted"}]),
        ("inclusion_exclusion.tsv.gz", INCLUSION_FIELDS, inclusions),
        ("near_duplicate_edges.tsv.gz", ["left_sequence_id", "right_sequence_id", "left_source_construct_id", "right_source_construct_id", "left_length", "right_length", "edit_distance", "threshold", "normalized_edit_distance"], edges),
        ("component_overlap_edges.tsv.gz", ["left_sequence_id", "right_sequence_id", "left_source_construct_id", "right_source_construct_id", "component", "component_sha256", "left_role", "right_role"], component_edges),
        ("role_assignments.tsv.gz", ["sequence_id", "source_construct_id", "source_family", "cluster_id", "role", "sequence_sha256"], constructs),
    ]:
        handle, writer = open_tsv_gz(output / filename, fields)
        with handle:
            for item in rows:
                write_row(writer, item)
    handle, writer = open_tsv_gz(output / "exact_duplicate_groups.tsv.gz", ["exact_group_id", "sequence_sha256", "member_sequence_ids", "member_source_construct_ids", "size"])
    with handle:
        for group in exact_groups:
            write_row(writer, {"exact_group_id": "exact_" + sha256_text("\n".join(group))[:16], "sequence_sha256": sha256_text(next(item["normalized_sequence"] for item in constructs if item["sequence_id"] == group[0])), "member_sequence_ids": group, "member_source_construct_ids": [item["source_construct_id"] for item in constructs if item["sequence_id"] in group], "size": len(group)})
    write_tsv(output / "field_dictionary.tsv", ["source_family", "source_file", "source_column", "classification", "description"], field_dictionary(config, pooled_fields, cds_fields))
    distribution_rows = []
    for family in ["pooled_233", "cds_24"]:
        for endpoint in ALL_ENDPOINTS:
            for status in ["measured", "predicted"]:
                selected = [item for item in coverage if item["source_family"] == family and item["endpoint"] == endpoint and item["status"] == status]
                stats = distribution([finite_float(item["numeric_value"]) for item in selected])
                if stats["n"]:
                    distribution_rows.append({"source_family": family, "endpoint": endpoint, "status": status, **stats})
    write_tsv(output / "distribution_summary.tsv", ["source_family", "endpoint", "status", "n", "mean", "sd", "min", "q1", "median", "q3", "max"], distribution_rows)

    tar_members = []
    with tarfile.open(source_path(config, "rdat_archive")) as archive:
        for member in archive.getmembers():
            if member.isfile():
                tar_members.append({"name": member.name, "bytes": member.size})
    with open(source_path(config, "sra_runinfo"), newline="") as handle:
        sra_runs = list(csv.DictReader(handle))
    require_fields = ["Run", "SampleName", "Experiment", "LibraryStrategy", "LibraryLayout", "Platform", "size_MB", "download_path"]
    run_index = [{key: row[key] for key in require_fields} for row in sra_runs]
    write_tsv(output / "sra_run_index.tsv", require_fields, run_index)
    soft_titles = {}
    current_sample = None
    import gzip as gzip_module
    with gzip_module.open(source_path(config, "geo_soft"), "rt") as handle:
        for line in handle:
            if line.startswith("^SAMPLE = "):
                current_sample = line.strip().split(" = ", 1)[1]
            elif line.startswith("!Sample_title = ") and current_sample:
                soft_titles[current_sample] = line.strip().split(" = ", 1)[1]
    require_sample_titles = {row["SampleName"] for row in sra_runs}
    if require_sample_titles != set(soft_titles):
        raise ValueError("SRA and SOFT sample accession mismatch")
    manifest = {
        "dataset_id": config["dataset_id"], "created_at": utc_now(), "config_path": str(config_path),
        "config_sha256": sha256_file(config_path), "sources": inputs, "geo_samples": {
            "series": "GSE173083", "sra": "SRP315847", "bioproject": "PRJNA723762",
            "sequencing_metadata_source": "GSE173083_family.soft.gz",
            "raw_fastq_downloaded": False,
            "raw_fastq_recomputation_performed": False,
            "runinfo_accessions": len(sra_runs),
            "runinfo_reported_size_MB": sum(int(row["size_MB"]) for row in sra_runs),
            "run_accessions": [row["Run"] for row in sra_runs],
            "soft_sample_titles": soft_titles,
            "note": "Published summary endpoints, RDAT and SRA RunInfo are audited here; approximately 50 GB of SRA reads are not claimed as downloaded or locally recomputed.",
        },
        "rdat_archive_members": tar_members,
        "primary_sheets": {"pooled": workbook_sheets(pooled_path), "cds": workbook_sheets(cds_path)},
        "raw_files_immutable": True, "prediction_never_relabeled_as_measurement": True,
    }
    write_json(output / "data_manifest.json", manifest)
    source_data_inventory = {}
    for sheet, _ in workbook_sheets(significance_path):
        rows = 0
        columns = 0
        for _, values, _ in iter_sheet_rows(significance_path, sheet):
            rows += 1
            columns = max(columns, len(values))
        source_data_inventory[sheet] = {"rows_including_headers_or_labels": rows, "maximum_columns": columns}
    write_json(output / "source_data_inventory.json", {
        "source_file": significance_path.name,
        "sheets": source_data_inventory,
        "Fig_4a_replicate_rows": source_data_inventory.get("Fig_4a", {}).get("rows_including_headers_or_labels", 1) - 1,
        "Fig_4a_integration": "Raw replicate records are preserved but not merged into the primary 233+24 role table because 18 additional IDs and chemistry/time contexts require a separate observation-level model.",
    })
    cross_role_near = sum(item["left_role"] != item["right_role"] for item in component_edges)
    overlap = {
        "rule": "full_length_normalized_DNA_Levenshtein_distance;threshold=max(1,floor(0.01*shorter_length));connected_components",
        "constructs": len(constructs), "unique_full_sequences": len(by_sequence := {item["normalized_sequence"] for item in constructs}),
        "exact_duplicate_groups": len(exact_groups), "near_duplicate_edges_excluding_exact": len(edges),
        "sequence_clusters": len(components), "component_overlap_edges": len(component_edges),
        "cross_role_component_overlap_edges": cross_role_near,
        "cross_role_full_or_near_duplicate_edges": 0,
        "component_overlap_warning": "Shared UTR/CDS components across roles are retained and reported; they preclude claiming fully component-independent generalization.",
    }
    write_json(output / "overlap_report.json", overlap)
    role_manifest = {
        "dataset_id": config["dataset_id"], "created_at": utc_now(), "seed": config["role_freeze"]["seed"],
        "targets": config["role_freeze"]["targets"], "counts": role_counts,
        "allocation_unit": "full_length_exact_and_near_duplicate_connected_component",
        "outcome_values_used_for_allocation": False, "frozen": True,
        "roles": {"development": "future training/tuning only", "candidate": "future unlabeled query until explicit reveal", "permanent_evaluation": "future evaluator-only final assessment"},
        "audit_table_access": "evaluator_level;future_learner_views_must_filter_candidate_and_permanent_labels",
    }
    write_json(output / "role_manifest.json", role_manifest)
    measured = [item for item in coverage if item["status"] == "measured" and item["numeric_value"] != ""]
    predicted_expression = [item for item in coverage if item["endpoint"] == "predicted_expression" and item["status"] == "predicted" and item["numeric_value"] != ""]
    decision = {
        "dataset_id": config["dataset_id"], "created_at": utc_now(), "all_source_sequence_rows_valid": all(item["assembly_valid"] and item["length_valid"] for item in constructs),
        "pooled_constructs": 233, "cds_constructs": 24, "measured_endpoint_values": len(measured),
        "pooled_measured_protein_expression_constructs": 0, "pooled_predicted_expression_constructs": len(predicted_expression),
        "cds_measured_6h_expression_constructs": sum(item["endpoint"] == "nluc_fluc_6h" for item in measured),
        "cds_measured_24h_expression_constructs": sum(item["endpoint"] == "nluc_fluc_24h" for item in measured),
        "sample2019_mrl_same_endpoint_GO": False,
        "sample2019_mrl_NO_GO_reason": "Full-length pooled PERSIST-seq ribosome load/stability is not Sample 2019 50-nt UTR MRL; reporter, construct, assay and endpoint differ; predicted expression is not measured MRL.",
        "cross_endpoint_descriptive_GO": True,
        "external_mechanism_GO": True,
        "authorized_claims": [
            "Construct-specific descriptive relationships among measured ribosome load, RNA stability and polysome ratios in the 233-member pool.",
            "Small-sample external mechanism audit relating measured RNA stability to measured Nluc/Fluc at 6h and 24h in 24 individual CDS designs.",
        ],
        "prohibited_claims": [
            "Measured protein expression for the pooled 233 constructs based on Predicted expression.",
            "Same-endpoint Sample 2019 MRL training, calibration, validation or regret evaluation.",
            "Component-independent or cross-study generalization without further evaluation.",
            "Independent recomputation of published endpoints from FASTQ.",
        ],
        "source_sentinel_exclusions": sum(item["reason"] == "source_sentinel_excluded" for item in inclusions),
        "source_sentinel_policy": config.get("source_sentinels", {}),
        "source_replicate_discrepancy": "Source Data Fig_4c reports half_life_n=2 for 1220198B1 (Astros-Brourd-Eli-mod3), while the Fig. 4 legend states n>=3 generically; exact endpoint-specific n is retained. Supplementary Data 5 also has name-value offsets relative to Source Data; these are reported and never used to select outcomes.",
        "new_independent_study_candidate": "Leppek2022_cross_endpoint_and_external_mechanism_v1",
        "model_training_or_inference_authorized": False, "outer_results_accessed": False,
        "positive_outcome_based_data_selection": False, "cuda_used": False,
        "qualification": "GO_ONLY_FOR_CROSS_ENDPOINT_DESCRIPTIVE_AND_EXTERNAL_MECHANISM;NO_GO_FOR_SAMPLE2019_MRL",
    }
    write_json(output / "gate0_decision.json", decision)
    summary = {
        "dataset_id": config["dataset_id"], "created_at": utc_now(), "config_sha256": sha256_file(config_path),
        "pooled_rows": len(pooled_rows), "cds_rows": len(cds_rows), "coverage_rows": len(coverage),
        "included_by_status": dict(collections.Counter(item["status"] for item in inclusions if item["included"])),
        "excluded_by_reason": dict(collections.Counter(item["reason"] for item in inclusions if not item["included"])),
        "assembly_rules": dict(collections.Counter(item["assembly_rule"] for item in constructs)),
        "near_duplicate_edges": len(edges), "clusters": len(components), "role_counts": role_counts,
        "conclusion": decision["qualification"], "model_runs": 0, "verification_required_rounds": 3,
    }
    write_json(output / "build_summary.json", summary)
    checksums = checksum_directory(output, exclude={"delivery_checksums.json"})
    write_json(output / "delivery_checksums.json", {"algorithm": "sha256", "files": checksums})
    print(canonical_json({"status": "built", "output": str(output), "constructs": len(constructs), "coverage": len(coverage), "near_edges": len(edges), "clusters": len(components), "qualification": decision["qualification"]}))


if __name__ == "__main__":
    main()
