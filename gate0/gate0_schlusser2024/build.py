import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import file_record, is_missing, load_config, normalize_sequence, open_table, sha256_file, terminal_50, write_json


def mapping_for(config, study):
    try:
        return config["field_mapping"][study]
    except KeyError as error:
        raise RuntimeError(f"No frozen field mapping for {study}") from error


def hamming1(left, right):
    return len(left) == len(right) == 50 and sum(a != b for a, b in zip(left, right)) == 1


def add_half_index(sequence, index):
    index[(0, sequence[:25])].append(sequence)
    index[(1, sequence[25:])].append(sequence)


def has_hamming1(sequence, index):
    for position, key in ((0, sequence[:25]), (1, sequence[25:])):
        for other in index.get((position, key), []):
            if hamming1(sequence, other):
                return True
    return False


def sample_sequences(paths):
    sequences = set()
    hashes = {}
    for value in paths:
        path = Path(value)
        hashes[str(path)] = sha256_file(path)
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sequences.add(row["utr50"])
    return sequences, hashes


def profile_study(config, study, definition, raw, derived):
    mapping = mapping_for(config, study)
    source = raw / definition["file"]
    sequence_column = mapping["sequence"]
    measured = mapping.get("measured", [])
    predicted = mapping.get("predicted", [])
    identifiers = mapping.get("identifiers", [])
    missing = Counter()
    lengths = Counter()
    terminal_lengths = Counter()
    sequences = set()
    terminal_sequences = set()
    source_ids = set()
    duplicate_source_ids = 0
    rows = 0
    invalid_sequences = 0
    eligible_sequences = 0
    frozen = config["frozen_outer"]["study"] == study
    sequence_path = derived / "learner" / f"{study}_sequences.tsv.gz" if frozen else None
    label_path = derived / "evaluator" / f"{study}_labels.tsv.gz" if frozen else None
    index_path = derived / "sequence_index" / f"{study}.tsv.gz"
    for path in (sequence_path, label_path, index_path):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    with open_table(source) as handle, gzip.open(index_path, "wt", newline="") as index_handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        required = [sequence_column, *measured, *predicted]
        absent = [column for column in required if column not in columns]
        if absent:
            raise RuntimeError(f"{study}: missing frozen columns {absent}")
        index_fields = ["row_id", "source_row_number", "study", "source_id", "system", "endpoint", "role", "utr_sequence", "utr_length", "utr50", "fixed_input_compatible"]
        index_writer = csv.DictWriter(index_handle, fieldnames=index_fields, delimiter="\t", lineterminator="\n")
        index_writer.writeheader()
        sequence_handle = gzip.open(sequence_path, "wt", newline="") if frozen else None
        label_handle = gzip.open(label_path, "wt", newline="") if frozen else None
        try:
            sequence_writer = label_writer = None
            if frozen:
                learner_fields = ["row_id", "source_id", "study", "utr_sequence", "utr50"]
                sequence_writer = csv.DictWriter(sequence_handle, fieldnames=learner_fields, delimiter="\t", lineterminator="\n")
                sequence_writer.writeheader()
                evaluator_fields = ["row_id", "source_id", "study", "endpoint", *measured]
                label_writer = csv.DictWriter(label_handle, fieldnames=evaluator_fields, delimiter="\t", lineterminator="\n")
                label_writer.writeheader()
            for row in reader:
                rows += 1
                row_id = f"{study}:{rows:08d}"
                source_id = "|".join(row.get(column, "") for column in identifiers)
                if source_id:
                    if source_id in source_ids:
                        duplicate_source_ids += 1
                    source_ids.add(source_id)
                sequence = normalize_sequence(row[sequence_column])
                utr50 = terminal_50(row[sequence_column])
                if sequence is None:
                    invalid_sequences += 1
                    length = 0
                else:
                    length = len(sequence)
                    sequences.add(sequence)
                    lengths[length] += 1
                if utr50 is not None:
                    eligible_sequences += 1
                    terminal_sequences.add(utr50)
                    terminal_lengths[len(sequence)] += 1
                for column in measured + predicted:
                    if is_missing(row[column]):
                        missing[column] += 1
                index_writer.writerow({"row_id": row_id, "source_row_number": rows, "study": study, "source_id": source_id, "system": definition["system"], "endpoint": definition["endpoint"], "role": definition["role"], "utr_sequence": sequence or "", "utr_length": length, "utr50": utr50 or "", "fixed_input_compatible": utr50 is not None})
                if frozen:
                    sequence_writer.writerow({"row_id": row_id, "source_id": source_id, "study": study, "utr_sequence": sequence or "", "utr50": utr50 or ""})
                    label_writer.writerow({"row_id": row_id, "source_id": source_id, "study": study, "endpoint": definition["endpoint"], **{column: row[column] for column in measured}})
        finally:
            if sequence_handle is not None:
                sequence_handle.close()
            if label_handle is not None:
                label_handle.close()
    coverage = {
        "study": study,
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "columns": columns,
        "rows": rows,
        "unique_full_utr_sequences": len(sequences),
        "invalid_sequences": invalid_sequences,
        "fixed_50nt_compatible_rows": eligible_sequences,
        "unique_terminal_50nt_sequences": len(terminal_sequences),
        "sequence_length_histogram": dict(sorted(lengths.items())),
        "compatible_sequence_length_histogram": dict(sorted(terminal_lengths.items())),
        "duplicate_source_id_rows": duplicate_source_ids,
        "measured_fields": {column: {"non_missing": rows - missing[column], "missing": missing[column], "status": "measured_or_measured_derived"} for column in measured},
        "prediction_fields": {column: {"non_missing": rows - missing[column], "missing": missing[column], "status": "model_prediction_not_measured"} for column in predicted},
        "label_distribution_inspected": False if frozen else None,
        "label_values_printed": False,
        "model_metrics_computed": False,
    }
    write_json(derived / "coverage" / f"{study}.json", coverage)
    return coverage


def profile_predictions(config, raw, derived):
    reports = []
    for mapping in config.get("prediction_file_mapping", []):
        path = raw / mapping["file"]
        missing = Counter()
        rows = 0
        with open_table(path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            columns = reader.fieldnames or []
            for column in mapping["measured"] + mapping["predicted"]:
                if column not in columns:
                    raise RuntimeError(f"{mapping['file']}: absent {column}")
            for row in reader:
                rows += 1
                for column in mapping["measured"] + mapping["predicted"]:
                    if is_missing(row[column]):
                        missing[column] += 1
        report = {"file": mapping["file"], "sha256": sha256_file(path), "rows": rows, "columns": columns, "measured_fields": {c: rows - missing[c] for c in mapping["measured"]}, "prediction_fields": {c: rows - missing[c] for c in mapping["predicted"]}, "predictions_promoted_to_labels": False}
        reports.append(report)
    write_json(derived / "prediction_field_coverage.json", {"files": reports})
    return reports


def freeze_outer(config, derived):
    study = config["frozen_outer"]["study"]
    learner = derived / "learner" / f"{study}_sequences.tsv.gz"
    evaluator = derived / "evaluator" / f"{study}_labels.tsv.gz"
    sample, sample_hashes = sample_sequences(config["sample_role_paths"])
    half_index = defaultdict(list)
    for sequence in sample:
        add_half_index(sequence, half_index)
    counts = Counter()
    excluded = Counter()
    outer_sequences = set()
    filtered_path = derived / "learner" / f"{study}_sequences_isolated.tsv.gz"
    inclusion_path = derived / "evaluator" / f"{study}_inclusion.tsv.gz"
    with gzip.open(learner, "rt", newline="") as handle, gzip.open(filtered_path, "wt", newline="") as filtered, gzip.open(inclusion_path, "wt", newline="") as inclusion:
        reader = csv.DictReader(handle, delimiter="\t")
        writer = csv.DictWriter(filtered, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        inclusion_writer = csv.DictWriter(inclusion, fieldnames=["row_id", "source_id", "included", "exclusion_reason"], delimiter="\t", lineterminator="\n")
        inclusion_writer.writeheader()
        for row in reader:
            counts["rows"] += 1
            sequence = row["utr50"]
            reasons = []
            if not sequence:
                reasons.append("fixed_50nt_input_incompatible")
            else:
                outer_sequences.add(sequence)
                if sequence in sample:
                    reasons.append("sample_exact_overlap")
                elif has_hamming1(sequence, half_index):
                    reasons.append("sample_hamming1_overlap")
            included = not reasons
            counts["included" if included else "excluded"] += 1
            for reason in reasons:
                excluded[reason] += 1
            if included:
                writer.writerow(row)
            inclusion_writer.writerow({"row_id": row["row_id"], "source_id": row["source_id"], "included": included, "exclusion_reason": ";".join(reasons)})
    exact = len(outer_sequences & sample)
    hamming = sum(1 for sequence in outer_sequences if sequence not in sample and has_hamming1(sequence, half_index))
    report = {"study": study, "sample_role_hashes": sample_hashes, "sample_unique_utr50": len(sample), "outer_unique_utr50": len(outer_sequences), "exact_overlap_unique_sequences": exact, "hamming1_overlap_unique_sequences_excluding_exact": hamming, "row_counts": dict(counts), "exclusion_counts": dict(excluded), "near_duplicate_rule": "exact and Hamming-1 on terminal 50nt against all Sample train/dev/calibration roles", "selection_uses_labels_or_results": False}
    write_json(derived / "overlap_report.json", report)
    vault_rows = 0
    label_columns = []
    with gzip.open(evaluator, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        label_columns = reader.fieldnames or []
        for _ in reader:
            vault_rows += 1
    if vault_rows != counts["rows"]:
        raise RuntimeError("Evaluator/learner row mismatch")
    role_manifest = {
        "schema_version": "1.0",
        "dataset": config["dataset"],
        "allocation_unit": "entire_study",
        "study_roles": {study: {k: v for k, v in definition.items() if k != "file"} for study, definition in config["studies"].items()},
        "frozen_outer": config["frozen_outer"],
        "sample_role_hashes": sample_hashes,
        "outer_metrics_computed": False,
        "labels_returned_to_learner": False,
        "evaluator_label_columns": label_columns,
        "learner_contains_TE_or_predictions": False,
        "future_reveal_requires_new_run": True,
        "history_audit": {"project_head_at_freeze": config.get("project_head_at_freeze"), "prior_schlusser_use": "citation_only_before_this_task", "prior_hepg2_training_or_tuning": False},
    }
    write_json(derived / "role_manifest.json", role_manifest)
    return report, role_manifest


def build(config_path):
    config = load_config(config_path)
    raw = Path(config["raw_root"])
    derived = Path(config["derived_root"])
    if derived.exists():
        raise FileExistsError(f"Refusing to overwrite {derived}")
    derived.mkdir(parents=True)
    coverage = {}
    for study, definition in config["studies"].items():
        coverage[study] = profile_study(config, study, definition, raw, derived)
    predictions = profile_predictions(config, raw, derived)
    overlap, roles = freeze_outer(config, derived)
    decision = {
        "dataset": config["dataset"],
        "confirmatory_ood_data_ready": overlap["row_counts"].get("included", 0) > 0,
        "authorized_model_execution": False,
        "outer_metrics_computed": False,
        "scientific_result_claimed": False,
        "reason": "HepG2 is frozen for a future separately authorized confirmatory evaluation; this task performs data audit only.",
        "gpu_used": False,
        "raw_fastq_reprocessing_claimed": False,
    }
    write_json(derived / "gate0_decision.json", decision)
    source_files = []
    for path in sorted(raw.rglob("*")):
        if path.is_file() and not any(".parts" in part for part in path.parts) and not path.name.endswith(".assembling"):
            source_files.append(file_record(path))
    manifest = {"schema_version": "1.0", "dataset": config["dataset"], "config_sha256": sha256_file(config_path), "project_head_at_freeze": config.get("project_head_at_freeze"), "article": {"doi": config["article_doi"], "pmcid": config["article_pmcid"], "license": "CC-BY-4.0"}, "upstream_commit": config["upstream_commit"], "licenses": config["licenses"], "source_files": source_files, "coverage": coverage, "prediction_files": predictions, "raw_fastq_downloaded": False, "all_source_studies_independent_outer": False, "frozen_outer_study": config["frozen_outer"]["study"]}
    write_json(derived / "data_manifest.json", manifest)
    delivery = []
    for path in sorted(derived.rglob("*")):
        if path.is_file() and path.name not in {"delivery_checksums.json", "build_summary.json"}:
            delivery.append(file_record(path, {"relative_path": str(path.relative_to(derived))}))
    write_json(derived / "delivery_checksums.json", {"files": delivery})
    summary = {"dataset": config["dataset"], "studies": len(coverage), "prediction_files": len(predictions), "outer_included_rows": overlap["row_counts"].get("included", 0), "outer_metrics_computed": False, "delivery_files": len(delivery)}
    write_json(derived / "build_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.config), sort_keys=True))


if __name__ == "__main__":
    main()
