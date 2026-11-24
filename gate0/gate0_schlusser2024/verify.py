import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from .build import has_hamming1, sample_sequences
from .common import is_missing, load_config, md5_file, normalize_sequence, open_table, sha256_file, terminal_50, write_json


def verify_round1(config, derived):
    raw = Path(config["raw_root"])
    errors = []
    records = []
    for spec in config["zenodo_files"]:
        path = raw / spec["name"]
        actual = {"name": spec["name"], "bytes": path.stat().st_size if path.exists() else None, "md5": md5_file(path) if path.exists() else None}
        if actual["bytes"] != spec["bytes"] or actual["md5"] != spec["md5"]:
            errors.append(spec["name"])
        records.append(actual)
    manifest = json.loads((derived / "data_manifest.json").read_text())
    if manifest["frozen_outer_study"] != "hepg2_te" or manifest["all_source_studies_independent_outer"]:
        errors.append("manifest_study_roles")
    return {"pass": not errors, "errors": errors, "zenodo_files_verified": len(records), "files": records}


def verify_round2(config, derived):
    errors = []
    summaries = {}
    for study, definition in config["studies"].items():
        mapping = config["field_mapping"][study]
        expected = json.loads((derived / "coverage" / f"{study}.json").read_text())
        rows = invalid = compatible = 0
        sequences = set()
        terminal = set()
        missing = Counter()
        with open_table(Path(config["raw_root"]) / definition["file"]) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if (reader.fieldnames or []) != expected["columns"]:
                errors.append(f"columns:{study}")
            for row in reader:
                rows += 1
                sequence = normalize_sequence(row[mapping["sequence"]])
                utr50 = terminal_50(row[mapping["sequence"]])
                if sequence is None:
                    invalid += 1
                else:
                    sequences.add(sequence)
                if utr50 is not None:
                    compatible += 1
                    terminal.add(utr50)
                for column in mapping.get("measured", []) + mapping.get("predicted", []):
                    if is_missing(row[column]):
                        missing[column] += 1
        actual = {"rows": rows, "unique_full_utr_sequences": len(sequences), "invalid_sequences": invalid, "fixed_50nt_compatible_rows": compatible, "unique_terminal_50nt_sequences": len(terminal)}
        for key, value in actual.items():
            if expected[key] != value:
                errors.append(f"{key}:{study}")
        for column in mapping.get("measured", []):
            if expected["measured_fields"][column]["missing"] != missing[column]:
                errors.append(f"measured_missing:{study}:{column}")
        for column in mapping.get("predicted", []):
            if expected["prediction_fields"][column]["missing"] != missing[column] or expected["prediction_fields"][column]["status"] != "model_prediction_not_measured":
                errors.append(f"prediction_semantics:{study}:{column}")
        summaries[study] = actual
    prediction_coverage = json.loads((derived / "prediction_field_coverage.json").read_text())
    if any(item["predictions_promoted_to_labels"] for item in prediction_coverage["files"]):
        errors.append("prediction_promotion")
    return {"pass": not errors, "errors": errors, "studies": summaries, "label_distributions_or_model_metrics_used": False}


def verify_round3(config, derived):
    errors = []
    study = config["frozen_outer"]["study"]
    role = json.loads((derived / "role_manifest.json").read_text())
    if role["outer_metrics_computed"] or role["labels_returned_to_learner"] or role["study_roles"][study]["role"] != "frozen_confirmatory_study_level_OOD":
        errors.append("role_contract")
    for other, definition in role["study_roles"].items():
        if other != study and definition["role"] == "frozen_confirmatory_study_level_OOD":
            errors.append(f"multiple_outer:{other}")
    forbidden = {"TE", "te", "translation_efficiency", "init_eff", "MRL", "rl", "prediction", "predicted"}
    rows = 0
    sequences = set()
    with gzip.open(derived / "learner" / f"{study}_sequences_isolated.tsv.gz", "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if forbidden & set(reader.fieldnames or []):
            errors.append("learner_label_columns")
        for row in reader:
            rows += 1
            if not row["utr50"] or len(row["utr50"]) != 50 or set(row["utr50"]) - set("ACGT"):
                errors.append("learner_sequence_contract")
            sequences.add(row["utr50"])
    sample, _ = sample_sequences(config["sample_role_paths"])
    if sequences & sample:
        errors.append("sample_exact_leakage")
    from collections import defaultdict
    index = defaultdict(list)
    for sequence in sample:
        index[(0, sequence[:25])].append(sequence)
        index[(1, sequence[25:])].append(sequence)
    if any(has_hamming1(sequence, index) for sequence in sequences):
        errors.append("sample_hamming1_leakage")
    inclusion_rows = 0
    included = 0
    with gzip.open(derived / "evaluator" / f"{study}_inclusion.tsv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            inclusion_rows += 1
            included += row["included"] == "True"
    if included != rows:
        errors.append("inclusion_row_mismatch")
    checksums = json.loads((derived / "delivery_checksums.json").read_text())
    for record in checksums["files"]:
        path = derived / record["relative_path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            errors.append(f"delivery_hash:{record['relative_path']}")
    decision = json.loads((derived / "gate0_decision.json").read_text())
    if decision["authorized_model_execution"] or decision["outer_metrics_computed"] or decision["scientific_result_claimed"]:
        errors.append("unauthorized_result_claim")
    return {"pass": not errors, "errors": errors, "isolated_learner_rows": rows, "evaluator_inclusion_rows": inclusion_rows, "outer_metrics_computed": False, "labels_returned_to_learner": False, "gpu_used": False}


def verify(config_path, round_number):
    config = load_config(config_path)
    derived = Path(config["derived_root"])
    function = {1: verify_round1, 2: verify_round2, 3: verify_round3}[round_number]
    report = function(config, derived)
    report.update({"dataset": config["dataset"], "round": round_number})
    target = derived / f"verification_round_{round_number}.json"
    attempt = 2
    while target.exists():
        target = derived / f"verification_round_{round_number}_attempt_{attempt}.json"
        attempt += 1
    report["report_path"] = str(target)
    write_json(target, report)
    if not report["pass"]:
        raise RuntimeError(f"Schlusser verification round {round_number} failed: {report['errors'][:10]}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--round", required=True, type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    print(json.dumps(verify(args.config, args.round), sort_keys=True))


if __name__ == "__main__":
    main()
