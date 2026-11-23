import argparse
import csv
import datetime
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import load_config, sha256_file
from .protocol import Evaluator, Learner, create_selection, read_revealed_labels


def rows(path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def expect_failure(name, function, failures):
    try:
        function()
    except (PermissionError, FileExistsError, ValueError):
        failures.append({"test": name, "expected": "reject", "observed": "reject", "pass": True})
    else:
        failures.append({"test": name, "expected": "reject", "observed": "accepted", "pass": False})


def round1(config, output):
    split = json.loads((output / "split_manifest.json").read_text())
    group_roles = defaultdict(set)
    sequence_roles = defaultdict(set)
    trajectory_roles = defaultdict(set)
    unit_roles = defaultdict(set)
    role_counts = Counter()
    official_counts = Counter()
    raw_ids = set()
    duplicate_ids = 0
    outer_strata = Counter()
    for row in rows(output / "allocation_units.tsv.gz"):
        if row["role"] == "outer":
            outer_strata[row["stratum"]] += 1
    for row in rows(output / "role_assignments.tsv.gz"):
        role = row["role"]
        role_counts[role] += 1
        group_roles[row["near_group_id"]].add(role)
        sequence_roles[row["sequence_id"]].add(role)
        unit_roles[row["allocation_unit_id"]].add(role)
        if row["trajectory_id"]:
            trajectory_roles[row["trajectory_id"]].add(role)
        official = row["official_model_role"]
        official_counts[official] += 1
        if official != "not_applicable":
            if role not in {"train", "official_seen_test", "official_boundary_ambiguous"} or row["new_blind_evaluation_allowed"] != "false":
                raise ValueError("Official-seen row leaked into a new blind role")
        if row["raw_row_id"] in raw_ids:
            duplicate_ids += 1
        raw_ids.add(row["raw_row_id"])
    violations = {
        "near_components_crossing_roles": sum(len(value) > 1 for value in group_roles.values()),
        "sequences_crossing_roles": sum(len(value) > 1 for value in sequence_roles.values()),
        "trajectories_crossing_roles": sum(len(value) > 1 for value in trajectory_roles.values()),
        "allocation_units_crossing_roles": sum(len(value) > 1 for value in unit_roles.values()),
        "duplicate_raw_row_ids": duplicate_ids,
    }
    learner = Learner(output / "learner_allowlist.json")
    negative = []
    expect_failure("learner_outer_labels", lambda: list(learner.read_labels("outer")), negative)
    expect_failure("learner_outer_features", lambda: list(learner.read_features("outer")), negative)
    expect_failure("learner_direct_query_labels", lambda: list(learner.read_labels("query")), negative)
    evaluator_manifest = json.loads((output / "evaluator_manifest.json").read_text())
    expect_failure("learner_outer_path", lambda: learner.read_arbitrary_path(evaluator_manifest["label_vaults"]["outer"]["path"]), negative)
    expect_failure("alpha_outer", lambda: learner.validate_parameter_origin("outer"), negative)
    expect_failure("threshold_query", lambda: learner.validate_parameter_origin("query"), negative)
    expect_failure("threshold_train", lambda: learner.validate_parameter_origin("train"), negative)
    for role in ["dev", "calibration"]:
        learner.validate_parameter_origin(role)
    first_train_label = next(learner.read_labels("train"))
    if set(first_train_label) != set(learner.allowlist["label_roles"]["train"]["columns"]):
        raise ValueError("Learner label column projection failed")
    evaluator = Evaluator(output / "evaluator_manifest.json", output / "split_manifest.json", output / "protocol_runs")
    expect_failure("outer_evaluation_disabled", lambda: evaluator.evaluate_outer({}), negative)
    role_manifest = json.loads((output / "role_manifest.json").read_text())
    limited_strata = {item.get("stratum") for item in role_manifest["limitations"]}
    insufficient_outer_without_limit = [stratum for stratum, count in outer_strata.items() if count < 2 and stratum not in limited_strata]
    overlap = json.loads((Path(config["split_ready_dir"]) / "overlap_report.json").read_text())
    result = {
        "round": 1,
        "role_counts": dict(role_counts),
        "official_model_role_counts": dict(official_counts),
        "violations": violations,
        "negative_tests": negative,
        "unmodified_designed_exact_overlap": overlap["unmodified_designed_reference"]["raw_exact_intersection"],
        "outer_units_by_stratum": dict(outer_strata),
        "insufficient_outer_without_limit": insufficient_outer_without_limit,
        "pass": sum(role_counts.values()) == 2753885 and not any(violations.values()) and all(test["pass"] for test in negative) and overlap["unmodified_designed_reference"]["raw_exact_intersection"] == 4906 and not insufficient_outer_without_limit,
    }
    write_json(output / "verification_round1_split.json", result)
    if not result["pass"]:
        raise ValueError("Split intersection/negative verification failed")
    return result


def round2(config, output):
    fixture = output / "protocol_negative_tests"
    fixture.mkdir()
    query_vault = fixture / "query_label_vault.tsv.gz"
    with gzip.open(query_vault, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", "measured_rl", "measured_status"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow({"sequence_id": "q1", "measured_rl": "1.25", "measured_status": "measured"})
        writer.writerow({"sequence_id": "q2", "measured_rl": "", "measured_status": "missing"})
    outer_vault = fixture / "outer_label_vault.tsv.gz"
    with gzip.open(outer_vault, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", "measured_rl", "measured_status"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow({"sequence_id": "o1", "measured_rl": "999", "measured_status": "measured"})
    manifest = {
        "label_vaults": {
            "query": {"path": str(query_vault)},
            "outer": {"path": str(outer_vault)},
        }
    }
    manifest_path = fixture / "evaluator_manifest.json"
    write_json(manifest_path, manifest)
    evaluator = Evaluator(manifest_path, output / "split_manifest.json", fixture / "runs")
    tests = []
    past = "2020-01-01T00:00:00+00:00"
    request1 = create_selection(fixture / "request1", "run1", [{"sequence_id": "q1"}], output / "split_manifest.json", past)
    record1 = evaluator.reveal(request1)
    tests.append({"test": "valid_query_reveal", "pass": record1["selection_hash_verified_before_reveal"] and not record1["outer_labels_returned"]})
    revealed1 = list(read_revealed_labels(record1))
    tests.append({"test": "query_label_only", "pass": revealed1 == [{"sequence_id": "q1", "measured_rl": "1.25", "measured_status": "measured"}]})
    run1_hash = sha256_file(fixture / "runs/run1/revealed_labels.jsonl")
    request2 = create_selection(fixture / "request2", "run2", [{"sequence_id": "q1"}], output / "split_manifest.json", past)
    record2 = evaluator.reveal(request2)
    tests.append({"test": "repeat_reveal_new_run_preserves_prior", "pass": record2["repeat_of_run_id"] == "run1" and sha256_file(fixture / "runs/run1/revealed_labels.jsonl") == run1_hash and record2["revealed_label_hash"] == record1["revealed_label_hash"]})
    request_same = create_selection(fixture / "request_same_id", "run1", [{"sequence_id": "q1"}], output / "split_manifest.json", past)
    expect_failure("same_run_id_rejected", lambda: evaluator.reveal(request_same), tests)
    tampered = create_selection(fixture / "request_tampered", "run3", [{"sequence_id": "q1"}], output / "split_manifest.json", past)
    with open(fixture / "request_tampered/selection.jsonl", "a") as handle:
        handle.write(json.dumps({"sequence_id": "q2"}) + "\n")
    expect_failure("tampered_selection_hash", lambda: evaluator.reveal(tampered), tests)
    future = create_selection(fixture / "request_future", "run4", [{"sequence_id": "q1"}], output / "split_manifest.json", "2999-01-01T00:00:00+00:00")
    expect_failure("not_before_future_rejected", lambda: evaluator.reveal(future), tests)
    outer_request = create_selection(fixture / "request_outer", "run5", [{"sequence_id": "o1"}], output / "split_manifest.json", past, request_role="outer")
    expect_failure("outer_reveal_rejected", lambda: evaluator.reveal(outer_request), tests)
    unknown_request = create_selection(fixture / "request_unknown", "run6", [{"sequence_id": "not_in_query"}], output / "split_manifest.json", past)
    expect_failure("nonquery_selection_rejected", lambda: evaluator.reveal(unknown_request), tests)
    empty_request = create_selection(fixture / "request_empty", "run7", [], output / "split_manifest.json", past)
    expect_failure("empty_selection_rejected", lambda: evaluator.reveal(empty_request), tests)
    revealed_ids = [item["sequence_id"] for path in (fixture / "runs").rglob("revealed_labels.jsonl") for item in read_revealed_labels({"request_role": "query", "outer_labels_returned": False, "revealed_labels_path": str(path)})]
    tests.append({"test": "no_outer_labels_in_reveal_outputs", "pass": revealed_ids == ["q1", "q1"]})
    result = {
        "round": 2,
        "valid_reveal_run_ids": [record1["run_id"], record2["run_id"]],
        "repeat_of_run_id": record2["repeat_of_run_id"],
        "negative_tests": tests,
        "outer_labels_returned": False,
        "outer_regret_computed": False,
        "pass": all(test["pass"] for test in tests),
    }
    write_json(output / "verification_round2_protocol.json", result)
    if not result["pass"]:
        raise ValueError("Protocol time/permission/reveal verification failed")
    return result


def round3(config, output):
    source = Path(config["split_ready_dir"])
    manifest = json.loads((output / "split_manifest.json").read_text())
    source_rows = rows(source / "sequence_index.tsv.gz")
    digest = hashlib.sha256()
    mismatches = []
    count = 0
    for assignment in rows(output / "role_assignments.tsv.gz"):
        source_row = next(source_rows)
        count += 1
        for key in ["raw_row_id", "raw_row_sha256", "sequence_id", "near_group_id", "trajectory_id", "round", "replicate_id"]:
            if assignment[key] != source_row[key] and len(mismatches) < 20:
                mismatches.append({"row": count, "field": key, "assignment": assignment[key], "source": source_row[key]})
        canonical = "\t".join([assignment["raw_row_id"], assignment["sequence_id"], assignment["near_group_id"], assignment["allocation_unit_id"], assignment["role"], assignment["official_model_role"]]) + "\n"
        digest.update(canonical.encode())
    extra = sum(1 for _ in source_rows)
    if digest.hexdigest() != manifest["split_hash"]:
        mismatches.append({"field": "split_hash"})
    if sha256_file(Path(config["input_manifest"])) != manifest["data_manifest_sha256"]:
        mismatches.append({"field": "data_manifest_sha256"})
    if sha256_file(Path(config["split_ready_checksums"])) != manifest["split_ready_checksums_sha256"]:
        mismatches.append({"field": "split_ready_checksums_sha256"})
    for name, expected in manifest["file_hashes"].items():
        if sha256_file(output / name) != expected:
            mismatches.append({"field": "file_hash", "name": name})
    role_counts = Counter()
    for row in rows(output / "role_assignments.tsv.gz"):
        role_counts[row["role"]] += 1
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
    view_counts = {}
    role_for_view = {"train": "train", "dev": "dev", "calibration": "calibration", "official_seen_test": "official_seen_test", "query": "query", "query_vault": "query", "outer_features": "outer", "outer_vault": "outer", "ambiguous_vault": "official_boundary_ambiguous"}
    for name, relative in view_paths.items():
        path = output / relative
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames or []
            if name in {"query", "outer_features"} and any(column in fields for column in ["measured_rl", "measured_status", "predicted_mrl", "predicted_status"]):
                mismatches.append({"field": "label_column_leak", "view": name})
            view_counts[name] = sum(1 for _ in reader)
        if view_counts[name] != role_counts[role_for_view[name]]:
            mismatches.append({"field": "view_row_count", "view": name, "observed": view_counts[name], "expected": role_counts[role_for_view[name]]})
    result = {
        "round": 3,
        "assignment_rows": count,
        "extra_source_rows": extra,
        "split_hash": digest.hexdigest(),
        "view_counts": view_counts,
        "mismatches": mismatches,
        "outer_labels_returned": False,
        "pass": count == 2753885 and extra == 0 and not mismatches,
    }
    write_json(output / "verification_round3_rows.json", result)
    if not result["pass"]:
        raise ValueError("Split row/hash verification failed")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["output_dir"])
    for name in ["verification_round1_split.json", "verification_round2_protocol.json", "verification_round3_rows.json", "verification_summary.json"]:
        if (output / name).exists():
            raise FileExistsError(output / name)
    first = round1(config, output)
    second = round2(config, output)
    third = round3(config, output)
    summary = {
        "status": "SPLIT_AND_VAULT_FROZEN_VERIFIED_E1_NOT_AUTHORIZED",
        "split_version": json.loads((output / "split_manifest.json").read_text())["split_version"],
        "split_hash": json.loads((output / "split_manifest.json").read_text())["split_hash"],
        "rounds": {"split_intersections_negative": first["pass"], "time_permissions_reveal": second["pass"], "row_conservation_hashes": third["pass"]},
        "role_counts": first["role_counts"],
        "limitations": json.loads((output / "role_manifest.json").read_text())["limitations"],
        "outer_regret_computed": False,
        "outer_labels_returned": False,
        "outer_results_accessed": False,
        "proxy_training_performed": False,
        "E1_authorized": False,
    }
    write_json(output / "verification_summary.json", summary)
    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "split_artifact_checksums.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(output / "split_artifact_checksums.json", files)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
