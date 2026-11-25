#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.workspace.resolve()
    source = root / "deliverables/reports/source_data/paper1_budget_matched_v1"
    manifest_path = source / "delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["contains_sequence_strings"] or manifest["contains_sequence_level_measured_labels"]:
        raise AssertionError("local privacy boundary")
    if len(manifest["files"]) < 25:
        raise AssertionError("incomplete delivery manifest")
    for item in manifest["files"]:
        path = root / item["local_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise AssertionError(f"remote/local delivery hash mismatch: {path}")
    results = json.loads((source / "results.json").read_text())
    audit = json.loads((source / "selection_audit.json").read_text())
    if results["new_blind_test"] is not False or results["execution"] != "CPU_only_no_training_no_inference":
        raise AssertionError("evidence or execution boundary")
    if audit["original_grid_records"] != 3200 or audit["new_random_n_records"] != 1600 or audit["cap1_random_n_id_regret_matches"] != 320:
        raise AssertionError("raw audit counts")
    if set(results["families"]) != {"bon_minus_random1", "bon_minus_random_n", "random_n_minus_random1"}:
        raise AssertionError("family mismatch")
    for family, rows in results["families"].items():
        if len(rows) != 20 or sum(row["cap"] == 1 for row in rows) != 4:
            raise AssertionError("all-20 family or cap1 mismatch")
        for row in rows:
            if row["cap"] == 1 and (row["effect"] != 0 or row["p"] != 1 or row["holm_p"] != 1 or row["bonferroni_p"] != 1):
                raise AssertionError("cap1 structural zero mismatch")
            if not (0 <= row["p"] <= 1 and 0 <= row["holm_p"] <= 1 and 0 <= row["bonferroni_p"] <= 1):
                raise AssertionError("invalid p-value")
    with (source / "pool_effects.tsv").open(newline="") as handle:
        pools = list(csv.DictReader(handle, delimiter="\t"))
    if len(pools) != 320:
        raise AssertionError("pool count")
    for row in pools:
        values = {key: float(row[key]) for key in ("bon_minus_random1", "bon_minus_random_n", "random_n_minus_random1")}
        if not np.isclose(values["bon_minus_random1"], values["bon_minus_random_n"] + values["random_n_minus_random1"], rtol=0, atol=1e-10):
            raise AssertionError("delivered pool decomposition")
    with (source / "context_summary.tsv").open(newline="") as handle:
        contexts = list(csv.DictReader(handle, delimiter="\t"))
    with (source / "family_effects.tsv").open(newline="") as handle:
        families = list(csv.DictReader(handle, delimiter="\t"))
    if len(contexts) != 20 or len(families) != 60:
        raise AssertionError("summary TSV counts")
    forbidden = {"selected_id", "selected_index", "utr50", "measured_rl", "sequence_id"}
    for filename in ("pool_effects.tsv", "context_summary.tsv", "family_effects.tsv"):
        with (source / filename).open(newline="") as handle:
            if forbidden.intersection(csv.DictReader(handle, delimiter="\t").fieldnames or []):
                raise AssertionError(f"sequence-level field in {filename}")
    if (source / "paired_records.tsv").exists():
        raise AssertionError("remote-only selected-ID/regret audit was exported")
    report = (root / "deliverables/reports/docs/PAPER1_BUDGET_MATCHED_V1.md").read_text()
    for phrase in ("retrospective sensitivity", "not a new blind test", "CPU-only", "No Random(N)−Random(1) contrast survives", "cap-1 structural zeros"):
        if phrase not in report:
            raise AssertionError(f"missing report boundary: {phrase}")
    for family in ("bon_minus_random_n", "random_n_minus_random1"):
        for row in results["families"][family]:
            for value in (row["effect"], *row["nominal_95_ci"], row["holm_p"], row["bonferroni_p"]):
                if f"{value:.6f}" not in report:
                    raise AssertionError(f"report value missing: {family}/{row['context']}/{row['cap']}")
    for round_number in (2, 3):
        validation = json.loads((source / f"validation/independent_round{round_number}.json").read_text())
        if validation["status"] != "PASS" or validation["round"] != round_number or not validation["independent_implementation"]:
            raise AssertionError("independent validation missing")
    failed_logs = list((source / "logs").glob("*FAILED*.log")) + [source / "logs/test_round1.log", source / "logs/independent_round2.log", source / "logs/independent_round2_retry1.log"]
    if any(not path.exists() or path.stat().st_size == 0 for path in failed_logs):
        raise AssertionError("failure logs missing")
    output = {
        "round": 3,
        "status": "PASS",
        "delivery_files_sha256_verified": len(manifest["files"]),
        "pool_decomposition_checks": len(pools),
        "context_rows": len(contexts),
        "family_rows": len(families),
        "failure_logs_retained": len(failed_logs),
        "local_sequence_strings": False,
        "local_sequence_level_labels": False,
        "remote_only_pair_audit": True,
        "cpu_only": True,
    }
    print(json.dumps(output))
    target = source / "validation/delivery_round3.json"
    if not args.check:
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
