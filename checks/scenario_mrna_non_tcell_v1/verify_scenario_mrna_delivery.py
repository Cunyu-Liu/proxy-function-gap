#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CAPS = [1, 2, 4, 8, 16]
CONTEXTS = ["HepG2|TGFBR2", "HepG2|PDCD1", "K562|TGFBR2", "K562|PDCD1"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    checks = {}
    execution = json.loads((root / "verification" / "round1_execution.json").read_text())
    independent = json.loads((root / "verification" / "round2_independent.json").read_text())
    if execution["status"] != "PASS" or independent["status"] != "PASS":
        raise AssertionError("Upstream verification not PASS")
    if execution["cuda_used"] is not False or execution["cpu_mlp_inference"] is not True or execution["new_training"] is not False:
        raise AssertionError("Execution device/training declaration")
    if execution["schlusser_recomputed"] is not False or execution["new_blind_test"] is not False:
        raise AssertionError("Evidence boundary")
    protocol_hash = sha256(root / "protocol" / "frozen_protocol.json")
    if protocol_hash != execution["protocol_sha256"]:
        raise AssertionError("Protocol changed after execution")
    locked_log = json.loads((root / "logs" / "round1_analysis_locked_final.log").read_text())
    if locked_log["status"] != "PASS" or abs(float(locked_log["elapsed_seconds"]) - float(execution["elapsed_seconds"])) > 1e-9:
        raise AssertionError("Execution log/JSON elapsed-time identity failed")
    checks["execution_log_identity"] = {"elapsed_seconds": execution["elapsed_seconds"], "status": locked_log["status"]}
    checks["protocol_sha256"] = protocol_hash
    per_seed = pd.read_csv(root / "results" / "decision_per_seed.tsv", sep="\t")
    summary = pd.read_csv(root / "results" / "decision_summary.tsv", sep="\t")
    maximum_summary_error = 0.0
    for row in summary.itertuples(index=False):
        selected = per_seed.loc[(per_seed["pool"] == row.pool) & (per_seed["context"] == row.context) & (per_seed["cap"] == row.cap)]
        if len(selected) != 200:
            raise AssertionError("Algorithm seed count")
        for field in ("bon_regret", "random_n_regret", "random1_regret", "bon_minus_random_n"):
            maximum_summary_error = max(maximum_summary_error, abs(float(selected[field].mean()) - float(getattr(row, field))))
        if not np.allclose(selected["bon_minus_random_n"], selected["bon_regret"] - selected["random_n_regret"], rtol=0, atol=1e-12):
            raise AssertionError("Per-seed regret identity")
    if maximum_summary_error > 1e-12:
        raise AssertionError("Summary arithmetic")
    checks["decision_summary"] = {"rows": len(summary), "per_seed_rows": len(per_seed), "maximum_absolute_mean_error": maximum_summary_error}
    primary = summary.loc[(summary["pool"] == "indel_kmer_48") & summary["context"].str.startswith("primary|")]
    if len(primary) != 20 or set(primary["cap"]) != set(CAPS) or set(primary["context"]) == set():
        raise AssertionError("Incomplete indel grid")
    stability = summary.loc[(summary["pool"] == "stability_kmer_27") & (summary["context"] == "primary|integrated_abundance")]
    if len(stability) != 5 or set(stability["cap"]) != set(CAPS):
        raise AssertionError("Incomplete stability grid")
    measured = summary.loc[summary["pool"] == "indel_measured_12"]
    if set(measured["cap"]) != {1, 2, 4, 8} or 16 in set(measured["cap"]):
        raise AssertionError("n12 cap boundary")
    checks["complete_grids"] = {"indel_primary_rows": len(primary), "stability_primary_rows": len(stability), "measured_mrl_oracle_caps": sorted(set(measured["cap"]))}
    candidate = pd.read_csv(root / "results" / "candidate_master.tsv", sep="\t")
    stability_rows = candidate.loc[candidate["integrated_abundance"].notna()]
    if len(candidate) != 48 or len(stability_rows) != 27 or candidate["sequence_id"].str.contains("spike").any():
        raise AssertionError("Candidate n or spike-in exclusion")
    if candidate["measured_mrl"].notna().sum() != 12 or candidate["published_predicted_mrl"].notna().sum() != 45 or stability_rows["measured_mrl"].notna().sum() != 8:
        raise AssertionError("MRL role coverage")
    checks["candidate_roles"] = {"panel_utrs": 48, "stability_experimental": 27, "spikein_candidates": 0, "measured_mrl_panel": 12, "measured_mrl_stability": 8, "published_predicted_mrl": 45}
    bootstrap = pd.read_json(root / "results" / "decision_bootstrap.json")
    if len(bootstrap) != 70 or len(bootstrap.loc[bootstrap["pool"] == "indel_kmer_48"]) != 20 or len(bootstrap.loc[bootstrap["pool"] == "stability_kmer_27"]) != 5:
        raise AssertionError("Bootstrap grid incomplete")
    cap1 = bootstrap.loc[bootstrap["cap"] == 1]
    if not all(abs(float(x["bootstrap_delta_mean"])) < 1e-12 and max(abs(float(x["ci95"][0])), abs(float(x["ci95"][1]))) < 1e-12 for x in cap1.to_dict("records")):
        raise AssertionError("cap1 structural zero")
    checks["bootstrap_rows"] = len(bootstrap)
    report = (root / "REPORT.md").read_text()
    required_text = ["不是新盲测", "未重新揭盲", "没有新训练", "27 experimental+8 spike-in", "48 UTR", "Random(N)", "Strong Kozak", "功效限制", "top-tail", "Schlusser"]
    missing = [x for x in required_text if x not in report]
    if missing:
        raise AssertionError(f"Report required language missing: {missing}")
    forbidden = ["本轮新盲测", "重新揭盲 Schlusser", "CUDA 推理（本轮）", "35 条候选"]
    present = [x for x in forbidden if x in report]
    if present:
        raise AssertionError(f"Forbidden report language: {present}")
    checks["report"] = {"bytes": len(report.encode()), "required_assertions": len(required_text), "forbidden_assertions": len(forbidden)}
    required_files = [
        "REPORT.md", "protocol/frozen_protocol.json", "protocol/isolation_audit.json", "predictions/predictions.tsv", "predictions/execution_device.json",
        "results/indel_summary_primary.tsv", "results/indel_summary_all_flagged.tsv", "results/indel_summary_absolute_2pmol.tsv", "results/stability_summary.tsv",
        "results/decision_per_seed.tsv", "results/decision_summary.tsv", "results/decision_bootstrap.json", "results/decision_diagnostics.json",
        "results/mrl_frozen_regret.tsv", "results/mrl_frozen_diagnostics.json", "results/unified_scenario_metrics.json", "results/schlusser_historical_summary.json",
        "verification/round1_execution.json", "verification/round2_independent.json", "source_snapshot/scenario_mrna_non_tcell.py", "source_snapshot/verify_scenario_mrna_independent.py",
        "logs/round1_analysis_attempt1_failed.log", "protocol/frozen_protocol_attempt1_failed.json", "logs/tests_round1.log",
        "verification/concurrent_output_incident.json", "logs/round1_analysis_locked_final.log", "quarantine/concurrent_mixed_output/verification/round1_execution.json",
    ]
    absent = [path for path in required_files if not (root / path).is_file()]
    if absent:
        raise AssertionError(f"Required files missing: {absent}")
    if args.local:
        manifest = json.loads((root / "artifact_manifest.json").read_text())
        mismatches = []
        for relative, record in manifest.items():
            path = root / relative
            if not path.is_file() or sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                mismatches.append(relative)
        current_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name != "artifact_manifest.json"}
        extra = sorted(current_files - set(manifest))
        if mismatches or extra:
            raise AssertionError(f"Local manifest mismatch: {mismatches}; extra: {extra}")
        checks["local_manifest"] = {"files": len(manifest), "mismatches": mismatches, "extra": extra}
    result = {"status": "PASS", "mode": "local_delivery" if args.local else "remote_delivery", "checks": checks}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.write_manifest:
        result["manifest_files"] = sum(1 for path in root.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.write_manifest:
        records = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "artifact_manifest.json":
                records[str(path.relative_to(root))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        (root / "artifact_manifest.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
