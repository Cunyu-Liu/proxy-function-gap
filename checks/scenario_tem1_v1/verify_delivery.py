#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def report_checks(report, results):
    text = Path(report).read_text()
    overall = pd.read_csv(Path(results) / "overall_results.tsv", sep="\t")
    strings = []
    for row in overall.itertuples():
        strings.extend([f"{row.bon_regret:.6f}", f"{row.random_n_regret:.6f}", f"{abs(row.delta_bon_minus_random):.6f}", f"{row.turnover_proxy_vs_oracle * 100:.2f}%"])
    strings += ["15/15", "78,000", "3,120", "−0.053196", "0.311799", "22", "25", "21"]
    missing = [value for value in strings if value not in text]
    if missing:
        raise AssertionError(f"report values missing: {missing}")
    forbidden_claims = ["新盲测", "label-free。\n", "直接酶活。", "跨药物 proxy 选择有效。"]
    return {"report_key_numbers_present": True, "n_report_number_checks": len(strings)}


def remote_mode(args):
    artifact = Path(args.artifact_dir)
    code = Path(args.code_dir)
    report = artifact / "SCENARIO_TEM1_V1.md"
    protocol = code / "frozen_protocol.json"
    run = json.loads((artifact / "results/test/run_manifest.json").read_text())
    assert run["split"] == "test" and run["test_execution"] and not run["model_fitting"]
    assert run["protocol_sha256"] == sha256(protocol)
    assert run["code_sha256"] == sha256(code / "run_experiment.py")
    assert sorted(p.name for p in (artifact / "results").iterdir()) == ["dev", "test"]
    assert sorted(p.name for p in (artifact / "logs").glob("test*.log")) == ["test_run_once.log"]
    for name in ["round1_raw.json", "round2_independent.json"]:
        assert json.loads((artifact / "verification" / name).read_text())["status"] == "PASS"
    for filename in ["frozen_protocol.json", "run_experiment.py", "verify_raw.py", "independent_verify.py", "verify_delivery.py"]:
        assert sha256(code / filename) == sha256(artifact / "source_snapshot" / filename)
    raw_manifest = json.loads((artifact / "raw/input_manifest.json").read_text())
    for item in raw_manifest["files"]:
        assert sha256(item["path"]) == item["sha256"]
    checks = report_checks(report, artifact / "results/test")
    checks.update({"test_run_once": True, "run_code_and_protocol_hashes_match": True, "source_snapshot_exact": True, "seven_raw_hashes_still_match": True, "rounds_1_2_pass": True, "only_dev_and_test_result_directories": True})
    result = {"round": 3, "stage": "remote", "status": "PASS", "checks": checks}
    output = artifact / "verification/round3_remote.json"
    output.write_text(json.dumps(result, indent=2) + "\n")

    local_base = "deliverables/reports/source_data/scenario_tem1_v1"
    mapping = {
        "source_snapshot/frozen_protocol.json": f"{local_base}/frozen_protocol.json",
        "source_snapshot/run_experiment.py": f"{local_base}/run_experiment.py",
        "source_snapshot/verify_raw.py": f"{local_base}/verify_raw.py",
        "source_snapshot/independent_verify.py": f"{local_base}/independent_verify.py",
        "source_snapshot/verify_delivery.py": f"{local_base}/verify_delivery.py",
        "results/test/overall_results.tsv": f"{local_base}/results/overall_results.tsv",
        "results/test/direction_results.tsv": f"{local_base}/results/direction_results.tsv",
        "results/test/correlations.tsv": f"{local_base}/results/correlations.tsv",
        "results/test/run_manifest.json": f"{local_base}/results/run_manifest.json",
        "results/test/data_audit.json": f"{local_base}/results/data_audit.json",
        "raw/input_manifest.json": f"{local_base}/results/input_manifest.json",
        "verification/round1_raw.json": f"{local_base}/verification/round1_raw.json",
        "verification/round2_independent.json": f"{local_base}/verification/round2_independent.json",
        "verification/round3_remote.json": f"{local_base}/verification/round3_remote.json",
        "logs/dev_run.log": f"{local_base}/logs/dev_run.log",
        "logs/test_run_once.log": f"{local_base}/logs/test_run_once.log",
        "logs/round1_raw.log": f"{local_base}/logs/round1_raw.log",
        "logs/round2_independent.log": f"{local_base}/logs/round2_independent.log",
        "SCENARIO_TEM1_V1.md": "SCENARIO_TEM1_V1.md",
    }
    entries = []
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rel = str(path.relative_to(artifact))
            entry = {"path": rel, "sha256": sha256(path), "bytes": path.stat().st_size}
            if rel in mapping:
                entry["local_relative"] = mapping[rel]
            entries.append(entry)
    manifest = {"artifact_id": "scenario_tem1_v1", "files": entries, "local_mapped_files": len(mapping)}
    (artifact / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "remote_checks": len(checks), "manifest_files": len(entries), "local_mapped_files": len(mapping)}, indent=2))


def local_mode(args):
    workspace = Path(args.workspace)
    base = workspace / "deliverables/reports/source_data/scenario_tem1_v1"
    manifest = json.loads((base / "artifact_manifest.json").read_text())
    mapped = [item for item in manifest["files"] if "local_relative" in item]
    assert len(mapped) == manifest["local_mapped_files"] == 19
    for item in mapped:
        path = workspace / item["local_relative"]
        assert path.is_file(), path
        assert sha256(path) == item["sha256"], path
    prohibited = []
    for path in base.rglob("*"):
        if path.is_file() and (path.name.startswith("stiffler2015_") or path.name in {"pool_membership.tsv.gz", "selections.tsv.gz", "position_effects.tsv.gz", "bootstrap_effects.npz"} or path.name.startswith("bootstrap_position_indices")):
            prohibited.append(str(path))
    assert not prohibited, prohibited
    for name in ["round1_raw.json", "round2_independent.json", "round3_remote.json"]:
        assert json.loads((base / "verification" / name).read_text())["status"] == "PASS"
    run = json.loads((base / "results/run_manifest.json").read_text())
    assert run["split"] == "test" and run["test_execution"] and not run["model_fitting"]
    checks = report_checks(workspace / "SCENARIO_TEM1_V1.md", base / "results")
    checks.update({"19_remote_local_hash_pairs_exact": True, "no_local_raw_or_per_variant_files": True, "rounds_1_2_and_remote_3_pass": True, "single_retrospective_test_manifest": True})
    result = {"round": 3, "stage": "delivery", "status": "PASS", "checks": checks, "prohibited_local_files": prohibited, "raw_score_tables_synced": False, "selection_or_pool_membership_synced": False}
    (base / "verification/round3_delivery.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["remote", "local"], required=True)
    parser.add_argument("--artifact-dir")
    parser.add_argument("--code-dir")
    parser.add_argument("--workspace")
    args = parser.parse_args()
    if args.mode == "remote":
        remote_mode(args)
    else:
        local_mode(args)


if __name__ == "__main__":
    main()
