import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_tie_audit_v1_1 import (
    AFFECTED_SEEDS,
    AUDIT_DIR,
    INPUT_HASHES,
    RESULTS,
    SEEDS,
    V1_ROOT,
    WORKSPACE,
    sha256_file,
)


def read_tsv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main(finalize_manifest=False):
    original_manifest = json.loads((V1_ROOT / "artifact_manifest.json").read_text())
    preexisting_mismatches = {}
    for name, record in original_manifest.items():
        path = V1_ROOT / name
        actual = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if actual == record:
            continue
        known_stale_delivery_record = (
            name == "delivery_verification.json"
            and record == {"bytes": 225, "sha256": "12b1ac143f3b1e3e6ba884c0bee92c8c3299e4a6e7ea5ea8706d2ef64303487e"}
            and actual == {"bytes": 225, "sha256": "8accdec714da2c3423db1753a2efcd4179dd4d8865f5d874480dbb966765f13b"}
        )
        assert known_stale_delivery_record, f"Unexpected v1 manifest mismatch: {name}"
        preexisting_mismatches[name] = {"manifest": record, "observed": actual}
    assert set(preexisting_mismatches) <= {"delivery_verification.json"}
    for name, expected in INPUT_HASHES.items():
        assert sha256_file(V1_ROOT / name) == expected, name

    code_hashes = json.loads((AUDIT_DIR / "frozen_code_hashes.json").read_text())
    original_code_manifest = json.loads((V1_ROOT / "verification" / "code_manifest.json").read_text())
    assert code_hashes["git_metadata_present"] is False
    assert code_hashes["git_commit"] is None
    assert code_hashes["actual_v1_tie_rule"]["preregistered"] is True
    for name, record in original_code_manifest.items():
        assert code_hashes["files"][name]["sha256"] == record["sha256"], name
        assert code_hashes["files"][name]["bytes"] == record["bytes"], name
        if name != "finalize_artifacts.py":
            assert sha256_file(AUDIT_DIR / "source_snapshot" / name) == record["sha256"], name

    original_summary = json.loads((V1_ROOT / "results" / "summary.json").read_text())
    alternate_summary = json.loads((RESULTS / "summary_v1_1.json").read_text())
    comparison = json.loads((RESULTS / "comparison_v1_vs_v1_1.json").read_text())
    original_boot = np.load(V1_ROOT / "results" / "bootstrap_replicates.npy")
    alternate_boot = np.load(RESULTS / "bootstrap_v1_1.npy")
    original_per_seed = read_tsv(V1_ROOT / "results" / "per_seed.tsv")
    alternate_per_seed = read_tsv(RESULTS / "per_seed_v1_1.tsv")
    ties = read_tsv(RESULTS / "tie_seeds.tsv")
    affected = read_tsv(RESULTS / "affected_seeds.tsv")

    assert len(original_per_seed) == len(alternate_per_seed) == 200
    assert [int(row["algorithm_seed"]) for row in alternate_per_seed] == SEEDS
    assert len(ties) == 9
    assert [int(row["algorithm_seed"]) for row in affected] == AFFECTED_SEEDS
    assert all(row["selection_changed"] == "false" for row in affected)
    assert original_boot.shape == alternate_boot.shape == (10000, 8)
    assert np.array_equal(original_per_seed, alternate_per_seed)
    assert np.array_equal(original_boot[:, [1, 2, 3, 4, 6, 7]], alternate_boot[:, [1, 2, 3, 4, 6, 7]])
    assert np.sum(np.any(original_boot != alternate_boot, axis=1)) == 78
    assert np.sum(original_boot != alternate_boot) == 156
    assert comparison["bootstrap_changed_element_count"] == 156
    assert comparison["per_seed_changed_count"] == 0
    assert comparison["v1_1_minus_v1_point"] == 0.0
    assert alternate_summary["primary"]["bon_minus_random_regret"] == original_summary["primary"]["bon_minus_random_regret"]
    assert alternate_summary["primary"]["ci95"] == original_summary["primary"]["ci95"]
    assert alternate_summary["primary"]["p_two_sided_sign"] == original_summary["primary"]["p_two_sided_sign"]
    assert alternate_summary["primary"]["significant_benefit"] is False
    assert alternate_summary["retrospective_sensitivity_only"] is True
    assert alternate_summary["version_selection_by_direction"] is False
    ci = np.percentile(alternate_boot[:, 0], [2.5, 97.5]).tolist()
    assert ci == alternate_summary["primary"]["ci95"]
    delta = alternate_boot[:, 0]
    p_value = min(1.0, 2 * min(((delta >= 0).sum() + 1) / 10001, ((delta <= 0).sum() + 1) / 10001))
    assert p_value == alternate_summary["primary"]["p_two_sided_sign"]

    difference_audit = json.loads((RESULTS / "bootstrap_difference_audit.json").read_text())
    assert difference_audit["status"] == "PASS"
    assert difference_audit["cause"]["scientific_data_changed"] is False
    assert difference_audit["rng_and_sampling"]["remote_original_environment_bitwise_cells"] == 80000
    remote_reproduction = json.loads((AUDIT_DIR / "logs" / "remote_bootstrap_reproduction.log").read_text())
    assert remote_reproduction["status"] == "PASS"
    assert remote_reproduction["remote_writes"] is False
    for name in ["analysis_attempt1.log", "analysis_attempt2.log", "bootstrap_difference_attempt1.log",
                 "bootstrap_difference_attempt2.log", "bootstrap_difference_attempt3.log",
                 "bootstrap_difference_attempt4.log", "setup_ssh_px2_failure.log", "round1_tests.log",
                 "round1_tests_retry1.log", "round1_tests_retry2.log", "round2_replay_hashes.log",
                 "round3_delivery.log"]:
        assert (AUDIT_DIR / "logs" / name).is_file(), name

    report_path = WORKSPACE / "PERSIST_TIE_AUDIT_V1_1.md"
    report_text = report_path.read_text()
    for required in ["20261147", "0.10472255084999998", "0.24517548245175483", "SHA256", "CPU", "94f667185a48abd035fc757c93d0fef6e455e4e1bc4d4bd981d0b56f999334da"]:
        assert required in report_text, required

    verification = {
        "status": "PASS",
        "round": 3,
        "original_v1_files_verified_against_frozen_manifest": len(original_manifest) - len(preexisting_mismatches),
        "original_v1_preexisting_manifest_mismatches": preexisting_mismatches,
        "original_v1_data_or_results_changed": False,
        "remote_source_files_verified": len(original_code_manifest),
        "remote_git_commit": None,
        "base_seed_rows": 200,
        "exact_tie_seeds": 9,
        "stable_first_order_sensitive_seeds": 5,
        "base_seed_effect_changes": 0,
        "bootstrap_replicates": 10000,
        "bootstrap_changed_replicates": 78,
        "bootstrap_changed_elements": 156,
        "primary_point_v1": original_summary["primary"]["bon_minus_random_regret"],
        "primary_point_v1_1": alternate_summary["primary"]["bon_minus_random_regret"],
        "primary_ci95_v1_1": alternate_summary["primary"]["ci95"],
        "primary_p_v1_1": alternate_summary["primary"]["p_two_sided_sign"],
        "bootstrap_difference_validator": "PASS",
        "remote_original_environment_bootstrap_bitwise_cells": 80000,
        "report": str(report_path.relative_to(WORKSPACE)),
        "report_sha256": sha256_file(report_path),
        "compute": "CPU only; no training or inference",
        "version_selection_by_direction": False,
    }
    verification_dir = AUDIT_DIR / "verification"
    verification_dir.mkdir(exist_ok=True)
    (verification_dir / "round3_delivery.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")

    if finalize_manifest:
        records = {}
        for path in sorted(AUDIT_DIR.rglob("*")):
            if not path.is_file() or path.name in {"artifact_manifest.json", ".DS_Store"} or "__pycache__" in path.parts:
                continue
            records[str(path.relative_to(AUDIT_DIR))] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        records["../../PERSIST_TIE_AUDIT_V1_1.md"] = {"sha256": sha256_file(report_path), "bytes": report_path.stat().st_size}
        (AUDIT_DIR / "artifact_manifest.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-manifest", action="store_true")
    args = parser.parse_args()
    main(args.finalize_manifest)
