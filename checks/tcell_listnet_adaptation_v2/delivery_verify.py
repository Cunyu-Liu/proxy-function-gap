from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from adaptation import digest, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_hash = digest(config_path)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    workspace = Path(__file__).resolve().parent
    prefreeze = json.loads((root / "verification/prefreeze.json").read_text())
    prepare = json.loads((root / "prepare_manifest.json").read_text())
    training = json.loads((root / "TRAINING_FROZEN.json").read_text())
    predictions = json.loads((root / "TEST_PREDICTIONS_FROZEN.json").read_text())
    started = json.loads((root / "target_v2/ONE_EVALUATION_STARTED.json").read_text())
    completed = json.loads((root / "target_v2/ONE_EVALUATION_COMPLETED.json").read_text())
    results = json.loads((root / "target_v2/results.json").read_text())
    assert prefreeze["config_sha256"] == prepare["config_sha256"] == training["config_sha256"] == predictions["config_sha256"] == results["config_sha256"] == config_hash
    for name, expected in prefreeze["code_sha256"].items():
        assert digest(workspace / name) == expected, f"code changed after prefreeze: {name}"
    assert prepare["test_v2_to_v1_exact_h1"] == prepare["test_sample_exact"] == prepare["test_sample_h1"] == prepare["cross_test_pool_exact_h1"] == 0
    assert prepare["counts"] == {"train": 50000, "dev": 10000, "test_v2": 1024}
    assert training["run_count"] == len(training["runs"]) == 81
    for run in training["runs"]:
        path = root / run["path"]
        summary = json.loads(path.read_text())
        assert digest(path) == run["summary_sha256"]
        assert summary["converged_dev_plateau"] and summary["scientific_max_epoch"] is None
        assert summary["gpu"]["actual_cuda"] and not summary["gpu"]["cpu_fallback"]
        assert summary["test_v2_mrl_read"] is False and summary["test_inference"] is False
        assert digest(path.parent / "best.pt") == summary["best_checkpoint_sha256"]
        assert digest(path.parent / "history.jsonl") == summary["history_sha256"]
    assert predictions["test_v2_mrl_read"] is False and predictions["post_freeze_temperature_weight_budget_changes"] == 0
    for method, row in predictions["methods"].items():
        assert digest(root / "test_predictions" / f"{method}_scores.npy") == row["score_sha256"]
    assert started["evaluation_count"] == completed["evaluation_count"] == results["evaluation_count"] == 1
    assert completed["results_sha256"] == digest(root / "target_v2/results.json")
    assert results["primary"]["post_evaluation_protocol_changes"] == completed["post_evaluation_protocol_changes"] == 0
    assert results["historical_retrospective_boundary_retained"]
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_listnet.py"], cwd=workspace, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = workspace / "logs/round3_tests.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(test.stdout)
    if test.returncode:
        raise RuntimeError(f"round3 tests failed; see {log}")
    report = {
        "status": "PASS", "analysis_id": config["analysis_id"], "config_sha256": config_hash,
        "code_hashes_unchanged_since_prefreeze": True, "training_runs": 81,
        "all_training_hashes_and_natural_cuda_convergence_pass": True,
        "test_v2_prediction_freeze_pass": True, "evaluation_count": 1,
        "post_evaluation_protocol_changes": 0,
        "historical_retrospective_boundary_retained": True,
        "primary_success": results["primary"]["success"],
        "clean_process_tests": test.stdout.strip().splitlines()[-1],
        "code_push_performed": False,
    }
    save_json(root / "verification/round3_delivery.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
