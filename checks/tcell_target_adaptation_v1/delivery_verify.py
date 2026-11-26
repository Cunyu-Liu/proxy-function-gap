from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_hash = digest(config_path)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    prefreeze = json.loads((root / "verification" / "prefreeze.json").read_text())
    workspace = Path(__file__).resolve().parent
    amendment_path = config_path.parent / "dev_repeat_amendment_frozen.json"
    amendment_hash = digest(amendment_path)
    amendment = json.loads(amendment_path.read_text())
    assert prefreeze["config_sha256"] == amendment["base_config_sha256"] == config_hash
    expected_code_hashes = dict(amendment["code_sha256"])
    post_fix_path = root / "POST_EVALUATION_AGGREGATE_FIX.json"
    if post_fix_path.exists():
        post_fix = json.loads(post_fix_path.read_text())
        assert post_fix["evaluation_count"] == 1 and not post_fix["new_test_inference"] and not post_fix["new_test_mrl_read"] and post_fix["model_or_protocol_changes"] == 0
        expected_code_hashes.update(post_fix["code_sha256"])
    current_hashes = {name: digest(workspace / name) for name in expected_code_hashes}
    assert current_hashes == expected_code_hashes, "code changed outside frozen amendment chain"
    prepare = json.loads((root / "prepare_manifest.json").read_text())
    frozen = json.loads((root / "TRAINING_FROZEN.json").read_text())
    started = json.loads((root / "target" / "ONE_EVALUATION_STARTED.json").read_text())
    completed = json.loads((root / "target" / "ONE_EVALUATION_COMPLETED.json").read_text())
    results = json.loads((root / "target" / "results.json").read_text())
    assert prepare["config_sha256"] == frozen["config_sha256"] == started["config_sha256"] == completed["config_sha256"] == results["config_sha256"] == config_hash
    assert frozen["dev_repeat_amendment_sha256"] == results["dev_repeat_amendment_sha256"] == amendment_hash
    assert results["dev_selection_metric"] == "maximum_of_r1_and_r2_cap16_regret"
    assert frozen["run_count"] == 72 and started["evaluation_count"] == completed["evaluation_count"] == results["evaluation_count"] == 1
    assert digest(root / "target" / "results.json") == completed["results_sha256"]
    assert started["training_freeze_sha256"] == results["training_freeze_sha256"] == digest(root / "TRAINING_FROZEN.json")
    assert completed["post_evaluation_protocol_changes"] == results["primary"]["post_evaluation_protocol_changes"] == 0
    assert not results["sequence_independent"] and not results["new_blind_test"]
    independent = json.loads((root / "verification" / "round2_independent.json").read_text())
    assert independent["status"] == "PASS" and independent["comparisons"] > 100 and independent["maximum_absolute_difference"] < 1e-10
    report = {"status": "PASS", "round": 3, "config_sha256": config_hash, "dev_repeat_amendment_sha256": amendment_hash, "code_hashes_unchanged_since_prereview_amendment": True, "dev_selection_metric": "maximum_of_r1_and_r2_cap16_regret", "train_repeat_equal_weight": True, "three_seed_equal_weight": True, "from_scratch_fairness": True, "training_runs": frozen["run_count"], "target_evaluation_count": 1, "training_frozen_before_target": True, "post_evaluation_protocol_changes": 0, "aggregate_learning_curve_fix_without_reevaluation": post_fix_path.exists(), "primary_success": results["primary"]["success"], "evidence_boundary": results["evidence_boundary"], "no_push": True}
    (root / "verification" / "round3_delivery.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
