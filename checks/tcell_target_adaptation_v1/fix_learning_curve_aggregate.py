from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from adaptation import digest, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["artifact_root"])
    target = root / "target"
    if (root / "POST_EVALUATION_AGGREGATE_FIX.json").exists():
        raise RuntimeError("aggregate fix already applied")
    original = target / "results.json"
    secondary_path = target / "secondary.json"
    archive = root / "failed_attempts" / "aggregate_learning_curve_attempt1"
    archive.mkdir(parents=True, exist_ok=False)
    for path in (original, secondary_path, target / "ONE_EVALUATION_COMPLETED.json"):
        shutil.copyfile(path, archive / path.name)
    results = json.loads(original.read_text())
    secondary = json.loads(secondary_path.read_text())
    row = next(row for row in results["nested_learning_curve"] if row["budget"] == 50000)
    primary_index = results["caps"].index(16)
    for repeat in ("defined_tcell_r1", "defined_tcell_r2"):
        method = np.load(target / f"{repeat}_main_50k_regret.npy", allow_pickle=False)[primary_index]
        baseline = np.load(target / f"{repeat}_original_cnn_regret.npy", allow_pickle=False)[primary_index]
        expected_regret = results["by_repeat_and_cap"][repeat]["16"]["mean_regret"]["main_50k"]
        expected_effect = results["primary"]["repeat_main_minus_original_effects"][repeat]
        if float(method.mean()) != expected_regret or float(np.mean(method - baseline)) != expected_effect:
            raise ValueError("saved cap16 arrays do not match frozen primary")
        row[f"{repeat}_cap16_regret"] = float(method.mean())
        row[f"{repeat}_minus_original"] = float(np.mean(method - baseline))
    secondary["nested_learning_curve"] = results["nested_learning_curve"]
    save_json(original, results)
    save_json(secondary_path, secondary)
    completed_path = target / "ONE_EVALUATION_COMPLETED.json"
    completed = json.loads(completed_path.read_text())
    completed["results_sha256"] = digest(original)
    completed["aggregate_fix"] = "50k nested row now uses cap16 index only; no new inference or label read"
    save_json(completed_path, completed)
    report = {"status": "PASS", "reason": "Delivery review found the nested 50k learning-curve row averaged all caps. Corrected that row from already-saved cap16 regret arrays; primary and all other scientific results unchanged.", "evaluation_count": 1, "new_test_inference": False, "new_test_mrl_read": False, "model_or_protocol_changes": 0, "archived_original": str(archive.relative_to(root)), "corrected_nested_50k": row, "code_sha256": {}}
    save_json(root / "POST_EVALUATION_AGGREGATE_FIX.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
