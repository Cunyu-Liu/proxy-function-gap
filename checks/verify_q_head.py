import argparse
import json
from pathlib import Path

import numpy as np
import torch

from proxy_function_gap.pool_manifest import load_pool_manifest
from proxy_function_gap.q_head import pool_features
from proxy_function_gap.selection_eval import SelectionRecord
from proxy_function_gap.small_model import SmallModelConfig, apply_platt, log_loss, predict_with_folds
from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--round", required=True)
    args = parser.parse_args()
    root = Path(args.artifacts)
    manifest = json.loads((root / "q_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha256_file(root / name) == expected
    unique_pools = {(item["path"], item["sha256"]) for item in manifest["input_pools"]}
    assert len(unique_pools) == 48 and len(manifest["selection_sources"]) == 48
    assert manifest["calibrator_loss_type"] == "log_loss" and not manifest["outer_labels_read"] and not manifest["E1_enabled"]
    folds = torch.load(root / "q_fold_models.pt", weights_only=False)
    platt = torch.load(root / "q_platt.pt", weights_only=False)
    assert all(item["gpu"]["cpu_fallback"] is False and item["history"][-1]["plateau_epochs"] >= 3 for item in folds)
    pools = []
    records = []
    for index in range(args.count):
        pool_id = f"calibration_{index:03d}"
        pool = load_pool_manifest(Path(args.pool_root) / pool_id / "pool_manifest.json")
        payload = json.loads((Path(args.selection_dir) / f"{pool_id}.json").read_text())
        selected = [SelectionRecord(**row) for row in payload["records"] if row["strategy"] == "fp-rej" and row["alpha"] == 20 and row["proposal_cap"] == 16]
        assert len(selected) == 5
        pools.append(pool)
        records.extend(selected)
    values = np.asarray([pool_features(pool, record) for pool in pools for record in records if record.pool_id == pool.pool_id])
    config = SmallModelConfig(model_type="logistic")
    raw = predict_with_folds(folds, values, config)
    calibrated = apply_platt(platt, raw, config)
    expected = np.load(root / "q_calibration_predictions.npz")
    assert np.allclose(raw, expected["raw_q"], rtol=0, atol=1e-6)
    assert np.allclose(calibrated, expected["calibrated_q"], rtol=0, atol=1e-6)
    report = {"round": args.round, "pass": True, "calibration_pools": args.count, "calibration_selection_rows": len(records), "raw_log_loss": log_loss(expected["target"], raw), "calibrated_log_loss": log_loss(expected["target"], calibrated), "cuda_recomputed": True, "outer_labels_read": False, "scientific_result": False}
    output = root / f"verification_round{args.round}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
