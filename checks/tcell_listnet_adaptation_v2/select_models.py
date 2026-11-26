from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adaptation import cap_regret, digest, run_name, save_json


def candidates(config):
    return [
        (learning_rate, temperature, mse_weight)
        for learning_rate in config["model"]["learning_rate_grid"]
        for temperature in config["model"]["temperature_grid"]
        for mse_weight in config["model"]["mse_weight_grid"]
    ]


def dev_metric(root, config, learning_rate, temperature, mse_weight):
    name = run_name(50000, learning_rate, temperature, mse_weight)
    predictions = []
    losses = []
    hashes = []
    epochs = []
    for seed in config["model"]["seeds"]:
        run = root / "training" / name / f"seed_{seed}"
        summary = json.loads((run / "summary.json").read_text())
        if summary["status"] != "completed" or not summary["converged_dev_plateau"] or summary["best_epoch"] < 1 or not summary["gpu"]["actual_cuda"] or summary["gpu"]["cpu_fallback"] or summary["scientific_max_epoch"] is not None:
            raise RuntimeError(f"training gate failed: {name} seed {seed}")
        if digest(run / "best.pt") != summary["best_checkpoint_sha256"] or digest(run / "history.jsonl") != summary["history_sha256"] or digest(run / "dev_predictions.npy") != summary["dev_predictions_sha256"]:
            raise ValueError("training hash mismatch")
        predictions.append(np.load(run / "dev_predictions.npy", allow_pickle=False))
        history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        losses.append(next(row["dev"]["loss"] for row in history if row["epoch"] == summary["best_epoch"]))
        hashes.append(summary["best_model_hash"])
        epochs.append(summary["best_epoch"])
    dev_indices = np.load(root / "data" / "dev_indices.npy", allow_pickle=False)
    labels = np.load(root / "data" / "train_dev_mean_labels.npy", allow_pickle=False)
    pools_global = np.load(root / "data" / "dev_pools.npy", allow_pickle=False)
    complete = pools_global[(pools_global >= 0).all(axis=1)]
    lookup = np.full(len(labels), -1, dtype=np.int64)
    lookup[dev_indices] = np.arange(len(dev_indices))
    pools = lookup[complete]
    proposals = np.load(root / "data" / "dev_proposals.npy", allow_pickle=False)
    if len(proposals) != len(pools):
        raise ValueError("dev proposal/pool mismatch")
    ensemble = np.mean(predictions, axis=0)
    mean_regret, _ = cap_regret(ensemble, labels[dev_indices], pools, proposals)
    return {
        "name": name, "variant": "listnet", "budget": 50000,
        "learning_rate": learning_rate, "temperature": temperature, "mse_weight": mse_weight,
        "dev_cap16_regret_mean_replicate": float(mean_regret.mean()),
        "dev_combined_loss_mean": float(np.mean(losses)),
        "complete_dev_pools": len(pools), "seed_best_epochs": epochs,
        "seed_model_hashes": hashes,
    }


def ordered(rows):
    return sorted(rows, key=lambda row: (
        round(row["dev_cap16_regret_mean_replicate"] / 1e-12) * 1e-12,
        row["dev_combined_loss_mean"], row["learning_rate"],
        row["temperature"], row["mse_weight"],
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_hash = digest(config_path)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    prepare = json.loads((root / "prepare_manifest.json").read_text())
    if prepare["test_v2_mrl_read"] or prepare["target_evaluation_count"] != 0:
        raise RuntimeError("test-v2 has already been touched")
    rows = ordered([dev_metric(root, config, *job) for job in candidates(config)])
    save_json(root / "selection" / "listnet.json", {
        "variant": "listnet", "selected": rows[0], "candidates": rows,
        "config_sha256": config_hash,
        "selection_metric": "minimum_mean_replicate_cap16_regret",
        "tie_rule": "1e-12 regret tolerance, then mean dev combined loss, lower LR, lower temperature, lower MSE weight",
        "test_used": False, "selection_role": "v1_target_train_dev_only",
    })
    runs = []
    for summary_path in sorted((root / "training").glob("*/seed_*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary["config_sha256"] != config_hash or not summary["converged_dev_plateau"] or summary["test_v2_mrl_read"] or summary["test_inference"]:
            raise RuntimeError("final training audit failed")
        runs.append({
            "path": str(summary_path.relative_to(root)), "summary_sha256": digest(summary_path),
            "history_sha256": summary["history_sha256"],
            "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
        })
    expected = len(config["model"]["learning_rate_grid"]) * len(config["model"]["temperature_grid"]) * len(config["model"]["mse_weight_grid"]) * len(config["model"]["seeds"])
    if len(runs) != expected:
        raise ValueError(f"expected {expected} complete runs, found {len(runs)}")
    save_json(root / "TRAINING_FROZEN.json", {
        "status": "PASS", "config_sha256": config_hash, "runs": runs,
        "run_count": len(runs), "selection_sha256": digest(root / "selection" / "listnet.json"),
        "three_seed_equal_weight": True,
        "train_repeat_equal_weight": "v1 frozen arithmetic mean labels; ListNet target is the same mean-replicate MRL",
        "test_v2_mrl_read": False, "test_mrl_read": False,
        "test_inference": False, "target_evaluation_count": 0,
        "test_prediction_freeze_now_permitted": True,
    })
    print(json.dumps({"selected": rows[0], "candidate_count": len(rows), "training_runs": len(runs), "test_v2_mrl_read": False}, indent=2))


if __name__ == "__main__":
    main()
