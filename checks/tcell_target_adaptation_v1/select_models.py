from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adaptation import cap_regret, digest, make_pools, make_proposals, run_name, save_json


def candidates(config, variant):
    result = []
    if variant in ("main", "from_scratch"):
        for learning_rate in config["model"]["learning_rate_grid"]:
            for lambda_ in config["model"]["lambda_grid"]:
                result.append((variant, 50000, learning_rate, lambda_))
    elif variant == "mse_only":
        for learning_rate in config["model"]["learning_rate_grid"]:
            result.append((variant, 50000, learning_rate, 0.0))
    else:
        raise ValueError(variant)
    return result


def dev_metric(root, config, variant, budget, learning_rate, lambda_):
    name = run_name(variant, budget, learning_rate, lambda_)
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
        values = np.load(run / "dev_predictions.npy", allow_pickle=False)
        predictions.append(values)
        history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        losses.append(next(row["dev"]["loss"] for row in history if row["epoch"] == summary["best_epoch"]))
        hashes.append(summary["best_model_hash"])
        epochs.append(summary["best_epoch"])
    sequences = np.load(root / "data" / "sequences.npy", allow_pickle=False)
    dev_indices = np.load(root / "data" / "dev_indices.npy", allow_pickle=False)
    labels = np.load(root / "data" / "train_dev_mean_labels.npy", allow_pickle=False)
    repeat_labels = {
        repeat: np.load(root / "data" / f"dev_{repeat}_labels.npy", allow_pickle=False)
        for repeat in ("defined_tcell_r1", "defined_tcell_r2")
    }
    pools_global = make_pools(dev_indices, sequences, "dev", 10000)
    complete = pools_global[(pools_global >= 0).all(axis=1)]
    lookup = np.full(len(sequences), -1, dtype=np.int64)
    lookup[dev_indices] = np.arange(len(dev_indices))
    pools = lookup[complete]
    proposals = make_proposals(len(pools), 64, config["selection"]["proposal_seeds"], "dev")
    ensemble = np.mean(predictions, axis=0)
    repeat_regrets = {}
    for repeat, repeat_label in repeat_labels.items():
        regret, _ = cap_regret(ensemble, repeat_label, pools, proposals)
        repeat_regrets[repeat] = float(regret.mean())
    mean_regret, _ = cap_regret(ensemble, labels[dev_indices], pools, proposals)
    return {"name": name, "variant": variant, "budget": budget, "learning_rate": learning_rate, "lambda": lambda_, "dev_cap16_regret": float(mean_regret.mean()), "dev_repeat_cap16_regret": repeat_regrets, "dev_cap16_regret_worst_repeat": float(max(repeat_regrets.values())), "dev_combined_loss_mean": float(np.mean(losses)), "complete_dev_pools": len(pools), "seed_best_epochs": epochs, "seed_model_hashes": hashes}


def ordered(candidates):
    return sorted(candidates, key=lambda row: (round(row["dev_cap16_regret_worst_repeat"] / 1e-12) * 1e-12, round(row["dev_cap16_regret"] / 1e-12) * 1e-12, row["dev_combined_loss_mean"], row["learning_rate"], row["lambda"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["grid", "nested"], required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_hash = digest(config_path)
    config = json.loads(config_path.read_text())
    amendment_path = config_path.parent / "dev_repeat_amendment_frozen.json"
    amendment_hash = digest(amendment_path)
    root = Path(config["artifact_root"])
    prepare = json.loads((root / "prepare_manifest.json").read_text())
    dev_repeat = json.loads((root / "DEV_REPEAT_SELECTION_FROZEN.json").read_text())
    if dev_repeat["amendment_sha256"] != amendment_hash or dev_repeat["test_mrl_read"]:
        raise RuntimeError("dev repeat amendment gate failed")
    if prepare["test_mrl_read"] or prepare["target_evaluation_count"] != 0:
        raise RuntimeError("test has already been touched")
    if args.stage == "grid":
        for variant in ("main", "mse_only", "from_scratch"):
            rows = [dev_metric(root, config, *job) for job in candidates(config, variant)]
            rows = ordered(rows)
            save_json(root / "selection" / f"{variant}.json", {"variant": variant, "selected": rows[0], "candidates": rows, "config_sha256": config_hash, "dev_repeat_amendment_sha256": amendment_hash, "selection_metric": "maximum_of_r1_and_r2_cap16_regret", "test_used": False, "selection_role": "target_train_dev_only"})
        print(json.dumps({variant: json.loads((root / "selection" / f"{variant}.json").read_text())["selected"] for variant in ("main", "mse_only", "from_scratch")}, indent=2))
    else:
        selected = json.loads((root / "selection" / "main.json").read_text())["selected"]
        rows = [dev_metric(root, config, "nested", budget, selected["learning_rate"], selected["lambda"]) for budget in (1000, 5000, 20000)]
        rows.append(dev_metric(root, config, "main", 50000, selected["learning_rate"], selected["lambda"]))
        save_json(root / "selection" / "nested_learning_curve_dev.json", {"selected_main_hyperparameters": {"learning_rate": selected["learning_rate"], "lambda": selected["lambda"]}, "budgets": rows, "config_sha256": config_hash, "dev_repeat_amendment_sha256": amendment_hash, "selection_metric": "maximum_of_r1_and_r2_cap16_regret", "test_used": False})
        for row in rows:
            if row["budget"] < 50000:
                save_json(root / "selection" / f"nested_{row['budget']}.json", {"selected": row, "config_sha256": config_hash, "dev_repeat_amendment_sha256": amendment_hash, "test_used": False})
        runs = []
        for summary_path in sorted((root / "training").glob("*/seed_*/summary.json")):
            summary = json.loads(summary_path.read_text())
            if summary["config_sha256"] != config_hash or not summary["converged_dev_plateau"] or summary["test_mrl_read"] or summary["test_inference"]:
                raise RuntimeError("final training audit failed")
            runs.append({"path": str(summary_path.relative_to(root)), "summary_sha256": digest(summary_path), "history_sha256": summary["history_sha256"], "best_checkpoint_sha256": summary["best_checkpoint_sha256"]})
        if len(runs) != 72:
            raise ValueError(f"expected 72 complete runs, found {len(runs)}")
        save_json(root / "TRAINING_FROZEN.json", {"status": "PASS", "config_sha256": config_hash, "dev_repeat_amendment_sha256": amendment_hash, "dev_repeat_selection_manifest_sha256": digest(root / "DEV_REPEAT_SELECTION_FROZEN.json"), "runs": runs, "run_count": len(runs), "selections": {name: digest(root / "selection" / f"{name}.json") for name in ("main", "mse_only", "from_scratch", "nested_learning_curve_dev")}, "three_seed_equal_weight": True, "train_repeat_equal_weight": "arithmetic mean labels gives each repeat 1/2 weight in MSE", "from_scratch_fairness": "same architecture, grid, seeds, data, plateau and dev metric", "test_mrl_read": False, "test_inference": False, "target_evaluation_count": 0, "test_now_permitted_once": True})
        print(json.dumps({"nested": rows, "training_runs": len(runs), "test_now_permitted_once": True}, indent=2))


if __name__ == "__main__":
    main()
