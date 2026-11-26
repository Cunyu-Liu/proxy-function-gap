from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def regret(scores, labels, pools, proposals):
    prefix = proposals[:, :, :16]
    proposed_scores = np.take_along_axis(scores[pools][:, None, :], prefix, axis=2)
    proposed_labels = np.take_along_axis(labels[pools][:, None, :], prefix, axis=2)
    chosen = np.argmax(proposed_scores, axis=2)
    selected = np.take_along_axis(proposed_labels, chosen[..., None], axis=2)[..., 0]
    return labels[pools].max(axis=1)[:, None] - selected


def bootstrap(values, seed):
    clusters = np.asarray(values, dtype=np.float64).mean(axis=1)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = clusters[rng.integers(0, len(clusters), size=(10000, len(clusters)))].mean(axis=1)
    return float(clusters.mean()), *map(float, np.percentile(draws, [2.5, 97.5]))


def proposals_independent(pools, context, role, seeds):
    import random
    result = np.empty((len(pools), len(seeds), 16), dtype=np.int64)
    for pool in range(len(pools)):
        for seed_index, seed in enumerate(seeds):
            rng = random.Random(f"{context}|{role}|{pool}|{seed}")
            result[pool, seed_index] = [int(rng.random() * 64) for _ in range(16)]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["artifact_root"])
    maximum_difference = 0.0
    comparisons = 0
    training_runs = 0
    for run in sorted((root / "training").glob("*/seed_*")):
        if not run.is_dir():
            continue
        summary = json.loads((run / "summary.json").read_text())
        history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        assert len(history) == summary["total_epochs"] == summary["history_records"]
        assert all(np.isfinite([row["train"]["loss"], row["dev"]["loss"], row["train"]["mse"], row["dev"]["mse"]]).all() for row in history)
        best = min(history, key=lambda row: row["dev"]["loss"])
        assert best["epoch"] == summary["best_epoch"]
        assert summary["converged_dev_plateau"] and summary["stop_reason"] and summary["gpu"]["actual_cuda"] and not summary["gpu"]["cpu_fallback"]
        assert summary["scientific_max_epoch"] is None
        training_runs += 1
    assert training_runs == 21
    for variant in ("main", "original_mse_tail_rank"):
        saved = json.loads((root / "selection" / f"{variant}.json").read_text())
        for candidate in saved["candidates"]:
            name = candidate["name"]
            seed_predictions = [np.load(root / "training" / name / f"seed_{seed}" / "dev_predictions.npy", allow_pickle=False) for seed in config["model"]["seeds"]]
            ensemble = np.mean(seed_predictions, axis=0)
            prefix = "original" if variant == "original_mse_tail_rank" else "extended"
            with np.load(root / "data" / f"{prefix}_dev.npz", allow_pickle=False) as pack:
                labels = pack["labels"]
                pools = pack["selection_pools"]
                contexts = pack["selection_pool_contexts"]
                names = pack["context_names"].tolist()
            values = []
            summary = json.loads((root / "training" / name / f"seed_{config['model']['seeds'][0]}" / "summary.json").read_text())
            for context_id, context in enumerate(names):
                selected = contexts == context_id
                context_pools = pools[selected]
                proposals = proposals_independent(context_pools, context, "dev", config["auxiliary"]["proposal_seeds"])
                value = float(regret(ensemble, labels, context_pools, proposals).mean() / summary["scalers"][str(context_id)]["scale"])
                values.append(value)
                maximum_difference = max(maximum_difference, abs(value - candidate["sample_dev"][context]["normalized_cap16_regret"]))
                comparisons += 1
            for source in config["auxiliary"]["contexts"]:
                base = root / "auxiliary" / source / "dev"
                predictions = np.load(root / "selection" / name / f"{source}_dev_predictions.npy", allow_pickle=False)
                value = float(regret(predictions, np.load(base / "labels.npy"), np.load(base / "pools.npy"), np.load(base / "proposals.npy")).mean() / np.load(root / "auxiliary" / source / "train" / "labels.npy").std(ddof=0))
                values.append(value)
                maximum_difference = max(maximum_difference, abs(value - candidate["source_inner_dev"][source]["normalized_cap16_regret"]))
                comparisons += 1
            metric = float(np.mean(values))
            maximum_difference = max(maximum_difference, abs(metric - candidate["selection_metric"]))
            comparisons += 1
        recomputed = sorted(saved["candidates"], key=lambda row: (round(row["selection_metric"] / 1e-12) * 1e-12, row["dev_combined_loss_mean"], row["lambda"] if row["lambda"] is not None else 0.0))[0]
        assert recomputed["name"] == saved["selected"]["name"]
    gate = json.loads((root / "auxiliary_gate.json").read_text())
    main_name = gate["selected_main"]["name"]
    for direction, saved in gate["directions"].items():
        destination = saved["destination"]
        base = root / "auxiliary" / destination / "test"
        predictions = np.load(root / "auxiliary" / f"{destination}_test_main_predictions.npy", allow_pickle=False)
        main_regret = regret(predictions, np.load(base / "labels.npy"), np.load(base / "pools.npy"), np.load(base / "proposals.npy"))
        baseline_regret = regret(np.load(base / "baseline.npy"), np.load(base / "labels.npy"), np.load(base / "pools.npy"), np.load(base / "proposals.npy"))
        effect, low, high = bootstrap(main_regret - baseline_regret, config["auxiliary"]["gate"]["bootstrap_seed"])
        for actual, expected in zip((effect, low, high), (saved["effect"]["effect"], saved["effect"]["ci_low"], saved["effect"]["ci_high"])):
            maximum_difference = max(maximum_difference, abs(actual - expected))
            comparisons += 1
        assert (effect < 0) == saved["point_pass"] and (high < 0) == saved["ci_pass"]
    passed = all(row["point_pass"] and row["ci_pass"] for row in gate["directions"].values())
    assert passed == gate["both_directions_pass"]
    if not passed:
        assert not (root / "target").exists()
    report = {"status": "PASS", "round": 2, "independent_import_of_training_modules": False, "training_runs": training_runs, "comparisons": comparisons, "maximum_absolute_difference": maximum_difference, "selection_replay": True, "destination_bootstrap_replay": True, "dual_gate_pass": passed, "target_directory_exists": (root / "target").exists(), "target_evaluation_count": 0}
    (root / "verification").mkdir(exist_ok=True)
    (root / "verification" / "round2_independent.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
