#!/usr/bin/env python3
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from gb1_common import load_labeled_csv, load_unlabeled_csv, pool_metrics, regression_metrics, sha256_file, write_json


def mean(values):
    return float(np.mean(values))


def mean_optional(values):
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def aggregate_pool_metrics(per_seed, budgets):
    output = {}
    for budget in budgets:
        key = str(budget)
        rows = [pool["metrics"][key] for seed in per_seed for pool in seed["pools"]]
        output[key] = {
            "budget": budget,
            "bon_regret": mean([row["bon_regret"] for row in rows]),
            "random_regret": mean([row["random_regret"] for row in rows]),
            "delta_regret_bon_minus_random": mean([row["delta_regret"] for row in rows]),
            "hit_at_budget": mean([row["hit_at_budget"] for row in rows]),
            "selected_mean_lnw": mean([row["selected_mean_lnw"] for row in rows]),
            "top16_recall": mean([row["top16_recall"] for row in rows]),
            "top_decile_recall": mean([row["top_decile_recall"] for row in rows]),
            "bon_selected_censored_fraction": mean([row["bon_selected_censored_fraction"] for row in rows]),
            "oracle_lnw": mean([row["oracle_lnw"] for row in rows]),
        }
    return output


def evaluate(mutants, labels, seeds, scores, pools, config):
    if len(set(map(str, mutants))) != len(mutants):
        raise RuntimeError("Duplicate evaluation identifiers")
    indices = {str(mutant): i for i, mutant in enumerate(mutants)}
    if scores.shape != (len(seeds), len(mutants)) or not np.isfinite(scores).all():
        raise RuntimeError("Prediction shape or fineness failure")
    covered = [mutant for pool in pools for mutant in pool["mutants"]]
    if len(covered) != len(set(covered)) or not set(covered).issubset(indices):
        raise RuntimeError("Pool coverage failure")
    pool_mutants = np.asarray(covered)
    pool_indices = np.asarray([indices[mutant] for mutant in covered])
    pool_labels = labels[pool_indices]
    per_seed = []
    for seed_index, seed in enumerate(seeds):
        predictions = scores[seed_index]
        metrics = regression_metrics(pool_labels, predictions[pool_indices], config["lnw_floor"])
        seed_pools = []
        for pool in pools:
            ids = pool["mutants"]
            idx = np.asarray([indices[mutant] for mutant in ids])
            seed_pools.append({"pool_id": pool["pool_id"], "position_pair": pool["position_pair"], "metrics": pool_metrics(ids, labels[idx], predictions[idx], int(seed), config["budgets"], config["lnw_floor"])})
        per_seed.append({"seed": int(seed), "point_metrics": metrics, "pools": seed_pools})
    budgets = aggregate_pool_metrics(per_seed, config["budgets"])
    cluster_deltas = {}
    bootstrap = {}
    rng = np.random.default_rng(config["bootstrap_seed"])
    for budget in config["budgets"]:
        key = str(budget)
        per_pool = np.asarray([[pool["metrics"][key]["delta_regret"] for pool in seed["pools"]] for seed in per_seed])
        cluster_values = per_pool.mean(axis=0)
        cluster_deltas[key] = cluster_values.tolist()
        draws = rng.integers(0, len(cluster_values), size=(config["bootstrap_replicates"], len(cluster_values)))
        distribution = cluster_values[draws].mean(axis=1)
        bootstrap[key] = {"cluster_unit": "position-pair pool", "replicates": config["bootstrap_replicates"], "method": "percentile bootstrap over seed-averaged pools", "ci95": [float(np.percentile(distribution, 2.5)), float(np.percentile(distribution, 97.5))]}
    primary = str(config["primary_budget"])
    primary_pool_values = np.asarray(cluster_deltas[primary])
    result = {
        "schema_version": config["schema_version"],
        "assay": config["assay"],
        "label": "lnW",
        "label_semantics": config["label_semantics"],
        "retrospective_public_dms": True,
        "prospective_blinded_experiment": False,
        "n_pools": len(pools),
        "pool_size": config["pool_size"],
        "n_unique_pool_variants": len(covered),
        "seeds": list(map(int, seeds)),
        "budgets": budgets,
        "primary_budget": config["primary_budget"],
        "primary_delta_regret": budgets[primary]["delta_regret_bon_minus_random"],
        "primary_ci95": bootstrap[primary]["ci95"],
        "bootstrap": bootstrap,
        "point_metrics_mean_over_seeds": {
            "mae": mean([seed["point_metrics"]["mae"] for seed in per_seed]),
            "spearman": mean([seed["point_metrics"]["spearman"] for seed in per_seed]),
            "censored_fraction": mean([seed["point_metrics"]["censored_fraction"] for seed in per_seed]),
            "censored_mae": mean([seed["point_metrics"]["censored"]["mae"] for seed in per_seed]),
            "uncensored_mae": mean([seed["point_metrics"]["uncensored"]["mae"] for seed in per_seed]),
            "censored_spearman": mean_optional([seed["point_metrics"]["censored"]["spearman"] for seed in per_seed]),
            "uncensored_spearman": mean_optional([seed["point_metrics"]["uncensored"]["spearman"] for seed in per_seed]),
        },
        "single_pool_diagnostics": {
            "max_absolute_pool_delta": float(np.max(np.abs(primary_pool_values))),
            "max_absolute_share_of_total_absolute_delta": float(np.max(np.abs(primary_pool_values)) / np.sum(np.abs(primary_pool_values))),
            "pools_with_negative_delta": int(np.sum(primary_pool_values < 0)),
            "pools_with_positive_delta": int(np.sum(primary_pool_values > 0)),
        },
        "per_seed_point_metrics": [{"seed": seed["seed"], **seed["point_metrics"]} for seed in per_seed],
        "per_seed_budgets": [{"seed": seed["seed"], "budgets": aggregate_pool_metrics([seed], config["budgets"])} for seed in per_seed],
        "per_seed": per_seed,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run-dev", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    data, output = Path(args.data), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Evaluation output is not empty")
    output.mkdir(parents=True, exist_ok=True)
    if args.dry_run_dev:
        mutants, labels = load_labeled_csv(data / "roles/dev.csv")
        pools = json.loads((data / "pools/dev_pools.json").read_text())
        seeds = np.asarray(config["seeds"])
        scores = np.asarray([[int(hashlib.sha256(f"dry-run:{seed}:{mutant}".encode()).hexdigest()[:16], 16) / 2**64 for mutant in mutants] for seed in seeds])
        result = evaluate(mutants, labels, seeds, scores, pools, config)
        result["dry_run_dev_only"] = True
        write_json(output / "results.json", result)
        print(json.dumps({"dry_run_dev_only": True, "primary_delta_regret": result["primary_delta_regret"]}, indent=2))
        return
    sentinel = output / "test_opened.json"
    if sentinel.exists():
        raise RuntimeError("Permanent test was already opened")
    prediction_path = Path(args.predictions) / "test_predictions.npz"
    prediction_manifest = json.loads((Path(args.predictions) / "manifest.json").read_text())
    if prediction_manifest["prediction_sha256"] != sha256_file(prediction_path):
        raise RuntimeError("Prediction hash mismatch")
    opening = {"opened_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "config_sha256": sha256_file(args.config), "prediction_sha256": sha256_file(prediction_path), "single_opening": True}
    write_json(sentinel, opening)
    mutants, labels = load_labeled_csv(data / "evaluator/test_labels.csv")
    unlabeled = load_unlabeled_csv(data / "roles/test_unlabeled.csv")
    if set(map(str, mutants)) != set(map(str, unlabeled)):
        raise RuntimeError("Evaluator vault and unlabeled test IDs differ")
    pools = json.loads((data / "evaluator/test_pools.json").read_text())
    with np.load(prediction_path) as archive:
        seeds = archive["seeds"]
        scores = archive["scores"]
        prediction_mutants = archive["mutants"]
    if set(map(str, prediction_mutants)) != set(map(str, mutants)):
        raise RuntimeError("Prediction IDs and evaluator IDs differ")
    prediction_order = {str(mutant): i for i, mutant in enumerate(prediction_mutants)}
    scores = scores[:, [prediction_order[str(mutant)] for mutant in mutants]]
    result = evaluate(mutants, labels, seeds, scores, pools, config)
    result["opening"] = opening
    result["prediction_manifest"] = prediction_manifest
    write_json(output / "results.json", result)
    compact = {key: result[key] for key in ("label", "n_pools", "pool_size", "budgets", "primary_delta_regret", "primary_ci95", "point_metrics_mean_over_seeds", "single_pool_diagnostics", "per_seed_budgets", "per_seed_point_metrics")}
    write_json(output / "aggregate.json", compact)
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
