#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row["mutant"] for row in rows], np.asarray([float(row["DMS_score"]) for row in rows])


def expected_max(values, budget):
    ordered = sorted(values)
    n = len(ordered)
    total = math.comb(n, budget)
    result = 0.0
    for index in range(budget - 1, n):
        below_or_equal = math.comb(index + 1, budget)
        below = math.comb(index, budget) if index >= budget else 0
        result += ordered[index] * (below_or_equal - below) / total
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    frozen = json.loads((Path(args.frozen) / "frozen_config.json").read_text())
    result = json.loads((Path(args.results) / "results.json").read_text())
    checks = {}
    convergence_files = sorted(Path(args.training).glob("lambda_*/seed_*/summary.json"))
    checks["all_35_training_runs_present"] = len(convergence_files) == len(config["seeds"]) * len(config["ridge_lambdas"])
    summaries = [json.loads(path.read_text()) for path in convergence_files]
    checks["all_cuda_runs_converged"] = all(summary["convergence"]["converged"] and summary["convergence"]["device"] == "cuda:0" and summary["convergence"]["relative_residual_l2"] <= config["cg_relative_tolerance"] for summary in summaries)
    checks["five_frozen_seed_models"] = set(frozen["models"]) == set(map(str, config["seeds"]))
    checks["frozen_model_hashes"] = all(file_hash(Path(args.frozen) / f"additive_ridge_seed_{seed}.npy") == frozen["models"][str(seed)] for seed in config["seeds"])
    checks["single_test_opening"] = (Path(args.results) / "test_opened.json").exists() and result["opening"]["single_opening"]
    with np.load(Path(args.predictions) / "test_predictions.npz") as archive:
        prediction_ids = list(map(str, archive["mutants"]))
        seeds = list(map(int, archive["seeds"]))
        scores = archive["scores"]
    mutants, labels = load_labels(Path(args.data) / "evaluator/test_labels.csv")
    pools = json.loads((Path(args.data) / "evaluator/test_pools.json").read_text())
    indices = {mutant: index for index, mutant in enumerate(mutants)}
    prediction_indices = {mutant: index for index, mutant in enumerate(prediction_ids)}
    checks["identifier_alignment"] = set(prediction_ids) == set(mutants) and seeds == config["seeds"]
    mismatches = []
    pool_deltas = {str(budget): [] for budget in config["budgets"]}
    for pool_index, pool in enumerate(pools):
        ids = pool["mutants"]
        idx = [indices[mutant] for mutant in ids]
        pool_labels = labels[idx]
        oracle = max(pool_labels)
        for seed_index, seed in enumerate(seeds):
            pidx = [prediction_indices[mutant] for mutant in ids]
            pool_scores = scores[seed_index, pidx]
            order = sorted(range(len(ids)), key=lambda i: (-float(pool_scores[i]), hashlib.sha256(f"gb1-bon-tie-v1:{seed}:{ids[i]}".encode()).hexdigest()))
            for budget in config["budgets"]:
                selected = order[:budget]
                best = max(float(pool_labels[i]) for i in selected)
                random_max = expected_max(pool_labels, budget)
                expected = {"bon_regret": oracle - best, "random_regret": oracle - random_max, "delta_regret": random_max - best}
                recorded = result["per_seed"][seed_index]["pools"][pool_index]["metrics"][str(budget)]
                for key, value in expected.items():
                    if not math.isclose(value, recorded[key], rel_tol=0.0, abs_tol=1e-10):
                        mismatches.append({"pool": pool["pool_id"], "seed": seed, "budget": budget, "metric": key, "independent": value, "recorded": recorded[key]})
                pool_deltas[str(budget)].append(random_max - best)
    checks["all_1100_pool_seed_budget_regrets_recomputed"] = not mismatches
    aggregate_errors = {}
    for budget in config["budgets"]:
        key = str(budget)
        independent = float(np.mean(pool_deltas[key]))
        recorded = result["budgets"][key]["delta_regret_bon_minus_random"]
        aggregate_errors[key] = independent - recorded
    checks["aggregate_regrets"] = all(abs(error) < 1e-12 for error in aggregate_errors.values())
    pool_ids = [mutant for pool in pools for mutant in pool["mutants"]]
    point_indices = [indices[mutant] for mutant in pool_ids]
    point_prediction_indices = [prediction_indices[mutant] for mutant in pool_ids]
    maes, rhos = [], []
    for seed_index in range(len(seeds)):
        predicted = scores[seed_index, point_prediction_indices]
        truth = labels[point_indices]
        maes.append(float(np.mean(np.abs(predicted - truth))))
        rhos.append(float(spearmanr(predicted, truth).statistic))
    checks["mae_recomputed"] = abs(float(np.mean(maes)) - result["point_metrics_mean_over_seeds"]["mae"]) < 1e-12
    checks["spearman_recomputed"] = abs(float(np.mean(rhos)) - result["point_metrics_mean_over_seeds"]["spearman"]) < 1e-12
    checks["lnw_semantics"] = result["label"] == "lnW" and "not Kd" in result["label_semantics"]
    checks["retrospective_semantics"] = result["retrospective_public_dms"] and not result["prospective_blinded_experiment"]
    report = {"round": 2, "scope": "independent post-freeze arithmetic without production imports", "pass": all(checks.values()), "checks": checks, "mismatches": mismatches[:20], "aggregate_errors": aggregate_errors, "n_training_summaries": len(summaries), "n_pool_seed_budget_checks": sum(len(config["budgets"]) * len(seeds) for _ in pools)}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
