#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from gb1_common import load_labeled_csv, parse_mutant, pool_metrics, regression_metrics, sha256_file, write_json


def feature_vocabulary(train_mutants):
    single_count = sum(len(parse_mutant(str(mutant))) == 1 for mutant in train_mutants)
    if single_count != 741:
        raise RuntimeError(f"Expected 741 train single substitutions, found {single_count}")
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    vocabulary = [(wild, 0, replacement) for wild in alphabet for replacement in alphabet if replacement != wild]
    return {token: i + 1 for i, token in enumerate(vocabulary)}


def make_features(mutants, vocabulary):
    features = np.zeros((len(mutants), len(vocabulary) + 1), dtype=np.float64)
    features[:, 0] = 1.0
    for row, mutant in enumerate(mutants):
        for wild, _, replacement in parse_mutant(str(mutant)):
            token = (wild, 0, replacement)
            if token not in vocabulary:
                raise RuntimeError(f"Feature absent from substitution vocabulary: {token}")
            features[row, vocabulary[token]] += 1.0
    return features


def fit_pcg(gram, rhs, regularizer, y_second_moment, seed, relative_tolerance, max_iterations):
    generator = torch.Generator(device=gram.device)
    generator.manual_seed(seed)
    weights = 0.01 * torch.randn(gram.shape[0], generator=generator, dtype=gram.dtype, device=gram.device)
    matrix = gram + regularizer

    def matvec(vector):
        return matrix @ vector

    residual = rhs - matvec(weights)
    residual_norm = torch.linalg.vector_norm(residual)
    target = relative_tolerance * max(1.0, float(torch.linalg.vector_norm(rhs)))
    diagonal = torch.diagonal(matrix)
    if bool((diagonal <= 0).any()):
        raise RuntimeError("Nonpositive PCG preconditioner")
    z = residual / diagonal
    direction = z.clone()
    rz = torch.dot(residual, z)
    history = []
    converged = bool(residual_norm <= target)
    iteration = 0
    start = time.time()
    while not converged and iteration < max_iterations:
        iteration += 1
        product = matvec(direction)
        denominator = torch.dot(direction, product)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            raise RuntimeError("PCG lost positive definiteness")
        alpha = rz / denominator
        weights = weights + alpha * direction
        residual = residual - alpha * product
        residual_norm = torch.linalg.vector_norm(residual)
        objective = 0.5 * torch.dot(weights, matvec(weights)) - torch.dot(rhs, weights) + 0.5 * y_second_moment
        record = {"iteration": iteration, "relative_residual_l2": float(residual_norm / max(1.0, float(torch.linalg.vector_norm(rhs)))), "objective": float(objective)}
        if iteration == 1 or iteration % 10 == 0 or residual_norm <= target:
            history.append(record)
        converged = bool(residual_norm <= target)
        if not converged:
            z = residual / diagonal
            rz_new = torch.dot(residual, z)
            direction = z + (rz_new / rz) * direction
            rz = rz_new
    gradient_inf = float(torch.linalg.vector_norm(rhs - matvec(weights), ord=float("inf")))
    if not converged or not math.isfinite(gradient_inf):
        raise RuntimeError(f"PCG failed to converge: seed={seed}, iterations={iteration}, residual={residual_norm}")
    return weights, {"seed": seed, "device": str(gram.device), "dtype": str(gram.dtype), "converged": converged, "iterations": iteration, "relative_tolerance": relative_tolerance, "relative_residual_l2": float(residual_norm / max(1.0, float(torch.linalg.vector_norm(rhs)))), "gradient_inf": gradient_inf, "elapsed_seconds": time.time() - start, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    data = Path(args.data)
    output = Path(args.output)
    frozen = Path(args.frozen)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Training output is not empty")
    if frozen.exists() and any(frozen.iterdir()):
        raise RuntimeError("Frozen output is not empty")
    output.mkdir(parents=True, exist_ok=True)
    frozen.mkdir(parents=True, exist_ok=True)
    train_mutants, train_y = load_labeled_csv(data / "roles/train.csv")
    dev_mutants, dev_y = load_labeled_csv(data / "roles/dev.csv")
    vocabulary = feature_vocabulary(train_mutants)
    train_x_np = make_features(train_mutants, vocabulary)
    dev_x_np = make_features(dev_mutants, vocabulary)
    train_x = torch.from_numpy(train_x_np).cuda()
    train_y_torch = torch.from_numpy(train_y).cuda()
    dev_x = torch.from_numpy(dev_x_np).cuda()
    n = len(train_y)
    gram = (train_x.T @ train_x) / n
    rhs = (train_x.T @ train_y_torch) / n
    y_second_moment = torch.dot(train_y_torch, train_y_torch) / n
    dev_pools = json.loads((data / "pools/dev_pools.json").read_text())
    dev_label_map = {str(mutant): float(label) for mutant, label in zip(dev_mutants, dev_y)}
    dev_prediction_map = {str(mutant): i for i, mutant in enumerate(dev_mutants)}
    summaries = []
    for regularization in config["ridge_lambdas"]:
        diagonal = torch.zeros(gram.shape[0], dtype=gram.dtype, device=gram.device)
        diagonal[1:] = regularization
        regularizer = torch.diag(diagonal)
        seed_results = []
        for seed in config["seeds"]:
            weights, convergence = fit_pcg(gram, rhs, regularizer, y_second_moment, seed, config["cg_relative_tolerance"], config["cg_max_iterations"])
            run_dir = output / f"lambda_{regularization:g}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            model_path = run_dir / "weights.npy"
            np.save(model_path, weights.cpu().numpy())
            dev_predictions = (dev_x @ weights).cpu().numpy()
            metrics = regression_metrics(dev_y, dev_predictions, config["lnw_floor"])
            per_pool = []
            for pool in dev_pools:
                ids = pool["mutants"]
                labels = np.asarray([dev_label_map[mutant] for mutant in ids])
                pool_predictions = np.asarray([dev_predictions[dev_prediction_map[mutant]] for mutant in ids])
                per_pool.append({"pool_id": pool["pool_id"], "metrics": pool_metrics(ids, labels, pool_predictions, seed, config["budgets"], config["lnw_floor"])})
            train_objective = float(0.5 * torch.dot(weights, (gram + regularizer) @ weights) - torch.dot(rhs, weights) + 0.5 * y_second_moment)
            result = {"lambda": regularization, "seed": seed, "convergence": convergence, "train_objective": train_objective, "dev_metrics": metrics, "dev_pools": per_pool, "model_sha256": sha256_file(model_path)}
            write_json(run_dir / "summary.json", result)
            seed_results.append(result)
        primary = str(config["primary_budget"])
        summary = {
            "lambda": regularization,
            "all_seeds_converged": all(result["convergence"]["converged"] for result in seed_results),
            "mean_dev_cap16_delta_regret": float(np.mean([pool["metrics"][primary]["delta_regret"] for result in seed_results for pool in result["dev_pools"]])),
            "mean_dev_cap16_bon_regret": float(np.mean([pool["metrics"][primary]["bon_regret"] for result in seed_results for pool in result["dev_pools"]])),
            "mean_dev_spearman": float(np.mean([result["dev_metrics"]["spearman"] for result in seed_results])),
            "max_iterations": max(result["convergence"]["iterations"] for result in seed_results),
            "max_relative_residual": max(result["convergence"]["relative_residual_l2"] for result in seed_results),
        }
        summaries.append(summary)
        write_json(output / f"lambda_{regularization:g}" / "aggregate.json", summary)
    best = summaries[0]
    for candidate in summaries[1:]:
        difference = candidate["mean_dev_cap16_delta_regret"] - best["mean_dev_cap16_delta_regret"]
        if difference < -1e-12 or (abs(difference) <= 1e-12 and (candidate["mean_dev_spearman"], -candidate["lambda"]) > (best["mean_dev_spearman"], -best["lambda"])):
            best = candidate
    selected_lambda = best["lambda"]
    for seed in config["seeds"]:
        source = output / f"lambda_{selected_lambda:g}" / f"seed_{seed}" / "weights.npy"
        shutil.copyfile(source, frozen / f"additive_ridge_seed_{seed}.npy")
    code_hashes = {path.name: sha256_file(path) for path in Path(__file__).parent.glob("*.py")}
    frozen_config = {
        "schema_version": config["schema_version"],
        "config_sha256": sha256_file(config_path),
        "train_sha256": sha256_file(data / "roles/train.csv"),
        "dev_sha256": sha256_file(data / "roles/dev.csv"),
        "test_unlabeled_sha256": sha256_file(data / "roles/test_unlabeled.csv"),
        "model": "additive directed-substitution-type one-hot ridge; intercept unregularized; float64 CUDA preconditioned conjugate gradient",
        "feature_vocabulary": ["intercept"] + [f"{wild}{position}{replacement}" for wild, position, replacement in sorted(vocabulary, key=vocabulary.get)],
        "selected_lambda": selected_lambda,
        "selection_rule": config["selection_rule"],
        "selection_summary": summaries,
        "seeds": config["seeds"],
        "budgets": config["budgets"],
        "primary_budget": config["primary_budget"],
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "code_sha256": code_hashes,
        "models": {str(seed): sha256_file(frozen / f"additive_ridge_seed_{seed}.npy") for seed in config["seeds"]},
        "test_labels_used": False,
    }
    write_json(frozen / "frozen_config.json", frozen_config)
    print(json.dumps({"selected_lambda": selected_lambda, "selection_summary": summaries, "device": frozen_config["device"]}, indent=2))


if __name__ == "__main__":
    main()
