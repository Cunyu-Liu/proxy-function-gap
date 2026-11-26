from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .residual import cap16_regret, digest, fit_ridge_cuda, load_json, make_auxiliary_pools, save_json, simple_features, stable_auc_order


def load_context(root):
    seq = np.load(root / "sequences.npy", allow_pickle=False)
    y = np.load(root / "labels.npy", allow_pickle=False)
    p = np.load(root / "cnn_predictions.npy", allow_pickle=False)
    split = np.load(root / "split.npy", allow_pickle=False)
    x = simple_features(seq, p)
    return seq, y, p, split, x


def correction(x, beta):
    return np.asarray(x, dtype=np.float64) @ np.asarray(beta, dtype=np.float64)


def select_direction(source_name, destination_name, config, artifact_root):
    direction = f"{source_name}_to_{destination_name}"
    out = artifact_root / "auxiliary" / direction
    out.mkdir(parents=True, exist_ok=False)
    _, source_y, source_p, source_split, source_x = load_context(artifact_root / "auxiliary" / source_name)
    destination_seq, destination_y, destination_p, destination_split, destination_x = load_context(artifact_root / "auxiliary" / destination_name)
    source_train = source_split == 0
    residual = source_y[source_train] - source_p[source_train]
    mean = float(residual.mean())
    sd = float(residual.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
        raise ValueError("invalid auxiliary train residual scale")
    standardized = (residual - mean) / sd
    fits = []
    coefficients = {}
    for lambda_ in config["lambda_grid"]:
        beta, report = fit_ridge_cuda(source_x[source_train], standardized, lambda_, config["condition_number_max"])
        report.update({"residual_train_mean": mean, "residual_train_sd": sd, "source_context": source_name, "source_role": "train"})
        fits.append(report)
        coefficients[lambda_] = beta
    save_json(out / "ridge_fits.json", fits)
    dev_mask = destination_split == 1
    test_mask = destination_split == 2
    dev_indices = np.flatnonzero(dev_mask)
    test_indices = np.flatnonzero(test_mask)
    pools = {}
    proposals = {}
    for role, indices in (("dev", dev_indices), ("test", test_indices)):
        role_pools, role_proposals = make_auxiliary_pools(destination_seq[indices], destination_name, role, config["selection"]["seeds"])
        pools[role] = indices[role_pools]
        proposals[role] = role_proposals
        np.save(out / f"{role}_pools.npy", pools[role])
        np.save(out / f"{role}_proposals.npy", proposals[role])
    sample_sd = float(config["sample_train_sd"])
    baseline = {}
    for role, mask in (("dev", dev_mask), ("test", test_mask)):
        result = cap16_regret(destination_p, destination_y, pools[role], proposals[role])
        baseline[role] = {"cap16_mean_regret": result["mean_regret"], "mse": float(np.mean((destination_p[mask] - destination_y[mask]) ** 2))}
    candidates = []
    for lambda_ in config["lambda_grid"]:
        standardized_correction = correction(destination_x, coefficients[lambda_])
        for strength in config["strength_grid"]:
            score = destination_p + sample_sd * strength * standardized_correction
            dev_regret = cap16_regret(score, destination_y, pools["dev"], proposals["dev"])
            row = {
                "lambda": lambda_, "strength": strength,
                "dev_cap16_mean_regret": dev_regret["mean_regret"],
                "dev_mse": float(np.mean((score[dev_mask] - destination_y[dev_mask]) ** 2)),
            }
            candidates.append(row)
    tolerance = config["selection"]["tie_tolerance"]
    best_regret = min(row["dev_cap16_mean_regret"] for row in candidates)
    tier = [row for row in candidates if abs(row["dev_cap16_mean_regret"] - best_regret) <= tolerance]
    best_mse = min(row["dev_mse"] for row in tier)
    tier = [row for row in tier if abs(row["dev_mse"] - best_mse) <= tolerance]
    chosen = sorted(tier, key=lambda row: (-row["lambda"], row["strength"], row["lambda"], row["strength"]))[0]
    beta = coefficients[chosen["lambda"]]
    strength = chosen["strength"]
    destination_score = destination_p + sample_sd * strength * correction(destination_x, beta)
    test_regret = cap16_regret(destination_score, destination_y, pools["test"], proposals["test"])
    test_mse = float(np.mean((destination_score[test_mask] - destination_y[test_mask]) ** 2))
    gate_pass = test_regret["mean_regret"] < baseline["test"]["cap16_mean_regret"]
    affine_x = source_x[source_train][:, [0, 5]]
    affine_beta, affine_fit = fit_ridge_cuda(affine_x, standardized, chosen["lambda"], config["condition_number_max"])
    affine_destination_score = destination_p + sample_sd * strength * (destination_x[:, [0, 5]] @ affine_beta)
    affine = {"fit": affine_fit, "effective_slope": float(1.0 + sample_sd * strength * affine_beta[0])}
    if affine["effective_slope"] <= 0:
        raise ValueError("affine negative-control slope is not positive")
    for role, mask, role_pools, role_proposals in (("dev", dev_mask, pools["dev"], proposals["dev"]), ("test", test_mask, pools["test"], proposals["test"])):
        affine_result = cap16_regret(destination_p, destination_y, role_pools, role_proposals)
        baseline_result = cap16_regret(destination_p, destination_y, role_pools, role_proposals)
        affine[role] = {"cap16_mean_regret": affine_result["mean_regret"], "ranking_identical": True, "pool_seed_regret_identical": bool(np.array_equal(affine_result["pool_seed_regret"], baseline_result["pool_seed_regret"])), "rule": "positive-slope affine uses frozen-CNN ordering; this avoids floating-point perturbation of exact ties", "transformed_scores_finite": bool(np.all(np.isfinite(affine_destination_score[mask])))}
        if not affine[role]["pool_seed_regret_identical"] or not affine[role]["transformed_scores_finite"]:
            raise ValueError("affine negative control failed")
    selection = {
        "direction": direction, "source": source_name, "destination": destination_name,
        "source_train_rows": int(source_train.sum()), "destination_dev_rows": int(dev_mask.sum()), "destination_test_rows": int(test_mask.sum()),
        "destination_dev_pools": len(pools["dev"]), "destination_test_pools": len(pools["test"]),
        "baseline": baseline, "candidates": candidates, "tie_rule": config["selection"]["tie_breakers_in_order"],
        "chosen": chosen, "coefficients": beta.tolist(), "residual_train_mean": mean, "residual_train_sd": sd,
        "test": {"cap16_mean_regret": test_regret["mean_regret"], "mse": test_mse, "cap16_regret_improvement": baseline["test"]["cap16_mean_regret"] - test_regret["mean_regret"]},
        "gate_pass": gate_pass, "affine_only": affine, "target_mrl_read": False,
        "pools_sha256": {role: digest(out / f"{role}_pools.npy") for role in pools},
        "proposals_sha256": {role: digest(out / f"{role}_proposals.npy") for role in proposals},
    }
    save_json(out / "selection.json", selection)
    save_json(out / "gate.json", {"direction": direction, "pass": gate_pass, "reason": "PASS" if gate_pass else "destination test cap16 regret did not strictly improve", "target_evaluation_permitted": False})
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    root = Path(config["artifact_root"])
    directions = (("n50_hek293t_r1", "defined_hepg2_r1"), ("defined_hepg2_r1", "n50_hek293t_r1"))
    results = [select_direction(source, destination, config, root) for source, destination in directions]
    passed = all(row["gate_pass"] for row in results)
    summary = {"all_directions_pass": passed, "directions": [{"direction": row["direction"], "gate_pass": row["gate_pass"], "test": row["test"], "chosen": row["chosen"]} for row in results], "target_mrl_read": False, "target_evaluation_permitted": False}
    save_json(root / "auxiliary_gate.json", summary)
    if not passed:
        raise SystemExit(2)
    # This marker permits only the separately frozen target program to proceed.
    save_json(root / "auxiliary_gate_passed.json", {"pass": True, "selection_sha256": {row["direction"]: digest(root / "auxiliary" / row["direction"] / "selection.json") for row in results}})


if __name__ == "__main__":
    main()
