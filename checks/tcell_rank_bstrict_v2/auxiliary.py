from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rank import (
    DIRECTIONS, cap_regret, cluster_bootstrap, fixed_tail, fit_pairwise_newton,
    load_context, make_pools, pair_arrays, pairwise_loss_numpy, save_json,
    sha256_file, simple_features,
)


def role_pools(sequences, split, indices, context, role, seeds):
    local_pools, proposals = make_pools(sequences[indices], context, role, seeds)
    return indices[local_pools], proposals


def run_direction(source_name, destination_name, config):
    direction = f"{source_name}_to_{destination_name}"
    v1_root = Path(config["v1_input_root"])
    root = Path(config["artifact_root"]) / "auxiliary" / direction
    root.mkdir(parents=True, exist_ok=False)
    source_seq, source_y, source_p, source_split = load_context(v1_root / "auxiliary" / source_name)
    train_indices = np.flatnonzero(source_split == 0)
    dev_indices = np.flatnonzero(source_split == 1)
    source_x = simple_features(source_seq, source_p)
    mean = source_x[train_indices, :5].mean(axis=0)
    scale = source_x[train_indices, :5].std(axis=0, ddof=0)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise FloatingPointError("invalid source train feature scaling")
    source_z = (source_x[:, :5] - mean) / scale
    seeds = config["selection"]["seeds"]
    pools = {}
    proposals = {}
    for role, indices in (("train", train_indices), ("dev", dev_indices)):
        pools[role], proposals[role] = role_pools(source_seq, source_split, indices, source_name, role, seeds)
        np.save(root / f"source_{role}_pools.npy", pools[role])
        np.save(root / f"source_{role}_proposals.npy", proposals[role])
    train_tail = fixed_tail(source_p, pools["train"], config["pair_construction"]["tail_k"])
    dev_tail = fixed_tail(source_p, pools["dev"], config["pair_construction"]["tail_k"])
    np.save(root / "source_train_tail.npy", train_tail)
    np.save(root / "source_dev_tail.npy", dev_tail)
    train_pairs = pair_arrays(source_y, source_p, source_z, train_tail)
    dev_pairs = pair_arrays(source_y, source_p, source_z, dev_tail)
    save_json(root / "pair_manifest.json", {
        "train_pools": len(pools["train"]), "dev_pools": len(pools["dev"]),
        "train_tail_rows": len(train_tail), "dev_tail_rows": len(dev_tail),
        "train_nontied_pairs": len(train_pairs["labels"]), "dev_nontied_pairs": len(dev_pairs["labels"]),
        "train_weight_sum": float(train_pairs["weights"].sum()),
        "dev_weight_sum": float(dev_pairs["weights"].sum()),
        "tail_frozen_before_labels": True, "tail_k": 16,
        "feature_mean": mean.tolist(), "feature_population_sd": scale.tolist(),
        "train_pools_sha256": sha256_file(root / "source_train_pools.npy"),
        "train_tail_sha256": sha256_file(root / "source_train_tail.npy"),
    })
    fits = []
    coefficients = {}
    for lambda_ in config["lambda_grid"]:
        beta, report = fit_pairwise_newton(train_pairs, lambda_, config)
        report.update({"source": source_name, "role": "train", "direction": direction})
        fits.append(report)
        coefficients[lambda_] = beta
        save_json(root / "fits.json", fits)
        print(json.dumps(report), flush=True)
    baseline_dev, baseline_selected = cap_regret(source_p, source_y, pools["dev"], proposals["dev"])
    candidates = []
    for lambda_ in config["lambda_grid"]:
        correction = source_z @ coefficients[lambda_]
        for strength in config["strength_grid"]:
            score = source_p + config["sample_train_sd"] * strength * correction
            regret, selected = cap_regret(score, source_y, pools["dev"], proposals["dev"])
            candidates.append({
                "lambda": lambda_, "strength": strength,
                "source_dev_cap16_mean_regret": float(regret.mean()),
                "source_dev_pairwise_logistic_loss": pairwise_loss_numpy(dev_pairs, coefficients[lambda_], config["sample_train_sd"], strength, lambda_),
                "source_dev_selection_turnover": float(np.mean(selected != baseline_selected)),
            })
    tolerance = config["selection"]["tie_tolerance"]
    best = min(row["source_dev_cap16_mean_regret"] for row in candidates)
    tier = [row for row in candidates if abs(row["source_dev_cap16_mean_regret"] - best) <= tolerance]
    best_loss = min(row["source_dev_pairwise_logistic_loss"] for row in tier)
    tier = [row for row in tier if abs(row["source_dev_pairwise_logistic_loss"] - best_loss) <= tolerance]
    chosen = sorted(tier, key=lambda row: (-row["lambda"], row["strength"]))[0]
    beta = coefficients[chosen["lambda"]]
    full_beta = np.concatenate((beta, [0.0]))
    save_json(root / "selection.json", {
        "direction": direction, "source": source_name, "destination": destination_name,
        "selection_role": "source_inner_dev_only", "destination_dev_used": False,
        "baseline_source_dev_cap16_regret": float(baseline_dev.mean()),
        "candidates": candidates, "chosen": chosen,
        "coefficients_nonintercept": beta.tolist(),
        "coefficients_in_order_p_p2_gc_uAUG_polyU_intercept": full_beta.tolist(),
        "feature_mean": mean.tolist(), "feature_population_sd": scale.tolist(),
        "target_mrl_read": False,
    })

    destination_seq, destination_y_all, destination_p_all, destination_split = load_context(v1_root / "auxiliary" / destination_name)
    test_indices = np.flatnonzero(destination_split == 2)
    destination_y = destination_y_all[test_indices]
    destination_p = destination_p_all[test_indices]
    destination_x = simple_features(destination_seq[test_indices], destination_p)
    destination_z = (destination_x[:, :5] - mean) / scale
    expected_test_pools, expected_test_proposals = make_pools(destination_seq[test_indices], destination_name, "test", seeds)
    test_pools = np.load(v1_root / "auxiliary" / direction / "test_pools.npy", allow_pickle=False)
    test_proposals = np.load(v1_root / "auxiliary" / direction / "test_proposals.npy", allow_pickle=False)
    if not np.array_equal(test_pools, test_indices[expected_test_pools]) or not np.array_equal(test_proposals, expected_test_proposals):
        raise ValueError("destination test pools differ from frozen deterministic construction")
    local_test_pools = expected_test_pools
    frozen_regret, frozen_selected = cap_regret(destination_p, destination_y, local_test_pools, test_proposals)
    corrected_score = destination_p + config["sample_train_sd"] * chosen["strength"] * (destination_z @ beta)
    corrected_regret, corrected_selected = cap_regret(corrected_score, destination_y, local_test_pools, test_proposals)
    effect = cluster_bootstrap(corrected_regret - frozen_regret, config["auxiliary_gate"]["bootstrap_seed"], config["auxiliary_gate"]["bootstrap_replicates"])
    gate_pass = effect["effect"] < 0 and effect["ci_high"] < 0
    gate = {
        "direction": direction, "test_role": "destination_auxiliary_test_once",
        "test_evaluations": 1, "destination_dev_used": False,
        "frozen_cap16_mean_regret": float(frozen_regret.mean()),
        "corrected_cap16_mean_regret": float(corrected_regret.mean()),
        "corrected_minus_frozen": effect,
        "selection_turnover": float(np.mean(corrected_selected != frozen_selected)),
        "pass": bool(gate_pass),
        "reason": "PASS" if gate_pass else "cap16 point improvement and pool-cluster CI upper bound must both be strictly negative",
        "target_evaluation_permitted": False, "target_mrl_read": False,
    }
    save_json(root / "gate.json", gate)
    np.save(root / "test_frozen_pool_mean_regret.npy", frozen_regret.mean(axis=1))
    np.save(root / "test_corrected_pool_mean_regret.npy", corrected_regret.mean(axis=1))
    print(json.dumps(gate, indent=2), flush=True)
    return {"direction": direction, "gate_pass": bool(gate_pass), "chosen": chosen, "test": gate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if config["analysis_id"] != "tcell_rank_bstrict_v2" or config["selection"]["role"] != "source_inner_dev_only":
        raise ValueError("wrong frozen protocol")
    root = Path(config["artifact_root"])
    results = [run_direction(source, destination, config) for source, destination in DIRECTIONS]
    passed = all(row["gate_pass"] for row in results)
    save_json(root / "auxiliary_gate.json", {
        "analysis_id": config["analysis_id"], "config_sha256": sha256_file(config_path),
        "all_directions_pass": passed, "directions": results,
        "target_mrl_read": False, "target_evaluation_permitted": False,
    })
    if not passed:
        raise SystemExit(2)
    save_json(root / "auxiliary_gate_passed.json", {
        "pass": True, "config_sha256": sha256_file(config_path),
        "gate_sha256": {row["direction"]: sha256_file(root / "auxiliary" / row["direction"] / "gate.json") for row in results},
        "target_mrl_read": False, "target_evaluation_permitted": True,
    })


if __name__ == "__main__":
    main()
