import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def random_seed(config, pool_id, budget, replicate):
    text = f"{config['evaluation']['random_seed']}:{pool_id}:{budget}:{replicate}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def safe_metrics(y, prediction):
    correlation = spearmanr(y, prediction).statistic if len(y) >= 2 else float("nan")
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(y - prediction))), "spearman": float(correlation)}


def percentile_interval(values):
    return {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}


def evaluate_method(name, variant_ids, pool_ids, orders, predictions, labels, config):
    rng_bootstrap = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
    unique_pools = sorted(set(pool_ids))
    pair_rows = []
    tail_rows = []
    seed_diagnostics = []
    predictions_2d = predictions if predictions.ndim == 2 else predictions[None, :]
    for budget in config["evaluation"]["budgets"]:
        for member in predictions_2d:
            member_deltas = []
            for pool_id in unique_pools:
                idx = np.flatnonzero(pool_ids == pool_id)
                ids = variant_ids[idx]
                scores = member[idx]
                y = labels[idx]
                order = np.lexsort((ids, -scores))
                selected = order[:budget]
                oracle = float(np.max(y))
                bon_regret = oracle - float(np.max(y[selected]))
                random_regrets = []
                for rep in range(config["evaluation"]["random_replicates_per_pool_budget"]):
                    rng = np.random.default_rng(random_seed(config, pool_id, budget, rep))
                    random_idx = rng.choice(len(idx), size=budget, replace=False)
                    random_regrets.append(oracle - float(np.max(y[random_idx])))
                member_deltas.append(bon_regret - float(np.mean(random_regrets)))
            seed_diagnostics.append({"budget": budget, "mean_bon_minus_random_regret": float(np.mean(member_deltas))})
        for pool_id in unique_pools:
            idx = np.flatnonzero(pool_ids == pool_id)
            ids = variant_ids[idx]
            scores = predictions[idx] if predictions.ndim == 1 else np.mean(predictions, axis=0)[idx]
            y = labels[idx]
            oracle = float(np.max(y))
            order = np.lexsort((ids, -scores))
            selected = order[:budget]
            bon_regret = oracle - float(np.max(y[selected]))
            random_regrets = []
            random_hits = []
            for rep in range(config["evaluation"]["random_replicates_per_pool_budget"]):
                rng = np.random.default_rng(random_seed(config, pool_id, budget, rep))
                random_idx = rng.choice(len(idx), size=budget, replace=False)
                random_regrets.append(oracle - float(np.max(y[random_idx])))
                random_hits.append(bool(np.any(y[random_idx] == oracle)))
            mean_random_regret = float(np.mean(random_regrets))
            pair_rows.append({"method": name, "pool_id": pool_id, "budget": budget, "bon_regret": bon_regret, "random_regret_mean": mean_random_regret, "bon_minus_random_regret": bon_regret - mean_random_regret, "bon_oracle_hit": bool(np.any(y[selected] == oracle)), "random_oracle_hit_rate": float(np.mean(random_hits)), "pool_oracle": oracle})
            for fraction in config["evaluation"]["top_tail_fractions"]:
                k = max(1, int(np.ceil(fraction * len(idx))))
                true_tail = set(np.argsort(-y, kind="stable")[:k].tolist())
                hits = sum(int(i) in true_tail for i in selected)
                expected_hits = budget * k / len(idx)
                tail_rows.append({"method": name, "pool_id": pool_id, "budget": budget, "fraction": fraction, "tail_size": k, "bon_hits": hits, "random_expected_hits": expected_hits, "bon_precision_in_selected": hits / budget, "bon_recall_of_tail": hits / k, "lift_over_random": hits / expected_hits})
    primary = []
    for budget in config["evaluation"]["budgets"]:
        rows = [r for r in pair_rows if r["budget"] == budget]
        bon = np.asarray([r["bon_regret"] for r in rows])
        random = np.asarray([r["random_regret_mean"] for r in rows])
        delta = bon - random
        indices = rng_bootstrap.integers(0, len(rows), size=(config["evaluation"]["bootstrap_pool_resamples"], len(rows)))
        primary.append({"method": name, "budget": budget, "pool_count": len(rows), "mean_bon_regret": float(bon.mean()), "mean_random_regret": float(random.mean()), "mean_bon_minus_random_regret": float(delta.mean()), "ci95_bon_minus_random_regret": percentile_interval(delta[indices].mean(axis=1)), "ci95_bon_regret": percentile_interval(bon[indices].mean(axis=1)), "ci95_random_regret": percentile_interval(random[indices].mean(axis=1)), "bon_oracle_hit_rate": float(np.mean([r["bon_oracle_hit"] for r in rows])), "random_oracle_hit_rate": float(np.mean([r["random_oracle_hit_rate"] for r in rows])), "pool_wins_bon_lower_regret": int(np.sum(delta < 0)), "pool_ties": int(np.sum(delta == 0)), "pool_losses_bon_higher_regret": int(np.sum(delta > 0))})
    point = safe_metrics(labels, predictions if predictions.ndim == 1 else np.mean(predictions, axis=0))
    by_order = []
    for order in sorted(set(orders.tolist())):
        idx = orders == order
        by_order.append({"order": int(order), **safe_metrics(labels[idx], (predictions if predictions.ndim == 1 else np.mean(predictions, axis=0))[idx])})
    return {"method": name, "point_metrics": point, "point_metrics_by_order": by_order, "primary": primary, "seed_diagnostics": seed_diagnostics}, pair_rows, tail_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = Path(config["paths"]["artifacts"])
    freeze_path = root / "predictions" / "PREDICTIONS_FROZEN.json"
    freeze = json.loads(freeze_path.read_text())
    assert freeze["status"] == "FROZEN_BEFORE_TEST_LABEL_EVALUATION"
    assert freeze["config_sha256"] == sha256_file(config_path)
    for relative, expected in freeze["code_sha256"].items():
        assert sha256_file(Path(config["paths"]["code"]) / relative) == expected, f"Code changed after freeze: {relative}"
    for filename, expected in freeze["prediction_files"].items():
        assert sha256_file(root / "predictions" / filename) == expected
    started = root / "results" / "EVALUATION_STARTED.json"
    assert not started.exists(), "Permanent test evaluation may only be started once"
    started.parent.mkdir(parents=True, exist_ok=True)
    started.write_text(json.dumps({"started_at_utc": datetime.now(timezone.utc).isoformat(), "freeze_manifest_sha256": sha256_file(freeze_path), "evaluation_count": 1}, indent=2) + "\n")
    label_rows = read_csv(root / "evaluator" / "high_order_labels.csv")
    label_map = {r["variant_id"]: float(r["DMS_score"]) for r in label_rows}
    results = {"schema_version": config["schema_version"], "study_status": config["study_status"], "primary_difference_definition": config["evaluation"]["primary_difference"], "methods": []}
    pair_rows = []
    tail_rows = []
    for method, filename in [("cnn_ensemble", "cnn_ensemble.npz"), ("additive", "additive.npz")]:
        data = np.load(root / "predictions" / filename)
        variant_ids = data["variant_id"]
        pool_ids = data["pool_id"]
        orders = data["order"]
        labels = np.asarray([label_map[str(v)] for v in variant_ids])
        method_result, method_pairs, method_tail = evaluate_method(method, variant_ids, pool_ids, orders, data["prediction"], labels, config)
        results["methods"].append(method_result)
        pair_rows.extend(method_pairs)
        tail_rows.extend(method_tail)
    seed_data = np.load(root / "predictions" / "cnn_seed_predictions.npz")
    seed_labels = np.asarray([label_map[str(v)] for v in seed_data["variant_id"]])
    seed_result, _, _ = evaluate_method("cnn_seed_members", seed_data["variant_id"], seed_data["pool_id"], seed_data["order"], seed_data["prediction"], seed_labels, config)
    results["cnn_seed_member_diagnostics"] = seed_result["seed_diagnostics"]
    results["interpretation_constraints"] = ["Negative BoN-minus-RandomN regret favors proxy selection; positive favors RandomN.", "Pool bootstrap intervals resample 16 fixed pools, not wet-lab replicates.", "Historical public measurements are retrospectively unblinded; this is not a new blind test.", "Results apply to the measured local high-order landscape and fixed pools, not the complete combinatorial space."]
    (root / "results" / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=True) + "\n")
    for filename, rows in [("paired_pool_results.json", pair_rows), ("top_tail_results.json", tail_rows)]:
        (root / "results" / filename).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    primary = {"study_status": config["study_status"], "primary_difference_definition": config["evaluation"]["primary_difference"], "methods": [{"method": m["method"], "point_metrics": m["point_metrics"], "point_metrics_by_order": m["point_metrics_by_order"], "primary": m["primary"]} for m in results["methods"]]}
    (root / "results" / "primary.json").write_text(json.dumps(primary, indent=2, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps(primary, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
