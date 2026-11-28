import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def rank_average(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def independent_spearman(left, right):
    a = rank_average(left)
    b = rank_average(right)
    return float(np.corrcoef(a, b)[0, 1])


def close(actual, expected, tolerance=1e-10):
    if isinstance(expected, float) and math.isnan(expected):
        return math.isnan(actual)
    return abs(actual - expected) <= tolerance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["paths"]["artifacts"])
    assert (root / "results" / "EVALUATION_STARTED.json").exists()
    raw = root / "raw" / "GFP_AEQVI_Sarkisyan_2016.csv"
    assert sha256_file(raw) == config["source"]["expected_csv_sha256"]
    raw_rows = read_csv(raw)
    assert len(raw_rows) == 51714
    raw_counts = Counter(len(r["mutant"].split(":")) for r in raw_rows)
    high = {r["mutant"] for r in raw_rows if len(r["mutant"].split(":")) >= 5}
    low = {r["mutant"] for r in raw_rows if len(r["mutant"].split(":")) <= 2}
    assert len(high) == 16130 and not (high & low)
    membership = read_csv(root / "evaluator" / "pool_membership.csv")
    assert len(membership) == 16000
    assert len({r["variant_id"] for r in membership}) == 16000
    assert all(r["variant_id"] in high for r in membership)
    pool_counts = Counter(r["pool_id"] for r in membership)
    assert len(pool_counts) == 16 and set(pool_counts.values()) == {1000}
    labels_map = {r["mutant"]: float(r["DMS_score"]) for r in raw_rows}
    low_rows = read_csv(root / "learner" / "low_order_labeled.csv")
    low_by_id = {r["variant_id"]: r for r in low_rows}
    single_by_position_aa = {}
    for r in low_rows:
        if int(r["order"]) == 1:
            position = int(r["variant_id"][1:-1])
            aa = r["mutated_sequence"][position - 1]
            single_by_position_aa[(position, aa)] = r["variant_id"]
    crossing_edges = 0
    for r in low_rows:
        if int(r["order"]) == 2:
            tokens = r["variant_id"].split(":")
            for removed in tokens:
                position = int(removed[1:-1])
                original = r["mutated_sequence"][position - 1]
                neighbor = single_by_position_aa.get((position, original))
                if neighbor is not None and low_by_id[neighbor]["role"] != r["role"]:
                    crossing_edges += 1
    assert crossing_edges == 0
    role_positions = {role: {int(token[1:-1]) for r in low_rows if r["role"] == role for token in r["variant_id"].split(":")} for role in ["train", "dev"]}
    assert role_positions["train"].isdisjoint(role_positions["dev"])
    assert min(int(r["order"]) for r in low_rows) + min(int(r["order"]) for r in low_rows) >= 2
    boundary = read_csv(root / "evaluator" / "low_order_boundary_excluded.csv")
    assert len(low_rows) + len(boundary) == 13861
    results = json.loads((root / "results" / "results.json").read_text())
    paired = json.loads((root / "results" / "paired_pool_results.json").read_text())
    comparisons = 0
    max_difference = 0.0
    for method_result in results["methods"]:
        filename = "cnn_ensemble.npz" if method_result["method"] == "cnn_ensemble" else "additive.npz"
        data = np.load(root / "predictions" / filename)
        ids = data["variant_id"].astype(str)
        pools = data["pool_id"].astype(str)
        prediction = data["prediction"]
        y = np.asarray([labels_map[v] for v in ids])
        mae = float(np.mean(np.abs(y - prediction)))
        rho = independent_spearman(y, prediction)
        assert close(mae, method_result["point_metrics"]["mae"])
        assert close(rho, method_result["point_metrics"]["spearman"], 1e-12)
        for primary in method_result["primary"]:
            budget = primary["budget"]
            rows = [r for r in paired if r["method"] == method_result["method"] and r["budget"] == budget]
            assert len(rows) == 16
            deltas = []
            for row in rows:
                idx = np.flatnonzero(pools == row["pool_id"])
                local_ids = ids[idx]
                local_y = y[idx]
                local_pred = prediction[idx]
                chosen = np.lexsort((local_ids, -local_pred))[:budget]
                bon = float(np.max(local_y) - np.max(local_y[chosen]))
                assert close(bon, row["bon_regret"])
                random_regrets = []
                for rep in range(config["evaluation"]["random_replicates_per_pool_budget"]):
                    key = f"{config['evaluation']['random_seed']}:{row['pool_id']}:{budget}:{rep}"
                    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                    rng = np.random.default_rng(seed)
                    random_idx = rng.choice(len(idx), size=budget, replace=False)
                    random_regrets.append(float(np.max(local_y) - np.max(local_y[random_idx])))
                random_mean = float(np.mean(random_regrets))
                assert close(random_mean, row["random_regret_mean"])
                deltas.append(bon - random_mean)
                comparisons += 1
                max_difference = max(max_difference, abs((bon - random_mean) - row["bon_minus_random_regret"]))
            assert close(float(np.mean(deltas)), primary["mean_bon_minus_random_regret"])
    output = {"status": "PASS", "source_sha256": sha256_file(raw), "raw_rows": len(raw_rows), "raw_order_counts": dict(sorted(raw_counts.items())), "eligible_high_order": len(high), "pool_counts": dict(sorted(pool_counts.items())), "exact_low_test_overlap": 0, "minimum_possible_low_test_hamming": 3, "low_order_train_dev_hamming1_crossings": crossing_edges, "low_order_train_dev_position_overlap": len(role_positions["train"] & role_positions["dev"]), "low_order_boundary_excluded": len(boundary), "independent_pair_comparisons": comparisons, "maximum_absolute_pair_difference": max_difference, "point_metrics_recomputed": True, "primary_means_recomputed": True, "production_modules_imported": False}
    (root / "verification" / "round2_independent.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
