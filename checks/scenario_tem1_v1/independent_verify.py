#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def corr(x, y):
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


def spearman(x, y):
    return corr(rankdata(x), rankdata(y))


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = Path(args.raw_dir)
    test = Path(args.test_dir)
    protocol = json.loads(Path(args.protocol).read_text())
    scores = {}
    for name, filename in protocol["data"]["conditions"].items():
        f = pd.read_csv(raw / filename)
        scores[name] = pd.to_numeric(f.score, errors="coerce").set_axis(f.hgvs_pro)
    frame = pd.DataFrame(scores)
    frame["position"] = [int(re.search(r"p\.[A-Za-z*]{3}(\d+)", x).group(1)) for x in frame.index]
    positions = sorted(frame.position.unique())
    ordered = sorted(positions, key=lambda p: hashlib.sha256(f"scenario_tem1_v1|split|20261127|{p}".encode()).hexdigest())
    test_positions = set(ordered[int(len(ordered) * 0.6) + int(len(ordered) * 0.2):])
    test_frame = frame[frame.position.isin(test_positions)]

    expected = pd.read_csv(test / "selections.tsv.gz", sep="\t")
    membership = pd.read_csv(test / "pool_membership.tsv.gz", sep="\t")
    membership_groups = {pool_id: sorted(group.hgvs_pro) for pool_id, group in membership.groupby("pool_id")}
    actual_rows = []
    for pool_id, pool in expected.groupby("pool_id", sort=False):
        chosen = membership_groups[pool_id]
        values = frame.loc[chosen]
        for row in pool.itertuples():
            source = row.source
            target = row.target
            proxy_pick = sorted(chosen, key=lambda x: (-values.loc[x, source], x))[0]
            target_oracle = sorted(chosen, key=lambda x: (-values.loc[x, target], x))[0]
            source_oracle = sorted(chosen, key=lambda x: (-values.loc[x, source], x))[0]
            oracle_value = float(values.loc[target_oracle, target])
            bon = oracle_value - float(values.loc[proxy_pick, target])
            random_n = oracle_value - float(values[target].mean())
            actual_rows.append({"pool_id": pool_id, "source": source, "target": target, "N": row.N, "position": row.position, "proxy_pick": proxy_pick, "target_oracle": target_oracle, "source_oracle": source_oracle, "bon_regret": bon, "random_n_regret": random_n, "delta_bon_minus_random": bon - random_n, "turnover_proxy_vs_oracle": float(proxy_pick != target_oracle), "turnover_source_oracle_vs_target_oracle": float(source_oracle != target_oracle)})
    actual = pd.DataFrame(actual_rows)
    keys = ["pool_id", "source", "target", "N", "position"]
    expected_sorted = expected.sort_values(keys).reset_index(drop=True)
    actual_sorted = actual.sort_values(keys).reset_index(drop=True)
    assert expected_sorted[keys].equals(actual_sorted[keys])
    for column in ["proxy_pick", "target_oracle", "source_oracle"]:
        assert (expected_sorted[column] == actual_sorted[column]).all(), column
    numeric_columns = ["bon_regret", "random_n_regret", "delta_bon_minus_random", "turnover_proxy_vs_oracle", "turnover_source_oracle_vs_target_oracle"]
    numeric_differences = {column: maxdiff(expected_sorted[column], actual_sorted[column]) for column in numeric_columns}
    assert max(numeric_differences.values()) < 1e-12

    expected_corr = pd.read_csv(test / "correlations.tsv", sep="\t")
    corr_diffs = []
    for row in expected_corr.itertuples():
        pair = test_frame[[row.source, row.target]].dropna()
        assert len(pair) == row.n_paired
        estimates = {"pearson": corr(pair[row.source], pair[row.target]), "spearman": spearman(pair[row.source], pair[row.target])}
        for label, column in [("source_top10", row.source), ("target_top10", row.target)]:
            ordered_ids = sorted(pair.index, key=lambda x: (-pair.loc[x, column], x))
            tail_ids = ordered_ids[:math.ceil(len(pair) * 0.1)]
            tail = pair.loc[tail_ids]
            assert len(tail) == getattr(row, f"{label}_n")
            estimates[f"{label}_spearman"] = spearman(tail[row.source], tail[row.target])
        for metric, estimate in estimates.items():
            corr_diffs.append(abs(estimate - getattr(row, metric)))
    assert max(corr_diffs) < 1e-12

    draws_by_budget = {}
    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    complete = frame.dropna()
    for budget in protocol["pools"]["budgets"]:
        eligible = sorted(p for p in complete.position.unique() if p in test_positions and sum(complete.position == p) >= budget)
        draws = rng.integers(0, len(eligible), size=(10000, len(eligible)))
        saved = np.load(test / f"bootstrap_position_indices_N{budget}.npy")
        assert np.array_equal(draws, saved)
        draws_by_budget[budget] = (eligible, draws)

    archived = np.load(test / "bootstrap_effects.npz")
    direction_expected = pd.read_csv(test / "direction_results.tsv", sep="\t")
    bootstrap_diffs = []
    point_diffs = []
    ci_diffs = []
    direction_keys = []
    for source, target, budget in expected[["source", "target", "N"]].drop_duplicates().itertuples(index=False, name=None):
        subset = actual[(actual.source == source) & (actual.target == target) & (actual.N == budget)]
        eligible, draws = draws_by_budget[budget]
        by_position = subset.groupby("position")[numeric_columns].mean().loc[eligible]
        matrix = by_position.to_numpy()
        sampled = matrix[draws].mean(axis=1)
        point = matrix.mean(axis=0)
        row = direction_expected[(direction_expected.source == source) & (direction_expected.target == target) & (direction_expected.N == budget)].iloc[0]
        for j, metric in enumerate(numeric_columns):
            key = f"{source}__{target}__N{budget}__{metric}"
            bootstrap_diffs.append(maxdiff(archived[key], sampled[:, j]))
            point_diffs.append(abs(point[j] - row[metric]))
            ci = np.percentile(sampled[:, j], [2.5, 97.5])
            ci_diffs.append(maxdiff(ci, [row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]]))
        direction_keys.append((source, target, budget))
    assert len(direction_keys) == 75
    assert max(bootstrap_diffs + point_diffs + ci_diffs) < 1e-12

    overall = pd.read_csv(test / "overall_results.tsv", sep="\t")
    overall_diffs = []
    for budget in protocol["pools"]["budgets"]:
        row = overall[overall.N == budget].iloc[0]
        for j, metric in enumerate(numeric_columns):
            arrays = [archived[f"{source}__{target}__N{budget}__{metric}"] for source, target, b in direction_keys if b == budget]
            combined = np.mean(arrays, axis=0)
            overall_diffs.append(maxdiff(combined, archived[f"overall__N{budget}__{metric}"]))
            points = [direction_expected[(direction_expected.source == source) & (direction_expected.target == target) & (direction_expected.N == budget)][metric].iloc[0] for source, target, b in direction_keys if b == budget]
            overall_diffs.append(abs(np.mean(points) - row[metric]))
            overall_diffs.append(maxdiff(np.percentile(combined, [2.5, 97.5]), [row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]]))
    assert max(overall_diffs) < 1e-12

    result = {
        "round": 2, "status": "PASS", "production_module_imported": False,
        "selection_rows_recomputed": int(len(actual)), "selected_identities_exact": True,
        "numeric_selection_max_abs_diff": numeric_differences,
        "correlations_recomputed": int(len(expected_corr)), "correlation_max_abs_diff": float(max(corr_diffs)),
        "direction_bootstrap_combinations": len(direction_keys), "bootstrap_resamples_per_combination": 10000,
        "bootstrap_max_abs_diff": float(max(bootstrap_diffs)), "point_estimate_max_abs_diff": float(max(point_diffs)),
        "interval_max_abs_diff": float(max(ci_diffs)), "overall_max_abs_diff": float(max(overall_diffs)),
        "position_cluster_bootstrap_indices_exact": True,
        "cross_condition_independence_assumption": "not used; all directional outcomes remain paired by fixed pool and position"
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
