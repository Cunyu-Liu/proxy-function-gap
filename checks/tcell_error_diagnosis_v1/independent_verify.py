#!/usr/bin/env python3
"""Independent raw-record reconstruction; deliberately does not import run_analysis."""
import csv
import gzip
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap")
OUT = BASE / "tcell_error_diagnosis_v1"
CONTEXTS = ("defined_tcell_r1", "defined_tcell_r2")
MODELS = ("cnn", "mlp")
CAPS = (1, 2, 4, 8, 16)
SEEDS = (101, 202, 303, 404, 505)
FIELD = {"cnn": "proxy_mean", "mlp": "kmer_mean"}
HIST_INDEX = {"defined_tcell_r1": 2, "defined_tcell_r2": 3}
comparisons = 0
max_difference = 0.0


def close(actual, expected, name, tolerance=2e-10):
    global comparisons, max_difference
    if actual is None or (isinstance(actual, float) and not math.isfinite(actual)):
        assert expected is None or not math.isfinite(expected), name
        comparisons += 1
        return
    difference = abs(float(actual) - float(expected))
    assert difference <= tolerance, (name, actual, expected, difference)
    max_difference = max(max_difference, difference)
    comparisons += 1


def correlation(x, y, rank=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if rank:
        x = rankdata(x)
        y = rankdata(y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return np.corrcoef(x, y)[0, 1]


def load(context):
    pool_root = BASE / "data/derived/castillo_fixed_pools_v3_strict_h1_retry1" / context
    result_root = BASE / "paper1_v3_strict_h1_retry1" / context
    with gzip.open(pool_root / "evaluator/labels.tsv.gz", "rt") as handle:
        labels = {r["sequence_id"]: float(r["measured_rl"]) for r in csv.DictReader(handle, delimiter="\t")}
    pools = []
    for pool_index, learner_path in enumerate(sorted((pool_root / "learner").glob("*.jsonl.gz"))):
        with gzip.open(learner_path, "rt") as handle:
            learner = [json.loads(line) for line in handle]
        pool_id = learner[0]["pool_id"]
        cnn = json.loads((result_root / "cnn_scores" / f"{pool_id}.json").read_text())["records"]
        mlp = json.loads((result_root / "cross_architecture" / f"{pool_id}_kmer.json").read_text())["records"]
        prediction_maps = {}
        for model, source in (("cnn", cnn), ("mlp", mlp)):
            prediction_maps[model] = {r["sequence_id"]: float(r[FIELD[model]]) for r in source}
        pools.append({"pool_id": pool_id, "ids": [r["sequence_id"] for r in learner],
                      "y": np.array([labels[r["sequence_id"]] for r in learner]),
                      "pred": {m: np.array([prediction_maps[m][r["sequence_id"]] for r in learner]) for m in MODELS}})
    return pools


def original_prefix(seed):
    generator = random.Random(seed)
    prefix = []
    for _ in range(16):
        target = generator.random()
        cumulative = 0.0
        chosen = 63
        for index in range(64):
            cumulative += 1.0 / 64
            if target < cumulative:
                chosen = index
                break
        prefix.append(chosen)
    return prefix


def random_position(context, pool, seed_index, cap_index, cap):
    if cap == 1:
        return 0
    sequence = np.random.SeedSequence(20261125, spawn_key=(HIST_INDEX[context], pool, seed_index, cap_index))
    return int(np.random.Generator(np.random.PCG64(sequence)).integers(0, cap))


def independent_bootstrap(pool_values, seed):
    values = np.asarray(pool_values, float)
    if np.all(values == 0):
        return 0.0, 0.0, 0.0, 1.0
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, 16, size=(10000, 16))
    draws = np.empty(10000)
    for b in range(10000):
        draws[b] = np.mean([values[i] for i in indices[b]])
    p = min(1.0, 2 * min((1 + np.sum(draws <= 0)) / 10001, (1 + np.sum(draws >= 0)) / 10001))
    return np.mean(values), np.percentile(draws, 2.5), np.percentile(draws, 97.5), p


def main():
    pointwise_expected = pd.read_csv(OUT / "analysis/pointwise_metrics.tsv", sep="\t")
    selection_expected = pd.read_csv(OUT / "analysis/selection_summary.tsv", sep="\t")
    effect_expected = pd.read_csv(OUT / "analysis/selection_effects.tsv", sep="\t")
    tail_expected = pd.read_csv(OUT / "analysis/top_tail_summary.tsv", sep="\t")
    outlier_expected = pd.read_csv(OUT / "analysis/outlier_summary.tsv", sep="\t")
    all_strict = {}
    selection_id_checks = 0
    for context in CONTEXTS:
        pools = load(context)
        all_strict[context] = pools
        for model in MODELS:
            pred = np.concatenate([p["pred"][model] for p in pools])
            y = np.concatenate([p["y"] for p in pools])
            error = pred - y
            expected = pointwise_expected[(pointwise_expected.context == context)
                                          & (pointwise_expected.model == model)].iloc[0]
            values = {"n": len(y), "prediction_mean": np.mean(pred), "prediction_sd": np.std(pred),
                      "mrl_mean": np.mean(y), "mrl_sd": np.std(y), "bias": np.mean(error),
                      "median_error": np.median(error), "mae": np.mean(np.abs(error)),
                      "rmse": np.sqrt(np.mean(error ** 2)), "mse": np.mean(error ** 2),
                      "bias_squared": np.mean(error) ** 2, "centered_mse": np.var(error),
                      "pearson": correlation(pred, y), "spearman": correlation(pred, y, True),
                      "mean_within_pool_spearman": np.mean([correlation(p["pred"][model], p["y"], True) for p in pools]),
                      "oracle_global_median_shift_mae": np.mean(np.abs(error - np.median(error)))}
            for key, value in values.items():
                close(value, expected[key], f"pointwise:{context}:{model}:{key}")
            absolute = np.abs(error)
            q1, q3 = np.quantile(absolute, [0.25, 0.75])
            threshold = q3 + 1.5 * (q3 - q1)
            mask = absolute > threshold
            expected = outlier_expected[(outlier_expected.context == context)
                                        & (outlier_expected.model == model)].iloc[0]
            for key, value in {"threshold": threshold, "outlier_n": np.sum(mask),
                               "outlier_fraction": np.mean(mask),
                               "mae_contribution": np.sum(absolute[mask]) / np.sum(absolute),
                               "mse_contribution": np.sum(error[mask] ** 2) / np.sum(error ** 2),
                               "mae_excluding_outliers_oracle_diagnostic": np.mean(absolute[~mask])}.items():
                close(value, expected[key], f"outlier:{context}:{model}:{key}")
            tail_values = {cap: [] for cap in CAPS}
            for pool in pools:
                ids = np.asarray(pool["ids"])
                order = sorted(range(64), key=lambda i: (-pool["pred"][model][i], ids[i], i))
                for cap in CAPS:
                    indices = order[:cap]
                    full_oracle = np.max(pool["y"])
                    tail_oracle = np.max(pool["y"][indices])
                    winner = pool["y"][indices[0]]
                    tail_values[cap].append([full_oracle - winner, full_oracle - tail_oracle,
                                             tail_oracle - winner,
                                             np.mean(np.abs(pool["pred"][model][indices] - pool["y"][indices])),
                                             np.mean(pool["pred"][model][indices] - pool["y"][indices]),
                                             float(np.any(pool["y"][indices] > winner))])
            for cap in CAPS:
                actual = np.mean(tail_values[cap], axis=0)
                expected = tail_expected[(tail_expected.context == context) & (tail_expected.model == model)
                                         & (tail_expected.cap == cap)].iloc[0]
                for key, value in zip(("regret", "opportunity_loss", "within_tail_ranking_loss", "mae", "bias",
                                       "winner_flipped_within_tail"), actual):
                    close(value, expected[key], f"tail:{context}:{model}:{cap}:{key}")
            pool_records = {(pool_index, seed_index, cap_index, strategy): None
                            for pool_index in range(16) for seed_index in range(5)
                            for cap_index in range(5) for strategy in ("bon", "random_n", "random_1")}
            for pool_index, pool in enumerate(pools):
                original = json.loads((BASE / "paper1_v3_strict_h1_retry1" / context / "selection"
                                       / f"{pool['pool_id']}.json").read_text())["records"]
                original_map = {(r["strategy"], r["seed"], r["proposal_cap"]): r for r in original if r["alpha"] == 0}
                for seed_index, seed in enumerate(SEEDS):
                    prefix = original_prefix(seed)
                    for cap_index, cap in enumerate(CAPS):
                        current = prefix[:cap]
                        bon = max(current, key=lambda i: pool["pred"][model][i])
                        random_1 = current[0]
                        random_n = current[random_position(context, pool_index, seed_index, cap_index, cap)]
                        for strategy, chosen in (("bon", bon), ("random_n", random_n), ("random_1", random_1)):
                            pool_records[(pool_index, seed_index, cap_index, strategy)] = np.max(pool["y"]) - pool["y"][chosen]
                        if model == "cnn":
                            assert original_map[("bon", seed, cap)]["selected_id"] == pool["ids"][bon]
                            assert original_map[("random", seed, cap)]["selected_id"] == pool["ids"][random_1]
                            selection_id_checks += 2
            for cap_index, cap in enumerate(CAPS):
                for strategy in ("bon", "random_n", "random_1"):
                    pool_means = [np.mean([pool_records[(p, s, cap_index, strategy)] for s in range(5)]) for p in range(16)]
                    expected = selection_expected[(selection_expected.context == context)
                                                  & (selection_expected.model == model)
                                                  & (selection_expected.cap == cap)
                                                  & (selection_expected.strategy == strategy)].iloc[0]
                    close(np.mean(pool_means), expected["regret"], f"selection:{context}:{model}:{cap}:{strategy}")
                for contrast, left, right, offset in (("bon_minus_random_n", "bon", "random_n", 0),
                                                      ("bon_minus_random_1", "bon", "random_1", 1000),
                                                      ("random_n_minus_random_1", "random_n", "random_1", 2000)):
                    differences = []
                    for p in range(16):
                        l = np.mean([pool_records[(p, s, cap_index, left)] for s in range(5)])
                        r = np.mean([pool_records[(p, s, cap_index, right)] for s in range(5)])
                        differences.append(l - r)
                    seed = 20261126 + 100 * CONTEXTS.index(context) + 10 * MODELS.index(model) + cap_index + offset
                    actual = independent_bootstrap(differences, seed)
                    expected = effect_expected[(effect_expected.context == context) & (effect_expected.model == model)
                                               & (effect_expected.cap == cap) & (effect_expected.contrast == contrast)].iloc[0]
                    for key, value in zip(("effect", "ci_low", "ci_high", "p"), actual):
                        close(value, expected[key], f"effect:{context}:{model}:{cap}:{contrast}")
    architecture_expected = json.loads((OUT / "analysis/architecture_comparison.json").read_text())
    for context in CONTEXTS:
        pools = all_strict[context]
        expected = next(r for r in architecture_expected if r["context"] == context)
        cnn = np.concatenate([p["pred"]["cnn"] for p in pools])
        mlp = np.concatenate([p["pred"]["mlp"] for p in pools])
        y = np.concatenate([p["y"] for p in pools])
        cnn_error = cnn - y
        mlp_error = mlp - y
        for key, value in {"prediction_pearson": correlation(cnn, mlp),
                           "prediction_spearman": correlation(cnn, mlp, True),
                           "error_pearson": correlation(cnn_error, mlp_error),
                           "error_spearman": correlation(cnn_error, mlp_error, True),
                           "error_sign_agreement": np.mean(np.sign(cnn_error) == np.sign(mlp_error)),
                           "cnn_minus_mlp_mae": np.mean(np.abs(cnn_error)) - np.mean(np.abs(mlp_error)),
                           "cnn_minus_mlp_bias": np.mean(cnn_error) - np.mean(mlp_error)}.items():
            close(value, expected[key], f"architecture:{context}:{key}")
        tops = {}
        for model in MODELS:
            selected = set()
            for pool in pools:
                ids = np.asarray(pool["ids"])
                order = sorted(range(64), key=lambda i: (-pool["pred"][model][i], ids[i], i))
                selected.update(ids[order[:16]])
            tops[model] = selected
        close(len(tops["cnn"] & tops["mlp"]), expected["top16_overlap_n"], f"architecture:{context}:top16")
        same = []
        for pool_index, pool in enumerate(pools):
            for seed in SEEDS:
                prefix = original_prefix(seed)[:16]
                winners = [max(prefix, key=lambda i: pool["pred"][model][i]) for model in MODELS]
                same.append(winners[0] == winners[1])
        close(np.mean(same), expected["cap16_bon_same_selected_fraction"], f"architecture:{context}:selection")
    r1rows = {}
    r2rows = {}
    for context, target in (("defined_tcell_r1", r1rows), ("defined_tcell_r2", r2rows)):
        for pool in all_strict[context]:
            for i, sid in enumerate(pool["ids"]):
                target[sid] = (pool["y"][i], pool["pred"]["cnn"][i], pool["pred"]["mlp"][i])
    shared = sorted(set(r1rows) & set(r2rows))
    replicate = json.loads((OUT / "analysis/replicate_comparison.json").read_text())
    close(len(shared), replicate["strict_shared_sequence_n"], "replicate:n")
    if shared:
        y1 = np.array([r1rows[s][0] for s in shared])
        y2 = np.array([r2rows[s][0] for s in shared])
        close(np.mean(y2-y1), replicate["strict_shared_r2_minus_r1_mean"], "replicate:bias")
        close(np.mean(np.abs(y2-y1)), replicate["strict_shared_paired_mae"], "replicate:mae")
        close(correlation(y1, y2), replicate["strict_shared_mrl_pearson"], "replicate:pearson")
    result = {"status": "PASS", "independent_implementation": True,
              "numeric_comparisons": comparisons, "maximum_absolute_difference": max_difference,
              "original_cnn_selected_id_checks": selection_id_checks,
              "candidate_rows_reconstructed": 2048, "bootstrap_replicates_per_effect": 10000,
              "tolerance": 2e-10}
    (OUT / "verification/round2_independent.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
