#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap")
POOL_ROOT = BASE / "data/derived/castillo_fixed_pools_v3_strict_h1_retry1"
RESULT_ROOT = BASE / "paper1_v3_strict_h1_retry1"
OUT = BASE / "tcell_error_diagnosis_v1"
CODE = Path("/home/proxy-function-gap-tcell-diagnosis")
CONTEXTS = ("defined_tcell_r1", "defined_tcell_r2")
HISTORICAL_CONTEXT_INDEX = {c: i for c, i in zip(CONTEXTS, (2, 3))}
MODELS = ("cnn", "mlp")
MODEL_FIELD = {"cnn": "proxy_mean", "mlp": "kmer_mean"}
CAPS = (1, 2, 4, 8, 16)
SEEDS = (101, 202, 303, 404, 505)
B = 10000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    if method == "spearman":
        return float(stats.spearmanr(x, y).statistic)
    raise ValueError(method)


def draw_index(rng: random.Random, q_values) -> int:
    total = float(sum(q_values))
    draw = rng.random() * total
    cumulative = 0.0
    last_positive = None
    for i, q in enumerate(q_values):
        if q > 0:
            last_positive = i
        cumulative += float(q)
        if draw < cumulative:
            return i
    assert last_positive is not None
    return last_positive


def proposal_prefix(pool_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    q = [1.0 / pool_size] * pool_size
    return [draw_index(rng, q) for _ in range(max(CAPS))]


def random_n_position(context: str, pool_index: int, seed_index: int, cap_index: int, cap: int) -> int:
    if cap == 1:
        return 0
    ss = np.random.SeedSequence(
        entropy=20261125,
        spawn_key=(HISTORICAL_CONTEXT_INDEX[context], pool_index, seed_index, cap_index),
    )
    return int(np.random.Generator(np.random.PCG64(ss)).integers(0, cap))


def bootstrap(values, seed: int, replicates: int = B):
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) and np.all(np.isfinite(values))
    effect = float(values.mean())
    if np.all(values == 0):
        return {"effect": effect, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0}
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = values[indices].mean(axis=1)
    p = min(1.0, 2.0 * min((1 + int(np.sum(draws <= 0))) / (replicates + 1),
                           (1 + int(np.sum(draws >= 0))) / (replicates + 1)))
    return {"effect": effect, "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)), "p": float(p)}


def add_multiplicity(rows: list[dict]) -> None:
    for model in MODELS:
        for contrast in sorted({r["contrast"] for r in rows if r["model"] == model}):
            family = [r for r in rows if r["model"] == model and r["contrast"] == contrast]
            assert len(family) == len(CONTEXTS) * len(CAPS)
            for r in family:
                r["bonferroni_p"] = min(1.0, len(family) * r["p"])
            ordered = sorted(family, key=lambda r: (r["p"], r["context"], r["cap"]))
            running = 0.0
            for rank, r in enumerate(ordered):
                running = max(running, min(1.0, (len(family) - rank) * r["p"]))
                r["holm_p"] = running


def load_context(context: str):
    pool_dir = POOL_ROOT / context
    result_dir = RESULT_ROOT / context
    labels = pd.read_csv(pool_dir / "evaluator/labels.tsv.gz", sep="\t")
    assert labels["sequence_id"].is_unique
    label_map = dict(zip(labels["sequence_id"], labels["measured_rl"].astype(float)))
    rows = []
    manifest_inputs = [pool_dir / "pool_manifest.json", pool_dir / "evaluator/labels.tsv.gz"]
    for pool_index, path in enumerate(sorted((pool_dir / "learner").glob("*.jsonl.gz"))):
        with gzip.open(path, "rt") as f:
            learner = [json.loads(line) for line in f]
        assert len(learner) == 64 and len({r["sequence_id"] for r in learner}) == 64
        pool_id = learner[0]["pool_id"]
        cnn_path = result_dir / "cnn_scores" / f"{pool_id}.json"
        mlp_path = result_dir / "cross_architecture" / f"{pool_id}_kmer.json"
        selection_path = result_dir / "selection" / f"{pool_id}.json"
        evaluation_path = result_dir / "evaluation" / f"{pool_id}.json"
        manifest_inputs.extend([path, cnn_path, mlp_path, selection_path, evaluation_path])
        cnn = json.loads(cnn_path.read_text())["records"]
        mlp = json.loads(mlp_path.read_text())["records"]
        cnn_by_id = {r["sequence_id"]: r for r in cnn}
        mlp_by_id = {r["sequence_id"]: r for r in mlp}
        assert len(cnn_by_id) == len(mlp_by_id) == 64
        for candidate_index, base in enumerate(learner):
            sid = base["sequence_id"]
            c, m = cnn_by_id[sid], mlp_by_id[sid]
            assert base["utr50"] == c["utr50"] == m["utr50"]
            assert base["utr50"] == labels.loc[labels["sequence_id"] == sid, "utr50"].iloc[0]
            row = {"context": context, "pool_index": pool_index, "pool_id": pool_id,
                   "candidate_index": candidate_index, "sequence_id": sid, "utr50": base["utr50"],
                   "y": float(label_map[sid]), "gc": float(c["gc"]),
                   "uAUG": bool(c["uAUG"]), "longest_poly_u": int(c["longest_poly_u"]),
                   "support": float(c["support"])}
            for model, source in (("cnn", c), ("mlp", m)):
                prefix = "proxy" if model == "cnn" else "kmer"
                row[f"{model}_pred"] = float(source[f"{prefix}_mean"])
                row[f"{model}_std"] = float(source[f"{prefix}_std"])
                row[f"{model}_range"] = float(source[f"{prefix}_range"])
            rows.append(row)
    df = pd.DataFrame(rows)
    assert len(df) == 1024 and df["sequence_id"].is_unique
    return df, manifest_inputs


def pointwise_row(df: pd.DataFrame, model: str, context: str) -> dict:
    pred = df[f"{model}_pred"].to_numpy(float)
    y = df["y"].to_numpy(float)
    error = pred - y
    pool_spearman = [corr(g[f"{model}_pred"], g["y"], "spearman")
                     for _, g in df.groupby("pool_index", sort=True)]
    global_shifted = error - np.median(error)
    pool_shifted = np.concatenate([(g[f"{model}_pred"].to_numpy(float) - g["y"].to_numpy(float)
                                    - np.median(g[f"{model}_pred"].to_numpy(float) - g["y"].to_numpy(float)))
                                   for _, g in df.groupby("pool_index", sort=True)])
    return {"context": context, "model": model, "n": len(df),
            "prediction_mean": pred.mean(), "prediction_sd": pred.std(),
            "mrl_mean": y.mean(), "mrl_sd": y.std(), "bias": error.mean(),
            "median_error": np.median(error), "mae": np.mean(np.abs(error)),
            "rmse": np.sqrt(np.mean(error ** 2)), "mse": np.mean(error ** 2),
            "bias_squared": error.mean() ** 2, "centered_mse": error.var(),
            "pearson": corr(pred, y), "spearman": corr(pred, y, "spearman"),
            "mean_within_pool_spearman": np.nanmean(pool_spearman),
            "oracle_global_median_shift_mae": np.mean(np.abs(global_shifted)),
            "oracle_pool_median_shift_mae": np.mean(np.abs(pool_shifted)),
            "mean_pool_mrl_sd": df.groupby("pool_index")["y"].std(ddof=0).mean(),
            "mae_over_mean_pool_sd": np.mean(np.abs(error)) / df.groupby("pool_index")["y"].std(ddof=0).mean()}


def top_tail(df: pd.DataFrame, context: str, model: str):
    records = []
    top16_ids = set()
    pred_col = f"{model}_pred"
    for pool_index, pool in df.groupby("pool_index", sort=True):
        ordered = pool.sort_values([pred_col, "sequence_id", "candidate_index"],
                                   ascending=[False, True, True])
        full_oracle = float(pool["y"].max())
        for cap in CAPS:
            tail = ordered.iloc[:cap]
            winner = tail.iloc[0]
            tail_oracle = float(tail["y"].max())
            y = tail["y"].to_numpy(float)
            non_tied = concordant = 0
            for i in range(cap):
                for j in range(i + 1, cap):
                    delta = y[i] - y[j]
                    if delta != 0:
                        non_tied += 1
                        concordant += int(delta > 0)
            records.append({"context": context, "model": model, "pool_index": pool_index, "cap": cap,
                            "full_oracle_mrl": full_oracle, "tail_oracle_mrl": tail_oracle,
                            "winner_mrl": float(winner["y"]), "winner_prediction": float(winner[pred_col]),
                            "regret": full_oracle - float(winner["y"]),
                            "opportunity_loss": full_oracle - tail_oracle,
                            "within_tail_ranking_loss": tail_oracle - float(winner["y"]),
                            "mae": float(np.mean(np.abs(tail[pred_col] - tail["y"]))),
                            "bias": float(np.mean(tail[pred_col] - tail["y"])),
                            "spearman": corr(tail[pred_col], tail["y"], "spearman"),
                            "pair_inversion_rate": 1.0 - concordant / non_tied if non_tied else float("nan"),
                            "winner_flipped_within_tail": float(np.any(y > y[0])),
                            "pool_mrl_sd": float(pool["y"].std(ddof=0)),
                            "pool_mrl_range": float(pool["y"].max() - pool["y"].min())})
            if cap == 16:
                top16_ids.update(tail["sequence_id"])
    detail = pd.DataFrame(records)
    assert np.max(np.abs(detail["regret"] - detail["opportunity_loss"] - detail["within_tail_ranking_loss"])) < 1e-12
    aggregated = []
    for cap, group in detail.groupby("cap", sort=True):
        row = {"context": context, "model": model, "cap": cap, "pools": len(group)}
        for col in ["full_oracle_mrl", "tail_oracle_mrl", "winner_mrl", "regret", "opportunity_loss",
                    "within_tail_ranking_loss", "mae", "bias", "spearman", "pair_inversion_rate",
                    "winner_flipped_within_tail", "pool_mrl_sd", "pool_mrl_range"]:
            row[col] = float(group[col].mean())
        ci_index = 5000 + CAPS.index(cap)
        for j, col in enumerate(["regret", "opportunity_loss", "within_tail_ranking_loss", "mae"]):
            interval = bootstrap(group[col].to_numpy(float),
                                 20261126 + 100 * CONTEXTS.index(context) + 10 * MODELS.index(model) + ci_index + j)
            row[f"{col}_ci_low"] = interval["ci_low"]
            row[f"{col}_ci_high"] = interval["ci_high"]
        aggregated.append(row)
    return detail, pd.DataFrame(aggregated), top16_ids


def selection_analysis(df: pd.DataFrame, context: str):
    record_rows = []
    original_checks = 0
    for pool_index, pool in df.groupby("pool_index", sort=True):
        pool = pool.sort_values("candidate_index").reset_index(drop=True)
        pool_id = pool["pool_id"].iloc[0]
        selection = json.loads((RESULT_ROOT / context / "selection" / f"{pool_id}.json").read_text())["records"]
        evaluation = json.loads((RESULT_ROOT / context / "evaluation" / f"{pool_id}.json").read_text())["records"]
        original = {(r["strategy"], r["seed"], r["proposal_cap"]): r for r in selection if r["alpha"] == 0}
        evaluated = {(r["strategy"], r["seed"], r["proposal_cap"]): r for r in evaluation if r["alpha"] == 0}
        prefixes = {seed: proposal_prefix(len(pool), seed) for seed in SEEDS}
        full_oracle = float(pool["y"].max())
        for seed_index, seed in enumerate(SEEDS):
            for cap_index, cap in enumerate(CAPS):
                prefix = prefixes[seed][:cap]
                prefix_oracle = float(pool.iloc[prefix]["y"].max())
                random_index = prefix[0]
                random_n_index = prefix[random_n_position(context, pool_index, seed_index, cap_index, cap)]
                for model in MODELS:
                    predictions = pool[f"{model}_pred"].to_numpy(float)
                    bon_index = max(prefix, key=lambda i: predictions[i])
                    for strategy, selected_index in (("bon", bon_index), ("random_n", random_n_index),
                                                     ("random_1", random_index)):
                        y = float(pool.iloc[selected_index]["y"])
                        record_rows.append({"context": context, "model": model, "pool_index": pool_index,
                                            "pool_id": pool_id, "seed": seed, "cap": cap, "strategy": strategy,
                                            "selected_index": selected_index,
                                            "selected_id": pool.iloc[selected_index]["sequence_id"],
                                            "selected_mrl": y, "full_oracle_mrl": full_oracle,
                                            "prefix_oracle_mrl": prefix_oracle, "regret": full_oracle - y,
                                            "opportunity_loss": full_oracle - prefix_oracle,
                                            "within_prefix_ranking_loss": prefix_oracle - y})
                    if model == "cnn":
                        for strategy, key, selected_index in (("bon", "bon", bon_index),
                                                              ("random_1", "random", random_index)):
                            old = original[(key, seed, cap)]
                            ev = evaluated[(key, seed, cap)]
                            assert old["selected_index"] == selected_index
                            assert old["selected_id"] == pool.iloc[selected_index]["sequence_id"]
                            assert ev["selected_index"] == selected_index
                            assert abs(ev["regret"] - (full_oracle - float(pool.iloc[selected_index]["y"]))) < 1e-12
                            original_checks += 4
    records = pd.DataFrame(record_rows)
    assert np.max(np.abs(records["regret"] - records["opportunity_loss"] - records["within_prefix_ranking_loss"])) < 1e-12
    pool_strategy = records.groupby(["context", "model", "pool_index", "cap", "strategy"], as_index=False).agg(
        regret=("regret", "mean"), selected_mrl=("selected_mrl", "mean"),
        opportunity_loss=("opportunity_loss", "mean"),
        within_prefix_ranking_loss=("within_prefix_ranking_loss", "mean"))
    summary_rows = []
    for (model, cap, strategy), group in pool_strategy.groupby(["model", "cap", "strategy"], sort=True):
        assert len(group) == 16
        row = {"context": context, "model": model, "cap": cap, "strategy": strategy, "pools": 16}
        for col in ["regret", "selected_mrl", "opportunity_loss", "within_prefix_ranking_loss"]:
            row[col] = float(group[col].mean())
        interval = bootstrap(group["regret"].to_numpy(float),
                             20261126 + 100 * CONTEXTS.index(context) + 10 * MODELS.index(model) + CAPS.index(cap) + 4000)
        row["regret_ci_low"] = interval["ci_low"]
        row["regret_ci_high"] = interval["ci_high"]
        summary_rows.append(row)
    effect_rows = []
    for model in MODELS:
        model_pool = pool_strategy[pool_strategy["model"] == model]
        for cap in CAPS:
            for contrast, a, b in (("bon_minus_random_n", "bon", "random_n"),
                                   ("bon_minus_random_1", "bon", "random_1"),
                                   ("random_n_minus_random_1", "random_n", "random_1")):
                left = model_pool[(model_pool["cap"] == cap) & (model_pool["strategy"] == a)].sort_values("pool_index")
                right = model_pool[(model_pool["cap"] == cap) & (model_pool["strategy"] == b)].sort_values("pool_index")
                values = left["regret"].to_numpy() - right["regret"].to_numpy()
                result = bootstrap(values, 20261126 + 100 * CONTEXTS.index(context) + 10 * MODELS.index(model)
                                   + CAPS.index(cap) + {"bon_minus_random_n": 0, "bon_minus_random_1": 1000,
                                                        "random_n_minus_random_1": 2000}[contrast])
                effect_rows.append({"context": context, "model": model, "cap": cap, "contrast": contrast, **result})
    for cap in CAPS:
        left = pool_strategy[(pool_strategy["model"] == "mlp") & (pool_strategy["cap"] == cap)
                             & (pool_strategy["strategy"] == "bon")].sort_values("pool_index")
        right = pool_strategy[(pool_strategy["model"] == "cnn") & (pool_strategy["cap"] == cap)
                              & (pool_strategy["strategy"] == "bon")].sort_values("pool_index")
        result = bootstrap(left["regret"].to_numpy() - right["regret"].to_numpy(),
                           20261126 + 100 * CONTEXTS.index(context) + 10 * MODELS.index("mlp") + CAPS.index(cap) + 3000)
        effect_rows.append({"context": context, "model": "mlp", "cap": cap,
                            "contrast": "mlp_minus_cnn_bon", **result})
    loo_rows = []
    for model in MODELS:
        for cap in (4, 8, 16):
            bon = pool_strategy[(pool_strategy["model"] == model) & (pool_strategy["cap"] == cap)
                                & (pool_strategy["strategy"] == "bon")].sort_values("pool_index")
            rnd = pool_strategy[(pool_strategy["model"] == model) & (pool_strategy["cap"] == cap)
                                & (pool_strategy["strategy"] == "random_n")].sort_values("pool_index")
            values = bon["regret"].to_numpy() - rnd["regret"].to_numpy()
            loo = [(values.sum() - values[i]) / 15 for i in range(16)]
            loo_rows.append({"context": context, "model": model, "cap": cap,
                             "full_effect": values.mean(), "loo_min": np.min(loo), "loo_max": np.max(loo),
                             "loo_min_pool_index": int(np.argmin(loo)), "loo_max_pool_index": int(np.argmax(loo)),
                             "pools_with_negative_effect": int(np.sum(values < 0)),
                             "pools_with_positive_effect": int(np.sum(values > 0))})
    return records, pool_strategy, pd.DataFrame(summary_rows), pd.DataFrame(effect_rows), pd.DataFrame(loo_rows), original_checks


def feature_analysis(df: pd.DataFrame, context: str, top16: dict):
    frame = df.copy()
    for col in ("gc", "support"):
        try:
            frame[col + "_stratum"] = pd.qcut(frame[col], 4, labels=["q1", "q2", "q3", "q4"]).astype(str)
        except ValueError:
            frame[col + "_stratum"] = pd.qcut(frame[col].rank(method="first"), 4,
                                               labels=["q1", "q2", "q3", "q4"]).astype(str)
    frame["uAUG_stratum"] = frame["uAUG"].map({False: "false", True: "true"})
    frame["longest_poly_u_stratum"] = frame["longest_poly_u"].map(lambda x: str(x) if x < 3 else ">=3")
    output = []
    contrasts = []
    for model in MODELS:
        pred = frame[f"{model}_pred"]
        frame["error"] = pred - frame["y"]
        frame["abs_error"] = frame["error"].abs()
        frame["top16"] = frame["sequence_id"].isin(top16[model])
        for feature in ("gc", "uAUG", "longest_poly_u", "support"):
            col = feature + "_stratum"
            for scope, scoped in (("all", frame), ("top16", frame[frame["top16"]])):
                for stratum, group in scoped.groupby(col, sort=True):
                    output.append({"context": context, "model": model, "feature": feature,
                                   "stratum": stratum, "scope": scope, "n": len(group),
                                   "prediction_mean": group[f"{model}_pred"].mean(), "mrl_mean": group["y"].mean(),
                                   "bias": group["error"].mean(), "mae": group["abs_error"].mean(),
                                   "feature_mean": group[feature].mean(),
                                   "scope_fraction": len(group) / len(scoped)})
            top = frame[frame["top16"]]
            rest = frame[~frame["top16"]]
            contrasts.append({"context": context, "model": model, "feature": feature,
                              "top16_n": len(top), "rest_n": len(rest),
                              "top16_feature_mean": top[feature].mean(), "rest_feature_mean": rest[feature].mean(),
                              "top16_minus_rest": top[feature].mean() - rest[feature].mean(),
                              "top16_bias": top["error"].mean(), "rest_bias": rest["error"].mean(),
                              "top16_mae": top["abs_error"].mean(), "rest_mae": rest["abs_error"].mean()})
    return pd.DataFrame(output), pd.DataFrame(contrasts)


def outlier_analysis(df: pd.DataFrame, selections: pd.DataFrame, context: str):
    rows = []
    for model in MODELS:
        error = df[f"{model}_pred"] - df["y"]
        absolute = error.abs()
        q1, q3 = absolute.quantile([0.25, 0.75])
        threshold = q3 + 1.5 * (q3 - q1)
        mask = absolute > threshold
        selected = selections[(selections["model"] == model) & (selections["cap"] == 16)
                              & (selections["strategy"] == "bon")]
        selected_ids = df.set_index("sequence_id").loc[selected["selected_id"]].index
        outlier_ids = set(df.loc[mask, "sequence_id"])
        rows.append({"context": context, "model": model, "abs_error_q1": q1, "abs_error_q3": q3,
                     "threshold": threshold, "outlier_n": int(mask.sum()), "outlier_fraction": mask.mean(),
                     "mae_contribution": absolute[mask].sum() / absolute.sum(),
                     "mse_contribution": (error[mask] ** 2).sum() / (error ** 2).sum(),
                     "mae_excluding_outliers_oracle_diagnostic": absolute[~mask].mean(),
                     "cap16_bon_selection_events": len(selected),
                     "cap16_bon_outlier_events": sum(sid in outlier_ids for sid in selected_ids),
                     "cap16_bon_outlier_fraction": np.mean([sid in outlier_ids for sid in selected_ids])})
    return pd.DataFrame(rows)


def paired_stats(frame: pd.DataFrame, prefix: str, include_errors: bool):
    delta = frame["y_r2"] - frame["y_r1"]
    result = {f"{prefix}_n": len(frame), f"{prefix}_r2_minus_r1_mean": delta.mean(),
              f"{prefix}_r2_minus_r1_median": delta.median(), f"{prefix}_r2_minus_r1_sd": delta.std(ddof=0),
              f"{prefix}_paired_mae": delta.abs().mean(), f"{prefix}_mrl_pearson": corr(frame["y_r1"], frame["y_r2"]),
              f"{prefix}_mrl_spearman": corr(frame["y_r1"], frame["y_r2"], "spearman")}
    if include_errors:
        for model in MODELS:
            e1 = frame[f"{model}_pred_r1"] - frame["y_r1"]
            e2 = frame[f"{model}_pred_r2"] - frame["y_r2"]
            result[f"{prefix}_{model}_error_pearson"] = corr(e1, e2)
            result[f"{prefix}_{model}_error_sign_agreement"] = np.mean(np.sign(e1) == np.sign(e2))
            result[f"{prefix}_{model}_max_prediction_difference"] = np.max(np.abs(
                frame[f"{model}_pred_r1"] - frame[f"{model}_pred_r2"]))
    return result


def replicate_audit(r1: pd.DataFrame, r2: pd.DataFrame):
    strict = r1.merge(r2, on="sequence_id", suffixes=("_r1", "_r2"))
    result = {"strict_shared_sequence_n": len(strict)}
    if len(strict):
        result.update(paired_stats(strict, "strict_shared", True))
    raw = []
    for context in CONTEXTS:
        manifest = json.loads((POOL_ROOT / context / "pool_manifest.json").read_text())
        path = Path(manifest["raw_file"])
        assert digest(path) == manifest["raw_file_sha256"]
        raw.append(pd.read_csv(path, usecols=["utr", "rl"]))
    raw_counts = []
    for context, frame in zip(CONTEXTS, raw):
        raw_counts.append({"context": context, "rows": len(frame), "unique_sequences": frame["utr"].nunique(),
                           "duplicate_rows": len(frame) - frame["utr"].nunique(),
                           "raw_sha256": digest(Path(json.loads((POOL_ROOT / context / "pool_manifest.json").read_text())["raw_file"]))})
    merged = raw[0].merge(raw[1], on="utr", suffixes=("_r1", "_r2"))
    merged = merged.rename(columns={"rl_r1": "y_r1", "rl_r2": "y_r2"})
    result.update(paired_stats(merged, "raw_shared_auxiliary", False))
    result["raw_counts"] = raw_counts
    return result


def main():
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for d in ("raw_audit", "analysis", "verification", "logs", "export"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    protocol = CODE / "frozen_protocol.json"
    assert protocol.exists()
    input_manifest = []
    frames = {}
    pointwise = []
    tail_details = []
    tail_summaries = []
    selections = []
    pool_selections = []
    selection_summaries = []
    effects = []
    loo_rows = []
    features = []
    feature_contrasts = []
    outliers = []
    original_checks = 0
    top16_by_context = {}
    for context in CONTEXTS:
        df, inputs = load_context(context)
        frames[context] = df
        input_manifest.extend({"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size} for p in inputs)
        for model in MODELS:
            pointwise.append(pointwise_row(df, model, context))
            detail, summary, top16 = top_tail(df, context, model)
            tail_details.append(detail)
            tail_summaries.append(summary)
            top16_by_context[(context, model)] = top16
        rec, pool_sel, sel_sum, eff, loo, checks = selection_analysis(df, context)
        original_checks += checks
        selections.append(rec)
        pool_selections.append(pool_sel)
        selection_summaries.append(sel_sum)
        effects.append(eff)
        loo_rows.append(loo)
        feat, con = feature_analysis(df, context, {m: top16_by_context[(context, m)] for m in MODELS})
        features.append(feat)
        feature_contrasts.append(con)
        outliers.append(outlier_analysis(df, rec, context))
    effect_frame = pd.concat(effects, ignore_index=True)
    effect_records = effect_frame.to_dict("records")
    add_multiplicity(effect_records)
    effect_frame = pd.DataFrame(effect_records)
    selection_frame = pd.concat(selections, ignore_index=True)
    pointwise_frame = pd.concat([pd.DataFrame(pointwise)], ignore_index=True)
    tail_summary_frame = pd.concat(tail_summaries, ignore_index=True)
    selection_summary_frame = pd.concat(selection_summaries, ignore_index=True)
    pool_selection_frame = pd.concat(pool_selections, ignore_index=True)
    feature_frame = pd.concat(features, ignore_index=True)
    feature_contrast_frame = pd.concat(feature_contrasts, ignore_index=True)
    outlier_frame = pd.concat(outliers, ignore_index=True)
    loo_frame = pd.concat(loo_rows, ignore_index=True)
    architecture_comparison = []
    for context in CONTEXTS:
        frame = frames[context]
        cnn_error = frame["cnn_pred"] - frame["y"]
        mlp_error = frame["mlp_pred"] - frame["y"]
        cnn_top = top16_by_context[(context, "cnn")]
        mlp_top = top16_by_context[(context, "mlp")]
        bon = selection_frame[(selection_frame["context"] == context) & (selection_frame["cap"] == 16)
                              & (selection_frame["strategy"] == "bon")]
        bon_wide = bon.pivot_table(index=["pool_index", "seed"], columns="model", values="selected_id", aggfunc="first")
        architecture_comparison.append({"context": context,
            "prediction_pearson": corr(frame["cnn_pred"], frame["mlp_pred"]),
            "prediction_spearman": corr(frame["cnn_pred"], frame["mlp_pred"], "spearman"),
            "error_pearson": corr(cnn_error, mlp_error), "error_spearman": corr(cnn_error, mlp_error, "spearman"),
            "error_sign_agreement": float(np.mean(np.sign(cnn_error) == np.sign(mlp_error))),
            "cnn_minus_mlp_mae": float(cnn_error.abs().mean() - mlp_error.abs().mean()),
            "cnn_minus_mlp_bias": float(cnn_error.mean() - mlp_error.mean()),
            "top16_overlap_n": len(cnn_top & mlp_top), "top16_overlap_fraction": len(cnn_top & mlp_top) / 256,
            "top16_jaccard": len(cnn_top & mlp_top) / len(cnn_top | mlp_top),
            "cap16_bon_same_selected_fraction": float(np.mean(bon_wide["cnn"] == bon_wide["mlp"]))})
    replicate = replicate_audit(frames[CONTEXTS[0]], frames[CONTEXTS[1]])
    raw_audit = {"status": "PASS", "original_cnn_field_checks": original_checks,
                 "candidate_rows_per_context": 1024, "pools_per_context": 16,
                 "within_pool_duplicate_counts": {c: int(1024 - frames[c]["sequence_id"].nunique()) for c in CONTEXTS},
                 "cross_replicate_strict_duplicate_n": replicate["strict_shared_sequence_n"],
                 "input_files": input_manifest}
    tables = {"pointwise_metrics.tsv": pointwise_frame, "top_tail_summary.tsv": tail_summary_frame,
              "selection_summary.tsv": selection_summary_frame, "selection_effects.tsv": effect_frame,
              "pool_selection_summary.tsv": pool_selection_frame, "leave_one_pool_out.tsv": loo_frame,
              "feature_strata.tsv": feature_frame, "feature_tail_contrasts.tsv": feature_contrast_frame,
              "outlier_summary.tsv": outlier_frame}
    for name, table in tables.items():
        table.to_csv(OUT / "analysis" / name, sep="\t", index=False, float_format="%.17g")
        export = table.drop(columns=[c for c in ("pool_id",) if c in table.columns], errors="ignore")
        export.to_csv(OUT / "export" / name, sep="\t", index=False, float_format="%.17g")
    tail_detail = pd.concat(tail_details, ignore_index=True)
    tail_detail.to_csv(OUT / "analysis" / "top_tail_pool_detail.tsv.gz", sep="\t", index=False,
                       float_format="%.17g", compression="gzip")
    selection_frame.to_csv(OUT / "analysis" / "selection_records.tsv.gz", sep="\t", index=False,
                           float_format="%.17g", compression="gzip")
    save_json(OUT / "raw_audit/raw_audit.json", clean(raw_audit))
    save_json(OUT / "analysis/replicate_comparison.json", clean(replicate))
    save_json(OUT / "export/replicate_comparison.json", clean(replicate))
    save_json(OUT / "analysis/architecture_comparison.json", clean(architecture_comparison))
    save_json(OUT / "export/architecture_comparison.json", clean(architecture_comparison))
    summary = {"status": "PASS", "analysis": "tcell_error_diagnosis_v1", "new_blind_test": False,
               "training_or_inference": False, "candidate_rows": 2048,
               "original_cnn_field_checks": original_checks,
               "pointwise": clean(pointwise), "selection_effects": clean(effect_records),
               "architecture_comparison": clean(architecture_comparison),
               "replicate_comparison": clean(replicate), "outliers": clean(outlier_frame.to_dict("records")),
               "negative_results_retained": True}
    save_json(OUT / "analysis/summary.json", summary)
    save_json(OUT / "export/summary.json", summary)
    save_json(OUT / "verification/environment.json", {"python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": stats.__version__ if hasattr(stats, "__version__") else "1.10.1",
            "cpu_only": True, "cuda_invoked": False})
    print(json.dumps({"status": "PASS", "candidate_rows": 2048, "original_cnn_field_checks": original_checks,
                      "strict_shared_sequence_n": replicate["strict_shared_sequence_n"],
                      "raw_shared_sequence_n": replicate["raw_shared_auxiliary_n"]}, indent=2))


if __name__ == "__main__":
    main()
