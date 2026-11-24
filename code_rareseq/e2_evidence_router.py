#!/usr/bin/env python3
"""E2 频率 vs d_hom 证据 vs 联合路由；全部选择仅使用 val。"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from e1_frozen_baselines import metrics
from log_run import RunLogger, sha256_file


def load_predictions(paths, identity):
    base_keys = ["experiment", "rbp", "tier", "fold", "track", "split", "window_id", "label"]
    keys = base_keys + ["occurrence"]
    merged = None
    for model in ("bce", "knn"):
        frame = pd.concat([pd.read_csv(paths[split][model], sep="\t") for split in ("val", "test")], ignore_index=True)
        frame["occurrence"] = frame.groupby(base_keys).cumcount()
        frame = frame[keys + ["score"]].rename(columns={"score": f"score_{model}"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=keys, validate="one_to_one")
    merged["d_hom"] = merged.window_id.map(identity)
    from sklearn.linear_model import LogisticRegression
    for track, group in merged.groupby("track"):
        val_mask = merged.split.eq("val") & merged.track.eq(track)
        for model in ("bce", "knn"):
            raw = merged[f"score_{model}"].to_numpy()
            calibration = LogisticRegression(C=1.0, max_iter=1000)
            calibration.fit(raw[val_mask].reshape(-1, 1), merged.loc[val_mask, "label"])
            merged[f"score_{model}"] = calibration.predict_proba(raw.reshape(-1, 1))[:, 1]
    return merged


def scores_for(frame, method, config):
    if method == "frequency":
        selected = [config[(track, tier)] for track, tier in zip(frame.track, frame.tier)]
        return np.array([row[f"score_{model}"] for model, (_, row) in zip(selected, frame.iterrows())])
    threshold = np.array([config[track] for track in frame.track]) if method == "evidence" else np.array(
        [config[(track, tier)] for track, tier in zip(frame.track, frame.tier)])
    use_knn = (frame.d_hom.to_numpy() >= threshold) & np.isfinite(frame.d_hom.to_numpy())
    return np.where(use_knn, frame.score_knn.to_numpy(), frame.score_bce.to_numpy())


def macro_metrics(frame, config, method):
    rows = []
    for exp, group in frame.groupby("experiment"):
        score = scores_for(group, method, config)
        auroc, auprc = metrics(group.label.to_numpy(), score)
        rows.append((exp, auroc, auprc))
    values = pd.DataFrame(rows, columns=["experiment", "auroc", "auprc"])
    return values.auroc.mean(), values.auprc.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-bce", required=True)
    ap.add_argument("--val-knn", required=True)
    ap.add_argument("--test-bce", required=True)
    ap.add_argument("--test-knn", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="0,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.1")
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E2", kind="evidence-router")
    config_log = vars(args).copy(); config_log["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config_log, seed=None, data_version=args.data_version,
             inputs=[args.val_bce, args.val_knn, args.test_bce, args.test_knn, args.identity],
             code_version=config_log["code_sha256"])
    identity_frame = pd.read_csv(args.identity, sep="\t", usecols=["window_id", "nn_identity"])
    identity = identity_frame.set_index("window_id").nn_identity
    paths = {"val": {"bce": args.val_bce, "knn": args.val_knn},
             "test": {"bce": args.test_bce, "knn": args.test_knn}}
    frame = load_predictions(paths, identity)
    thresholds = [float(x) for x in args.thresholds.split(",")]
    candidate_rows, configs = [], {"frequency": {}, "evidence": {}, "joint": {}}
    for track, group in frame.groupby("track"):
        for tier, subgroup in group.groupby("tier"):
            frequency_scores = {}
            for model in ("bce", "knn"):
                auroc, auprc = macro_metrics(subgroup, {(track, tier): model}, "frequency")
                frequency_scores[model] = auprc
                candidate_rows.append({"method": "frequency", "track": track, "tier": tier,
                                       "candidate": model, "val_auroc": auroc, "val_auprc": auprc})
            configs["frequency"][(track, tier)] = max(frequency_scores, key=frequency_scores.get)
            best = max(((t, *macro_metrics(subgroup, {(track, tier): t}, "joint")) for t in thresholds),
                       key=lambda x: x[2])
            configs["joint"][(track, tier)] = best[0]
            candidate_rows.append({"method": "joint", "track": track, "tier": tier,
                                   "candidate": best[0], "val_auroc": best[1], "val_auprc": best[2]})
        best = max(((t, *macro_metrics(group, {track: t}, "evidence")) for t in thresholds), key=lambda x: x[2])
        configs["evidence"][track] = best[0]
        candidate_rows.append({"method": "evidence", "track": track, "tier": "all",
                               "candidate": best[0], "val_auroc": best[1], "val_auprc": best[2]})
    pd.DataFrame(candidate_rows).to_csv(os.path.join(args.out, "validation_selection.tsv"), sep="\t", index=False)
    serializable = {method: {f"{key[0]}|{key[1]}" if isinstance(key, tuple) else key: value
                             for key, value in mapping.items()} for method, mapping in configs.items()}
    with open(os.path.join(args.out, "selected_config.json"), "w") as handle:
        json.dump({"selection": "validation only", "calibration": "per-track Platt on validation only",
                   "configs": serializable}, handle, indent=2)
    print("E2_SELECTED", serializable, flush=True)
    test = frame[frame.split == "test"].copy()
    result_rows = []
    for method in ("frequency", "evidence", "joint"):
        test["score"] = scores_for(test, method, configs[method])
        test["method"] = method
        test.to_csv(os.path.join(args.out, f"{method}_test_predictions.tsv.gz"), sep="\t", index=False,
                    compression="gzip")
        for keys, group in test.groupby(["experiment", "rbp", "tier", "fold", "track"]):
            auroc, auprc = metrics(group.label.to_numpy(), group.score.to_numpy())
            result_rows.append({"method": method, "experiment": keys[0], "rbp": keys[1], "tier": keys[2],
                                "fold": keys[3], "track": keys[4], "split": "test", "n_eval": len(group),
                                "auroc": auroc, "auprc": auprc})
    results = pd.DataFrame(result_rows)
    results.to_csv(os.path.join(args.out, "router_results.tsv"), sep="\t", index=False)
    results.groupby(["method", "track", "tier"])[["auroc", "auprc"]].mean().round(6).to_csv(
        os.path.join(args.out, "router_summary.tsv"), sep="\t")
    lg.finish(metrics={"n_test_predictions": len(test) // 3, "selected": serializable}, outputs=[args.out])
    print(results.groupby(["method", "track", "tier"])[["auroc", "auprc"]].mean().round(6).to_string())


if __name__ == "__main__":
    main()
