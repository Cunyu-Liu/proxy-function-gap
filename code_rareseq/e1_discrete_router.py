#!/usr/bin/env python3
"""E1 Meta-Expert 式频率分层离散路由：仅由验证集选择路径，测试集不参与。"""
import argparse
import json
import os

import pandas as pd

from e1_frozen_baselines import metrics
from log_run import RunLogger, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", action="append", required=True, help="model=results.tsv")
    ap.add_argument("--prediction", action="append", required=True, help="model=test_predictions.tsv.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    result_models = {spec.split("=", 1)[0] for spec in args.result}
    prediction_models = {spec.split("=", 1)[0] for spec in args.prediction}
    if result_models != prediction_models:
        raise ValueError("Result and prediction model sets differ")
    lg = RunLogger(args.log, stage="E1", kind="discrete-router")
    config = vars(args).copy(); config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=None, data_version=args.data_version,
             inputs=args.result + args.prediction, code_version=config["code_sha256"])
    val_frames = []
    for spec in args.result:
        model, path = spec.split("=", 1)
        frame = pd.read_csv(path, sep="\t")
        frame = frame[frame.split == "val"].copy()
        frame["model"] = model
        val_frames.append(frame)
    val = pd.concat(val_frames, ignore_index=True)
    selection = val.groupby(["track", "tier", "model"])[["auroc", "auprc"]].mean().reset_index()
    selection["selected"] = False
    for (track, tier), group in selection.groupby(["track", "tier"]):
        best = group.auprc.idxmax()
        selection.loc[best, "selected"] = True
    selected = selection[selection.selected].copy()
    selected.to_csv(os.path.join(args.out, "router_selected_before_test.tsv"), sep="\t", index=False)
    selection.to_csv(os.path.join(args.out, "router_validation_candidates.tsv"), sep="\t", index=False)
    selected_map = {(r.track, r.tier): r.model for r in selected.itertuples()}
    with open(os.path.join(args.out, "router_config.json"), "w") as handle:
        json.dump({"selection_metric": "macro validation AUPRC", "selected_by_track_tier": {
            f"{track}|{tier}": model for (track, tier), model in selected_map.items()},
            "test_used_for_selection": False}, handle, indent=2)
    print("ROUTER_SELECTED", selected_map, flush=True)
    base_keys = ["experiment", "rbp", "tier", "fold", "track", "window_id", "label"]
    keys = base_keys + ["occurrence"]
    merged = None
    for spec in args.prediction:
        model, path = spec.split("=", 1)
        frame = pd.read_csv(path, sep="\t", usecols=base_keys + ["score"])
        frame["occurrence"] = frame.groupby(base_keys).cumcount()
        frame = frame[keys + ["score"]].rename(columns={"score": f"score_{model}"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=keys, validate="one_to_one")
    merged["selected_model"] = [selected_map[(track, tier)] for track, tier in zip(merged.track, merged.tier)]
    merged["score"] = [row[f"score_{model}"] for model, (_, row) in zip(merged.selected_model, merged.iterrows())]
    merged.to_csv(os.path.join(args.out, "router_test_predictions.tsv.gz"), sep="\t", index=False, compression="gzip")
    result_rows = []
    for keys_values, group in merged.groupby(["experiment", "rbp", "tier", "fold", "track"]):
        experiment, rbp, tier, fold, track = keys_values
        auroc, auprc = metrics(group.label.to_numpy(), group.score.to_numpy())
        result_rows.append({"experiment": experiment, "rbp": rbp, "tier": tier, "fold": fold,
                            "track": track, "split": "test", "selected_model": group.selected_model.iloc[0],
                            "n_eval": len(group), "auroc": auroc, "auprc": auprc})
    results = pd.DataFrame(result_rows)
    results.to_csv(os.path.join(args.out, "router_results.tsv"), sep="\t", index=False)
    results.groupby(["track", "split", "tier"])[["auroc", "auprc"]].mean().round(6).to_csv(
        os.path.join(args.out, "router_summary.tsv"), sep="\t")
    lg.finish(metrics={"n_predictions": len(merged), "selected_by_track_tier": {
        f"{track}|{tier}": model for (track, tier), model in selected_map.items()}}, outputs=[args.out])
    print("ROUTER_DONE", len(merged), flush=True)


if __name__ == "__main__":
    main()
