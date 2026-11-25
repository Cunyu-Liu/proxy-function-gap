#!/usr/bin/env python3
"""E1 预测结果的频率/低同源分层；不读取训练数据，不重选模型。"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from e1_locked_logger import E1RunLogger as RunLogger
from log_run import sha256_file


def metric(y, score):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, score), average_precision_score(y, score)


def markdown_table(frame):
    columns = frame.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def latest_runs(log_path):
    runs = {}
    if not log_path or not os.path.exists(log_path):
        return runs
    with open(log_path) as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for out in rec.get("outputs", []):
                runs[out] = rec
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=prediction.tsv.gz")
    ap.add_argument("--identity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--low-hom-max", type=float, default=0.60)
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E1", kind="stratified-analysis")
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=None, data_version=args.data_version,
             inputs=args.run + [args.identity], code_version=config["code_sha256"])
    identity = pd.read_csv(args.identity, sep="\t", usecols=["window_id", "nn_identity"])
    identity = identity.set_index("window_id").nn_identity
    frames = []
    run_ids = latest_runs(args.log)
    for spec in args.run:
        name, path = spec.split("=", 1)
        frame = pd.read_csv(path, sep="\t")
        frame["model"] = name
        frame["prediction_path"] = path
        frames.append(frame)
    pred = pd.concat(frames, ignore_index=True)
    pred["d_hom"] = pred.window_id.map(identity)
    pred.to_csv(os.path.join(args.out, "predictions_with_identity.tsv.gz"), sep="\t", index=False,
                compression="gzip")
    rows = []
    for (model, track, tier), group in pred.groupby(["model", "track", "tier"]):
        per_exp = []
        for exp, g in group.groupby("experiment"):
            auroc, auprc = metric(g.label.to_numpy(), g.score.to_numpy())
            per_exp.append({"experiment": exp, "auroc": auroc, "auprc": auprc})
        values = pd.DataFrame(per_exp)
        rows.append({"model": model, "track": track, "slice": tier, "n_experiments": len(values),
                     "auroc": values.auroc.mean(), "auprc": values.auprc.mean(),
                     "auroc_sem": values.auroc.sem(), "auprc_sem": values.auprc.sem()})
    for (model, track), group in pred.groupby(["model", "track"]):
        per_exp = []
        for exp, g in group.groupby("experiment"):
            selected = g[(g.label == 0) | ((g.label == 1) & (g.d_hom < args.low_hom_max))]
            n_pos = int(((g.label == 1) & (g.d_hom < args.low_hom_max)).sum())
            if n_pos == 0:
                continue
            auroc, auprc = metric(selected.label.to_numpy(), selected.score.to_numpy())
            per_exp.append({"experiment": exp, "auroc": auroc, "auprc": auprc, "n_low_hom_pos": n_pos})
        values = pd.DataFrame(per_exp)
        rows.append({"model": model, "track": track, "slice": f"low_hom<{args.low_hom_max:.2f}",
                     "n_experiments": len(values), "auroc": values.auroc.mean(), "auprc": values.auprc.mean(),
                     "auroc_sem": values.auroc.sem(), "auprc_sem": values.auprc.sem()})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "stratified_summary.tsv"), sep="\t", index=False)
    occurrence_keys = ["model", "experiment", "rbp", "tier", "fold", "track", "window_id", "label"]
    pred["occurrence"] = pred.groupby(occurrence_keys, sort=False).cumcount()
    pivot = pred.pivot_table(index=occurrence_keys[1:] + ["occurrence"],
                             columns="model", values="score").reset_index()
    pair_rows = []
    models = sorted(pred.model.unique())
    if len(models) >= 2:
        reference = models[0]
        for candidate in models[1:]:
            if candidate == reference or candidate not in pivot:
                continue
            for (track, tier), group in pivot.groupby(["track", "tier"]):
                deltas = []
                for exp, g in group.groupby("experiment"):
                    a = metric(g.label.to_numpy(), g[reference].to_numpy())
                    b = metric(g.label.to_numpy(), g[candidate].to_numpy())
                    deltas.append({"experiment": exp, "d_auroc": b[0] - a[0], "d_auprc": b[1] - a[1]})
                values = pd.DataFrame(deltas)
                pair_rows.append({"reference": reference, "candidate": candidate, "track": track, "slice": tier,
                                  "n_experiments": len(values), "d_auroc": values.d_auroc.mean(),
                                  "d_auprc": values.d_auprc.mean(), "d_auroc_sem": values.d_auroc.sem(),
                                  "d_auprc_sem": values.d_auprc.sem()})
    pd.DataFrame(pair_rows).to_csv(os.path.join(args.out, "paired_deltas.tsv"), sep="\t", index=False)
    lines = ["# E1 误差分层（描述性）", "",
             "仅汇总已生成预测；频率切片的显著性只来自预指定配对 bootstrap，低同源切片不作显著性声明。", "",
             f"低同源定义：测试正例 d_hom < {args.low_hom_max:.2f}，配同实验全部测试负例；负例不按 d_hom 子采样。", "",
             markdown_table(summary.round(6)), "", "## Run ID"]
    for spec in args.run:
        name, path = spec.split("=", 1)
        directory = os.path.dirname(path)
        rec = run_ids.get(directory, {})
        lines.append(f"- {name}: `{rec.get('run_id', '未从日志解析')}`；`{path}`")
    with open(os.path.join(args.out, "E1_stratified_report.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    lg.finish(metrics={"n_summary_rows": len(summary), "n_pair_rows": len(pair_rows)}, outputs=[args.out])
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
