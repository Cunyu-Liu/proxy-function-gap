#!/usr/bin/env python3
"""汇总 E1 多折多种子结果；保留逐折配对差值，不重选配置。"""
import argparse
import os

import numpy as np
import pandas as pd

from log_run import RunLogger, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-root", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E1", kind="cv-summary")
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=None, data_version=args.data_version, code_version=config["code_sha256"])
    modern_frames, knn_frames = [], []
    for fold in [int(x) for x in args.folds.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            path = os.path.join(args.e1_root, f"pilot_f{fold}_modern_bce_s{seed}", "modern_results.tsv")
            frame = pd.read_csv(path, sep="\t")
            frame["seed"] = seed
            modern_frames.append(frame)
        path = os.path.join(args.e1_root, f"pilot_f{fold}_knn", "knn_results.tsv")
        knn_frames.append(pd.read_csv(path, sep="\t"))
    modern = pd.concat(modern_frames, ignore_index=True)
    knn = pd.concat(knn_frames, ignore_index=True)
    keys = ["experiment", "rbp", "tier", "fold", "track", "split"]
    paired = modern.merge(knn, on=keys, suffixes=("_modern", "_knn"), validate="many_to_one")
    paired["d_auroc"] = paired.auroc_modern - paired.auroc_knn
    paired["d_auprc"] = paired.auprc_modern - paired.auprc_knn
    metric_rows = []
    for group_keys, group in paired.groupby(["track", "split", "tier"]):
        row = dict(zip(["track", "split", "tier"], group_keys))
        for column in ("auroc_modern", "auprc_modern", "auroc_knn", "auprc_knn", "d_auroc", "d_auprc"):
            row[column + "_mean"] = group[column].mean()
            row[column + "_std"] = group[column].std()
        row["n_pairs"] = len(group)
        row["n_folds"] = group.fold.nunique()
        row["n_seeds"] = group.seed.nunique()
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(args.out, "cv_metrics.tsv"), sep="\t", index=False)
    by_fold = paired.groupby(["fold", "track", "split", "tier"])[
        ["auroc_modern", "auprc_modern", "auroc_knn", "auprc_knn", "d_auroc", "d_auprc"]].mean().reset_index()
    by_fold.to_csv(os.path.join(args.out, "cv_by_fold.tsv"), sep="\t", index=False)
    paired.to_csv(os.path.join(args.out, "cv_paired_experiments.tsv"), sep="\t", index=False)
    lg.finish(metrics={"n_pairs": len(paired), "n_folds": paired.fold.nunique(),
                       "n_seeds": paired.seed.nunique()}, outputs=[args.out])
    test = by_fold[by_fold.split == "test"]
    print(test[test.tier.isin(["head", "tail"])].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
