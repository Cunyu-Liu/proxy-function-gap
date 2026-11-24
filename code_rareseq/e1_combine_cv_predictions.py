#!/usr/bin/env python3
"""合并冻结五折的逐窗预测，供统一 CV 分层与配对 bootstrap。"""
import argparse
import os

import pandas as pd

from log_run import RunLogger, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    ap.add_argument("--prefix", default="pilot_", help="run 目录前缀；主集用空串")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E1", kind="cv-prediction-combine")
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=None, data_version=args.data_version, code_version=config["code_sha256"])
    folds = [int(x) for x in args.folds.split(",")]
    counts = {}
    for seed in [int(x) for x in args.seeds.split(",")]:
        frames = []
        for fold in folds:
            path = os.path.join(args.e1_root, f"{args.prefix}f{fold}_modern_bce_s{seed}",
                                "modern_test_predictions.tsv.gz")
            frames.append(pd.read_csv(path, sep="\t"))
        out = os.path.join(args.out, f"modern_bce_s{seed}.tsv.gz")
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(out, sep="\t", index=False, compression="gzip")
        counts[f"modern_s{seed}"] = len(frame)
    frames = []
    for fold in folds:
        path = os.path.join(args.e1_root, f"{args.prefix}f{fold}_knn", "knn_test_predictions.tsv.gz")
        frames.append(pd.read_csv(path, sep="\t"))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(os.path.join(args.out, "knn.tsv.gz"), sep="\t", index=False, compression="gzip")
    counts["knn"] = len(frame)
    lg.finish(metrics=counts, outputs=[args.out])
    print(counts)


if __name__ == "__main__":
    main()
