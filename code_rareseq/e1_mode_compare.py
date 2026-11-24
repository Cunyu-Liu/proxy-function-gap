#!/usr/bin/env python3
"""E1 同折同种子 frozen/LoRA/full 实验级配对比较。"""
import argparse
import os

import pandas as pd

from log_run import RunLogger, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="mode=results.tsv")
    ap.add_argument("--reference", default="frozen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E1", kind="mode-comparison")
    config = vars(args).copy(); config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=None, data_version=args.data_version,
             inputs=args.run, code_version=config["code_sha256"])
    keys = ["experiment", "rbp", "tier", "fold", "track", "split"]
    merged = None
    for spec in args.run:
        mode, path = spec.split("=", 1)
        frame = pd.read_csv(path, sep="\t")
        columns = keys + ["auroc", "auprc"]
        frame = frame[columns].rename(columns={"auroc": f"auroc_{mode}", "auprc": f"auprc_{mode}"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=keys, validate="one_to_one")
    modes = [spec.split("=", 1)[0] for spec in args.run]
    for mode in modes:
        if mode == args.reference:
            continue
        merged[f"d_auroc_{mode}"] = merged[f"auroc_{mode}"] - merged[f"auroc_{args.reference}"]
        merged[f"d_auprc_{mode}"] = merged[f"auprc_{mode}"] - merged[f"auprc_{args.reference}"]
    merged.to_csv(os.path.join(args.out, "mode_paired_experiments.tsv"), sep="\t", index=False)
    value_columns = [c for c in merged if c.startswith(("auroc_", "auprc_", "d_auroc_", "d_auprc_"))]
    summary = merged.groupby(["track", "split", "tier"])[value_columns].mean().reset_index()
    summary.to_csv(os.path.join(args.out, "mode_comparison.tsv"), sep="\t", index=False)
    lg.finish(metrics={"n_pairs": len(merged), "modes": modes}, outputs=[args.out])
    print(summary[summary.tier.isin(["head", "tail"])].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
