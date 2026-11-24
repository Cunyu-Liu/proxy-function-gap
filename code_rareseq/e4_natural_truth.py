#!/usr/bin/env python3
"""E4 天然真值：源系负例位点在目标系同 RBP 为正例的候选集合。"""
import argparse
import bisect
import os

import pandas as pd

from log_run import RunLogger, sha256_file


def key(window_id):
    return "|".join(window_id.split("|")[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--exps-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E4", kind="natural-truth-audit")
    config = vars(args).copy(); config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    config["match_definition"] = "source negative 101nt window contains target-cell same-RBP crosslink; exact-center match was zero"
    lg.start(config=config, seed=None, data_version=args.data_version,
             inputs=[args.exps_file], code_version=config["code_sha256"])
    exps = [x.strip() for x in open(args.exps_file) if x.strip()]
    positive = {}
    for name in exps:
        frame = pd.read_csv(os.path.join(args.windows_dir, f"{name}.windows.tsv"), sep="\t",
                            usecols=["window_id", "chrom", "start", "end", "strand"])
        positions = {}
        for row in frame.itertuples():
            crosslink = int(row.window_id.split("|")[2])
            positions.setdefault((row.chrom, row.strand), []).append(crosslink)
        positive[name] = {key: sorted(values) for key, values in positions.items()}
    candidates, summary = [], []
    for source in exps:
        rbp, cell = source.rsplit("_", 1)
        target = rbp + ("_HepG2" if cell == "K562" else "_K562")
        target_positions = positive[target]
        neg = pd.read_csv(os.path.join(args.neg_dir, f"{source}.negatives.tsv"), sep="\t")
        for track, group in neg.groupby("track"):
            def contains_target(row):
                positions = target_positions.get((row.chrom, row.strand), [])
                i = bisect.bisect_left(positions, row.start)
                return i < len(positions) and positions[i] < row.end
            matched = group[group.apply(contains_target, axis=1)].copy()
            matched["experiment"] = source
            matched["label"] = 0
            matched["target_experiment"] = target
            candidates.append(matched)
            summary.append({"experiment": source, "target_experiment": target, "track": track,
                            "n_negative": len(group), "n_candidates": len(matched),
                            "candidate_fraction": len(matched) / max(len(group), 1)})
    candidate = pd.concat(candidates, ignore_index=True)
    candidate.to_csv(os.path.join(args.out, "natural_truth_candidates.tsv"), sep="\t", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out, "natural_truth_summary.tsv"), sep="\t", index=False)
    metrics = summary.groupby("track")[["n_negative", "n_candidates"]].sum()
    metrics["candidate_fraction"] = metrics.n_candidates / metrics.n_negative
    metrics.to_csv(os.path.join(args.out, "natural_truth_totals.tsv"), sep="\t")
    lg.finish(metrics=metrics.to_dict(), outputs=[args.out])
    print(metrics.to_string())


if __name__ == "__main__":
    main()
