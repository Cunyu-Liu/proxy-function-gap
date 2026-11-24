#!/usr/bin/env python3
"""E4 跨系天然阳性候选召回：val 校准，test 候选上计算 FNR。"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from log_run import RunLogger, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=modern_output_dir")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E4", kind="candidate-recall")
    config = vars(args).copy(); config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=None, data_version=args.data_version,
             inputs=args.run + [args.candidates], code_version=config["code_sha256"])
    candidates = pd.read_csv(args.candidates, sep="\t")
    base_keys = ["experiment", "window_id", "track", "fold", "label"]
    candidates["occurrence"] = candidates.groupby(base_keys).cumcount()
    totals, per_experiment = [], []
    for spec in args.run:
        name, directory = spec.split("=", 1)
        val = pd.read_csv(os.path.join(directory, "modern_val_predictions.tsv.gz"), sep="\t")
        test = pd.read_csv(os.path.join(directory, "modern_test_predictions.tsv.gz"), sep="\t")
        test["occurrence"] = test.groupby(base_keys).cumcount()
        for track, val_group in val.groupby("track"):
            calibration = LogisticRegression(C=1.0, max_iter=1000)
            calibration.fit(val_group.score.to_numpy().reshape(-1, 1), val_group.label)
            test_group = test[test.track == track].copy()
            test_group["calibrated"] = calibration.predict_proba(test_group.score.to_numpy().reshape(-1, 1))[:, 1]
            candidate_group = candidates[candidates.track == track]
            joined = candidate_group.merge(test_group[["experiment", "window_id", "track", "fold", "label",
                                                        "occurrence", "calibrated", "rbp", "tier"]],
                                           on=base_keys + ["occurrence"], validate="one_to_one")
            joined["model"] = name
            joined.to_csv(os.path.join(args.out, f"candidate_scores_{name}_{track}.tsv.gz"),
                          sep="\t", index=False, compression="gzip")
            for experiment, group in joined.groupby("experiment"):
                per_experiment.append({"model": name, "track": track, "experiment": experiment,
                                       "n_candidates": len(group), "recall": (group.calibrated >= 0.5).mean(),
                                       "fnr": (group.calibrated < 0.5).mean(), "mean_score": group.calibrated.mean()})
            totals.append({"model": name, "track": track, "n_candidates": len(joined),
                           "recall": (joined.calibrated >= 0.5).mean(), "fnr": (joined.calibrated < 0.5).mean(),
                           "mean_score": joined.calibrated.mean()})
    totals = pd.DataFrame(totals)
    per_experiment = pd.DataFrame(per_experiment)
    totals.to_csv(os.path.join(args.out, "candidate_metrics.tsv"), sep="\t", index=False)
    per_experiment.to_csv(os.path.join(args.out, "candidate_by_experiment.tsv"), sep="\t", index=False)
    lg.finish(metrics={"n_rows": len(totals), "results": totals.to_dict(orient="records")}, outputs=[args.out])
    print(totals.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
