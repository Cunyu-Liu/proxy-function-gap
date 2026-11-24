#!/usr/bin/env python3
"""Validation-only M1a/M1b gates; this program never reads test predictions."""
import argparse
import json
import os

import numpy as np
import pandas as pd

from m1_evaluate import read_predictions


def validation_table(configs):
    rows = []
    for name, path in configs.items():
        val = read_predictions(path, "val")
        for (track, seed), part in val.groupby(["track", "seed"]):
            tail = part[part.tier == "tail"]
            for experiment, group in tail.groupby("experiment"):
                if group.label.nunique() == 2:
                    from sklearn.metrics import average_precision_score, roc_auc_score
                    rows.append({"config": name, "track": track, "seed": int(seed),
                                 "experiment": experiment,
                                 "auprc": average_precision_score(group.label, group.score),
                                 "auroc": roc_auc_score(group.label, group.score)})
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No evaluable tail-RBP validation rows")
    return result


def compare(table, candidate, reference):
    keys = ["track", "seed", "experiment"]
    a = table[table.config == candidate].drop(columns="config")
    b = table[table.config == reference].drop(columns="config")
    joined = a.merge(b, on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one")
    if len(joined) != len(a) or len(joined) != len(b):
        raise ValueError(f"Incomplete validation pairing: {candidate} vs {reference}")
    joined["delta_auprc"] = joined.auprc_candidate - joined.auprc_reference
    joined["delta_auroc"] = joined.auroc_candidate - joined.auroc_reference
    return joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["m1a", "m1b"])
    ap.add_argument("--config", action="append", required=True, help="name=run-directory")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    configs = {}
    for item in args.config:
        name, sep, path = item.partition("=")
        if not sep or not name or name in configs or not os.path.isdir(path):
            raise ValueError(f"Invalid or duplicate --config: {item}")
        configs[name] = path
    for name in [args.reference] + args.candidate:
        if name not in configs:
            raise ValueError(f"Missing configuration: {name}")
    os.makedirs(args.out, exist_ok=False)
    table = validation_table(configs)
    table.to_csv(os.path.join(args.out, "validation_per_experiment.tsv"), sep="\t", index=False)
    comparisons, decisions = [], []
    for candidate in args.candidate:
        delta = compare(table, candidate, args.reference)
        delta["candidate"] = candidate
        comparisons.append(delta)
        for track, part in delta.groupby("track"):
            per_seed = part.groupby("seed")[["delta_auprc", "delta_auroc"]].mean()
            mean_delta = float(per_seed.delta_auprc.mean())
            if args.stage == "m1a":
                passed = mean_delta >= -0.02 and len(per_seed) == 3
                criterion = "three seeds; mean validation tail delta AUPRC >= -0.02"
            else:
                passed = (len(per_seed) == 3 and (per_seed.delta_auprc >= 0).all()
                          and mean_delta >= 0.005)
                criterion = "three seeds; every seed >=0 and mean validation tail delta AUPRC >=0.005"
            decisions.append({"stage": args.stage, "candidate": candidate, "reference": args.reference,
                              "track": track, "n_seeds": len(per_seed),
                              "mean_delta_auprc": mean_delta,
                              "per_seed_delta_auprc": {str(k): float(v) for k, v in per_seed.delta_auprc.items()},
                              "passed": bool(passed), "criterion": criterion,
                              "test_accessed": False})
    pd.concat(comparisons).to_csv(os.path.join(args.out, "validation_paired_deltas.tsv"), sep="\t", index=False)
    selected = {}
    for track in {row["track"] for row in decisions}:
        eligible = [row for row in decisions if row["track"] == track]
        selected[track] = max(eligible, key=lambda row: row["mean_delta_auprc"])["candidate"]
    gated = [row for row in decisions if args.stage == "m1a" or selected[row["track"]] == row["candidate"]]
    gate_pass = bool(gated) and all(row["passed"] for row in gated)
    with open(os.path.join(args.out, "gate_decision.json"), "w") as handle:
        json.dump({"stage": args.stage, "passed": gate_pass, "decisions": decisions,
                   "selected_by_validation": selected, "next_stage_allowed": gate_pass,
                   "evidence_level": "validation-only pilot fold0"},
           handle, indent=2)
    print(json.dumps({"stage": args.stage, "passed": gate_pass, "test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
