#!/usr/bin/env python3
"""Independent aggregate verification for the completed RNAcompete five-fold matrix."""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from rnacompete_ray2013_tasklog import atomic_json


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_metrics(root, track, arch, fold, seed):
    if arch == "knn":
        path = os.path.join(root, "knn", f"{track}_f{fold}", "test_metrics.tsv")
    else:
        path = os.path.join(root, "runs", f"{track}_{arch}_f{fold}_s{seed}", "test_metrics.tsv")
    frame = pd.read_csv(path, sep="\t")
    if frame.split.nunique() != 1 or frame.split.iloc[0] != "test":
        raise ValueError(f"Non-test metric input: {path}")
    return frame


def summarize_runs(root):
    run_rows = []
    for track in ("sequence", "rbp_ood"):
        for arch in ("linear", "mlp", "knn"):
            for fold in range(5):
                for seed in ((None,) if arch == "knn" else (0, 1, 2)):
                    frame = run_metrics(root, track, arch, fold, seed)
                    tiers = ("Overall", "head", "mid", "tail") if track == "sequence" else ("Overall",)
                    for probe_set in ("All", "SetA", "SetB"):
                        selected = frame[frame.probe_set == probe_set]
                        for tier in tiers:
                            subset = selected if tier == "Overall" else selected[selected.frequency_tier == tier]
                            row = {"track": track, "arch": arch, "fold": fold, "seed": seed,
                                   "probe_set": probe_set, "tier": tier,
                                   "n_experiments": len(subset),
                                   "spearman": subset.spearman.mean(), "pearson": subset.pearson.mean(),
                                   "mse": subset.mse.mean()}
                            if track == "rbp_ood":
                                groups = subset.groupby("species_gene")[["spearman", "pearson", "mse"]].mean()
                                row.update({"group_spearman": groups.spearman.mean(),
                                            "group_pearson": groups.pearson.mean(),
                                            "group_mse": groups.mse.mean(), "n_groups": len(groups)})
                            run_rows.append(row)
    runs = pd.DataFrame(run_rows)
    output = []
    group_columns = ["track", "arch", "probe_set", "tier"]
    for keys, frame in runs.groupby(group_columns):
        row = dict(zip(group_columns, keys))
        row["n_runs"] = len(frame)
        if keys[0] == "sequence":
            if (frame.n_experiments != frame.n_experiments.iloc[0]).any():
                raise ValueError(f"Experiment support changed across runs: {keys}")
            row["n_experiments"] = int(frame.n_experiments.iloc[0])
        else:
            row["n_experiments_min"] = int(frame.n_experiments.min())
            row["n_experiments_max"] = int(frame.n_experiments.max())
        for metric in ("spearman", "pearson", "mse"):
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_sd"] = frame[metric].std(ddof=1)
        if keys[0] == "rbp_ood":
            row["n_groups_min"] = int(frame.n_groups.min())
            row["n_groups_max"] = int(frame.n_groups.max())
            for metric in ("group_spearman", "group_pearson", "group_mse"):
                row[f"{metric}_mean"] = frame[metric].mean()
                row[f"{metric}_sd"] = frame[metric].std(ddof=1)
        output.append(row)
    return runs, pd.DataFrame(output).sort_values(group_columns).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--report-pointer", required=True)
    args = parser.parse_args()
    gate_path = os.path.join(args.root, "final_gate.json")
    with open(gate_path) as handle:
        gate = json.load(handle)
    if gate["status"] != "passed" or gate["scope"] != "five_fold_three_seed":
        raise ValueError("Final five-fold gate is not accepted")
    if gate["n_evaluations"] != 70 or len(gate["runs"]) != 70:
        raise ValueError("Expected 60 neural and 10 kNN evaluations")
    if gate["selection_audit"]["test_used_for_selection"] is not False:
        raise ValueError("Test contact in selection")
    if not all(record["replay_passed"] and record.get("checkpoint_replay_passed", True)
               for record in gate["runs"]):
        raise ValueError("Prediction or checkpoint replay failure")
    neural = [record for record in gate["runs"] if "checkpoint_replay_passed" in record]
    if len(neural) != 60:
        raise ValueError("Expected 60 checkpoint replays")
    runs, summary = summarize_runs(args.root)
    if len(runs) != 525 or len(summary) != 45:
        raise ValueError(f"Unexpected aggregate shapes: {runs.shape} {summary.shape}")
    for track, expected in (("sequence", 15), ("rbp_ood", 15)):
        neural_runs = summary[(summary.track == track) & (summary.arch != "knn")].n_runs.unique()
        knn_runs = summary[(summary.track == track) & (summary.arch == "knn")].n_runs.unique()
        if not np.array_equal(neural_runs, [expected]) or not np.array_equal(knn_runs, [5]):
            raise ValueError(f"Wrong run support for {track}: {neural_runs} {knn_runs}")
    run_path = os.path.join(args.root, "final_independent_run_metrics.tsv.gz")
    summary_path = os.path.join(args.root, "final_independent_summary.tsv")
    verification_path = os.path.join(args.root, "final_independent_verification.json")
    runs.to_csv(run_path, sep="\t", index=False, compression="gzip")
    summary.to_csv(summary_path, sep="\t", index=False, float_format="%.10g")
    result = {"status": "passed", "scope": "independent_five_fold_three_seed_aggregate",
              "n_evaluations": 70, "n_neural_checkpoint_replays": 60,
              "n_run_metric_rows": len(runs), "n_summary_rows": len(summary),
              "selection_test_used": False, "final_gate_sha256": sha256_file(gate_path),
              "run_metrics_sha256": sha256_file(run_path), "summary_sha256": sha256_file(summary_path),
              "report_pointer": args.report_pointer,
              "evidence_boundary": "Five-fold descriptive benchmark; not SOTA or mechanistic causal evidence"}
    atomic_json(verification_path, result)
    print("RNACOMPETE_FINAL_INDEPENDENT_VERIFICATION_OK", json.dumps(result))


if __name__ == "__main__":
    main()
