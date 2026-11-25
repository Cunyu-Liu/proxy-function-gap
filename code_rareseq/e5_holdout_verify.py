#!/usr/bin/env python3
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from e5_holdout_ext_summary import cross_seed_alignment, verify_alignment, verify_run
from e5_rbp_holdout import METRIC_VERSION, TRAIN_FOLD_POLICY


def manual_quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def manual_metrics(y_ood, y_class, probability, score):
    n_known = int((y_ood == 0).sum())
    n_unknown = int((y_ood == 1).sum())
    ranks = rankdata(score, method="average")
    auroc = (ranks[y_ood == 1].sum() - n_unknown * (n_unknown + 1) / 2) / (n_known * n_unknown)
    threshold = manual_quantile(score[y_ood == 1], 0.05)
    fpr = float(np.mean(score[y_ood == 0] >= threshold))
    order = np.argsort(score, kind="mergesort")
    errors = ((probability >= 0.5).astype(np.int8) != y_class).astype(np.int8)
    cumulative = 0
    previous_risk = None
    area = 0.0
    n = len(score)
    for index, position in enumerate(order, 1):
        cumulative += int(errors[position])
        risk = cumulative / index
        if previous_risk is not None:
            area += (previous_risk + risk) / (2 * n)
        previous_risk = risk
    return {"auroc_ood": float(auroc), "fpr95": fpr, "aurc": float(area)}


def replay_history(history, selection):
    best = -np.inf
    stale = 0
    best_epoch = 0
    for row in history.itertuples(index=False):
        if row.val_tail_auprc > best + selection["min_delta"]:
            best = row.val_tail_auprc
            best_epoch = int(row.epoch)
            stale = 0
        else:
            stale += 1
        if stale != int(row.stale_epochs) or bool(row.is_best) != (stale == 0):
            raise ValueError("History early-stop replay mismatch")
    if best_epoch != selection["best_epoch"] or not math.isclose(best, selection["best_val_tail_auprc"], abs_tol=1e-12):
        raise ValueError("History best-checkpoint replay mismatch")
    if selection["stop_reason"] == "validation_early_stop" and stale != selection["patience"]:
        raise ValueError("Early stop did not exhaust patience")
    if len(history) != selection["epochs_run"]:
        raise ValueError("History length mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--summary-dir", required=True)
    args = parser.parse_args()
    with open(os.path.join(args.summary_dir, "config.json")) as handle:
        summary_config = json.load(handle)["args"]
    seeds = [int(value) for value in summary_config["seeds"].split(",")]
    tracks = summary_config["tracks"].split(",")
    holdouts = summary_config["holdouts"].split(",")
    runs = {}
    rows = []
    for seed in seeds:
        for track in tracks:
            run_dir = os.path.join(args.run_root, f"holdout_convergence_s{seed}", track)
            if not os.path.exists(os.path.join(run_dir, "holdout_results.tsv")):
                continue
            run = verify_run(run_dir, seed, track, holdouts)
            verify_alignment(run, seed, track, holdouts)
            runs[seed, track] = run
            for holdout, manifest in run["manifests"].items():
                replay_history(run["histories"][holdout], manifest["selection"])
                prefix = f"{holdout}_{track}"
                pair_path = os.path.join(run_dir, "pairs", prefix + ".tsv")
                pair = pd.read_csv(pair_path, sep="\t")
                score_path = os.path.join(run_dir, "scores", prefix + ".npz")
                with np.load(score_path, allow_pickle=False) as saved:
                    if int(saved["metric_version"]) != METRIC_VERSION or manifest["train_fold_policy"] != TRAIN_FOLD_POLICY:
                        raise ValueError("Invalid metric or training-fold version")
                    y_ood = saved["y_ood"].astype(np.int8)
                    y_class = saved["class_label"].astype(np.int8)
                    probability = saved["probability"].astype(np.float64)
                    for method in ("msp", "energy", "deep_knn"):
                        score = saved[method].astype(np.float64)
                        direct = {"auroc_ood": float(roc_auc_score(y_ood, score)),
                                  "fpr95": float(np.mean(score[y_ood == 0] >= np.quantile(score[y_ood == 1], 0.05)))}
                        order = np.argsort(score, kind="mergesort")
                        errors = ((probability >= 0.5).astype(np.int8) != y_class).astype(np.int8)
                        risk = np.cumsum(errors[order]) / np.arange(1, len(score) + 1)
                        direct["aurc"] = float(np.trapezoid(risk, np.arange(1, len(score) + 1) / len(score)))
                        manual = manual_metrics(y_ood, y_class, probability, score)
                        saved_row = pair[pair.method == method].iloc[0]
                        for metric in direct:
                            if not (math.isclose(direct[metric], manual[metric], rel_tol=0, abs_tol=1e-12) and
                                    math.isclose(direct[metric], float(saved_row[metric]), rel_tol=0, abs_tol=1e-10)):
                                raise ValueError(f"Three-pass mismatch: {seed}/{track}/{holdout}/{method}/{metric}")
                        rows.append({"seed": seed, "track": track, "holdout": holdout, "method": method,
                                     **direct, "manual_passed": True})
    cross_seed_alignment(runs)
    expected = len(seeds) * len(tracks) * len(holdouts)
    completed = len(rows) // 3
    result = {"status": "partial" if completed < expected else "complete",
              "n_pairs_completed": completed, "n_pairs_expected": expected,
              "pass1_schema_hashes_checkpoint": True,
              "pass2_independent_vector_recalculation": True,
              "pass3_independent_rank_loop_and_history_replay": True,
              "train_fold_policy": TRAIN_FOLD_POLICY, "metric_version": METRIC_VERSION}
    temporary = os.path.join(args.summary_dir, "independent_verification.json.tmp")
    with open(temporary, "w") as handle:
        json.dump(result, handle, indent=2)
    os.replace(temporary, os.path.join(args.summary_dir, "independent_verification.json"))
    pd.DataFrame(rows).to_csv(os.path.join(args.summary_dir, "independent_window_recalculation.tsv"), sep="\t", index=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
