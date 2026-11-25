#!/usr/bin/env python3
"""复用冻结 pilot 五折预测，完成低同源三种子描述性统计与三检。"""
import argparse
import csv
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from log_run import RunLogger, sha256_file


KEYS = ["experiment", "rbp", "tier", "fold", "track", "window_id", "label"]


def atomic_json(value, path):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def atomic_csv(frame, path):
    temporary = path + ".tmp"
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def frame_signature(frame):
    hashed = pd.util.hash_pandas_object(frame[KEYS], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def source_run_ids(log_path, output_dir):
    records, malformed = {}, []
    if not log_path or not os.path.exists(log_path):
        return records, malformed
    with open(log_path) as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed.append(line_number)
                continue
            if output_dir in record.get("outputs", []):
                records[record["run_id"]] = record
    return records, malformed


def low_hom_rows(path, seed, model, identity, expected_rows, low_hom_max):
    frame = pd.read_csv(path, sep="\t")
    required = set(KEYS + ["score"])
    if required - set(frame.columns):
        raise ValueError(f"{path}: missing {sorted(required - set(frame.columns))}")
    if len(frame) != expected_rows or not np.isfinite(frame.score).all() or not frame.label.isin([0, 1]).all():
        raise ValueError(f"{path}: row count, labels or scores invalid")
    if set(frame.fold) != {0, 1, 2, 3, 4}:
        raise ValueError(f"{path}: incomplete folds")
    frame["d_hom"] = frame.window_id.map(identity)
    positive = frame.label == 1
    if frame.loc[positive, "d_hom"].isna().any():
        raise ValueError(f"{path}: missing positive identity")
    selected = frame[(~positive) | (positive & (frame.d_hom < low_hom_max))]
    rows = []
    for (experiment, rbp, tier, track), group in selected.groupby(["experiment", "rbp", "tier", "track"]):
        labels = group.label.to_numpy()
        scores = group.score.to_numpy()
        if len(np.unique(labels)) != 2 or set(group.fold) != {0, 1, 2, 3, 4}:
            raise ValueError(f"{path}: invalid pooled fivefold slice {experiment}/{track}")
        rows.append({"seed": seed, "model": model, "experiment": experiment, "rbp": rbp,
                     "tier": tier, "n_folds": group.fold.nunique(), "track": track,
                     "auroc": float(roc_auc_score(labels, scores)),
                     "auprc": float(average_precision_score(labels, scores)),
                     "n_low_hom_pos": int((group.label == 1).sum()), "n_negative": int((group.label == 0).sum())})
    result = pd.DataFrame(rows)
    if result.experiment.nunique() != 40 or len(result) != 80:
        raise ValueError(f"{path}: expected 40 experiments and 80 experiment×track rows")
    return result, frame_signature(frame), len(frame)


def recalculate(per_seed_path, multiseed_path):
    groups = {}
    with open(per_seed_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["track"], row["model"])
            groups.setdefault(key, {"auroc": [], "auprc": []})
            groups[key]["auroc"].append(float(row["auroc"]))
            groups[key]["auprc"].append(float(row["auprc"]))
    checked = 0
    with open(multiseed_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["track"], row["model"])
            for metric in ("auroc", "auprc"):
                expected = float(np.mean(groups[key][metric]))
                if not math.isclose(float(row[metric + "_mean"]), expected, rel_tol=0, abs_tol=5e-12):
                    raise ValueError(f"Low-homology recalculation mismatch: {key}/{metric}")
                checked += 1
    return checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modern", action="append", required=True, help="seed=modern_bce.tsv.gz")
    ap.add_argument("--knn", required=True)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--low-hom-max", type=float, default=0.60)
    ap.add_argument("--expected-rows", type=int, default=2419417)
    ap.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    code_hash = sha256_file(os.path.abspath(__file__))
    lg = RunLogger(args.log, stage="E1", kind="low-hom-multiseed")
    config = vars(args).copy()
    config.update({"code_sha256": code_hash, "evidence_level": "pilot fivefold test descriptive multiseed",
                   "model_selection": "none; reuse frozen BCE/kNN predictions", "test_used_for_selection": False})
    run_id = lg.start(config=config, data_version=args.data_version,
                      inputs=[value for _, value in (spec.split("=", 1) for spec in args.modern)] + [args.knn, args.identity],
                      code_version=code_hash)
    try:
        identity_frame = pd.read_csv(args.identity, sep="\t", usecols=["window_id", "nn_identity"])
        if identity_frame.window_id.duplicated().any() or not np.isfinite(identity_frame.nn_identity).all():
            raise ValueError("Identity table has duplicate IDs or non-finite values")
        identity = identity_frame.set_index("window_id").nn_identity
        frames, signatures, input_rows = [], {}, {}
        for spec in sorted(args.modern):
            seed_text, path = spec.split("=", 1)
            seed = int(seed_text)
            frame, signature, n_rows = low_hom_rows(path, seed, "bce", identity, args.expected_rows, args.low_hom_max)
            frames.append(frame)
            signatures[f"bce_s{seed}"] = signature
            input_rows[f"bce_s{seed}"] = n_rows
        knn, signature, n_rows = low_hom_rows(args.knn, 0, "knn", identity, args.expected_rows, args.low_hom_max)
        frames.append(knn)
        signatures["knn"] = signature
        input_rows["knn"] = n_rows
        if len(set(signatures.values())) != 1:
            raise ValueError(f"Prediction keys/labels are not aligned across seeds/models: {signatures}")
        per_experiment = pd.concat(frames, ignore_index=True)
        if per_experiment.duplicated(["seed", "model", "experiment", "track"]).any():
            raise ValueError("Duplicate per-experiment low-homology rows")
        per_seed = per_experiment.groupby(["seed", "model", "track"], as_index=False).agg(
            n_experiments=("experiment", "nunique"), auroc=("auroc", "mean"), auprc=("auprc", "mean"),
            n_low_hom_pos=("n_low_hom_pos", "sum"))
        multiseed = per_seed.groupby(["model", "track"])[["auroc", "auprc"]].agg(["mean", "std"])
        multiseed.columns = [f"{metric}_{stat}" for metric, stat in multiseed.columns]
        multiseed = multiseed.reset_index()
        multiseed["n_seeds"] = per_seed.groupby(["model", "track"]).seed.nunique().to_numpy()
        multiseed["n_folds"] = 5
        bce = per_experiment[per_experiment.model == "bce"].drop(columns="model")
        knn = per_experiment[per_experiment.model == "knn"].drop(columns=["model", "seed"])
        paired = bce.merge(knn, on=["experiment", "rbp", "tier", "track"],
                           suffixes=("_bce", "_knn"), validate="many_to_one")
        paired["d_auroc"] = paired.auroc_bce - paired.auroc_knn
        paired["d_auprc"] = paired.auprc_bce - paired.auprc_knn
        paired_per_seed = paired.groupby(["seed", "track"], as_index=False).agg(
            n_experiments=("experiment", "nunique"), d_auroc=("d_auroc", "mean"), d_auprc=("d_auprc", "mean"),
            d_auroc_sem=("d_auroc", "sem"), d_auprc_sem=("d_auprc", "sem"))
        paired_multi = paired_per_seed.groupby("track")[["d_auroc", "d_auprc"]].agg(["mean", "std"])
        paired_multi.columns = [f"{metric}_{stat}" for metric, stat in paired_multi.columns]
        paired_multi = paired_multi.reset_index()
        paired_multi["n_seeds"] = paired_per_seed.groupby("track").seed.nunique().to_numpy()
        atomic_csv(per_experiment, os.path.join(args.out, "low_hom_per_experiment.tsv"))
        atomic_csv(per_seed, os.path.join(args.out, "low_hom_per_seed.tsv"))
        atomic_csv(multiseed, os.path.join(args.out, "low_hom_multiseed.tsv"))
        atomic_csv(paired, os.path.join(args.out, "low_hom_paired_per_experiment.tsv"))
        atomic_csv(paired_per_seed, os.path.join(args.out, "low_hom_paired_per_seed.tsv"))
        atomic_csv(paired_multi, os.path.join(args.out, "low_hom_paired_multiseed.tsv"))
        checked = recalculate(os.path.join(args.out, "low_hom_per_seed.tsv"),
                              os.path.join(args.out, "low_hom_multiseed.tsv"))
        source_dir = os.path.dirname(args.knn)
        source_runs, malformed_log_lines = source_run_ids(args.log, source_dir)
        verification = {"run_id": run_id, "status": "complete", "evidence_level": config["evidence_level"],
                        "pass1_schema_finite": True,
                        "pass2_fold_seed_window_label_alignment": {"signature": next(iter(signatures.values())),
                                                                    "input_rows": input_rows, "passed": True},
                        "pass3_report_recalculation": {"values_checked": checked, "passed": True},
                        "source_prediction_run_ids": sorted(source_runs),
                        "source_log_malformed_lines_skipped": malformed_log_lines, "test_used_for_selection": False,
                        "significance_test": "none; descriptive mean/std across three seeds"}
        atomic_json(verification, os.path.join(args.out, "verification.json"))
        atomic_json({"run_id": run_id, "args": vars(args), "code_sha256": code_hash},
                    os.path.join(args.out, "config.json"))
        lg.finish(metrics={"n_seeds": len(args.modern), "n_experiments": 40, "n_folds": 5,
                           "n_per_experiment_rows": len(per_experiment)}, outputs=[args.out])
        print("LOW_HOM_DONE", run_id)
        print(multiseed.round(6).to_string(index=False))
        print(paired_multi.round(6).to_string(index=False))
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
