#!/usr/bin/env python3
"""E5 留出扩展三检与多种子汇总；只接受 validation 早停新协议产物。"""
import argparse
import csv
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from e1_frozen_baselines import stable_seed
from e5_rbp_holdout import TRAIN_FOLD_POLICY
from e5_holdout_retention import materialize_checkpoint
from log_run import RunLogger, sha256_file


HOLDOUTS = "HNRNPC,PTBP1,U2AF2,SRSF1,RBFOX2,QKI,IGF2BP1,LIN28B,TAF15,UPF1,CSTF2T,YBX3"
METRICS = ["auroc_ood", "fpr95", "aurc", "ece", "brier"]
RESULT_COLUMNS = {"run_id", "holdout", "track", "method", "seed", "stable_seed", "best_epoch",
                  "best_val_tail_auprc", "epochs_run", "stop_reason", "n_train", "n_validation",
                  "n_known", "n_unknown", "ood_tpr_at_known_fpr5", "metric_version", *METRICS}
HISTORY_COLUMNS = {"run_id", "holdout", "track", "seed", "stable_seed", "epoch", "train_loss",
                   "val_tail_auroc", "val_tail_auprc", "stale_epochs", "is_best"}


def atomic_json(value, path):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def atomic_csv(frame, path):
    temporary = path + ".tmp"
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def require_columns(frame, required, label):
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")


def finite_numbers(frame, label):
    numeric = frame.select_dtypes(include=np.number)
    if numeric.empty or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{label}: non-finite numeric values")


def recheck_window_scores(score_path, pair, manifest):
    with np.load(score_path, allow_pickle=False) as saved:
        if int(saved["metric_version"]) != 2 or manifest.get("metric_version") != 2:
            raise ValueError(f"{score_path}: unsupported metric version")
        y_ood = saved["y_ood"].astype(np.int8)
        y_class = saved["class_label"].astype(np.int8)
        probability = saved["probability"].astype(np.float64)
        window_id = saved["window_id"].astype(str)
        occurrence = pd.Series(window_id).groupby(window_id).cumcount().to_numpy()
        if not np.array_equal(occurrence, saved["occurrence"]):
            raise ValueError(f"{score_path}: occurrence mismatch")
        n = len(y_ood)
        if n != manifest["score_rows"] or not np.isin(y_ood, [0, 1]).all() or not np.isin(y_class, [0, 1]).all():
            raise ValueError(f"{score_path}: invalid rows/labels")
        coverage = np.arange(1, n + 1) / n
        for method in ("msp", "energy", "deep_knn"):
            score = saved[method].astype(np.float64)
            if not np.isfinite(score).all():
                raise ValueError(f"{score_path}: non-finite {method}")
            if method == "energy" and not np.array_equal(score, -np.abs(saved["logit"].astype(np.float64))):
                raise ValueError(f"{score_path}: energy direction mismatch")
            order = np.argsort(score, kind="mergesort")
            errors = ((probability >= 0.5).astype(np.int8) != y_class).astype(np.int8)
            risk = np.cumsum(errors[order]) / np.arange(1, n + 1)
            threshold = np.quantile(score[y_ood == 1], 0.05)
            legacy_threshold = np.quantile(score[y_ood == 0], 0.95)
            expected = {"auroc_ood": float(roc_auc_score(y_ood, score)),
                        "fpr95": float((score[y_ood == 0] >= threshold).mean()),
                        "ood_tpr_at_known_fpr5": float((score[y_ood == 1] >= legacy_threshold).mean()),
                        "aurc": float(np.trapezoid(risk, coverage))}
            row = pair[pair.method == method].iloc[0]
            for metric, value in expected.items():
                if not math.isclose(float(row[metric]), value, rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"{score_path}: {method}/{metric} does not match saved scores")


def verify_run(run_dir, seed, track, holdouts):
    import torch
    config_path = os.path.join(run_dir, "config.json")
    result_path = os.path.join(run_dir, "holdout_results.tsv")
    history_path = os.path.join(run_dir, "history.tsv")
    with open(config_path) as handle:
        config = json.load(handle)
    results = pd.read_csv(result_path, sep="\t")
    require_columns(results, RESULT_COLUMNS, result_path)
    if results.duplicated(["holdout", "track", "method"]).any():
        raise ValueError(f"{result_path}: duplicate keys")
    finite_numbers(results, result_path)
    if set(results.method) != {"msp", "energy", "deep_knn"}:
        raise ValueError(f"{result_path}: incomplete methods")
    if not set(results.holdout).issubset(set(holdouts)):
        raise ValueError(f"{result_path}: unexpected holdout")
    if not results.track.eq(track).all() or not results.seed.eq(seed).all():
        raise ValueError(f"{result_path}: seed/track mismatch")
    if not results.metric_version.eq(2).all():
        raise ValueError(f"{result_path}: withdrawn OOD metric implementation")
    for metric in ("auroc_ood", "fpr95", "aurc", "ece", "brier", "ood_tpr_at_known_fpr5"):
        if not results[metric].between(0, 1).all():
            raise ValueError(f"{result_path}: {metric} outside [0,1]")
    pair_counts = results.groupby(["holdout", "track"]).size()
    if not pair_counts.eq(3).all():
        raise ValueError(f"{result_path}: each pair must have exactly three methods")
    manifests, histories = {}, {}
    for holdout in sorted(results.holdout.unique()):
        prefix = f"{holdout}_{track}"
        manifest_path = os.path.join(run_dir, "manifests", prefix + ".json")
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        manifests[holdout] = manifest
        pair_path = os.path.join(run_dir, "pairs", prefix + ".tsv")
        history_path = os.path.join(run_dir, "history", prefix + ".tsv")
        checkpoint_path = os.path.join(run_dir, "checkpoints", prefix + ".pt")
        score_path = os.path.join(run_dir, "scores", prefix + ".npz")
        history = pd.read_csv(history_path, sep="\t")
        require_columns(history, HISTORY_COLUMNS, history_path)
        finite_numbers(history, history_path)
        histories[holdout] = history
        if history.epoch.tolist() != list(range(1, len(history) + 1)):
            raise ValueError(f"{history_path}: non-consecutive epochs")
        pair = pd.read_csv(pair_path, sep="\t")
        if len(pair) != 3 or set(pair.method) != {"msp", "energy", "deep_knn"}:
            raise ValueError(f"{pair_path}: invalid pair rows")
        with materialize_checkpoint(checkpoint_path, manifest["artifacts"]["checkpoint_sha256"]) as resolved_checkpoint:
            checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=True)
        for key, value in checkpoint["state_dict"].items():
            if not torch.isfinite(value).all():
                raise ValueError(f"{checkpoint_path}: non-finite state {key}")
        selection = manifest["selection"]
        if manifest.get("train_fold_policy") != TRAIN_FOLD_POLICY:
            raise ValueError(f"{manifest_path}: unassigned training positives are not excluded")
        if selection["stop_reason"] not in {"validation_early_stop", "max_epochs_failsafe"}:
            raise ValueError(f"{manifest_path}: unknown stop reason")
        if len(history) != selection["epochs_run"] or selection["best_epoch"] not in history.epoch.tolist():
            raise ValueError(f"{manifest_path}: history/selection mismatch")
        best = history.loc[history.epoch == selection["best_epoch"]].iloc[0]
        if not math.isclose(best.val_tail_auprc, selection["best_val_tail_auprc"], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"{manifest_path}: best AUPRC mismatch")
        if checkpoint["selection"] != selection or checkpoint["test_used_for_selection"]:
            raise ValueError(f"{checkpoint_path}: selection metadata mismatch")
        for name, artifact in (("pair_sha256", pair_path), ("history_sha256", history_path),
                               ("scores_sha256", score_path)):
            if manifest["artifacts"][name] != sha256_file(artifact):
                raise ValueError(f"{manifest_path}: {name} mismatch")
        recheck_window_scores(score_path, pair, manifest)
    return {"config": config, "results": results, "manifests": manifests, "histories": histories,
            "run_dir": run_dir}


def verify_alignment(run, seed, track, holdouts):
    args = run["config"]["args"]
    expected_seed = stable_seed
    if args["seed"] != seed or args["tracks"] != track or args["holdouts"].split(",") != holdouts:
        raise ValueError("Configuration seed/track/holdout mismatch")
    if args["max_epochs"] != 1000 or args["patience"] != 20 or not math.isclose(args["min_delta"], 1e-4):
        raise ValueError("Run does not use the frozen convergence configuration")
    if run["config"].get("test_used_for_selection"):
        raise ValueError("Test was marked as used for selection")
    if run["config"].get("train_fold_policy") != TRAIN_FOLD_POLICY:
        raise ValueError("Run predates the assigned-fold training policy")
    for holdout, manifest in run["manifests"].items():
        pair_seed = expected_seed(seed, {"experiment": holdout, "fold": 0, "track": track})
        if manifest["stable_seed"] != pair_seed or manifest["heldout"] != holdout or manifest["track"] != track:
            raise ValueError(f"Stable seed or identity mismatch: {holdout}/{track}")
        known = [task for task in manifest["tasks"] if task["role"] == "known"]
        held = [task for task in manifest["tasks"] if task["role"] == "heldout_test"]
        if not known or not held or any(task["rbp"] == holdout for task in known) or any(task["rbp"] != holdout for task in held):
            raise ValueError(f"RBP leakage or missing heldout task: {holdout}/{track}")
        rows = run["results"][run["results"].holdout == holdout]
        if set(rows.run_id) != {manifest["run_id"]} or set(rows.stable_seed) != {pair_seed}:
            raise ValueError(f"Run ID/seed alignment mismatch: {holdout}/{track}")
        if not rows.best_epoch.eq(manifest["selection"]["best_epoch"]).all():
            raise ValueError(f"Result/checkpoint epoch mismatch: {holdout}/{track}")


def canonical_train_alignment(manifest):
    rows = []
    for task in manifest["tasks"]:
        if task["role"] == "known":
            rows.append({key: value for key, value in task.items()
                         if (key.startswith("train_") and key != "train_unassigned_positives_removed")
                         or key in ("experiment", "track")})
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def evaluation_task_alignment(manifest):
    rows = []
    for task in manifest["tasks"]:
        row = {"experiment": task["experiment"], "role": task["role"]}
        for split in ("validation", "test"):
            for suffix in ("n", "ids_sha256", "positives"):
                key = f"{split}_{suffix}"
                row[key] = task[key]
        rows.append(row)
    return rows


def compare_pair_across_seeds(reference_manifest, current_manifest, reference_path, current_path):
    if evaluation_task_alignment(reference_manifest) != evaluation_task_alignment(current_manifest):
        raise ValueError("Cross-seed per-task validation/test alignment mismatch")
    if reference_manifest["score_rows"] != current_manifest["score_rows"]:
        raise ValueError("Cross-seed score_rows mismatch")
    with np.load(reference_path, allow_pickle=False) as reference, np.load(current_path, allow_pickle=False) as current:
        if set(reference.files) != set(current.files):
            raise ValueError("Cross-seed NPZ keys mismatch")
        for key in reference.files:
            left, right = reference[key], current[key]
            if left.shape != right.shape:
                raise ValueError(f"Cross-seed NPZ shape mismatch: {key}")
            if np.issubdtype(left.dtype, np.number) and not np.isfinite(left).all():
                raise ValueError(f"Non-finite reference NPZ array: {key}")
            if np.issubdtype(right.dtype, np.number) and not np.isfinite(right).all():
                raise ValueError(f"Non-finite current NPZ array: {key}")
        for key in ("window_id", "occurrence", "sample_role", "y_ood", "class_label"):
            if not np.array_equal(reference[key], current[key]):
                raise ValueError(f"Cross-seed NPZ identity/label mismatch: {key}")
        n = len(reference["window_id"])
        if reference_manifest["score_rows"] != n:
            raise ValueError("Manifest score_rows does not match NPZ")


def cross_seed_alignment(all_runs):
    reference = {}
    comparisons = 0
    for (seed, track), run in sorted(all_runs.items()):
        for holdout, manifest in run["manifests"].items():
            keys = ("train_alignment_sha256", "validation_alignment_sha256", "heldout_test_ids_sha256",
                    "n_train", "n_validation", "n_known_test", "n_heldout_test")
            value = {key: manifest[key] for key in keys}
            value["train_alignment_sha256"] = canonical_train_alignment(manifest)
            identity = (track, holdout)
            if identity in reference:
                reference_value, reference_manifest, reference_run = reference[identity]
                if reference_value != value:
                    raise ValueError(f"Cross-seed split/alignment mismatch: {identity}")
                prefix = f"{holdout}_{track}.npz"
                compare_pair_across_seeds(
                    reference_manifest, manifest,
                    os.path.join(reference_run["run_dir"], "scores", prefix),
                    os.path.join(run["run_dir"], "scores", prefix))
                comparisons += 1
            reference[identity] = (value, manifest, run)
    return comparisons


def summarize(all_runs):
    results = pd.concat([run["results"] for run in all_runs.values()], ignore_index=True)
    per_seed = results.groupby(["seed", "track", "method"], as_index=False)[METRICS].mean()
    multi = per_seed.groupby(["track", "method"])[METRICS].agg(["mean", "std"])
    multi.columns = [f"{metric}_{stat}" for metric, stat in multi.columns]
    multi = multi.reset_index()
    counts = per_seed.groupby(["track", "method"]).seed.nunique().rename("n_seeds").reset_index()
    multi = multi.merge(counts, on=["track", "method"], validate="one_to_one")
    convergence = results[["seed", "track", "holdout", "best_epoch", "epochs_run", "stop_reason"]].drop_duplicates()
    convergence_summary = convergence.groupby(["seed", "track"]).agg(
        n_pairs=("holdout", "nunique"), n_early_stop=("stop_reason", lambda x: int((x == "validation_early_stop").sum())),
        n_failsafe=("stop_reason", lambda x: int((x == "max_epochs_failsafe").sum())),
        min_epochs_run=("epochs_run", "min"), median_epochs_run=("epochs_run", "median"),
        max_epochs_run=("epochs_run", "max"), max_best_epoch=("best_epoch", "max")).reset_index()
    conflicts = []
    for track, group in multi.groupby("track"):
        row = {"track": track}
        winners = []
        for metric in ("auroc_ood", "fpr95", "aurc"):
            column = metric + "_mean"
            index = group[column].idxmax() if metric == "auroc_ood" else group[column].idxmin()
            row[metric + "_winner"] = multi.loc[index, "method"]
            row[metric + "_value"] = multi.loc[index, column]
            winners.append(multi.loc[index, "method"])
        row["winner_conflict"] = len(set(winners)) > 1
        conflicts.append(row)
    return results, per_seed, multi, convergence, convergence_summary, pd.DataFrame(conflicts)


def independent_recalculation(per_seed_path, multi_path):
    groups = {}
    with open(per_seed_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["track"], row["method"])
            groups.setdefault(key, {metric: [] for metric in METRICS})
            for metric in METRICS:
                groups[key][metric].append(float(row[metric]))
    expected = {}
    for key, values in groups.items():
        expected[key] = {metric: float(np.mean(numbers)) for metric, numbers in values.items()}
    with open(multi_path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["track"], row["method"])
            for metric in METRICS:
                if not math.isclose(float(row[metric + "_mean"]), expected[key][metric], rel_tol=0, abs_tol=5e-12):
                    raise ValueError(f"Report recalculation mismatch: {key}/{metric}")
    return len(expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e5/holdout_convergence")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tracks", default="neg1c,neg2")
    ap.add_argument("--holdouts", default=HOLDOUTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    code_hash = sha256_file(os.path.abspath(__file__))
    lg = RunLogger(args.log, stage="E5", kind="holdout-convergence-summary")
    config = vars(args).copy()
    config.update({"code_sha256": code_hash, "evidence_level": "pilot fold0 test; descriptive multiseed",
                   "test_used_for_selection": False})
    run_id = lg.start(config=config, data_version=args.data_version, inputs=[args.run_root], code_version=code_hash)
    try:
        seeds = [int(value) for value in args.seeds.split(",")]
        tracks = args.tracks.split(",")
        holdouts = args.holdouts.split(",")
        all_runs, missing = {}, []
        for seed in seeds:
            for track in tracks:
                run_dir = os.path.join(args.run_root, f"holdout_convergence_s{seed}", track)
                if not os.path.exists(os.path.join(run_dir, "holdout_results.tsv")):
                    missing.append({"seed": seed, "track": track, "reason": "missing results"})
                    continue
                run = verify_run(run_dir, seed, track, holdouts)
                verify_alignment(run, seed, track, holdouts)
                all_runs[seed, track] = run
                found = set(run["results"].holdout)
                for holdout in holdouts:
                    if holdout not in found:
                        missing.append({"seed": seed, "track": track, "holdout": holdout, "reason": "incomplete pair"})
        if not all_runs:
            raise ValueError("No valid runs")
        cross_seed_comparisons = cross_seed_alignment(all_runs)
        if missing and not args.allow_partial:
            raise ValueError(f"Incomplete matrix: {missing}")
        results, per_seed, multi, convergence, convergence_summary, conflicts = summarize(all_runs)
        atomic_csv(results, os.path.join(args.out, "holdout_per_rbp.tsv"))
        atomic_csv(per_seed, os.path.join(args.out, "holdout_per_seed.tsv"))
        atomic_csv(multi, os.path.join(args.out, "holdout_multiseed.tsv"))
        atomic_csv(convergence, os.path.join(args.out, "convergence_per_pair.tsv"))
        atomic_csv(convergence_summary, os.path.join(args.out, "convergence_summary.tsv"))
        atomic_csv(conflicts, os.path.join(args.out, "metric_winner_conflicts.tsv"))
        n_groups = independent_recalculation(os.path.join(args.out, "holdout_per_seed.tsv"),
                                             os.path.join(args.out, "holdout_multiseed.tsv"))
        run_ids = sorted({manifest["run_id"] for run in all_runs.values() for manifest in run["manifests"].values()})
        n_completed = sum(len(run["results"].holdout.unique()) for run in all_runs.values())
        verification = {"run_id": run_id, "status": "partial" if missing else "complete",
                        "n_pairs_completed": n_completed, "n_pairs_expected": len(seeds) * len(tracks) * len(holdouts),
                        "train_fold_policy": TRAIN_FOLD_POLICY,
                        "evidence_level": "pilot fold0 test descriptive; not a final conclusion" if missing else "pilot fold0 test descriptive multiseed",
                        "pass1_schema_finite_checkpoint": True,
                        "pass2_config_split_seed_alignment": True,
                        "cross_seed_window_npz_alignment": {
                            "comparisons": cross_seed_comparisons,
                            "per_task_validation_test_ids_and_positives": True,
                            "npz_keys_window_occurrence_roles_labels_shapes_finite": True,
                            "model_scores_compared_for_equality": False},
                        "pass3_report_recalculation": {"groups": n_groups, "passed": True},
                        "missing": missing, "source_run_ids": run_ids,
                        "n_source_runs": len(run_ids), "test_used_for_selection": False,
                        "input_digest": hashlib.sha256(json.dumps(run_ids).encode()).hexdigest()}
        atomic_json(verification, os.path.join(args.out, "verification.json"))
        atomic_json({"run_id": run_id, "args": vars(args), "code_sha256": code_hash},
                    os.path.join(args.out, "config.json"))
        lg.finish(metrics={"status": verification["status"], "n_rows": len(results),
                           "n_source_runs": len(run_ids), "n_missing": len(missing)}, outputs=[args.out])
        print("SUMMARY_STATUS", verification["status"], "runs", len(run_ids), "missing", len(missing))
        print(multi.round(6).to_string(index=False))
        print(conflicts.round(6).to_string(index=False))
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
