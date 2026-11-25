#!/usr/bin/env python3
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from e1_frozen_baselines import stable_seed


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint verification")
    checks = {"pass1_files_schema_finite": {}, "pass2_alignment": {},
              "pass3_metric_recalculation": {}}
    all_results = []
    for seed in range(3):
        directory = os.path.join(args.base, f"seed{seed}")
        with open(os.path.join(directory, "run_config.json")) as handle:
            config_record = json.load(handle)
        config = config_record["config"]
        results = pd.read_csv(os.path.join(directory, "cross_protocol_results.tsv"), sep="\t")
        history_files = [f for f in os.listdir(directory) if f.endswith("_valhist.tsv")]
        checkpoint_files = os.listdir(os.path.join(directory, "checkpoints"))
        score_files = os.listdir(os.path.join(directory, "scores"))
        assert len(results) == 27
        assert len(history_files) == len(checkpoint_files) == 18
        assert len(score_files) == 27
        assert results.rbp.nunique() == 9
        assert set(results.eval_protocol) == {
            "eCLIP-neg1c-fold0", "eCLIP-neg2-fold0", "PARCLIP-otherRBP-geneMatched-excl300"}
        numeric = results.select_dtypes(include=[np.number])
        assert np.isfinite(numeric.to_numpy()).all()
        assert ((results.auroc >= 0) & (results.auroc <= 1)).all()
        assert ((results.auprc >= 0) & (results.auprc <= 1)).all()
        assert ((results.positive_prevalence > 0) & (results.positive_prevalence < 1)).all()
        checks["pass1_files_schema_finite"][f"seed{seed}"] = {
            "rows": len(results), "histories": len(history_files),
            "checkpoints": len(checkpoint_files), "score_files": len(score_files),
            "all_finite": True}

        assert config["seed"] == seed
        assert config["train_folds"] == "2,3,4" and config["val_fold"] == 1
        assert config["neg_fold"] == 0 and config["patience"] == args.patience
        assert config["min_delta"] == args.min_delta
        assert config["epochs"] == args.epochs and config["device"] == "cuda:0"
        with open(os.path.join(directory, "log_preflight.json")) as handle:
            log_preflight = json.load(handle)
        assert log_preflight["line_111_preexisting_malformed"]
        assert 111 in log_preflight["preexisting_malformed_lines"]
        assert log_preflight["new_write_mode"] == "fcntl.flock + O_APPEND + single os.write + fsync"
        assert config_record["overlap_rbps"] == sorted(results.rbp.unique())
        split = pd.read_csv(os.path.join(directory, "split_audit.tsv"), sep="\t")
        assert len(split) == 18
        for row in split.itertuples(index=False):
            expected_seed = stable_seed(seed, {"experiment": row.rbp, "fold": 0, "track": row.track})
            assert row.seed == expected_seed
            history = pd.read_csv(
                os.path.join(directory, f"{row.rbp}_{row.track}_valhist.tsv"), sep="\t")
            assert np.isfinite(history[["train_loss", "val_auroc", "val_auprc"]].to_numpy()).all()
            assert len(history) == row.stopped_epoch
            accepted_metric = -np.inf
            accepted_epoch = 0
            for historical in history.itertuples(index=False):
                if historical.val_auprc > accepted_metric + args.min_delta:
                    accepted_metric = historical.val_auprc
                    accepted_epoch = historical.epoch
            assert row.best_epoch == accepted_epoch
            assert abs(row.best_val_auprc - accepted_metric) < 1e-15
            assert row.stopped_epoch - row.best_epoch == args.patience
            assert row.stopped_epoch < args.epochs
            checkpoint = torch.load(os.path.join(
                directory, "checkpoints", f"{row.rbp}_{row.track}_best.pt"),
                map_location="cuda:0", weights_only=False)
            assert checkpoint["seed"] == expected_seed and checkpoint["base_seed"] == seed
            assert checkpoint["train_folds"] == [2, 3, 4]
            assert checkpoint["val_fold"] == 1 and checkpoint["neg_fold"] == 0
            assert checkpoint["best_epoch"] == row.best_epoch
            assert checkpoint["stopped_epoch"] == row.stopped_epoch
            assert np.isfinite(checkpoint["mean"]).all() and np.isfinite(checkpoint["std"]).all()
            assert (checkpoint["std"] > 0).all()
            assert all(value.is_cuda and torch.isfinite(value).all()
                       for value in checkpoint["model_state"].values())
            del checkpoint
        torch.cuda.empty_cache()
        checks["pass2_alignment"][f"seed{seed}"] = {
            "run_id": config_record["run_id"], "split_seed_history_checkpoint_aligned": True,
            "patience": args.patience, "min_delta": args.min_delta,
            "epoch_failsafe": args.epochs, "all_stopped_before_failsafe": True,
            "preexisting_log_line_111_recorded": True}

        max_auroc_error = 0.0
        max_auprc_error = 0.0
        for row in results.itertuples(index=False):
            path = os.path.join(directory, "scores", f"{row.rbp}_{row.train_track}_{row.eval_protocol}.tsv.gz")
            scores = pd.read_csv(path, sep="\t")
            assert len(scores) == row.n_parclip_pos + row.n_neg
            assert scores.loc[scores.y == 0, "window_id"].nunique() == row.n_unique_neg
            assert scores.loc[scores.y == 1, "window_id"].nunique() == row.n_parclip_pos
            assert np.isfinite(scores.score).all()
            assert scores.y.sum() == row.n_parclip_pos
            auroc = roc_auc_score(scores.y, scores.score)
            auprc = average_precision_score(scores.y, scores.score)
            max_auroc_error = max(max_auroc_error, abs(auroc - row.auroc))
            max_auprc_error = max(max_auprc_error, abs(auprc - row.auprc))
            assert abs(auroc - row.auroc) < 1e-12
            assert abs(auprc - row.auprc) < 1e-12
            assert abs(scores.y.mean() - row.positive_prevalence) < 1e-12
        macro = results.groupby("eval_protocol")[["auroc", "auprc"]].mean()
        written_macro = pd.read_csv(os.path.join(directory, "cross_protocol_macro.tsv"), sep="\t", index_col=0)
        assert np.allclose(macro, written_macro, rtol=0, atol=1e-15)
        checks["pass3_metric_recalculation"][f"seed{seed}"] = {
            "max_abs_auroc_error": max_auroc_error, "max_abs_auprc_error": max_auprc_error,
            "macro_recalculated": True}
        results.insert(0, "seed", seed)
        all_results.append(results)

    matching_hashes = []
    for seed in range(3):
        matching_hashes.append(sha256_file(os.path.join(
            args.base, f"seed{seed}", "parclip_neg2_matching.tsv")))
    assert len(set(matching_hashes)) == 1
    checks["pass2_alignment"]["fixed_external_matching_across_seeds"] = True

    combined = pd.concat(all_results, ignore_index=True)
    summary = combined.groupby(["eval_protocol", "seed"])[["auroc", "auprc"]].mean().reset_index()
    summary.to_csv(os.path.join(args.base, "three_seed_macro_by_seed.tsv"), sep="\t", index=False)
    aggregate = summary.groupby("eval_protocol")[["auroc", "auprc"]].agg(["mean", "std", "min", "max"])
    aggregate.to_csv(os.path.join(args.base, "three_seed_macro_summary.tsv"), sep="\t")
    per_rbp = combined.groupby(["eval_protocol", "rbp"])[["auroc", "auprc"]].agg(["mean", "std"])
    per_rbp.to_csv(os.path.join(args.base, "three_seed_per_rbp_summary.tsv"), sep="\t")

    old = pd.read_csv(os.path.join(args.old, "cross_protocol_results.tsv"), sep="\t")
    new0 = all_results[0][all_results[0].eval_protocol.str.startswith("eCLIP")].copy()
    new0["track"] = new0.train_track
    comparison = old.merge(new0, on=["rbp", "track"], suffixes=("_old", "_new"))
    assert len(comparison) == 18
    assert (comparison.n_eclip_neg == comparison.n_neg).all()
    comparison["auroc_abs_diff"] = (comparison.auroc_old - comparison.auroc_new).abs()
    comparison["auprc_abs_diff"] = (comparison.auprc_old - comparison.auprc_new).abs()
    comparison.to_csv(os.path.join(args.base, "old_vs_earlystop_seed0.tsv"), sep="\t", index=False)
    checks["old_seed0_comparison"] = {
        "max_abs_auroc_difference": float(comparison.auroc_abs_diff.max()),
        "max_abs_auprc_difference": float(comparison.auprc_abs_diff.max()),
        "best_epoch_same": bool((comparison.best_epoch_old == comparison.best_epoch_new).all())}

    run_ids = {json.load(open(os.path.join(args.base, f"seed{seed}", "run_config.json")))["run_id"]
               for seed in range(3)}
    latest = {}
    skipped_malformed = []
    with open(args.log) as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if any(run_id in line for run_id in run_ids):
                    raise
                skipped_malformed.append(line_number)
                continue
            if record["run_id"] in run_ids:
                latest[record["run_id"]] = record
    assert len(latest) == 3 and all(record["status"] == "completed" for record in latest.values())
    assert 111 in skipped_malformed
    code_hash = sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e5_parclip_eval.py"))
    assert all(record["code_version"] == code_hash for record in latest.values())
    checks["run_log_completed"] = {run_id: latest[run_id]["wall_seconds"] for run_id in sorted(run_ids)}
    checks["code_sha256"] = code_hash
    checks["skipped_malformed_unrelated_log_lines"] = skipped_malformed
    checks["status"] = "passed"
    with open(os.path.join(args.base, "three_pass_verification.json"), "w") as handle:
        json.dump(checks, handle, indent=2)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
