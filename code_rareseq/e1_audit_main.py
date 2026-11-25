#!/usr/bin/env python3
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from e1_frozen_baselines import load_tiers, rbp_of
from e1_locked_logger import E1RunLogger as RunLogger
from log_run import sha256_file


BASE_KEYS = ["experiment", "fold", "track", "window_id", "label"]
PRED_COLUMNS = ["experiment", "rbp", "tier", "fold", "track", "split", "window_id", "label", "score"]
RESULT_COLUMNS = ["experiment", "rbp", "tier", "fold", "track", "split", "auroc", "auprc"]
TASK_KEYS = ["experiment", "fold", "track"]
DATA_VERSION = "e0-v2-id90-strict3train-main"


def add_occurrence(frame):
    frame = frame.copy()
    frame["occurrence"] = frame.groupby(BASE_KEYS, sort=False).cumcount()
    if frame.duplicated(BASE_KEYS + ["occurrence"]).any():
        raise AssertionError("duplicate occurrence key")
    return frame


def comparable(frame):
    return add_occurrence(frame)[BASE_KEYS + ["occurrence"]].reset_index(drop=True)


def ids_sha256(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def expected_predictions(args, fold, track, split, tiers):
    fold_map = pd.read_csv(os.path.join(args.neg_dir, "window_fold_map.tsv"), sep="\t",
                           usecols=["window_id", "fold"])
    fold_by_wid = dict(zip(fold_map.window_id, fold_map.fold))
    val_fold = (fold + 1) % 5
    selected_fold = val_fold if split == "val" else fold
    frames = []
    for experiment in args.experiments:
        pos = pd.read_csv(os.path.join(args.windows_dir, f"{experiment}.windows.tsv"), sep="\t",
                          usecols=["window_id"])
        neg = pd.read_csv(os.path.join(args.neg_dir, f"{experiment}.negatives.tsv"), sep="\t",
                          usecols=["window_id", "track", "fold"])
        pos = pos[pos.window_id.map(fold_by_wid) == selected_fold]
        neg = neg[(neg.track == track) & (neg.fold == selected_fold)]
        ids = pos.window_id.tolist() + neg.window_id.tolist()
        labels = [1] * len(pos) + [0] * len(neg)
        frames.append(pd.DataFrame({
            "experiment": experiment, "rbp": rbp_of(experiment), "tier": tiers[rbp_of(experiment)],
            "fold": fold, "track": track, "split": split, "window_id": ids, "label": labels,
        }))
    return pd.concat(frames, ignore_index=True)


def expected_split_audit(args, fold, track, tiers):
    fold_map = pd.read_csv(os.path.join(args.neg_dir, "window_fold_map.tsv"), sep="\t",
                           usecols=["window_id", "fold"])
    fold_by_wid = dict(zip(fold_map.window_id, fold_map.fold))
    val_fold = (fold + 1) % 5
    train_folds = sorted(set(range(5)) - {fold, val_fold})
    rows = []
    for experiment in args.experiments:
        pos = pd.read_csv(os.path.join(args.windows_dir, f"{experiment}.windows.tsv"), sep="\t",
                          usecols=["window_id"])
        neg = pd.read_csv(os.path.join(args.neg_dir, f"{experiment}.negatives.tsv"), sep="\t",
                          usecols=["window_id", "track", "fold"])
        pos_fold = pos.window_id.map(fold_by_wid)
        pos_splits = [pos[pos_fold.isin(train_folds)], pos[pos_fold == val_fold], pos[pos_fold == fold]]
        neg_splits = [neg[(neg.track == track) & (neg.fold.isin(train_folds))],
                      neg[(neg.track == track) & (neg.fold == val_fold)],
                      neg[(neg.track == track) & (neg.fold == fold)]]
        row = {"experiment": experiment, "rbp": rbp_of(experiment), "tier": tiers[rbp_of(experiment)],
               "fold": fold, "val_fold": val_fold, "track": track,
               "train_folds": json.dumps(train_folds)}
        for split, p, n in zip(("train", "val", "test"), pos_splits, neg_splits):
            for name, frame, folds in (("pos", p, p.window_id.map(fold_by_wid)), ("neg", n, n.fold)):
                ids = frame.window_id.tolist()
                counts = {str(int(k)): int(v) for k, v in folds.value_counts().sort_index().items()}
                row[f"{split}_{name}_fold_counts"] = json.dumps(counts, sort_keys=True)
                row[f"{split}_{name}_id_sha256"] = ids_sha256(ids)
                row[f"n_{split}_{name}"] = len(ids)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(TASK_KEYS).reset_index(drop=True)


def completed_runs(path):
    runs = {}
    malformed = 0
    with open(path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if record.get("status") != "completed":
                continue
            config = record.get("config", {})
            key = (config.get("out"), config.get("folds"), config.get("seed"), config.get("command"))
            runs[key] = record
    return runs, malformed


def check_cuda(config, run):
    assert config["strict_train_folds"] is True
    assert config["data_version"] == DATA_VERSION
    cuda = config["cuda"]
    assert cuda["device"] == "cuda:0" and cuda["numeric_probe"] == [2.5, -6.0]
    assert run.get("metrics", {}).get("cuda", {}).get("device") == "cuda:0"


def check_history(history, results, out):
    import torch
    test = results[results.split == "test"].set_index(TASK_KEYS)
    assert len(test) == 292
    assert history.groupby(TASK_KEYS).size().eq(test.stopped_epoch).all()
    assert history.groupby(TASK_KEYS).epoch.apply(
        lambda x: np.array_equal(x.to_numpy(), np.arange(1, len(x) + 1))).all()
    assert np.isfinite(history[["val_auroc", "val_auprc", "train_loss"]].to_numpy()).all()
    for key, group in history.groupby(TASK_KEYS):
        best_metric = -np.inf
        best_epoch = 0
        for row in group.itertuples(index=False):
            if row.val_auprc > best_metric + 1e-4:
                best_metric = row.val_auprc
                best_epoch = row.epoch
        expected = test.loc[key]
        assert best_epoch == expected.best_epoch
        assert np.isclose(best_metric, expected.best_val_auprc, rtol=0, atol=1e-12)
        assert len(group) == best_epoch + 20
        assert expected.early_stopped and expected.stopped_epoch < 1000
        path = os.path.join(out, "checkpoints", f"{key[0]}.f{key[1]}.{key[2]}.best.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["best_epoch"] == best_epoch and checkpoint["stopped_epoch"] == len(group)
        assert checkpoint["training_device"]["device"] == "cuda:0"
        assert checkpoint["training_device"]["numeric_probe"] == [2.5, -6.0]
        assert checkpoint["task"]["train_folds"] == sorted(set(range(5)) - {key[1], (key[1] + 1) % 5})
        assert "model_state" in checkpoint and checkpoint["mean"].shape[1] == 640


def audit(args):
    report = {"modern": [], "knn": [], "prediction_folds": [], "checks": {}}
    tiers = load_tiers(args.positives_audit)
    runs, malformed = completed_runs(args.log)
    report["checks"]["source_log_malformed_lines"] = malformed
    for fold in range(5):
        references = {track: expected_split_audit(args, fold, track, tiers) for track in ("neg1c", "neg2")}
        for seed in range(3):
            name = f"f{fold}_modern_bce_s{seed}"
            directory = os.path.join(args.root, name)
            config = json.load(open(os.path.join(directory, "modern_config.json")))
            assert config["command"] == "modern" and config["folds"] == str(fold)
            assert config["seed"] == seed and config["tracks"] == "neg1c,neg2"
            assert config["epochs"] == 1000 and config["patience"] == 20
            assert config["min_delta"] == 1e-4 and config["loss"] == "bce" and config["model"] == "rnafm"
            assert os.path.samefile(config["exps_file"], args.exps_file)
            results = pd.read_csv(os.path.join(directory, "modern_results.tsv"), sep="\t")
            history = pd.read_csv(os.path.join(directory, "validation_history.tsv"), sep="\t")
            assert set(RESULT_COLUMNS).issubset(results.columns)
            assert len(results) == 584 and results.experiment.nunique() == 146
            assert set(results.experiment) == set(args.experiments)
            assert set(results.fold) == {fold} and set(results.track) == {"neg1c", "neg2"}
            assert set(results.split) == {"val", "test"}
            assert not results.duplicated(["experiment", "fold", "track", "split"]).any()
            assert np.isfinite(results[["auroc", "auprc"]].to_numpy()).all()
            check_history(history, results, directory)
            split_audit = pd.read_csv(os.path.join(directory, "split_audit.tsv"), sep="\t")
            for track, expected in references.items():
                actual = split_audit[split_audit.track == track].sort_values(TASK_KEYS).reset_index(drop=True)
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
            run = runs.get((directory, str(fold), seed, "modern"))
            assert run is not None and run.get("metrics", {}).get("n_tasks") == 292
            check_cuda(config, run)
            selected = results[results.split == "test"].groupby("track").best_epoch.agg(["min", "max"])
            report["modern"].append({
                "name": name, "run_id": run["run_id"], "code_sha256": run["code_version"],
                "best_neg1c_min": int(selected.loc["neg1c", "min"]),
                "best_neg1c_max": int(selected.loc["neg1c", "max"]),
                "best_neg2_min": int(selected.loc["neg2", "min"]),
                "best_neg2_max": int(selected.loc["neg2", "max"]),
                "history_rows": len(history), "result_rows": len(results), "checkpoints": 292,
            })
        name = f"f{fold}_knn"
        directory = os.path.join(args.root, name)
        config = json.load(open(os.path.join(directory, "knn_config.json")))
        results = pd.read_csv(os.path.join(directory, "knn_results.tsv"), sep="\t")
        validation = pd.read_csv(os.path.join(directory, "knn_validation.tsv"), sep="\t")
        assert len(results) == 584 and results.experiment.nunique() == 146
        assert set(results.experiment) == set(args.experiments)
        assert len(validation) == 292 * 3 and np.isfinite(validation[["auroc", "auprc"]].to_numpy()).all()
        assert np.isfinite(results[["auroc", "auprc"]].to_numpy()).all()
        selected = validation[validation.tier == "tail"].groupby(["track", "k"]).auprc.mean().groupby("track").idxmax()
        recomputed = {track: int(k) for track, k in selected}
        assert recomputed == config["selected_k_by_track"]
        run = runs.get((directory, str(fold), 0, "knn"))
        assert run is not None and run.get("metrics", {}).get("n_tasks") == 292
        check_cuda(config, run)
        split_audit = pd.read_csv(os.path.join(directory, "split_audit.tsv"), sep="\t")
        for track, expected in references.items():
            actual = split_audit[split_audit.track == track].sort_values(TASK_KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        report["knn"].append({"name": name, "run_id": run["run_id"], "best_k": recomputed,
                              "validation_rows": len(validation), "result_rows": len(results)})
        val_knn = pd.read_csv(os.path.join(directory, "knn_val_predictions.tsv.gz"), sep="\t", usecols=["score"])
        assert np.isfinite(val_knn.score.to_numpy()).all()
        assert len(val_knn) == results[results.split == "val"].n_eval.sum()
        for seed in range(3):
            val_modern = pd.read_csv(os.path.join(args.root, f"f{fold}_modern_bce_s{seed}", "modern_val_predictions.tsv.gz"),
                                     sep="\t", usecols=["score"])
            assert np.isfinite(val_modern.score.to_numpy()).all() and len(val_modern) == len(val_knn)
        knn_path = os.path.join(directory, "knn_test_predictions.tsv.gz")
        reference = pd.read_csv(knn_path, sep="\t")
        assert list(reference.columns) == PRED_COLUMNS and np.isfinite(reference.score.to_numpy()).all()
        for track in ("neg1c", "neg2"):
            expected = expected_predictions(args, fold, track, "test", tiers)
            subset = reference[reference.track == track].reset_index(drop=True)
            expected = expected[PRED_COLUMNS[:-1]].reset_index(drop=True)
            assert subset[PRED_COLUMNS[:-1]].equals(expected), f"raw split mismatch f{fold} {track} test"
        reference_meta = comparable(reference)
        for seed in range(3):
            path = os.path.join(args.root, f"f{fold}_modern_bce_s{seed}", "modern_test_predictions.tsv.gz")
            frame = pd.read_csv(path, sep="\t")
            assert list(frame.columns) == PRED_COLUMNS and np.isfinite(frame.score.to_numpy()).all()
            assert comparable(frame).equals(reference_meta), f"occurrence mismatch f{fold}s{seed} test"
            assert len(frame) == len(reference)
        report["prediction_folds"].append({"fold": fold, "split": "test", "rows": len(reference),
                                           "neg2_duplicate_ids": int(reference[reference.track == "neg2"].duplicated(
                                               ["experiment", "window_id", "label"]).sum())})
        print("PRED_OK", fold, "test", len(reference), flush=True)
    report["checks"].update({
        "n_experiments": len(args.experiments), "n_modern_runs": len(report["modern"]),
        "n_knn_runs": len(report["knn"]), "n_tracks": 2, "n_folds": 5, "n_seeds": 3,
        "all_early_stopped_below_1000": True, "all_cuda0_numeric_probes_passed": True,
        "all_train_strict_three_folds": True, "no_fold_minus1_in_train": True,
        "all_prediction_keys_aligned": True, "all_raw_splits_aligned": True, "all_scores_finite": True,
    })
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--neg-dir", required=True)
    parser.add_argument("--positives-audit", required=True)
    parser.add_argument("--exps-file", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.experiments = [line.strip() for line in open(args.exps_file) if line.strip()]
    assert len(args.experiments) == 146 and len(set(args.experiments)) == 146
    os.makedirs(args.out, exist_ok=True)
    logger = RunLogger(args.log, stage="E1", kind="main-independent-audit")
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    logger.start(config=config, data_version=DATA_VERSION, code_version=config["code_sha256"])
    try:
        report = audit(args)
        report["run_id"] = logger.rec["run_id"]
        with open(os.path.join(args.out, "main_audit.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        pd.DataFrame(report["modern"] + report["knn"]).to_csv(
            os.path.join(args.out, "run_manifest.tsv"), sep="\t", index=False)
        logger.finish(metrics=report["checks"], outputs=[args.out])
    except Exception as exc:
        logger.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
