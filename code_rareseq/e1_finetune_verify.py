#!/usr/bin/env python3
"""Independent schema/convergence/split checks for RNA-FM fine-tuning runs."""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from e1_frozen_baselines import metrics
from e1_finetune import DATA_VERSION, atomic_json
from log_run import sha256_file


def verify_run(run, require_eval=False, expected_rank=None):
    config_path = os.path.join(run, "finetune_config.json")
    config = json.load(open(config_path))
    args = config["args"]
    assert args["data_version"] == DATA_VERSION
    assert args["epochs"] == 1000 and args["patience"] == 20 and np.isclose(args["min_delta"], 1e-4)
    assert config["strict_train_folds"] is True and config["cuda"]["device"] == "cuda:0"
    assert config["cuda"]["backward_probe"] == "passed" and config["cuda"]["optimizer_probe"] == "passed"
    assert os.path.isfile(os.path.join(run, ".train_complete"))
    if expected_rank is not None:
        assert args["mode"] == "lora" and int(args["lora_r"]) == expected_rank
    validation_path = os.path.join(run, "finetune_validation_results.tsv")
    validation = pd.read_csv(validation_path, sep="\t")
    assert len(validation) > 0 and set(validation.split) == {"val"}
    metric_columns = ["auroc", "auprc"] + (["train_loss"] if "train_loss" in validation else [])
    assert np.isfinite(validation[metric_columns]).all().all()
    assert (validation.stopped_epoch == validation.best_epoch + 20).all()
    assert (validation.stopped_epoch < 1000).all() and validation.early_stopped.all()
    history = pd.read_csv(os.path.join(run, "validation_history.tsv"), sep="\t")
    assert np.isfinite(history[["train_loss", "val_auroc", "val_auprc"]]).all().all()
    counts = history.groupby(["experiment", "fold", "track"]).size().reset_index(name="n")
    merged = validation.merge(counts, on=["experiment", "fold", "track"], validate="one_to_one")
    assert (merged.n == merged.stopped_epoch).all()
    accepted = history[history.is_best].groupby(["experiment", "fold", "track"]).epoch.max().reset_index()
    accepted = accepted.rename(columns={"epoch": "replayed_best_epoch"})
    replayed = validation.merge(accepted, on=["experiment", "fold", "track"], validate="one_to_one")
    assert (replayed.best_epoch == replayed.replayed_best_epoch).all()
    split_audit = pd.read_csv(os.path.join(run, "split_audit.tsv"), sep="\t")
    assert len(split_audit) == len(validation)
    for _, row in split_audit.iterrows():
        expected = sorted(set(range(5)) - {int(row.fold), int(row.val_fold)})
        assert json.loads(row.train_folds) == expected
        for kind in ("pos", "neg"):
            counts_by_fold = json.loads(row[f"train_{kind}_fold_counts"])
            assert sorted(int(key) for key in counts_by_fold) == expected
            assert "-1" not in counts_by_fold
    task_dirs = sorted(glob.glob(os.path.join(run, "tasks", "*")))
    assert len(task_dirs) == len(validation)
    for directory in task_dirs:
        manifest = json.load(open(os.path.join(directory, "checkpoint_manifest.json")))
        checkpoint_path = os.path.join(directory, manifest["checkpoint"])
        assert manifest["format"] == "torch.save FP32 weights + zlib-1 stream"
        assert manifest["contains_optimizer_state"] is False
        assert manifest["contains_old_epoch_weights"] is False
        assert manifest["roundtrip_sha256_verified"] is True
        retention_path = os.path.join(directory, "retention.json")
        if os.path.isfile(retention_path):
            retention = json.load(open(retention_path))
            assert retention["raw_sha256"] == manifest["raw_sha256"]
            assert retention["compressed_sha256"] == manifest["compressed_sha256"]
            assert retention["checkpoint_validation_reproduced"] is True
            assert retention["checkpoint_used_for_test"] is True
            head_path = os.path.join(directory, "head.pt")
            assert sha256_file(head_path) == retention["head"]["sha256"]
            for artifact in retention["artifact_hashes"].values():
                artifact_path = os.path.join(run, artifact["path"])
                assert os.path.getsize(artifact_path) == artifact["bytes"]
                assert sha256_file(artifact_path) == artifact["sha256"]
            if retention["status"] == "retained_proof_only":
                assert not os.path.exists(checkpoint_path)
            else:
                assert retention["status"] == "retained_full_checkpoint"
                assert os.path.isfile(checkpoint_path)
        else:
            assert os.path.isfile(checkpoint_path)
        if os.path.isfile(checkpoint_path):
            assert os.path.getsize(checkpoint_path) == manifest["compressed_bytes"]
            assert sha256_file(checkpoint_path) == manifest["compressed_sha256"]
        assert not glob.glob(os.path.join(directory, "best.restore.*"))
        assert not os.path.exists(os.path.join(directory, "best.pt.tmp"))
        assert not os.path.exists(os.path.join(directory, "best.pt.zlib.tmp"))
        metadata = json.load(open(os.path.join(directory, "metadata.json")))
        assert metadata["training_device"]["device"] == "cuda:0"
        prediction = np.load(os.path.join(directory, "validation_predictions.npz"))
        assert np.isfinite(prediction["score"]).all()
        assert len(prediction["score"]) == metadata["n_eval"]
        auroc, auprc = metrics(prediction["label"], prediction["score"])
        selected = validation[(validation.experiment == metadata["experiment"]) &
                              (validation.fold == metadata["fold"]) &
                              (validation.track == metadata["track"])]
        assert len(selected) == 1
        assert np.allclose([auroc, auprc], selected.iloc[0][["auroc", "auprc"]].astype(float), atol=1e-12)
    result = {"run": run, "mode": args["mode"], "fold": args["folds"], "seed": args["seed"],
              "lora_r": args["lora_r"], "n_tasks": len(validation), "train_passed": True,
              "eval_passed": False}
    if require_eval:
        assert os.path.isfile(os.path.join(run, ".eval_complete"))
        results = pd.read_csv(os.path.join(run, "finetune_results.tsv"), sep="\t")
        assert len(results) == 2 * len(validation) and set(results.split) == {"val", "test"}
        assert np.isfinite(results[["auroc", "auprc"]]).all().all()
        test_tasks = sorted(glob.glob(os.path.join(run, "test_tasks", "*")))
        assert len(test_tasks) == len(validation)
        for directory in test_tasks:
            prediction = np.load(os.path.join(directory, "test_predictions.npz"))
            assert np.isfinite(prediction["score"]).all()
            auroc, auprc = metrics(prediction["label"], prediction["score"])
            metadata = json.load(open(os.path.join(directory, "metadata.json")))
            assert metadata["checkpoint_validation_reproduced"] is True
            assert metadata["checkpoint_used_for_test"] is True
            assert sha256_file(os.path.join(directory, "test_predictions.npz")) == metadata["test_predictions_sha256"]
            assert np.allclose([auroc, auprc], [metadata["auroc"], metadata["auprc"]], atol=1e-12)
        if args["mode"] == "lora":
            eval_config = json.load(open(os.path.join(run, "finetune_eval_config.json")))
            assert eval_config["selection_sha256"]
        result["eval_passed"] = True
    else:
        assert not os.path.exists(os.path.join(run, "finetune_test_predictions.tsv.gz"))
        assert not os.path.exists(os.path.join(run, ".eval_complete"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="Comma-separated run directories")
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument("--expected-rank", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = [path for path in args.runs.split(",") if path]
    report = {"status": "passed", "require_eval": args.require_eval,
              "runs": [verify_run(run, args.require_eval, args.expected_rank) for run in runs]}
    atomic_json(args.out, report)
    print("E1_FINETUNE_VERIFY_PASSED " + json.dumps({"n_runs": len(runs), "out": args.out}))


if __name__ == "__main__":
    main()
