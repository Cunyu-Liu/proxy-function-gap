#!/usr/bin/env python3
import json
import os
import tempfile
import hashlib

import numpy as np
import pandas as pd

from e1_finetune import (EarlyStopping, atomic_json, atomic_marker, atomic_npz,
                         load_checkpoint_compressed, save_checkpoint_compressed)
from e1_finetune_select import select_rank
from e1_finetune_verify import verify_run


def early_stopping_check():
    stopping = EarlyStopping(patience=20, min_delta=1e-4)
    assert stopping.update(1, 0.5, 0.7)
    assert not stopping.update(2, 0.50005, 0.8)
    assert stopping.update(3, 0.5002, 0.75)
    for epoch in range(4, 23):
        assert not stopping.update(epoch, 0.4, 0.6)
        assert stopping.should_stop(epoch) == (epoch == 23)
    info = stopping.info(23, 1000)
    assert info["best_epoch"] == 3 and info["stopped_epoch"] == 23
    try:
        stopping.info(1000, 1000)
        raise AssertionError("1000-epoch failsafe did not fire")
    except RuntimeError:
        pass


def selection_check():
    with tempfile.TemporaryDirectory() as root:
        runs = {}
        for rank, score in ((4, 0.4), (8, 0.5), (16, 0.5)):
            run = os.path.join(root, f"r{rank}")
            os.makedirs(run)
            config = {"args": {"mode": "lora", "lora_r": rank, "folds": "0", "seed": 0,
                               "data_version": "e0-v2-id90-strict3train-main"}}
            atomic_json(os.path.join(run, "finetune_config.json"), config)
            pd.DataFrame({"split": ["val", "val"], "tier": ["tail", "head"],
                          "auprc": [score, 0.0]}).to_csv(
                os.path.join(run, "finetune_validation_results.tsv"), sep="\t", index=False)
            atomic_marker(os.path.join(run, ".train_complete"))
            atomic_json(os.path.join(run, "independent_validation.json"), {"test_used": False})
            atomic_marker(os.path.join(run, ".independent_validation_complete"))
            runs[rank] = run
        selected = select_rank(runs)
        assert selected["selected_rank"] == 8 and selected["test_used"] is False
        with open(os.path.join(runs[8], "finetune_results.tsv"), "w") as handle:
            handle.write("test must not be present\n")
        try:
            select_rank(runs)
            raise AssertionError("Selection accepted test artifacts")
        except ValueError:
            pass


def verification_check():
    import torch
    import e1_finetune_retention as retention
    with tempfile.TemporaryDirectory() as root:
        run = os.path.join(root, "f1_full_s0")
        os.makedirs(run)
        args = {"data_version": "e0-v2-id90-strict3train-main", "epochs": 1000,
                "patience": 20, "min_delta": 1e-4, "mode": "full", "lora_r": 8,
                "folds": "1", "seed": 0}
        atomic_json(os.path.join(run, "finetune_config.json"),
                    {"args": args, "strict_train_folds": True,
                     "cuda": {"device": "cuda:0", "backward_probe": "passed", "optimizer_probe": "passed"}})
        atomic_marker(os.path.join(run, ".train_complete"))
        result = {"experiment": "RBP_Cell", "rbp": "RBP", "tier": "tail", "fold": 1,
                  "track": "neg1c", "split": "val", "best_epoch": 1, "stopped_epoch": 21,
                  "early_stopped": True, "auroc": 1.0, "auprc": 1.0}
        pd.DataFrame([result]).to_csv(os.path.join(run, "finetune_validation_results.tsv"), sep="\t", index=False)
        pd.DataFrame([{**result, "epoch": epoch, "train_loss": 0.1, "val_auroc": 0.7,
                       "val_auprc": 0.6, "is_best": epoch == 1}
                      for epoch in range(1, 22)]).to_csv(os.path.join(run, "validation_history.tsv"), sep="\t", index=False)
        audit = {"fold": 1, "val_fold": 2, "train_folds": json.dumps([0, 3, 4]),
                 "train_pos_fold_counts": json.dumps({"0": 1, "3": 1, "4": 1}),
                 "train_neg_fold_counts": json.dumps({"0": 1, "3": 1, "4": 1})}
        pd.DataFrame([audit]).to_csv(os.path.join(run, "split_audit.tsv"), sep="\t", index=False)
        directory = os.path.join(run, "tasks", "RBP_Cell.f1.neg1c")
        os.makedirs(directory)
        with open(os.path.join(directory, "history.jsonl"), "w") as handle:
            for epoch in range(1, 22):
                handle.write(json.dumps({**result, "epoch": epoch, "train_loss": 0.1,
                                         "val_auroc": 0.7, "val_auprc": 0.6,
                                         "is_best": epoch == 1}) + "\n")
        checkpoint = {"encoder_state": {"weight": torch.tensor([1.25, -2.0])},
                      "head_state": {"bias": torch.tensor([0.5])}}
        manifest = save_checkpoint_compressed(directory, checkpoint)
        atomic_json(os.path.join(directory, "metadata.json"),
                    {"training_device": {"device": "cuda:0"}, "n_eval": 2,
                     "experiment": "RBP_Cell", "rbp": "RBP", "tier": "tail", "fold": 1, "track": "neg1c"})
        atomic_npz(os.path.join(directory, "validation_predictions.npz"),
                   window_id=np.asarray(["a", "b"]), label=np.asarray([0, 1]), score=np.asarray([0.2, 0.8]))
        assert verify_run(run)["train_passed"] is True
        test_dir = os.path.join(run, "test_tasks", "RBP_Cell.f1.neg1c")
        os.makedirs(test_dir)
        test_prediction = os.path.join(test_dir, "test_predictions.npz")
        atomic_npz(test_prediction, window_id=np.asarray(["c", "d"]),
                   label=np.asarray([0, 1]), score=np.asarray([0.1, 0.9]))
        atomic_json(os.path.join(test_dir, "metadata.json"),
                    {"auroc": 1.0, "auprc": 1.0, "checkpoint_validation_reproduced": True,
                     "checkpoint_used_for_test": True, "checkpoint_raw_sha256": manifest["raw_sha256"],
                     "checkpoint_compressed_sha256": manifest["compressed_sha256"],
                     "test_predictions_sha256": hashlib.sha256(open(test_prediction, "rb").read()).hexdigest()})
        pd.DataFrame([result, {**result, "split": "test"}]).to_csv(
            os.path.join(run, "finetune_results.tsv"), sep="\t", index=False)
        atomic_json(os.path.join(run, "finetune_eval_config.json"), {"selection_sha256": None})
        atomic_marker(os.path.join(run, ".eval_complete"))
        verification = os.path.join(run, "pre_retention_verification.json")
        atomic_json(verification, {"status": "passed", "require_eval": True,
                                   "runs": [{"run": run, "eval_passed": True}]})
        outside = os.path.join(root, "must_not_delete")
        with open(outside, "w") as handle:
            handle.write("keep")
        retention.APPROVED_ROOT = root
        outcome = retention.retain_run(run, verification)
        assert outcome["retained_proof_only"] == 1
        assert not os.path.exists(os.path.join(directory, "best.pt.zlib"))
        assert os.path.isfile(os.path.join(directory, "head.pt")) and os.path.isfile(outside)
        assert verify_run(run, require_eval=True)["eval_passed"] is True
        link = os.path.join(run, "tasks", "escaping_link")
        os.symlink(root, link)
        try:
            retention.audited_checkpoint_path(run, link, {"checkpoint": "best.pt.zlib"})
            raise AssertionError("Task symlink escape was not rejected")
        except ValueError:
            pass


def representative_selection_check():
    import e1_finetune_retention as retention
    with tempfile.TemporaryDirectory() as run:
        atomic_json(os.path.join(run, "finetune_config.json"), {"args": {"folds": "0", "seed": 0}})
        frame = pd.DataFrame({"experiment": ["Z", "A", "M", "B", "T", "C"],
                              "tier": ["head", "head", "mid", "mid", "tail", "tail"],
                              "track": ["neg1c"] * 6})
        selected, policy = retention.representative_tasks(run, frame)
        assert selected == {("A", "neg1c"), ("B", "neg1c"), ("C", "neg1c")}
        assert policy == "fold0_seed0_tier_lexicographic_permanent"
        pilot = os.path.join(run, "pilot_cuda_full")
        selected, policy = retention.representative_tasks(pilot, frame)
        assert len(selected) == 6 and policy == "pilot_permanent"


def compression_roundtrip_check():
    import torch
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = {"encoder_state": {"weight": torch.tensor([1.25, -2.0], dtype=torch.float32)},
                      "head_state": {"bias": torch.tensor([0.5], dtype=torch.float32)},
                      "training_device": {"torch": str(torch.__version__)}}
        save_checkpoint_compressed(directory, checkpoint)
        recovered = load_checkpoint_compressed(directory)
        assert torch.equal(recovered["encoder_state"]["weight"], checkpoint["encoder_state"]["weight"])
        assert torch.equal(recovered["head_state"]["bias"], checkpoint["head_state"]["bias"])
        assert not [name for name in os.listdir(directory) if name.endswith(".tmp") or "best.restore." in name]


def main():
    early_stopping_check()
    selection_check()
    verification_check()
    representative_selection_check()
    compression_roundtrip_check()
    print("E1_FINETUNE_CONTRACT_OK: convergence, selection, artifact verification, lossless checkpoint recovery")


if __name__ == "__main__":
    main()
