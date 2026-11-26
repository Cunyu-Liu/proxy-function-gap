#!/usr/bin/env python3
import json
import os
import tempfile

import numpy as np
import pandas as pd

from e1_finetune import EarlyStopping, atomic_json, atomic_marker, atomic_npz
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
    with tempfile.TemporaryDirectory() as run:
        args = {"data_version": "e0-v2-id90-strict3train-main", "epochs": 1000,
                "patience": 20, "min_delta": 1e-4, "mode": "lora", "lora_r": 8,
                "folds": "0", "seed": 0}
        atomic_json(os.path.join(run, "finetune_config.json"),
                    {"args": args, "strict_train_folds": True,
                     "cuda": {"device": "cuda:0", "backward_probe": "passed", "optimizer_probe": "passed"}})
        atomic_marker(os.path.join(run, ".train_complete"))
        result = {"experiment": "RBP_Cell", "fold": 0, "track": "neg1c", "split": "val",
                  "best_epoch": 1, "stopped_epoch": 21, "early_stopped": True,
                  "auroc": 0.7, "auprc": 0.6}
        pd.DataFrame([result]).to_csv(os.path.join(run, "finetune_validation_results.tsv"), sep="\t", index=False)
        pd.DataFrame([{**result, "epoch": epoch, "train_loss": 0.1, "val_auroc": 0.7, "val_auprc": 0.6}
                      for epoch in range(1, 22)]).to_csv(os.path.join(run, "validation_history.tsv"), sep="\t", index=False)
        audit = {"fold": 0, "val_fold": 1, "train_folds": json.dumps([2, 3, 4]),
                 "train_pos_fold_counts": json.dumps({"2": 1, "3": 1, "4": 1}),
                 "train_neg_fold_counts": json.dumps({"2": 1, "3": 1, "4": 1})}
        pd.DataFrame([audit]).to_csv(os.path.join(run, "split_audit.tsv"), sep="\t", index=False)
        directory = os.path.join(run, "tasks", "RBP_Cell.f0.neg1c")
        os.makedirs(directory)
        with open(os.path.join(directory, "best.pt"), "wb") as handle:
            handle.write(b"checkpoint")
        atomic_json(os.path.join(directory, "metadata.json"),
                    {"training_device": {"device": "cuda:0"}, "n_eval": 2})
        atomic_npz(os.path.join(directory, "validation_predictions.npz"),
                   window_id=np.asarray(["a", "b"]), label=np.asarray([0, 1]), score=np.asarray([0.2, 0.8]))
        assert verify_run(run)["train_passed"] is True


def main():
    early_stopping_check()
    selection_check()
    verification_check()
    print("E1_FINETUNE_CONTRACT_OK: convergence, validation-only selection, independent artifact verification")


if __name__ == "__main__":
    main()
