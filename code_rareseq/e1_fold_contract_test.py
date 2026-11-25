#!/usr/bin/env python3
import os
import tempfile
from types import SimpleNamespace

import pandas as pd

from e1_frozen_baselines import TaskLoader


def main():
    with tempfile.TemporaryDirectory() as root:
        windows = os.path.join(root, "windows")
        neg_dir = os.path.join(root, "neg")
        os.makedirs(windows)
        os.makedirs(neg_dir)
        experiment = "RBP_Cell"
        pos_ids = [f"p{fold}" for fold in (-1, 0, 1, 2, 3, 4)]
        pd.DataFrame({"window_id": pos_ids}).to_csv(
            os.path.join(windows, f"{experiment}.windows.tsv"), sep="\t", index=False)
        pd.DataFrame({"window_id": pos_ids, "fold": [-1, 0, 1, 2, 3, 4]}).to_csv(
            os.path.join(neg_dir, "window_fold_map.tsv"), sep="\t", index=False)
        negatives = []
        for track in ("neg1c", "neg2"):
            for fold in (-1, 0, 1, 2, 3, 4):
                negatives.append({"window_id": f"{track}-n{fold}", "track": track, "fold": fold})
        pd.DataFrame(negatives).to_csv(os.path.join(neg_dir, f"{experiment}.negatives.tsv"),
                                      sep="\t", index=False)
        pd.DataFrame({"rbp": ["RBP"], "n_sites": [5]}).to_csv(
            os.path.join(root, "positives_audit.tsv"), sep="\t", index=False)
        exps_file = os.path.join(root, "exps.txt")
        with open(exps_file, "w") as handle:
            handle.write(experiment + "\n")
        for fold, expected_train in ((0, [2, 3, 4]), (4, [1, 2, 3])):
            args = SimpleNamespace(neg_dir=neg_dir, windows_dir=windows,
                                   positives_audit=os.path.join(root, "positives_audit.tsv"),
                                   exps_file=exps_file, folds=str(fold), tracks="neg1c,neg2")
            tasks = list(TaskLoader(args).tasks())
            assert len(tasks) == 2
            for task in tasks:
                assert task["train_folds"] == expected_train
                for split, expected in zip(("train", "val", "test"),
                                           (expected_train, [(fold + 1) % 5], [fold])):
                    ids, y = task["splits"][("train", "val", "test").index(split)]
                    actual_pos = sorted(ids[i] for i in range(len(ids)) if y[i] == 1)
                    actual_neg = sorted(ids[i] for i in range(len(ids)) if y[i] == 0)
                    assert actual_pos == sorted(f"p{f}" for f in expected)
                    assert actual_neg == sorted(f"{task['track']}-n{f}" for f in expected)
                    assert not any(i.endswith("p-1") or i.endswith("n-1") for i in ids)
        print("E1_FOLD_CONTRACT_OK: fold=-1 excluded; explicit train=3 folds; val/test unchanged")


if __name__ == "__main__":
    main()
