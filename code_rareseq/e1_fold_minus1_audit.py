#!/usr/bin/env python3
import argparse
import hashlib
import json
import os

import pandas as pd


def ids_sha256(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--neg-dir", required=True)
    parser.add_argument("--exps-file", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    experiments = [line.strip() for line in open(args.exps_file) if line.strip()]
    old_results = {(fold, seed): pd.read_csv(
        os.path.join(args.old_root, f"f{fold}_modern_bce_s{seed}", "modern_results.tsv"), sep="\t")
        for fold in range(5) for seed in range(3)}
    fold_map = pd.read_csv(os.path.join(args.neg_dir, "window_fold_map.tsv"), sep="\t",
                           usecols=["window_id", "fold"])
    fold_by_wid = dict(zip(fold_map.window_id, fold_map.fold))
    rows = []
    unique_fold_minus1 = 0
    for experiment in experiments:
        pos = pd.read_csv(os.path.join(args.windows_dir, f"{experiment}.windows.tsv"), sep="\t",
                          usecols=["window_id"])
        neg = pd.read_csv(os.path.join(args.neg_dir, f"{experiment}.negatives.tsv"), sep="\t",
                          usecols=["track", "fold"])
        pos_fold = pos.window_id.map(fold_by_wid)
        unique_fold_minus1 += int((pos_fold == -1).sum())
        for fold in range(5):
            val_fold = (fold + 1) % 5
            train_folds = sorted(set(range(5)) - {fold, val_fold})
            old_pos = pos[~pos_fold.isin([fold, val_fold])]
            strict_pos = pos[pos_fold.isin(train_folds)]
            extra_pos = old_pos[old_pos.window_id.map(fold_by_wid) == -1]
            assert len(old_pos) - len(strict_pos) == len(extra_pos) == int((pos_fold == -1).sum())
            for track in ("neg1c", "neg2"):
                old_neg = neg[(neg.track == track) & (~neg.fold.isin([fold, val_fold]))]
                strict_neg = neg[(neg.track == track) & (neg.fold.isin(train_folds))]
                assert old_neg.equals(strict_neg)
                row = {"experiment": experiment, "fold": fold, "track": track,
                       "train_folds": json.dumps(train_folds),
                       "old_n_train_pos": len(old_pos), "strict_n_train_pos": len(strict_pos),
                       "fold_minus1_n_train_pos": len(extra_pos), "n_train_neg": len(strict_neg),
                       "old_n_train_total": len(old_pos) + len(strict_neg),
                       "strict_n_train_total": len(strict_pos) + len(strict_neg),
                       "old_train_pos_id_sha256": ids_sha256(old_pos.window_id.tolist()),
                       "strict_train_pos_id_sha256": ids_sha256(strict_pos.window_id.tolist())}
                for seed in range(3):
                    results = old_results[(fold, seed)]
                    old = results[(results.experiment == experiment) & (results.track == track) &
                                  (results.split == "test")]
                    assert len(old) == 1 and old.n_train.iloc[0] == row["old_n_train_total"]
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(args.out, "fold_minus1_by_experiment_fold_track.tsv"), sep="\t", index=False)
    summary = {"n_experiments": len(experiments), "n_experiment_fold_track_rows": len(frame),
               "unique_fold_minus1_positives": unique_fold_minus1,
               "old_positive_train_occurrences": int(frame[frame.track == "neg1c"].old_n_train_pos.sum()),
               "strict_positive_train_occurrences": int(frame[frame.track == "neg1c"].strict_n_train_pos.sum()),
               "fold_minus1_train_occurrences": int(frame[frame.track == "neg1c"].fold_minus1_n_train_pos.sum()),
               "affected_experiment_folds": int((frame[frame.track == "neg1c"].fold_minus1_n_train_pos > 0).sum()),
               "all_15_old_modern_n_train_reproduced": True, "negative_train_unchanged": True,
               "per_fold": frame[frame.track == "neg1c"].groupby("fold")[
                   ["old_n_train_pos", "strict_n_train_pos", "fold_minus1_n_train_pos"]].sum().to_dict("index")}
    with open(os.path.join(args.out, "fold_minus1_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
