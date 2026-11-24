#!/usr/bin/env python3
"""E1 跨细胞系：源系 train/val，目标系同折 test；复用 frozen/kNN 实现。"""
import argparse
import os

import numpy as np
import pandas as pd

from e1_frozen_baselines import TaskLoader, run_knn, run_modern
from log_run import RunLogger, sha256_file


class CrossCellLoader(TaskLoader):
    def tasks(self):
        for task in super().tasks():
            source = task["experiment"]
            rbp, cell = source.rsplit("_", 1)
            if cell not in ("K562", "HepG2"):
                raise ValueError(source)
            target = rbp + ("_HepG2" if cell == "K562" else "_K562")
            if not os.path.exists(os.path.join(self.args.windows_dir, f"{target}.windows.tsv")):
                raise FileNotFoundError(target)
            task["source_experiment"] = source
            task["target_experiment"] = target
            task["experiment"] = source + "->" + target
            task["eval_protocol"] = "cross_cell"
            yield task

    def arrays(self, task):
        source_task = task.copy()
        source_task["experiment"] = task["source_experiment"]
        train, val, _ = super().arrays(source_task)
        target = task["target_experiment"]
        pos = pd.read_csv(os.path.join(self.args.windows_dir, f"{target}.windows.tsv"), sep="\t")
        neg = pd.read_csv(os.path.join(self.args.neg_dir, f"{target}.negatives.tsv"), sep="\t")
        fold = task["fold"]
        pos_fold = pos.window_id.map(self.fold_by_wid)
        pte = pos[pos_fold == fold]
        nte = neg[(neg.track == task["track"]) & (neg.fold == fold)]
        ids = pte.window_id.tolist() + nte.window_id.tolist()
        y = np.array([1] * len(pte) + [0] * len(nte), dtype=np.int64)
        if not ids:
            raise ValueError(f"{target}/fold{fold}/{task['track']}: empty target test")
        return train, val, (ids, self.embeddings(target, ids), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["modern", "knn"])
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--positives-audit", required=True)
    ap.add_argument("--split-clstr", required=True)
    ap.add_argument("--exps-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="rnafm")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--tracks", default="neg1c,neg2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E1", kind=f"cross-cell-{args.command}")
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    config["helper_sha256"] = sha256_file(os.path.abspath("e1_frozen_baselines.py"))
    lg.start(config=config, seed=args.seed, data_version=args.data_version,
             inputs=[args.split_clstr, args.exps_file], code_version=config["code_sha256"])
    try:
        loader = CrossCellLoader(args)
        result = run_modern(loader, args, lg) if args.command == "modern" else run_knn(loader, args, lg)
        lg.finish(metrics=result, outputs=[args.out])
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise
    print("CROSS_CELL_DONE", result, flush=True)


if __name__ == "__main__":
    main()
