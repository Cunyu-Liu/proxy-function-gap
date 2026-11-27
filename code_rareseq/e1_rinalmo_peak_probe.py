#!/usr/bin/env python3
import argparse
import json
import os

from e1_rinalmo_frozen_baselines import TaskLoader, require_cuda, train_modern
from e1_locked_logger import E1RunLogger as RunLogger
from log_run import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--neg-dir", required=True)
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--positives-audit", required=True)
    parser.add_argument("--exps-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    args.model = "rinalmo-giga"
    args.folds = "0"
    args.tracks = "neg1c,neg2"
    args.seed = 0
    args.loss = "bce"
    args.epochs = 2
    args.patience = 20
    args.min_delta = 1e-4
    args.batch = 512
    args.lr = 1e-3
    args.weight_decay = 1e-4
    args.hidden = 256
    args.dropout = 0.1
    args.label_smoothing = 0.02
    args.pi = 0.1
    args.focal_gamma = 2.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    logger = RunLogger(args.log, stage="E1", kind="rinalmo-bce-peak-calibration")
    logger.start(config=config, seed=0, data_version="e0-v2-id90-strict3train-main",
                 code_version=config["code_sha256"])
    try:
        import torch
        device, device_info = require_cuda()
        loader = TaskLoader(args)
        task = max(loader.tasks(), key=lambda item: sum(len(ids) for ids, _ in item["splits"]))
        arrays = loader.arrays(task)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        expected_failsafe = False
        try:
            train_modern(task, arrays, args, device)
        except RuntimeError as exc:
            if "Training failsafe reached" not in str(exc):
                raise
            expected_failsafe = True
        result = {
            "selection": "largest fold0 task by train+validation+test rows",
            "experiment": task["experiment"], "track": task["track"], "fold": task["fold"],
            "split_rows": [len(ids) for ids, _ in task["splits"]],
            "input_dim": int(arrays[0][1].shape[1]), "hidden": args.hidden, "batch": args.batch,
            "epochs": args.epochs, "loss": args.loss, "lr": args.lr,
            "weight_decay": args.weight_decay, "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "baseline_allocated_mib": baseline_allocated / 2**20,
            "baseline_reserved_mib": baseline_reserved / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "cuda": device_info,
            "scientific_config_changed": False,
            "expected_probe_epoch_failsafe": expected_failsafe,
            "calibration_only": True,
        }
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
        logger.finish(metrics=result, outputs=[args.out])
        print("RINALMO_PEAK_PROBE", json.dumps(result), flush=True)
    except Exception as exc:
        logger.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
