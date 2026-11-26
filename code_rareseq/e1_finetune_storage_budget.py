#!/usr/bin/env python3
"""Compute the RNA-FM best-checkpoint storage budget without changing artifacts."""
import argparse
import json
import os
import shutil
import time

from e1_finetune import atomic_json

RAW_MODEL_BYTES = 398_117_588
MEASURED_ZLIB1_BYTES = 249_910_990
PLANNING_CHECKPOINT_BYTES = int(420_000_000 * 0.70)
RESERVE_BYTES = 20 * 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams-path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    free = shutil.disk_usage(args.teams_path).free
    scopes = {"pilot_two_tasks": 2, "main_fold0_one_seed": 292,
              "main_fold0_three_seeds": 876, "main_five_fold_three_seeds": 4380}
    budget = {}
    for name, tasks in scopes.items():
        measured = MEASURED_ZLIB1_BYTES * tasks
        planning = PLANNING_CHECKPOINT_BYTES * tasks + RESERVE_BYTES
        budget[name] = {"tasks": tasks, "raw_best_weight_bytes": RAW_MODEL_BYTES * tasks,
                        "measured_zlib1_best_weight_bytes": measured,
                        "planning_bytes_with_reserve": planning,
                        "planning_deficit_bytes_at_observation": max(0, planning - free),
                        "capacity_ready_at_observation": free >= planning}
    report = {"observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "teams_path": args.teams_path, "free_bytes_at_observation": free,
              "checkpoint_policy": {"required": "every task best FP32 encoder/head weights",
                                    "optimizer_state_saved": False, "old_epoch_weights_saved": False,
                                    "temporary_restore_weights_retained": False,
                                    "format": "torch.save + zlib-1 + dual SHA256 manifest + bitwise recovery check"},
              "benchmark": {"raw_bytes": RAW_MODEL_BYTES, "zlib1_bytes": MEASURED_ZLIB1_BYTES,
                            "ratio": MEASURED_ZLIB1_BYTES / RAW_MODEL_BYTES,
                            "planning_ratio_on_420MB": 0.70, "reserve_bytes": RESERVE_BYTES},
              "scopes": budget,
              "matrix_policy": "never shrink matrix, delete existing data, or omit checkpoints; block when capacity is insufficient"}
    atomic_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
