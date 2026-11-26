#!/usr/bin/env python3
"""Select one global LoRA rank from fold0/seed0 tail validation AUPRC only."""
import argparse
import json
import os

import pandas as pd

from e1_finetune import atomic_json
from log_run import sha256_file


def select_rank(run_by_rank):
    means = {}
    inputs = {}
    for rank in sorted(run_by_rank):
        run = run_by_rank[rank]
        if not os.path.isfile(os.path.join(run, ".train_complete")):
            raise ValueError(f"Rank {rank}: training incomplete")
        if not os.path.isfile(os.path.join(run, ".independent_validation_complete")):
            raise ValueError(f"Rank {rank}: independent checkpoint validation missing")
        independent = json.load(open(os.path.join(run, "independent_validation.json")))
        if independent.get("test_used") is not False:
            raise ValueError(f"Rank {rank}: independent validation used test")
        forbidden = ["finetune_results.tsv", "finetune_test_predictions.tsv.gz", ".eval_complete"]
        if any(os.path.exists(os.path.join(run, name)) for name in forbidden):
            raise ValueError(f"Rank {rank}: test artifact found in selection input")
        config = json.load(open(os.path.join(run, "finetune_config.json")))
        args = config["args"]
        if args["mode"] != "lora" or int(args["lora_r"]) != rank:
            raise ValueError(f"Rank {rank}: config mismatch")
        if args["folds"] != "0" or int(args["seed"]) != 0:
            raise ValueError("Rank selection is restricted to fold0/seed0")
        if args["data_version"] != "e0-v2-id90-strict3train-main":
            raise ValueError("Rank selection requires strict3train data")
        path = os.path.join(run, "finetune_validation_results.tsv")
        frame = pd.read_csv(path, sep="\t")
        if frame.empty or set(frame.split) != {"val"} or frame.tier.isna().any():
            raise ValueError(f"Rank {rank}: invalid validation results")
        tail = frame[frame.tier == "tail"]
        if tail.empty or not tail.auprc.notna().all():
            raise ValueError(f"Rank {rank}: empty or invalid tail validation")
        means[rank] = float(tail.auprc.mean())
        inputs[str(rank)] = {"path": path, "sha256": sha256_file(path), "n_tail_tasks": len(tail)}
    selected = min(means, key=lambda rank: (-means[rank], rank))
    return {"selection_metric": "tail experiment-track macro mean validation AUPRC at accepted best epoch",
            "selection_fold": 0, "selection_seed": 0, "candidate_ranks": sorted(means),
            "tail_validation_auprc_by_rank": means, "selected_rank": selected,
            "tie_break": "smaller rank", "test_used": False, "inputs": inputs}


def main():
    parser = argparse.ArgumentParser()
    for rank in (4, 8, 16):
        parser.add_argument(f"--rank{rank}", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = select_rank({rank: getattr(args, f"rank{rank}") for rank in (4, 8, 16)})
    atomic_json(args.out, result)
    print("E1_FINETUNE_RANK_SELECTED " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
