#!/usr/bin/env python3
"""Aggregate the accepted five-fold three-seed RNA-FM fine-tuning matrix."""
import argparse
import json
import os

import pandas as pd

from e1_finetune import atomic_json
from e1_locked_logger import E1RunLogger as RunLogger
from log_run import sha256_file


def run_dir(root, mode, fold, seed, rank):
    if fold == 0 and seed == 0 and mode == "lora":
        return os.path.join(root, f"f0_lora_r{rank}_s0")
    return os.path.join(root, f"f{fold}_{mode}_s{seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    verification = json.load(open(os.path.join(args.root, "final_matrix_verification.json")))
    if verification.get("status") != "passed":
        raise ValueError("Final matrix has not passed independent verification")
    selection = json.load(open(os.path.join(args.root, "fold0_lora_rank_selection.json")))
    rank = int(selection["selected_rank"])
    lg = RunLogger(args.log, stage="E1", kind="finetune-matrix-summary")
    lg.start(config=vars(args), data_version="e0-v2-id90-strict3train-main",
             inputs=[os.path.join(args.root, "final_matrix_verification.json")],
             code_version=sha256_file(os.path.abspath(__file__)))
    frames, manifest = [], []
    for mode in ("lora", "full"):
        for fold in range(5):
            for seed in range(3):
                directory = run_dir(args.root, mode, fold, seed, rank)
                config = json.load(open(os.path.join(directory, "finetune_config.json")))
                frame = pd.read_csv(os.path.join(directory, "finetune_results.tsv"), sep="\t")
                frame["mode"] = mode
                frame["seed"] = seed
                frame["lora_r"] = rank if mode == "lora" else 0
                frames.append(frame)
                manifest.append({"mode": mode, "fold": fold, "seed": seed, "lora_r": frame.lora_r.iloc[0],
                                 "run_dir": directory, "n_rows": len(frame),
                                 "config_sha256": sha256_file(os.path.join(directory, "finetune_config.json")),
                                 "code_sha256": config["code_sha256"]})
    if len(frames) != 30:
        raise ValueError(f"Expected 30 runs, got {len(frames)}")
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(os.path.join(args.root, "final_results.tsv"), sep="\t", index=False)
    pd.DataFrame(manifest).to_csv(os.path.join(args.root, "final_run_manifest.tsv"), sep="\t", index=False)
    results.groupby(["mode", "track", "split", "tier"])[["auroc", "auprc"]].mean().round(6).to_csv(
        os.path.join(args.root, "final_summary.tsv"), sep="\t")
    results.groupby(["mode", "fold", "seed", "track", "split", "tier"])[["auroc", "auprc"]].mean().round(6).to_csv(
        os.path.join(args.root, "final_fold_seed_summary.tsv"), sep="\t")
    atomic_json(os.path.join(args.root, "final_summary_config.json"),
                {"n_runs": len(frames), "n_result_rows": len(results), "selected_rank": rank,
                 "selection_sha256": sha256_file(os.path.join(args.root, "fold0_lora_rank_selection.json")),
                 "data_version": "e0-v2-id90-strict3train-main",
                 "scope": "descriptive RNA-FM LoRA/full matrix; no SOTA claim",
                 "selection": "fold0/seed0 validation only; test excluded from all configuration choices"})
    lg.finish(metrics={"n_runs": len(frames), "n_result_rows": len(results), "selected_rank": rank},
              outputs=[args.root])
    print("E1_FINETUNE_MATRIX_SUMMARY_DONE", len(frames), len(results))


if __name__ == "__main__":
    main()
