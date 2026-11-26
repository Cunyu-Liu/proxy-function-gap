#!/usr/bin/env python3
"""Create deterministic experiment chunks and global representative mappings."""
import argparse
import json
import os
from e1_frozen_baselines import load_tiers, rbp_of, read_exps
from e1_finetune import atomic_json
from log_run import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--neg-dir", required=True)
    parser.add_argument("--positives-audit", required=True)
    parser.add_argument("--exps-file", required=True)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--tracks", default="neg1c,neg2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-experiments", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.chunk_experiments < 1:
        raise ValueError("chunk-experiments must be positive")
    os.makedirs(args.out, exist_ok=True)
    experiments = read_exps(args.exps_file)
    tiers = load_tiers(args.positives_audit)
    representatives = []
    if args.folds == "0" and args.seed == 0:
        for tier in ("head", "mid", "tail"):
            subset = [experiment for experiment in experiments if tiers[rbp_of(experiment)] == tier]
            if not subset:
                raise ValueError(f"Missing tier {tier}")
            experiment = min(subset)
            representatives.extend({"experiment": experiment, "track": track}
                                   for track in args.tracks.split(","))
    chunks = []
    for index, start in enumerate(range(0, len(experiments), args.chunk_experiments)):
        subset = experiments[start:start + args.chunk_experiments]
        path = os.path.join(args.out, f"chunk_{index:03d}.exps.txt")
        with open(path, "w") as handle:
            handle.write("\n".join(subset) + "\n")
        chunks.append({"index": index, "experiment_start": start, "experiments": subset,
                       "n_tasks": len(subset) * len(args.tracks.split(",")),
                       "exps_file": path, "exps_sha256": sha256_file(path)})
    atomic_json(os.path.join(args.out, "chunk_manifest.json"),
                {"source_exps_file": args.exps_file, "source_exps_sha256": sha256_file(args.exps_file),
                 "folds": args.folds, "seed": args.seed, "tracks": args.tracks,
                 "chunk_experiments": args.chunk_experiments, "n_experiments": len(experiments),
                 "n_tasks": len(experiments) * len(args.folds.split(",")) * len(args.tracks.split(",")),
                 "n_chunks": len(chunks), "chunks": chunks})
    atomic_json(os.path.join(args.out, "representatives.json"),
                {"policy": "fold0_seed0_tier_lexicographic_permanent", "tasks": representatives})
    print("E1_FINETUNE_CHUNK_PLAN_DONE", len(experiments),
          len(experiments) * len(args.folds.split(",")) * len(args.tracks.split(",")), len(chunks))


if __name__ == "__main__":
    main()
