#!/usr/bin/env python3
"""E4 天然候选 FNR 的逐候选配对 bootstrap；比较各 nnPU 与 naive。"""
import argparse
import os

import numpy as np
import pandas as pd

from log_run import RunLogger, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", action="append", required=True, help="seed=directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E4", kind="g3-bootstrap")
    config = vars(args).copy(); config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=args.seed, data_version=args.data_version,
             inputs=args.eval_dir, code_version=config["code_sha256"])
    frames = []
    for spec in args.eval_dir:
        seed, directory = spec.split("=", 1)
        for path in os.listdir(directory):
            if not path.startswith("candidate_scores_") or not path.endswith(".tsv.gz"):
                continue
            model = path[len("candidate_scores_"):].split("_neg")[0]
            track = "neg1c" if "neg1c" in path else "neg2"
            if model == "naive":
                continue
            method = pd.read_csv(os.path.join(directory, path), sep="\t")
            naive = pd.read_csv(os.path.join(directory, f"candidate_scores_naive_{track}.tsv.gz"), sep="\t")
            keys = ["experiment", "window_id", "occurrence"]
            method["method_fn"] = (method.calibrated < 0.5).astype(int)
            naive["naive_fn"] = (naive.calibrated < 0.5).astype(int)
            merged = method[keys + ["method_fn"]].merge(naive[keys + ["naive_fn"]], on=keys, validate="one_to_one")
            merged["model"] = model; merged["track"] = track; merged["seed"] = int(seed)
            frames.append(merged)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(os.path.join(args.out, "candidate_fn_pairs.tsv.gz"), sep="\t", index=False, compression="gzip")
    rng = np.random.default_rng(args.seed)
    rows = []
    for (model, track), group in data.groupby(["model", "track"]):
        experiments = [g for _, g in group.groupby("experiment")]
        observed = np.mean([np.mean(g.method_fn.to_numpy() - g.naive_fn.to_numpy()) for g in experiments])
        boot = []
        for _ in range(args.n_boot):
            deltas = []
            for g in experiments:
                method_delta = (g.method_fn.to_numpy() - g.naive_fn.to_numpy())
                ii = rng.integers(0, len(g), len(g))
                deltas.append(method_delta[ii].mean())
            boot.append(np.mean(deltas))
        boot = np.array(boot)
        p = min(1.0, 2 * (1 + min(np.sum(boot <= 0), np.sum(boot >= 0))) / (len(boot) + 1))
        rows.append({"model": model, "track": track, "n_experiments": len(experiments),
                     "n_seeds": group.seed.nunique(), "observed_delta_fnr": observed,
                     "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975), "p_two_sided": p})
        np.savez_compressed(os.path.join(args.out, f"bootstrap_{model}_{track}.npz"), deltas=boot)
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(args.out, "g3_bootstrap.tsv"), sep="\t", index=False)
    lg.finish(metrics={"results": result.to_dict(orient="records")}, outputs=[args.out])
    print(result.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
