#!/usr/bin/env python3
"""E1 多种子 vs 检索的逐窗配对 bootstrap；实验内重采样，实验/种子宏平均。"""
import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from e1_locked_logger import E1RunLogger as RunLogger
from log_run import sha256_file

_DATA = None


def both_metrics(y, score):
    if y.min() == y.max():
        return np.nan, np.nan
    return roc_auc_score(y, score), average_precision_score(y, score)


def one_draw(args):
    seed, groups = args
    rng = np.random.default_rng(seed) if seed is not None else None
    per_exp = []
    for y, reference, modern in groups:
        ii = rng.integers(0, len(y), len(y)) if rng is not None else np.arange(len(y))
        yr = y[ii]
        ref = np.array(both_metrics(yr, reference[ii]))
        mod = np.mean([both_metrics(yr, modern[j, ii]) for j in range(modern.shape[0])], axis=0)
        per_exp.append(mod - ref)
    return np.nanmean(per_exp, axis=0)


def init_worker(groups):
    global _DATA
    _DATA = groups


def worker(seed):
    return one_draw((seed, _DATA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modern", action="append", required=True, help="seed=predictions.tsv.gz")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", default="tail,head")
    ap.add_argument("--tracks", default="neg1c,neg2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    ap.add_argument("--stage", default="E1", choices=["E1", "E2"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage=args.stage, kind="paired-bootstrap")
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=args.seed, data_version=args.data_version,
             inputs=args.modern + [args.reference], code_version=config["code_sha256"])
    base_keys = ["experiment", "rbp", "tier", "fold", "track", "window_id", "label"]
    keys = base_keys + ["occurrence"]
    reference = pd.read_csv(args.reference, sep="\t")
    reference["occurrence"] = reference.groupby(base_keys).cumcount()
    reference = reference[keys + ["score"]].rename(columns={"score": "reference"})
    modern = None
    seeds = []
    for spec in args.modern:
        seed, path = spec.split("=", 1)
        seeds.append(int(seed))
        frame = pd.read_csv(path, sep="\t", usecols=base_keys + ["score"])
        frame["occurrence"] = frame.groupby(base_keys).cumcount()
        frame = frame.rename(columns={"score": f"seed_{seed}"})
        if modern is None:
            modern = frame
        else:
            modern = modern.merge(frame, on=keys, validate="one_to_one")
    data = reference.merge(modern, on=keys, validate="one_to_one")
    score_cols = [f"seed_{s}" for s in seeds]
    rows = []
    for track, tier in [(t, s) for t in args.tracks.split(",") for s in args.tiers.split(",")]:
        subset = data[(data.track == track) & (data.tier == tier)]
        groups = []
        for _, g in subset.groupby("experiment", sort=True):
            groups.append((g.label.to_numpy(), g.reference.to_numpy(), g[score_cols].to_numpy().T))
        observed = one_draw((None, groups))
        boot_path = os.path.join(args.out, f"bootstrap_{track}_{tier}.npz")
        if args.resume and os.path.exists(boot_path):
            boot = np.load(boot_path)["deltas"]
            if boot.shape != (args.n_boot, 2) or not np.isfinite(boot).all():
                raise ValueError(f"invalid resume artifact: {boot_path}")
            print("BOOT_REUSED", track, tier, flush=True)
        else:
            seeds_boot = np.random.SeedSequence(args.seed).generate_state(args.n_boot) + np.uint32(args.seed)
            if args.jobs > 1:
                with mp.Pool(args.jobs, initializer=init_worker, initargs=(groups,)) as pool:
                    boot = np.array(list(pool.imap_unordered(worker, seeds_boot, chunksize=20)))
            else:
                boot = np.array([one_draw((s, groups)) for s in seeds_boot])
            np.savez_compressed(boot_path, deltas=boot)
        for j, metric in enumerate(("auroc", "auprc")):
            values = boot[:, j]
            values = values[np.isfinite(values)]
            p = min(1.0, 2 * (1 + min(np.sum(values <= 0), np.sum(values >= 0))) / (len(values) + 1))
            rows.append({"track": track, "slice": tier, "metric": metric, "n_experiments": len(groups),
                         "n_seeds": len(seeds), "n_boot": len(values), "observed_delta": observed[j],
                         "ci_low": np.quantile(values, 0.025), "ci_high": np.quantile(values, 0.975),
                         "p_two_sided": p})
        print("BOOT_DONE", track, tier, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(args.out, "paired_bootstrap.tsv"), sep="\t", index=False)
    lg.finish(metrics={"n_rows": len(result), "results": result.to_dict(orient="records")},
              outputs=[args.out], notes="neg2 duplicate IDs retained with deterministic occurrence index")
    print(result.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
