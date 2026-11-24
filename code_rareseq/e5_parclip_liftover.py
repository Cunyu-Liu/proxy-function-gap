#!/usr/bin/env python3
"""Lift over frozen Mukherjee PAR-CLIP crosslinks from hg19 to hg38."""
import argparse
import os
import subprocess

import pandas as pd

from log_run import RunLogger, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parclip-dir", required=True, help="Mukherjee-PAR-CLIP/processed")
    ap.add_argument("--chain", required=True)
    ap.add_argument("--main-exps", required=True)
    ap.add_argument("--crossmap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=False)
    shared = {line.strip().rsplit("_", 1)[0] for line in open(args.main_exps) if line.strip()}
    experiments = sorted(name for name in os.listdir(args.parclip_dir)
                         if os.path.isdir(os.path.join(args.parclip_dir, name))
                         and name.split("_", 1)[0] in shared)
    if not experiments:
        raise ValueError("No PAR-CLIP experiments shared with the frozen main set")
    lg = RunLogger(args.log, stage="E5", kind="parclip-liftover")
    lg.start(config=vars(args), data_version="Mukherjee-PAR-CLIP-hg19-to-hg38",
             inputs=[args.chain, args.main_exps], code_version=sha256_file(os.path.abspath(__file__)))
    rows = []
    try:
        for experiment in experiments:
            source = os.path.join(args.parclip_dir, experiment, "peaks.crosslink.bed")
            target_dir = os.path.join(args.out, experiment)
            os.makedirs(target_dir)
            target = os.path.join(target_dir, "peaks.crosslink.bed")
            env = os.environ.copy()
            env["PYTHONPATH"] = "/home/ProxyFunctionGap/pydeps"
            result = subprocess.run([args.crossmap, "bed", args.chain, source, target],
                                    capture_output=True, text=True, env=env)
            if result.returncode:
                raise RuntimeError(f"{experiment}: {result.stderr[-2000:]}")
            columns = ["chrom", "start", "end", "name", "score", "strand"]
            original = pd.read_csv(source, sep="\t", header=None, names=columns)
            lifted = pd.read_csv(target, sep="\t", header=None, names=columns)
            if ((lifted.end - lifted.start) != 1).any():
                raise ValueError(f"{experiment}: lifted non-point crosslink")
            before, after = len(original), len(lifted)
            lifted = lifted.drop_duplicates(["chrom", "start", "end", "strand"]).sort_values(["chrom", "start"])
            lifted.to_csv(target, sep="\t", header=False, index=False)
            rows.append({"experiment": experiment, "rbp": experiment.split("_", 1)[0],
                         "input_sites": before, "lifted_sites": after, "unique_sites": len(lifted),
                         "failed_sites": before - after, "failure_rate": (before - after) / before,
                         "input_sha256": sha256_file(source), "output_sha256": sha256_file(target)})
            print(experiment, before, after, len(lifted), flush=True)
        table = pd.DataFrame(rows)
        table.to_csv(os.path.join(args.out, "liftover_audit.tsv"), sep="\t", index=False)
        result = {"n_experiments": len(table), "n_rbps": table.rbp.nunique(),
                  "input_sites": int(table.input_sites.sum()), "lifted_sites": int(table.lifted_sites.sum()),
                  "unique_sites": int(table.unique_sites.sum()), "failure_rate": float(table.failed_sites.sum() / table.input_sites.sum())}
        lg.finish(metrics=result, outputs=[args.out])
        print("LIFTOVER_DONE", result, flush=True)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
