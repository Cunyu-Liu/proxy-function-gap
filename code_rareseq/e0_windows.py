#!/usr/bin/env python3
"""RARE-Seq E0 — 正例 101nt 窗口提取（bbad307 规范峰 → slop50 → FASTA）。

输入: --repo（Benchmark-RBP 根）--genome（chr 命名 FASTA，需 .fai）--out
产出（每实验）:
  {RBP}_{CELL}.windows.tsv   window_id/chrom/start/end/strand/fc/seq/len/n_dup
  {RBP}_{CELL}.windows.fasta 聚类用
  window_stats.tsv           全实验汇总（输入/去重/裁剪/跳过/最终）
协议: 交联位点=峰5'端碱基；±50nt → 101nt；负链反向互补；T→U；非ACGU跳过。
去重策略: 同 (chrom,start,strand) 重复峰合并为一个窗口，n_dup 记录重数（审计用）。
"""
import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd
import pysam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_run import RunLogger  # noqa: E402
from rareseq_common import chrom_sizes_from_fai, extract_seq, read_bed, slop_bed, write_bed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--exps", default=None, help="可选: 逗号分隔实验名子集")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    proc = os.path.join(args.repo, "data/datasets/ENCODE/processed")
    stats = pd.read_csv(os.path.join(args.repo, "data/datasets/ENCODE/stats.ENCODE.csv"))
    names = [f"{r.rbp}_{r.cell}" for r in stats.itertuples()]
    if args.exps:
        keep = set(args.exps.split(","))
        names = [n for n in names if n in keep]

    if not os.path.exists(args.genome + ".fai"):
        pysam.faidx(args.genome)
    gfile = os.path.join(args.out, "chrom.sizes")
    chrom_sizes_from_fai(args.genome + ".fai", gfile)
    fasta = pysam.FastaFile(args.genome)

    lg = RunLogger(args.log, stage="E0", kind="pos-windows")
    lg.start(config={"repo": args.repo, "genome": args.genome, "slop": 50, "dedup": "by(chrom,start,strand)"},
             seed=None, data_version="bbad307-repo-main-2024-01-31", inputs=[proc])

    all_stats = []
    for name in names:
        fcl = os.path.join(proc, name, "peaks.crosslink.bed")
        if not os.path.exists(fcl):
            continue
        rows = read_bed(fcl)
        cnt = Counter((c, s, st) for c, s, e, _, _, st in rows)
        uniq = []
        seen = set()
        for c, s, e, n_, sc, st in rows:
            k = (c, s, st)
            if k in seen:
                continue
            seen.add(k)
            uniq.append((c, s, e, f"{name}|{c}|{s}|{st}", sc, st))
        tmp_in = os.path.join(args.out, f"{name}.cl.uniq.bed")
        tmp_slop = os.path.join(args.out, f"{name}.cl.slop50.bed")
        write_bed(uniq, tmp_in)
        slop_bed(tmp_in, tmp_slop, gfile, 50)
        slopped = read_bed(tmp_slop)
        n_clip = sum(1 for r in slopped if r[2] - r[1] != 101)
        n_skip = 0
        tsv = os.path.join(args.out, f"{name}.windows.tsv")
        fa = os.path.join(args.out, f"{name}.windows.fasta")
        n_final = 0
        with open(tsv, "w") as ft, open(fa, "w") as ff:
            ft.write("window_id\tchrom\tstart\tend\tstrand\tfc\tseq\tlen\tn_dup\n")
            for c, s, e, wid, sc, st in slopped:
                seq = extract_seq(fasta, c, s, e, st)
                if seq is None:
                    n_skip += 1
                    continue
                cl_pos = s + 50 if st == "+" else e - 51
                key = (c, cl_pos, st)
                ft.write(f"{wid}\t{c}\t{s}\t{e}\t{st}\t{sc}\t{seq}\t{len(seq)}\t{cnt.get(key,1)}\n")
                ff.write(f">{wid}\n{seq}\n")
                n_final += 1
        os.unlink(tmp_in)
        os.unlink(tmp_slop)
        all_stats.append({"experiment": name, "n_input": len(rows), "n_unique": len(uniq),
                          "n_dup_removed": len(rows) - len(uniq), "n_clipped": n_clip,
                          "n_nonACGU_skipped": n_skip, "n_final": n_final})
        if len(all_stats) % 25 == 0:
            print(f"progress {len(all_stats)}/{len(names)}", flush=True)
    sdf = pd.DataFrame(all_stats)
    sdf.to_csv(os.path.join(args.out, "window_stats.tsv"), sep="\t", index=False)
    tot = {k: int(sdf[k].sum()) for k in ("n_input", "n_unique", "n_dup_removed", "n_clipped", "n_nonACGU_skipped", "n_final")}
    json.dump(tot, open(os.path.join(args.out, "window_totals.json"), "w"), indent=2)
    lg.finish(metrics=tot, outputs=[args.out])
    print(json.dumps(tot), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()