#!/usr/bin/env python3
"""RARE-Seq E0 — 负例构造（bbad307 协议复现 + 审计修正版，三轨）。

轨道:
  neg1f  忠实复现: 在含正例(±300nt扩展)的转录本区域内均匀采 1bp，不做有效排除
         （原管线 subtract -s -u -wa 实测为 no-op，负例可落在正例邻域——E0 审计发现）
  neg1c  修正版: 真正 subtract ±300nt 正例扩展区后采样（bedtools subtract -s）
  neg2   其他 RBP（排除同 RBP 双系）的交联位点，去除与本 RBP 正例重叠者，1:1 有放回采样
协议参数: seed=42（neg1 长度加权+均匀位置；neg2 random.Random(42).choices 有放回，
         原管线 neg2 无种子——偏差已记录）；±300nt 扩展；窗口 ±50nt→101nt。
输入: --repo --genome --folds-dir（e0_splits 产出）--windows-dir --out
产出: {name}.negatives.tsv/fasta（fold/track 列）、negatives_stats.tsv、neg_audit.tsv、
      window_fold_map.tsv（正例窗口→折/区域）
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pysam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_run import RunLogger  # noqa: E402
from rareseq_common import bedtools, chrom_sizes_from_fai, extract_seq, read_bed, slop_bed, write_bed  # noqa: E402

SEED = 42
EXT = 300
SLOP = 50


def sample_positions(regions, n, seed):
    """长度加权选区域 + 区域内均匀 1bp；重复重采（忠实 generate_background_no-peak.py）。"""
    rng = np.random.default_rng(seed)
    rows = [(c, s, e, st) for c, s, e, _, _, st in regions if e - s - 1 > 0]
    if not rows or n <= 0:
        return []
    sizes = np.array([e - s - 1 for _, s, e, _ in rows], dtype=float)
    p = sizes / sizes.sum()
    seen = set()
    out = []
    while len(out) < n:
        i = rng.choice(len(rows), p=p)
        c, s, e, st = rows[i]
        pos = s + int(rng.integers(0, e - s))
        key = (c, pos)
        if key in seen:
            continue
        seen.add(key)
        out.append((c, pos, pos + 1, "neg", 0.0, st))
    return out


def bed_from_rows(rows, path):
    write_bed(rows, path)


def rows_from_stdout(out):
    rows = []
    for l in out.splitlines():
        if not l.strip():
            continue
        p = l.split("\t")
        if len(p) >= 6:
            rows.append((p[0], int(p[1]), int(p[2]), p[3], float(p[4]) if p[4] not in (".", "") else 0.0, p[5]))
    return rows


def process_experiment(task):
    (name, rbp, proc_dir, folds_dir, out_dir, genome, gfile, n_folds) = task
    fasta = pysam.FastaFile(genome)
    fcl = os.path.join(proc_dir, name, "peaks.crosslink.bed")
    all_rows = []
    stats = []
    audit = []
    tmp = tempfile.mkdtemp(prefix=f"neg_{name}_")
    try:
        for f in range(n_folds):
            pos_f = os.path.join(folds_dir, "per_exp", name, f"pos.fold{f}.bed")
            ext_f = os.path.join(folds_dir, "per_exp", name, f"ext.fold{f}.bed")
            frame_f = os.path.join(folds_dir, "per_exp", name, f"frame.fold{f}.bed")
            pool_f = os.path.join(folds_dir, f"pool.fold{f}.bed")
            if not os.path.exists(pos_f):
                continue
            pos = read_bed(pos_f)
            n_pos = len(pos)
            if n_pos == 0:
                continue
            frame = read_bed(frame_f)
            # ---- neg1f: 无排除采样 ----
            neg1f = sample_positions(frame, n_pos, SEED)
            # ---- neg1c: 真排除 ----
            sub = os.path.join(tmp, f"sub{f}.bed")
            bedtools(f"bedtools subtract -s -a {frame_f} -b {ext_f} | sort -k1,1 -k2,2n | bedtools merge -s -c 4,5,6 -o collapse,distinct,distinct -delim ';' -i - > {sub}")
            neg1c = sample_positions(read_bed(sub), n_pos, SEED)
            # ---- neg2: 其他 RBP 位点（严格忠实原管线顺序）----
            # 1) 按 name 排除同 RBP（双系）；2) intersect -v 去与本 RBP 1bp 正例重叠（无 -s）；3) slop -300
            other_ext = os.path.join(tmp, f"otherext{f}.bed")
            with open(pool_f) as fi, open(other_ext, "w") as fo:
                for line in fi:
                    p = line.split("\t")
                    if len(p) >= 4 and p[3].rsplit("_", 1)[0] != rbp:
                        fo.write(line)
            noov = os.path.join(tmp, f"noov{f}.bed")
            bedtools(f"bedtools intersect -v -a {other_ext} -b {pos_f} > {noov}")
            centered = os.path.join(tmp, f"center{f}.bed")
            bedtools(f"bedtools slop -b -{EXT} -g {gfile} -i {noov} > {centered}")
            pool_rows = [(c, s, e, n_, sc, st) for c, s, e, n_, sc, st in read_bed(centered) if e > s]
            if pool_rows:
                rng = random.Random(SEED)
                neg2 = [list(r) for r in rng.choices(pool_rows, k=n_pos)]
                neg2 = [(c, s, e, "neg", 0.0, st) for c, s, e, _, _, st in neg2]
            else:
                neg2 = []
            # ---- 窗口化 + 序列 ----
            for track, negs in (("neg1f", neg1f), ("neg1c", neg1c), ("neg2", neg2)):
                nb = os.path.join(tmp, f"{track}{f}.bed")
                sb = os.path.join(tmp, f"{track}{f}.slop.bed")
                bed_from_rows(negs, nb)
                slop_bed(nb, sb, gfile, SLOP)
                n_skip = 0
                n_final = 0
                for c, s, e, _, _, st in read_bed(sb):
                    seq = extract_seq(fasta, c, s, e, st)
                    if seq is None:
                        n_skip += 1
                        continue
                    cl = s + SLOP if st == "+" else e - SLOP - 1
                    wid = f"{name}|{c}|{cl}|{st}|{track}|f{f}"
                    all_rows.append((wid, c, s, e, st, track, f, seq, len(seq)))
                    n_final += 1
                stats.append({"experiment": name, "fold": f, "track": track,
                              "n_pos": n_pos, "n_sampled": len(negs), "n_nonACGU": n_skip, "n_final": n_final})
                # 审计: neg1f 落在本 RBP ±300nt 正例邻域的比例 + 最近正例距离
                if track == "neg1f" and negs:
                    inzone = bedtools(f"bedtools intersect -s -u -wa -a {nb} -b {ext_f}")
                    frac = sum(1 for l in inzone.splitlines() if l.strip()) / len(negs)
                    sub_n = negs[:2000]
                    nb2 = os.path.join(tmp, f"audit{f}.bed")
                    bed_from_rows(sub_n, nb2)
                    # closest 要求排序输入（crosslink BED 原序为 FC 降序）
                    nb2s, pos_fs = nb2 + ".srt", pos_f + ".srt"
                    subprocess.run(f"sort -k1,1 -k2,2n {nb2} > {nb2s}", shell=True, check=True)
                    subprocess.run(f"sort -k1,1 -k2,2n {pos_f} > {pos_fs}", shell=True, check=True)
                    dist_out = bedtools(f"bedtools closest -s -d -a {nb2s} -b {pos_fs}")
                    dists = [int(l.split("\t")[-1]) for l in dist_out.splitlines() if l.strip()]
                    dists = [d for d in dists if d >= 0]
                    audit.append({"experiment": name, "fold": f, "n_pos": n_pos,
                                  "neg1f_in_ext300_frac": round(frac, 4),
                                  "neg1f_dist_median": float(np.median(dists)) if dists else None,
                                  "neg1f_dist_frac_le100": round(float(np.mean([d <= 100 for d in dists])), 4) if dists else None})
        # 写输出
        with open(os.path.join(out_dir, f"{name}.negatives.tsv"), "w") as ft, \
             open(os.path.join(out_dir, f"{name}.negatives.fasta"), "w") as ff:
            ft.write("window_id\tchrom\tstart\tend\tstrand\ttrack\tfold\tseq\tlen\n")
            for wid, c, s, e, st, track, f, seq, ln in all_rows:
                ft.write(f"{wid}\t{c}\t{s}\t{e}\t{st}\t{track}\t{f}\t{seq}\t{ln}\n")
                ff.write(f">{wid}\n{seq}\n")
        return name, stats, audit, len(all_rows)
    finally:
        subprocess.run(f"rm -rf {tmp}", shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--folds-dir", required=True)
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--identity-tag", default="id90")
    ap.add_argument("--exps", default=None)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--skip-precompute", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    proc = os.path.join(args.repo, "data/datasets/ENCODE/processed")
    stats_df = pd.read_csv(os.path.join(args.repo, "data/datasets/ENCODE/stats.ENCODE.csv"))
    exps = [(r.rbp, r.cell) for r in stats_df.itertuples()]
    if args.exps:
        keep = set(args.exps.split(","))
        exps = [(r, c) for r, c in exps if f"{r}_{c}" in keep]
    if not os.path.exists(args.genome + ".fai"):
        pysam.faidx(args.genome)
    gfile = os.path.join(args.out, "chrom.sizes")
    chrom_sizes_from_fai(args.genome + ".fai", gfile)

    lg = RunLogger(args.log, stage="E0", kind="negatives")
    lg.start(config={"seed": SEED, "ext": EXT, "slop": SLOP, "tracks": ["neg1f", "neg1c", "neg2"],
                     "identity_tag": args.identity_tag, "neg2_replacement": True},
             seed=SEED, data_version="bbad307-repo-main-2024-01-31+gencode-v42-basic")

    fmap = pd.read_csv(os.path.join(args.folds_dir, f"region_fold_map.{args.identity_tag}.tsv"), sep="\t")
    rid2fold = dict(zip(fmap.region_id, fmap.fold))
    # 区域 BED 直接从 folds-dir 的 fold-k 文件合并
    all_regions = []
    for f in range(args.folds):
        all_regions += read_bed(os.path.join(args.folds_dir, f"fold-{f}.regions.{args.identity_tag}.bed"))
    allreg_bed = os.path.join(args.folds_dir, "all_regions.bed")
    write_bed(all_regions, allreg_bed)

    if not args.skip_precompute:
        # ---- 预计算: 每实验每折 pos/ext/frame + 每折 pool + 正例窗口折归属 ----
        pe = os.path.join(args.folds_dir, "per_exp")
        os.makedirs(pe, exist_ok=True)
        pool_rows = {f: [] for f in range(args.folds)}
        for i, (rbp, cell) in enumerate(exps):
            name = f"{rbp}_{cell}"
            d = os.path.join(pe, name)
            os.makedirs(d, exist_ok=True)
            fcl = os.path.join(proc, name, "peaks.crosslink.bed")
            if not os.path.exists(fcl):
                continue
            for f in range(args.folds):
                fr = os.path.join(args.folds_dir, f"fold-{f}.regions.{args.identity_tag}.bed")
                pos_f = os.path.join(d, f"pos.fold{f}.bed")
                ext_f = os.path.join(d, f"ext.fold{f}.bed")
                frame_f = os.path.join(d, f"frame.fold{f}.bed")
                bedtools(f"bedtools intersect -s -u -wa -a {fcl} -b {fr} > {pos_f}")
                slop_bed(pos_f, ext_f, gfile, EXT)
                bedtools(f"bedtools intersect -s -wa -u -a {fr} -b {ext_f} > {frame_f}")
                for r in read_bed(ext_f):
                    pool_rows[f].append((r[0], r[1], r[2], name, r[4], r[5]))
            if (i + 1) % 25 == 0:
                print(f"precompute {i+1}/{len(exps)}", flush=True)
        for f in range(args.folds):
            write_bed(pool_rows[f], os.path.join(args.folds_dir, f"pool.fold{f}.bed"))
        # 正例窗口 → 折/区域
        wmap = []
        for rbp, cell in exps:
            name = f"{rbp}_{cell}"
            wt = os.path.join(args.windows_dir, f"{name}.windows.tsv")
            if not os.path.exists(wt):
                continue
            df = pd.read_csv(wt, sep="\t")
            clbed = os.path.join(pe, name, "cl.all.bed")
            with open(clbed, "w") as fh:
                for r in df.itertuples():
                    cl = r.start + 50 if r.strand == "+" else r.end - 51
                    fh.write(f"{r.chrom}\t{cl}\t{cl+1}\t{r.window_id}\t0\t{r.strand}\n")
            out = bedtools(f"bedtools intersect -s -wa -wb -a {clbed} -b {allreg_bed}")
            w2r = {}
            for line in out.splitlines():
                p = line.split("\t")
                w2r[p[3]] = p[9]
            for r in df.itertuples():
                rid = w2r.get(r.window_id, "")
                wmap.append((r.window_id, name, rid, rid2fold.get(rid, -1)))
        pd.DataFrame(wmap, columns=["window_id", "experiment", "region_id", "fold"]).to_csv(
            os.path.join(args.out, "window_fold_map.tsv"), sep="\t", index=False)
        n_unassigned = sum(1 for w in wmap if w[3] == -1)
        print(f"window_fold_map: {len(wmap)} windows, unassigned={n_unassigned}", flush=True)

    # ---- 并行负例构造 ----
    tasks = [(f"{rbp}_{cell}", rbp, proc, args.folds_dir, args.out, args.genome, gfile, args.folds)
             for rbp, cell in exps]
    all_stats, all_audit = [], []
    total_neg = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_experiment, t): t[0] for t in tasks}
        for i, fu in enumerate(as_completed(futs)):
            name, st, au, n = fu.result()
            all_stats += st
            all_audit += au
            total_neg += n
            if (i + 1) % 20 == 0:
                print(f"negatives {i+1}/{len(tasks)} total={total_neg}", flush=True)
    pd.DataFrame(all_stats).to_csv(os.path.join(args.out, "negatives_stats.tsv"), sep="\t", index=False)
    adf = pd.DataFrame(all_audit)
    adf.to_csv(os.path.join(args.out, "neg_audit.tsv"), sep="\t", index=False)
    summ = {"total_neg_windows": total_neg,
            "neg1f_in_ext300_frac_mean": round(float(adf.neg1f_in_ext300_frac.mean()), 4) if len(adf) else None,
            "neg1f_in_ext300_frac_median": round(float(adf.neg1f_in_ext300_frac.median()), 4) if len(adf) else None}
    json.dump(summ, open(os.path.join(args.out, "negatives_summary.json"), "w"), indent=2)
    lg.finish(metrics=summ, outputs=[args.out])
    print(json.dumps(summ), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()