#!/usr/bin/env python3
"""RARE-Seq E0 — bbad307 规范正例集审计（无需基因组）。

输入:
  --repo   Benchmark-RBP 根目录（含 data/datasets/ENCODE/processed 与 stats.ENCODE.csv）
  --meta   ENCODE 当前快照元数据目录（experiments.tsv / dual_cellline_rbps.txt / snapshot_info.json）
  --out    输出目录
产出:
  positives_audit.tsv          每实验峰数/去重/FC分布/链平衡/染色体统计
  snapshot_alignment.json      223 规范实验 vs 当前 ENCODE 250 实验对齐
  dual_cellline_consistency.tsv 73 双系 RBP 峰重叠率与 FC 相关（§4.6）
  fc_sensitivity.tsv           FC 阈值 {2,3,5} × topN {10000,20000} 保留峰数（§4.6）
  rbp_distribution.json        长尾统计: Gini/失衡比/每细胞系实验数（§4.4）
  audit_summary.json           格式校验与异常汇总
所有 bedtools 调用同链(-s)。运行: python e0_positives.py --repo ... --meta ... --out ...
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_run import RunLogger, sha256_file  # noqa: E402


def read_bed6(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            rows.append((p[0], int(p[1]), int(p[2]), p[3], float(p[4]), p[5]))
    return rows


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


def bedtools_intersect_count(a, b, extra=""):
    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as fa:
        pa = fa.name
    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as fb:
        pb = fb.name
    try:
        with open(pa, "w") as f:
            for r in a:
                f.write("\t".join(map(str, r)) + "\n")
        with open(pb, "w") as f:
            for r in b:
                f.write("\t".join(map(str, r)) + "\n")
        cmd = f"bedtools intersect -s -u -wa {extra} -a {pa} -b {pb}"
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout
        return sum(1 for l in out.splitlines() if l.strip())
    finally:
        os.unlink(pa)
        os.unlink(pb)


def slop_rows(rows, b=50):
    return [(c, max(0, s - b), e + b, n, sc, st) for c, s, e, n, sc, st in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    enc = os.path.join(args.repo, "data/datasets/ENCODE")
    proc = os.path.join(enc, "processed")

    lg = RunLogger(args.log, stage="E0", kind="positives-audit")
    lg.start(config={"repo": args.repo, "meta": args.meta}, seed=None,
             data_version="bbad307-repo-main-2024-01-31", inputs=[proc])

    stats = pd.read_csv(os.path.join(enc, "stats.ENCODE.csv"))
    exps = [(r.rbp, r.cell) for r in stats.itertuples()]
    print(f"stats.ENCODE.csv experiments: {len(exps)}", flush=True)

    audit_rows = []
    problems = []
    peaks_cache = {}
    for rbp, cell in exps:
        name = f"{rbp}_{cell}"
        d = os.path.join(proc, name)
        fcl = os.path.join(d, "peaks.crosslink.bed")
        fpk = os.path.join(d, "peaks.bed")
        if not os.path.exists(fcl):
            problems.append(f"MISSING crosslink: {name}")
            continue
        rows = read_bed6(fcl)
        pkrows = read_bed6(fpk) if os.path.exists(fpk) else rows
        peaks_cache[name] = rows
        n = len(rows)
        pos = {(c, s, st) for c, s, e, _, _, st in rows}
        n_uniq = len(pos)
        bad_len = sum(1 for c, s, e, _, _, st in rows if e - s != 1)
        bad_strand = sum(1 for r in rows if r[5] not in "+-")
        fc = np.array([r[4] for r in pkrows])
        chroms = defaultdict(int)
        strands = defaultdict(int)
        for c, s, e, _, _, st in rows:
            chroms[c] += 1
            strands[st] += 1
        audit_rows.append({
            "rbp": rbp, "cell": cell, "n_sites": n, "n_unique": n_uniq,
            "dup_rate": round(1 - n_uniq / n, 4) if n else 0,
            "bad_interval_len": bad_len, "bad_strand": bad_strand,
            "fc_min": float(fc.min()) if len(fc) else None,
            "fc_median": float(np.median(fc)) if len(fc) else None,
            "fc_max": float(fc.max()) if len(fc) else None,
            "frac_fc_lt2": round(float((fc < 2).mean()), 4) if len(fc) else None,
            "n_chroms": len(chroms), "frac_plus": round(strands["+"] / n, 4) if n else None,
            "top_chrom": max(chroms, key=chroms.get) if chroms else "",
            "at_cap_20k": int(n >= 20000),
        })
        if bad_len or bad_strand:
            problems.append(f"FORMAT {name}: bad_len={bad_len} bad_strand={bad_strand}")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(os.path.join(args.out, "positives_audit.tsv"), sep="\t", index=False)
    print(f"audited {len(audit)} experiments; problems: {len(problems)}", flush=True)

    # ---- snapshot alignment ----
    cur = pd.read_csv(os.path.join(args.meta, "experiments.tsv"), sep="\t")
    cur_pairs = {(r.rbp, r.cell_line) for r in cur.itertuples() if r.n_peak_beds > 0}
    repo_pairs = set(exps)
    repo_rbps = {r for r, c in repo_pairs}
    cur_rbps = {r for r, c in cur_pairs}
    dual_repo = sorted({r for r in repo_rbps if (r, "K562") in repo_pairs and (r, "HepG2") in repo_pairs})
    dual_cur = sorted({r for r in cur_rbps if (r, "K562") in cur_pairs and (r, "HepG2") in cur_pairs})
    alignment = {
        "repo_experiments": len(repo_pairs), "repo_rbps": len(repo_rbps),
        "repo_dual_rbps": len(dual_repo),
        "current_experiments": len(cur_pairs), "current_rbps": len(cur_rbps),
        "current_dual_rbps": len(dual_cur),
        "repo_only_experiments": sorted(f"{r}_{c}" for r, c in repo_pairs - cur_pairs),
        "current_only_experiments": sorted(f"{r}_{c}" for r, c in cur_pairs - repo_pairs),
        "dual_repo_list": dual_repo,
        "dual_new_in_current": sorted(set(dual_cur) - set(dual_repo)),
        "repo_cell_counts": {c: int(sum(1 for _, cc in repo_pairs if cc == c)) for c in ("K562", "HepG2")},
    }
    json.dump(alignment, open(os.path.join(args.out, "snapshot_alignment.json"), "w"), indent=2, ensure_ascii=False)
    print(f"dual repo={len(dual_repo)} current={len(dual_cur)}; repo_only={len(alignment['repo_only_experiments'])} cur_only={len(alignment['current_only_experiments'])}", flush=True)

    # ---- dual cell-line consistency (§4.6) ----
    cons_rows = []
    for rbp in dual_repo:
        a = peaks_cache.get(f"{rbp}_K562")
        b = peaks_cache.get(f"{rbp}_HepG2")
        if not a or not b:
            continue
        aw, bw = slop_rows(a, 50), slop_rows(b, 50)
        a_hit = bedtools_intersect_count(aw, bw)
        b_hit = bedtools_intersect_count(bw, aw)
        # FC correlation at exact shared positions (same strand)
        bmap = {(c, s, st): sc for c, s, e, _, sc, st in b}
        shared = [(r[4], bmap[(r[0], r[1], r[5])]) for r in a if (r[0], r[1], r[5]) in bmap]
        rho = float(spearmanr([x for x, y in shared], [y for x, y in shared]).statistic) if len(shared) >= 10 else None
        cons_rows.append({
            "rbp": rbp, "n_K562": len(a), "n_HepG2": len(b),
            "overlap50_K562_frac": round(a_hit / len(a), 4),
            "overlap50_HepG2_frac": round(b_hit / len(b), 4),
            "n_exact_shared": len(shared), "fc_spearman": round(rho, 4) if rho is not None else None,
        })
    cons = pd.DataFrame(cons_rows)
    cons.to_csv(os.path.join(args.out, "dual_cellline_consistency.tsv"), sep="\t", index=False)
    print(f"dual consistency: {len(cons)} RBPs; median overlap50 K562={cons['overlap50_K562_frac'].median():.3f}", flush=True)

    # ---- FC sensitivity (§4.6) ----
    sens_rows = []
    for rbp, cell in exps:
        name = f"{rbp}_{cell}"
        fpk = os.path.join(proc, name, "peaks.bed")
        if not os.path.exists(fpk):
            continue
        fc = np.array([r[4] for r in read_bed6(fpk)])
        row = {"rbp": rbp, "cell": cell, "n_total": len(fc)}
        for t in (2.0, 3.0, 5.0):
            row[f"n_fc_ge{t:g}"] = int((fc >= t).sum())
        for n in (10000, 20000):
            row[f"n_top{n}"] = min(n, len(fc))
        sens_rows.append(row)
    pd.DataFrame(sens_rows).to_csv(os.path.join(args.out, "fc_sensitivity.tsv"), sep="\t", index=False)

    # ---- RBP distribution (§4.4) ----
    per_rbp = audit.groupby("rbp")["n_sites"].sum()
    per_rbp_max = audit.groupby("rbp")["n_sites"].max()
    dist = {
        "n_rbps": int(len(per_rbp)),
        "total_sites": int(audit["n_sites"].sum()),
        "total_unique_sites": int(audit["n_unique"].sum()),
        "gini_per_rbp_total": round(gini(per_rbp.values), 4),
        "gini_per_experiment": round(gini(audit["n_sites"].values), 4),
        "imbalance_ratio_max_min": round(float(per_rbp_max.max() / max(per_rbp_max.min(), 1)), 2),
        "sites_quantiles": {q: float(np.quantile(audit["n_sites"], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "n_at_cap": int(audit["at_cap_20k"].sum()),
        "cell_counts": alignment["repo_cell_counts"],
        "mean_dup_rate": round(float(audit["dup_rate"].mean()), 4),
        "max_dup_rate": round(float(audit["dup_rate"].max()), 4),
    }
    json.dump(dist, open(os.path.join(args.out, "rbp_distribution.json"), "w"), indent=2)
    json.dump({"problems": problems, "n_audited": len(audit)},
              open(os.path.join(args.out, "audit_summary.json"), "w"), indent=2, ensure_ascii=False)
    lg.finish(metrics={"n_experiments": len(audit), "n_dual": len(dual_repo),
                       "total_sites": dist["total_sites"], "gini": dist["gini_per_rbp_total"],
                       "n_problems": len(problems)},
              outputs=[os.path.join(args.out, f) for f in
                       ("positives_audit.tsv", "snapshot_alignment.json", "dual_cellline_consistency.tsv",
                        "fc_sensitivity.tsv", "rbp_distribution.json", "audit_summary.json")])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()