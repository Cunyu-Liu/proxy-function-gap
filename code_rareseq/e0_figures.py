#!/usr/bin/env python3
"""E0 分布图（§4.4/§4.6）: 长尾曲线 / n_eff vs 表观 / 组成偏差 / 双系一致性。"""
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = "/root/rareseq/data/e0"
FIG = "/root/rareseq/data/e0/figures"
os.makedirs(FIG, exist_ok=True)

# 1. 长尾曲线（RBP 级总位点，降序）
pa = pd.read_csv(f"{E}/positives_audit.tsv", sep="\t")
per_rbp = pa.groupby("rbp")["n_sites"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(per_rbp) + 1), per_rbp.values, lw=1.5)
ax.set_yscale("log"); ax.set_xlabel("RBP rank"); ax.set_ylabel("total crosslink sites (log)")
ax.set_title("RBP site-count long tail (223 experiments, 150 RBPs)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/longtail_rbp.png", dpi=150); plt.close(fig)

# 2. n_eff vs apparent（从 clusters.id90.clstr 按实验统计）
cl = f"{E}/splits/clusters.id90.clstr"
if os.path.exists(cl):
    from collections import defaultdict
    exp_clusters = defaultdict(set)
    exp_windows = defaultdict(int)
    sizes = []
    cur = 0
    cid = None
    for line in open(cl):
        if line.startswith(">"):
            if cur: sizes.append(cur)
            cur = 0
            cid = int(line.split()[1])
        else:
            cur += 1
            # 行形如: 0	101nt, >EXP|chr|pos|strand... at +/90%...
            gt = line.split(">")[1]
            wid = gt.split("...")[0]
            exp = wid.split("|")[0]
            exp_windows[exp] += 1
            exp_clusters[exp].add(cid)
    if cur: sizes.append(cur)
    rows = [{"experiment": e, "n_windows": exp_windows[e], "n_clusters": len(exp_clusters[e])}
            for e in exp_windows]
    cs = pd.DataFrame(rows)
    cs.to_csv(f"{E}/splits/cluster_stats.tsv", sep="\t", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(cs["n_windows"], cs["n_clusters"], s=8, alpha=0.6)
    lim = [1, max(cs["n_windows"].max(), cs["n_clusters"].max()) * 1.2]
    axes[0].plot(lim, lim, "k--", lw=0.8)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("apparent windows"); axes[0].set_ylabel("n_eff (90% clusters)")
    axes[0].set_title("Effective vs apparent sample size")
    axes[1].hist(np.log10(np.array(sizes) + 1), bins=40)
    axes[1].set_xlabel("log10(cluster size + 1)"); axes[1].set_ylabel("count")
    axes[1].set_title(f"Cluster size distribution (n={len(sizes)})")
    fig.tight_layout(); fig.savefig(f"{FIG}/neff_clusters.png", dpi=150); plt.close(fig)

# 3. 组成偏差（composition_bias 为 long 格式: experiment/track/U_frac/GC_frac → pivot 求 pos-neg）
cb_path = f"{E}/leakage/composition_bias.tsv"
if os.path.exists(cb_path):
    cb = pd.read_csv(cb_path, sep="\t")
    pos = cb[cb["track"] == "pos"].set_index("experiment")[["U_frac", "GC_frac"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for track, color in [("neg1c", "tab:blue"), ("neg2", "tab:orange")]:
        sub = cb[cb["track"] == track].set_index("experiment")
        if sub.empty: continue
        dU = (pos["U_frac"] - sub["U_frac"]).dropna()
        dGC = (pos["GC_frac"] - sub["GC_frac"]).dropna()
        axes[0].hist(dU, bins=40, alpha=0.5, label=track, color=color)
        axes[1].hist(dGC, bins=40, alpha=0.5, label=track, color=color)
    axes[0].axvline(0, color="k", lw=0.8); axes[1].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("U fraction: pos - neg"); axes[0].set_ylabel("experiments")
    axes[1].set_xlabel("GC fraction: pos - neg"); axes[0].legend(); axes[1].legend()
    axes[0].set_title("Nucleotide composition bias (pos - neg)")
    fig.tight_layout(); fig.savefig(f"{FIG}/composition_bias.png", dpi=150); plt.close(fig)

# 4. 双系一致性
dc = pd.read_csv(f"{E}/dual_cellline_consistency.tsv", sep="\t")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(dc["overlap50_K562_frac"].dropna(), bins=30, alpha=0.6, label="K562 in HepG2")
axes[0].hist(dc["overlap50_HepG2_frac"].dropna(), bins=30, alpha=0.6, label="HepG2 in K562")
axes[0].set_xlabel("fraction of sites covered (±50nt)"); axes[0].set_ylabel("RBPs"); axes[0].legend()
axes[0].set_title("Cross-cell-line site overlap (73 dual RBPs)")
axes[1].scatter(dc["overlap50_K562_frac"], dc["fc_spearman"], s=12, alpha=0.7)
axes[1].set_xlabel("overlap50 (K in H)"); axes[1].set_ylabel("Spearman FC (shared sites)")
axes[1].set_title("Overlap vs FC consistency")
fig.tight_layout(); fig.savefig(f"{FIG}/dual_consistency.png", dpi=150); plt.close(fig)

print("figures written:", sorted(os.listdir(FIG)))