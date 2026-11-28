#!/usr/bin/env python3
"""ProxyFunctionGap 主图 v2（2026-11-28）。

数据路径全部在下方常量区；每个关键数字与最终报告断言核对（第一遍核验），
绘图数值落盘 figures/verification_v2_20261128.json 供独立复算（第二/三遍）。
风格：scientific-figure-making PALETTE + plot-is-all-you-need 图库
oa-tech-pone-0258309-f05（点区间）与 oa-tech-pone-0255450-f01（双热图）演绎。
仅使用已最终验收结果；描述性结果不画显著性星号。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figures" / "data_20261128"
E0 = ROOT / "E0_数据审计" / "results"
OUT = ROOT / "figures"

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B", "red_strong": "#B64342",
    "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E",
    "blue_light": "#ABDDF1", "red_light": "#F6CFCB",
}
MODEL_COLORS = {"MLP": PALETTE["blue_main"], "linear": PALETTE["teal"],
                "kNN": PALETTE["red_strong"], "BCE": PALETTE["blue_main"],
                "deep-kNN": PALETTE["red_strong"]}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11, "axes.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "legend.frameon": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

VERIFICATION = {}


def check(name, computed, expected, tol=5e-6):
    assert abs(float(computed) - float(expected)) < tol, \
        f"{name}: computed {computed} != report {expected}"
    VERIFICATION[name] = {"computed": float(computed), "report": float(expected)}


def panel_label(ax, label):
    ax.text(-0.08, 1.06, label, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("SAVED", name)


# ---------------------------------------------------------------- Fig 1
def fig1():
    pos = pd.read_csv(E0 / "positives_audit.tsv", sep="\t")
    dist = json.loads((E0 / "rbp_distribution.json").read_text())
    cons = pd.read_csv(E0 / "dual_cellline_consistency.tsv", sep="\t")
    per_rbp = pos.groupby("rbp")["n_sites"].sum().sort_values(ascending=False)
    check("fig1_gini", dist["gini_per_rbp_total"], 0.3523, tol=1e-4)
    check("fig1_imbalance", dist["imbalance_ratio_max_min"], 45.98, tol=1e-2)
    overlap = np.concatenate([cons["overlap50_K562_frac"].values,
                              cons["overlap50_HepG2_frac"].values])
    med = float(np.median(overlap))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    ax = axes[0]
    x = np.arange(len(per_rbp))
    ax.fill_between(x, per_rbp.values, step="mid", color=PALETTE["blue_light"], alpha=0.9)
    ax.step(x, per_rbp.values, where="mid", color=PALETTE["blue_main"], lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("RBP rank (by total eCLIP sites)")
    ax.set_ylabel("Total sites per RBP (log)")
    ax.set_xlim(-1, len(per_rbp))
    ax.text(0.97, 0.95, f"Gini = {dist['gini_per_rbp_total']:.3f}\nmax/min = {dist['imbalance_ratio_max_min']:.2f}×",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PALETTE["neutral"], lw=0.8))
    panel_label(ax, "a")

    ax = axes[1]
    exp_sites = pos["n_sites"].values
    bp = ax.boxplot([exp_sites], vert=True, widths=0.45, patch_artist=True,
                    medianprops=dict(color=PALETTE["red_strong"], lw=1.6),
                    boxprops=dict(facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], lw=1.1),
                    whiskerprops=dict(color=PALETTE["blue_main"], lw=1.1),
                    capprops=dict(color=PALETTE["blue_main"], lw=1.1),
                    flierprops=dict(marker="o", ms=3, mfc=PALETTE["neutral"], mec="none"))
    rng = np.random.default_rng(7)
    ax.scatter(1 + rng.uniform(-0.13, 0.13, len(exp_sites)), exp_sites,
               s=7, color=PALETTE["blue_secondary"], alpha=0.45, zorder=3)
    ax.axhline(20000, color=PALETTE["neutral"], lw=1.0, ls="--")
    ax.text(1.28, 20000, "20k cap", va="center", fontsize=9, color="#666666")
    ax.set_xticks([1]); ax.set_xticklabels([f"experiments\n(n={len(exp_sites)})"])
    ax.set_ylabel("Sites per experiment")
    ax.set_ylim(0, 22500)
    panel_label(ax, "b")

    ax = axes[2]
    parts = ax.violinplot([cons["overlap50_K562_frac"].values, cons["overlap50_HepG2_frac"].values],
                          positions=[1, 2], showmedians=True, widths=0.7)
    for body, col in zip(parts["bodies"], [PALETTE["blue_secondary"], PALETTE["teal"]]):
        body.set_facecolor(col); body.set_edgecolor("none"); body.set_alpha(0.55)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color(PALETTE["blue_main"]); parts[key].set_linewidth(1.2)
    for i, col in enumerate(["overlap50_K562_frac", "overlap50_HepG2_frac"], start=1):
        vals = cons[col].values
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.09, 0.09, len(vals)), vals,
                   s=8, color=PALETTE["blue_main"], alpha=0.5, zorder=3)
    ax.axhline(med, color=PALETTE["red_strong"], lw=1.1, ls="--")
    ax.text(2.42, med, f"median {med:.1%}", va="center", fontsize=9, color=PALETTE["red_strong"])
    ax.set_xticks([1, 2]); ax.set_xticklabels(["K562→HepG2", "HepG2→K562"])
    ax.set_ylabel("Cross-cell-line site overlap (±50 nt)")
    ax.set_ylim(0, 1.0); ax.set_xlim(0.5, 2.75)
    panel_label(ax, "c")

    fig.suptitle("eCLIP data audit: frequency long tail and condition-dependent sites",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    VERIFICATION["fig1_overlap_median"] = {"computed": med, "n_rbp_pairs": int(len(cons))}
    save(fig, "Fig1_data_audit_v2_20261128")


# ---------------------------------------------------------------- Fig 2
def fig2():
    df = pd.read_csv(DATA / "paired_bootstrap.tsv", sep="\t")
    check("fig2_neg1c_tail_auroc", df.query("track=='neg1c' and slice=='tail' and metric=='auroc'")["observed_delta"].iloc[0], 0.031365)
    check("fig2_neg1c_tail_auprc", df.query("track=='neg1c' and slice=='tail' and metric=='auprc'")["observed_delta"].iloc[0], 0.043649)
    check("fig2_neg2_tail_auroc", df.query("track=='neg2' and slice=='tail' and metric=='auroc'")["observed_delta"].iloc[0], 0.024114)
    check("fig2_neg2_tail_auprc", df.query("track=='neg2' and slice=='tail' and metric=='auprc'")["observed_delta"].iloc[0], 0.034854)
    rows = [("tail", "auroc"), ("tail", "auprc"), ("mid", "auroc"),
            ("mid", "auprc"), ("head", "auroc"), ("head", "auprc")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, track, label in zip(axes, ["neg1c", "neg2"], ["a", "b"]):
        sub = df[df.track == track].set_index(["slice", "metric"])
        ypos = np.arange(len(rows))[::-1]
        for y, (sl, met) in zip(ypos, rows):
            r = sub.loc[(sl, met)]
            sig = r.ci_low > 0
            col = PALETTE["blue_main"] if sig else PALETTE["blue_light"]
            ax.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=2.2, solid_capstyle="round")
            ax.plot([r.ci_low, r.ci_low], [y - 0.12, y + 0.12], color=col, lw=1.4)
            ax.plot([r.ci_high, r.ci_high], [y - 0.12, y + 0.12], color=col, lw=1.4)
            ax.scatter([r.observed_delta], [y], s=64, color=col, zorder=4,
                       marker="o" if met == "auroc" else "^")
            ax.text(r.ci_high + 0.0012, y, f"{r.observed_delta:+.4f}", va="center", fontsize=9.5)
        ax.axvline(0, color="#888888", lw=1.0)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{sl} · {met.upper()}" for sl, met in rows])
        ax.set_xlabel("Δ (frozen RNA-FM BCE − cluster-dedup cosine kNN)")
        ax.set_title(f"{track} negatives", fontsize=12)
        ax.set_xlim(-0.004, 0.062)
        ax.grid(axis="y", color="#EEEEEE", lw=0.7)
        ax.set_axisbelow(True)
        panel_label(ax, label)
        VERIFICATION[f"fig2_{track}"] = [
            {"slice": sl, "metric": met,
             "delta": float(sub.loc[(sl, met)].observed_delta),
             "ci": [float(sub.loc[(sl, met)].ci_low), float(sub.loc[(sl, met)].ci_high)]}
            for sl, met in rows]
    handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE["blue_main"], label="ΔAUROC"),
               plt.Line2D([], [], marker="^", ls="", color=PALETTE["blue_main"], label="ΔAUPRC"),
               plt.Line2D([], [], color=PALETTE["blue_main"], lw=2.2, label="95% paired bootstrap CI (10,000 resamples)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Tail advantage of frozen RNA-FM over retrieval baseline survives leakage-controlled splits",
                 fontsize=12.5, y=1.03)
    fig.tight_layout()
    save(fig, "Fig2_eclip_tail_forest_v2_20261128")


# ---------------------------------------------------------------- Fig 3
def fig3():
    df = pd.read_csv(DATA / "holdout_per_rbp.tsv", sep="\t")
    ms = pd.read_csv(DATA / "holdout_multiseed.tsv", sep="\t")
    for track, auroc, fpr in [("neg1c", 0.507697, 0.947266), ("neg2", 0.508286, 0.945827)]:
        r = ms.query("track==@track and method=='deep_knn'").iloc[0]
        check(f"fig3_{track}_auroc", r.auroc_ood_mean, auroc)
        check(f"fig3_{track}_fpr95", r.fpr95_mean, fpr)
    check("fig3_neg1c_aurc_energy", ms.query("track=='neg1c' and method=='energy'").iloc[0].aurc_mean, 0.155127)
    check("fig3_neg2_aurc_msp", ms.query("track=='neg2' and method=='msp'").iloc[0].aurc_mean, 0.473079)

    mean = df.groupby(["holdout", "track", "method"])[["auroc_ood", "fpr95", "aurc"]].mean().reset_index()
    rbps = sorted(mean.holdout.unique())
    tracks = ["neg1c", "neg2"]
    knn = mean[mean.method == "deep_knn"].set_index(["holdout", "track"])
    auroc_m = np.array([[knn.loc[(r, t)].auroc_ood for t in tracks] for r in rbps])
    fpr_m = np.array([[knn.loc[(r, t)].fpr95 for t in tracks] for r in rbps])
    aurc_wide = mean.pivot_table(index=["holdout", "track"], columns="method", values="aurc")
    winner = aurc_wide.idxmin(axis=1)
    win_m = np.array([[winner.loc[(r, t)] for t in tracks] for r in rbps])
    n_conflict = 0
    for r in rbps:
        for t in tracks:
            row = aurc_wide.loc[(r, t)]
            best_aurc = row.idxmin()
            best_auroc = mean[(mean.holdout == r) & (mean.track == t)].sort_values("auroc_ood", ascending=False).iloc[0].method
            best_fpr = mean[(mean.holdout == r) & (mean.track == t)].sort_values("fpr95").iloc[0].method
            if not (best_aurc == best_auroc == best_fpr):
                n_conflict += 1
    check("fig3_conflicts", n_conflict, 23, tol=0.5)

    blue_cmap = LinearSegmentedColormap.from_list("blues_seq", ["#EAF3FB", "#98C6DF", "#0F4D92"])
    red_cmap = LinearSegmentedColormap.from_list("reds_seq", ["#FDEFEA", "#FAB59B", "#B64342"])
    fig = plt.figure(figsize=(12.6, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.42)
    for i, (mat, cmap, title, fmt, lo, hi) in enumerate([
            (auroc_m, blue_cmap, "AUROC-OOD (deep-kNN)", "{:.3f}", 0.45, 0.60),
            (fpr_m, red_cmap, "FPR95 (deep-kNN)", "{:.3f}", 0.90, 1.00)]):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        for a in range(mat.shape[0]):
            for b in range(mat.shape[1]):
                ax.text(b, a, fmt.format(mat[a, b]), ha="center", va="center", fontsize=8,
                        color="white" if (mat[a, b] - lo) / (hi - lo) > 0.62 else "#222222")
        ax.set_xticks([0, 1]); ax.set_xticklabels(tracks)
        ax.set_yticks(range(len(rbps)))
        ax.set_yticklabels(rbps if i == 0 else [], fontsize=9)
        ax.set_title(title, fontsize=11.5, pad=8)
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rbps), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.6)
        ax.tick_params(which="minor", length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.outline.set_linewidth(0.6)
        panel_label(ax, "ab"[i])
    ax = fig.add_subplot(gs[2])
    wcolors = {"energy": PALETTE["teal"], "msp": PALETTE["violet"], "deep_knn": PALETTE["blue_main"]}
    wmat = np.array([[list(wcolors).index(win_m[a, b]) for b in range(2)] for a in range(len(rbps))])
    wcmap = LinearSegmentedColormap.from_list("winners", [wcolors["energy"], wcolors["msp"], wcolors["deep_knn"]])
    ax.imshow(wmat, cmap=wcmap, vmin=-0.5, vmax=2.5, aspect="auto")
    for a in range(len(rbps)):
        for b in range(2):
            ax.text(b, a, win_m[a, b].replace("deep_knn", "d-kNN"), ha="center", va="center",
                    fontsize=8, color="white")
    ax.set_xticks([0, 1]); ax.set_xticklabels(tracks)
    ax.set_yticks(range(len(rbps))); ax.set_yticklabels([])
    ax.set_title("Best AURC method per RBP × track", fontsize=11.5, pad=8)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rbps), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=wcolors[k]) for k in wcolors]
    ax.legend(handles, ["energy", "MSP", "deep-kNN"], loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=9)
    ax.text(0.5, 1.0, "", transform=ax.transAxes)
    panel_label(ax, "c")
    fig.suptitle(f"Whole-RBP holdout: near-chance AUROC, high FPR95, and {n_conflict}/24 metric-winner conflicts",
                 fontsize=12.5, y=1.02)
    VERIFICATION["fig3_matrices"] = {"rbps": rbps, "auroc": auroc_m.tolist(),
                                     "fpr95": fpr_m.tolist(), "aurc_winner": win_m.tolist(),
                                     "n_conflict": int(n_conflict)}
    save(fig, "Fig5_ood_metric_conflicts_v2_20261128")


# ---------------------------------------------------------------- Fig 4
def fig4():
    buck = pd.read_csv(DATA / "five_split_bucket_summary.tsv", sep="\t")
    per = pd.read_csv(DATA / "five_split_buckets.tsv", sep="\t")
    pc = pd.read_csv(DATA / "three_seed_macro_summary.tsv", sep="\t", header=[0, 1], index_col=0)
    exp_b = {"tail_lt10": 0.270534, "tail_10_29": 0.344729, "mid_30_99": 0.313893,
             "head_ge100": 0.203189, "all": 0.278050}
    for k, v in exp_b.items():
        check(f"fig4_bucket_{k}", buck.set_index("bucket").loc[k].split_mean, v)
    check("fig4_ci_excludes_zero_total", buck.ci_excludes_zero_splits.sum(), 25, tol=0.5)
    check("fig4_parclip_otherRBP", pc.loc["PARCLIP-otherRBP-geneMatched-excl300", ("auroc", "mean")], 0.655701)
    check("fig4_parclip_neg1c", pc.loc["eCLIP-neg1c-fold0", ("auroc", "mean")], 0.646192)
    check("fig4_parclip_neg2", pc.loc["eCLIP-neg2-fold0", ("auroc", "mean")], 0.607869)

    order = ["tail_lt10", "tail_10_29", "mid_30_99", "head_ge100", "all"]
    labels = ["tail <10", "tail 10–29", "mid 30–99", "head ≥100", "all labels"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), width_ratios=[1.35, 1])
    ax = axes[0]
    rng = np.random.default_rng(11)
    for i, k in enumerate(order):
        vals = per[per.bucket == k].delta_macro_auprc.values
        row = buck.set_index("bucket").loc[k]
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.10, 0.10, len(vals)), vals,
                   s=26, color=PALETTE["blue_light"], edgecolor=PALETTE["blue_secondary"],
                   linewidth=0.7, zorder=3, label="per-split (5 official splits)" if i == 0 else None)
        ax.errorbar(i, row.split_mean, yerr=row.split_sd, fmt="o", ms=8,
                    color=PALETTE["blue_main"], ecolor=PALETTE["blue_main"],
                    elinewidth=1.8, capsize=4, zorder=4,
                    label="mean ± SD across splits" if i == 0 else None)
        ax.text(i + 0.16, row.split_mean, f"{row.split_mean:.3f}", va="center", fontsize=9)
    ax.axhline(0, color="#888888", lw=1.0)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Δ macro-AUPRC (frozen ESM-C BCE − deep-kNN)")
    ax.set_title("GRIMM EC: BCE ahead in 25/25 frequency buckets", fontsize=11.5)
    ax.set_ylim(-0.03, 0.42)
    ax.grid(axis="y", color="#EEEEEE", lw=0.7); ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    panel_label(ax, "a")

    ax = axes[1]
    prots = ["eCLIP-neg1c-fold0", "eCLIP-neg2-fold0", "PARCLIP-otherRBP-geneMatched-excl300"]
    plabels = ["eCLIP neg1c\n(train protocol)", "eCLIP neg2\n(train protocol)", "PAR-CLIP other-RBP\n(cross-protocol)"]
    means = [pc.loc[p, ("auroc", "mean")] for p in prots]
    sds = [pc.loc[p, ("auroc", "std")] for p in prots]
    cols = [PALETTE["blue_secondary"], PALETTE["teal"], PALETTE["violet"]]
    ypos = np.arange(3)[::-1]
    for y, m, s, c, lab in zip(ypos, means, sds, cols, plabels):
        ax.barh(y, m, height=0.52, color=c, alpha=0.85, xerr=s,
                error_kw=dict(ecolor="#333333", elinewidth=1.4, capsize=4))
        ax.text(m + s + 0.006, y, f"{m:.3f} ± {s:.3f}", va="center", fontsize=9.5)
    ax.set_yticks(ypos); ax.set_yticklabels(plabels, fontsize=9.5)
    ax.set_xlim(0.5, 0.72); ax.set_xlabel("Macro AUROC (9 overlapping RBPs)")
    ax.set_title("PAR-CLIP cross-protocol transfer (3 seeds, descriptive)", fontsize=11.5)
    ax.grid(axis="x", color="#EEEEEE", lw=0.7); ax.set_axisbelow(True)
    panel_label(ax, "b")
    fig.suptitle("Protein modality and cross-protocol evidence bound the RNA conclusions",
                 fontsize=12.5, y=1.03)
    fig.tight_layout()
    VERIFICATION["fig4"] = {"buckets": {k: float(buck.set_index('bucket').loc[k].split_mean) for k in order},
                            "parclip": {p: [float(pc.loc[p, ('auroc', 'mean')]), float(pc.loc[p, ('auroc', 'std')])] for p in prots}}
    save(fig, "Fig4_protein_crossprotocol_v2_20261128")


# ---------------------------------------------------------------- Fig 5
def fig5():
    df = pd.read_csv(DATA / "final_independent_summary.tsv", sep="\t")
    seq = df[(df.track == "sequence") & (df.probe_set == "All")].set_index(["arch", "tier"])
    ood = df[(df.track == "rbp_ood") & (df.probe_set == "All") & (df.tier == "Overall")].set_index("arch")
    for arch, v in [("mlp", 0.475605), ("linear", 0.430202), ("knn", 0.352559)]:
        check(f"fig5_seq_tail_{arch}", seq.loc[(arch, "tail")].spearman_mean, v)
    for arch, v in [("knn", 0.185360), ("mlp", 0.111699), ("linear", 0.098188)]:
        check(f"fig5_ood_{arch}", ood.loc[arch].spearman_mean, v)

    archs = ["linear", "mlp", "knn"]
    names = {"linear": "linear", "mlp": "MLP", "knn": "kNN"}
    tiers = ["Overall", "head", "mid", "tail"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), width_ratios=[1.5, 1])
    ax = axes[0]
    width = 0.26
    for j, arch in enumerate(archs):
        means = [seq.loc[(arch, t)].spearman_mean for t in tiers]
        sds = [seq.loc[(arch, t)].spearman_sd for t in tiers]
        xpos = np.arange(len(tiers)) + (j - 1) * width
        ax.bar(xpos, means, width, yerr=sds, color=MODEL_COLORS[names[arch]],
               alpha=0.88, error_kw=dict(ecolor="#333333", elinewidth=1.1, capsize=3),
               label=names[arch])
        for x, m in zip(xpos, means):
            ax.text(x, m + 0.012, f"{m:.3f}", ha="center", fontsize=8.2)
    ax.set_xticks(range(len(tiers))); ax.set_xticklabels(tiers)
    ax.set_ylabel("Spearman (per-experiment macro)")
    ax.set_ylim(0, 0.58)
    ax.set_title("Sequence-ID90 track: MLP leads in tail", fontsize=11.5)
    ax.legend(loc="upper right", ncol=3, fontsize=9.5)
    ax.grid(axis="y", color="#EEEEEE", lw=0.7); ax.set_axisbelow(True)
    panel_label(ax, "a")

    ax = axes[1]
    means = [ood.loc[a].spearman_mean for a in archs]
    sds = [ood.loc[a].spearman_sd for a in archs]
    xpos = np.arange(3)
    ax.bar(xpos, means, 0.55, yerr=sds, color=[MODEL_COLORS[names[a]] for a in archs],
           alpha=0.88, error_kw=dict(ecolor="#333333", elinewidth=1.2, capsize=4))
    for x, m in zip(xpos, means):
        ax.text(x, m + 0.032, f"{m:.3f}", ha="center", fontsize=9.5)
    ax.set_xticks(xpos); ax.set_xticklabels([names[a] for a in archs])
    ax.set_ylabel("Experiment-macro Spearman")
    ax.set_ylim(0, 0.26)
    ax.set_title("Whole-RBP OOD: ranking reverses", fontsize=11.5)
    ax.annotate("", xy=(0, 0.205), xytext=(2, 0.135),
                arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=1.4,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(1.0, 0.225, "task-dependent winner", ha="center", fontsize=9, color=PALETTE["red_strong"])
    ax.grid(axis="y", color="#EEEEEE", lw=0.7); ax.set_axisbelow(True)
    panel_label(ax, "b")
    fig.suptitle("RNAcompete Ray2013 (5 folds × 3 seeds): no universal winner across OOD axes",
                 fontsize=12.5, y=1.03)
    fig.tight_layout()
    VERIFICATION["fig5"] = {"sequence_tail": {names[a]: float(seq.loc[(a, 'tail')].spearman_mean) for a in archs},
                            "rbp_ood": {names[a]: float(ood.loc[a].spearman_mean) for a in archs}}
    save(fig, "Fig6_rnacompete_dual_track_v2_20261128")


# ---------------------------------------------------------------- Fig 3 (low homology)
def fig3_lowhom():
    """Low-homology descriptive stratification (accepted E1 report §6).

    Source of truth: E1_强基线/E1_主集严格三折冻结基线最终报告_2026-11-26.md §6,
    itself accepted via E1-stratified-analysis-20261126-104609-ZDLg6Y and the
    independent recalculation run. Values are parsed from the report table and
    cross-checked against constants; no bootstrap exists for this slice.
    """
    import re
    report = (ROOT / "E1_强基线" / "E1_主集严格三折冻结基线最终报告_2026-11-26.md").read_text()
    rows = {}
    for track in ("neg1c", "neg2"):
        m = re.search(rf"\| {track} \| ([0-9.]+) \| ([0-9.]+) \| \+([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| \+([0-9.]+) \|", report)
        assert m, f"low-homology row for {track} not found in accepted report"
        rows[track] = [float(x) for x in m.groups()]
    expected = {"neg1c": [0.875332, 0.830752, 0.044580, 0.781584, 0.697716, 0.083869],
                "neg2": [0.790567, 0.751782, 0.038785, 0.653712, 0.585196, 0.068516]}
    for track, vals in rows.items():
        for got, want in zip(vals, expected[track]):
            check(f"fig3_{track}_{want}", got, want)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    metrics = [("AUROC", 0, 1, 2), ("AUPRC", 3, 4, 5)]
    for ax, track, label in zip(axes, ["neg1c", "neg2"], ["a", "b"]):
        vals = rows[track]
        xpos = np.arange(2)
        bce = [vals[0], vals[3]]
        knn = [vals[1], vals[4]]
        ax.bar(xpos - 0.18, bce, 0.34, color=PALETTE["blue_main"], alpha=0.88, label="frozen RNA-FM BCE")
        ax.bar(xpos + 0.18, knn, 0.34, color=PALETTE["red_strong"], alpha=0.88, label="cluster-dedup cosine kNN")
        for x, v in zip(xpos - 0.18, bce):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.5)
        for x, v in zip(xpos + 0.18, knn):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.5)
        for x, d in zip(xpos, [vals[2], vals[5]]):
            ax.text(x, 0.06, f"Δ=+{d:.3f}", ha="center", fontsize=10, color=PALETTE["blue_main"], fontweight="bold")
        ax.set_xticks(xpos); ax.set_xticklabels(["AUROC", "AUPRC"])
        ax.set_title(f"{track} negatives", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#EEEEEE", lw=0.7); ax.set_axisbelow(True)
        panel_label(ax, label)
    axes[0].set_ylabel("Experiment macro mean (3 seeds)")
    axes[0].legend(loc="upper right", fontsize=9.5)
    fig.suptitle("Low-homology slice (test positives d_hom < 0.60): direction persists, descriptive only",
                 fontsize=12.5, y=1.03)
    fig.tight_layout()
    VERIFICATION["fig3_low_homology"] = rows
    save(fig, "Fig3_low_homology_stratification_v2_20261128")


if __name__ == "__main__":
    fig1(); fig2(); fig3_lowhom(); fig3(); fig4(); fig5()
    (OUT / "verification_v2_20261128.json").write_text(json.dumps(VERIFICATION, indent=2))
    print("ALL_FIGURES_DONE", len(VERIFICATION), "verification entries")