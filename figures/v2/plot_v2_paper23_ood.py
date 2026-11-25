#!/usr/bin/env python3
"""Paper2 / Paper3 / Schlusser / Leppek figures (v2 style), from frozen local source data."""
from __future__ import annotations

import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from delivery_scripts.publication_paths import DELIVERABLES

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_common_v2 import PALETTE, apply_style, light_grid, panel_label, save_figure

SRC = DELIVERABLES / "reports" / "source_data"
P3MD = DELIVERABLES / "reports" / "paper3_headroom_v4_20261124.md"


def fig_paper2():
    d = json.load(open(SRC / "paper2_baselines_v1" / "outer_designed_summary.json"))
    caps16 = {c["method"]: c for c in d["configurations"] if c["proposal_cap"] == 16}
    methods = ["bon", "lcb", "distance", "com", "diversity_lcb"]
    labels = {"bon": "BoN", "lcb": "LCB", "distance": "Distance", "com": "COM-style", "diversity_lcb": "Diversity-LCB"}
    colors = {"bon": PALETTE["blue"], "lcb": PALETTE["neutral"], "distance": PALETTE["neutral"],
              "com": PALETTE["teal"], "diversity_lcb": PALETTE["violet"]}
    paired = {p["method"]: p for p in d["paired_comparisons"] if p["reference"] == "bon" and p["metric"] == "regret"}

    fig = plt.figure(figsize=(11.4, 4.6))
    grid = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.15], wspace=0.30)

    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(methods))
    means = [caps16[m]["mean_regret"] for m in methods]
    ses = [caps16[m]["pool_standard_error"] for m in methods]
    ax.bar(x, means, yerr=ses, capsize=4, color=[colors[m] for m in methods],
           edgecolor="white", linewidth=0.8, error_kw=dict(lw=1.2, ecolor=PALETTE["ink"]))
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.02, f"{m:.3f}", ha="center", fontsize=8.4, color=PALETTE["ink"])
    ax.set_xticks(x, [labels[m] for m in methods], rotation=18, ha="right")
    ax.set_ylabel("Mean measured-MRL regret (cap 16)")
    ax.set_title("Designed outer: no method beats BoN", loc="left")
    light_grid(ax)
    ax.set_ylim(0, max(means) * 1.25)
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    pm = ["lcb", "distance", "com", "diversity_lcb"]
    y = np.arange(len(pm))[::-1]
    ax.axvline(0, color=PALETTE["zero"], lw=1.1, ls=(0, (4, 3)), zorder=1)
    for yi, m in zip(y, pm):
        p = paired[m]; lo, hi = p["pool_cluster_bootstrap_95_percent_interval"]
        ax.plot([lo, hi], [yi, yi], color=colors[m], lw=2.6, solid_capstyle="round", zorder=3)
        ax.scatter(p["mean_difference"], yi, s=64, color=colors[m], edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(0.115, yi, f"{p['mean_difference']:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
                va="center", ha="right", fontsize=7.8, color=PALETTE["ink"])
    ax.set_yticks(y, [labels[m] for m in pm])
    ax.set_xlim(-0.13, 0.12); ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("Paired regret difference vs BoN (negative favors method)")
    ax.set_title("All 95% intervals include zero", loc="left")
    light_grid(ax, "x")
    ax.spines.left.set_visible(False); ax.tick_params(axis="y", length=0)
    panel_label(ax, "b")

    fig.suptitle("Paper 2 selection baselines: no context-general regret advantage",
                 x=0.055, y=1.02, ha="left", fontsize=13.5, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.055, -0.045,
             "COM-style adaptation (not official COM weights); dev-frozen hyperparameters; retrospective sensitivity on previously inspected outer. "
             "Permutation negative controls retained (BoN observed 0.672 vs permuted 1.951). VGSC FP-REJ gate failure preserved separately.",
             ha="left", fontsize=7.5, color=PALETTE["subink"])
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.17)
    save_figure(fig, "Figure7_v2_paper2_baselines")


def _parse_p3_curve():
    txt = P3MD.read_text()
    block = txt.split("### kmer4：固定 5 池")[1].split("###")[0]
    rows = []
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0].isdigit():
            rows.append([float(c) for c in cells])
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1:]


def fig_paper3():
    d = json.load(open(SRC / "paper3_headroom_v4" / "headroom_v4_round1" / "summary.json"))
    cases = d["cases"]
    reps = ["summary4", "kmer4", "cnn64"]
    pools = [f"calibration_{i:03d}" for i in range(16)]
    mat = np.zeros((3, 16))
    for c in cases:
        if c["query_pool_id"] in pools:
            mat[reps.index(c["representation"]), pools.index(c["query_pool_id"])] = c["repair_headroom"]

    fig = plt.figure(figsize=(12.0, 4.8))
    grid = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1], wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("headroom", ["#F2F3F7", "#AADCA9", "#2A9D8F", "#1B6E63"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(mat.max(), 0.1), aspect="auto")
    for i in range(3):
        for j in range(16):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                        color="white" if v > mat.max() * 0.5 else PALETTE["ink"], fontweight="bold")
    ax.set_xticks(range(16), [p.replace("calibration_", "") for p in pools], fontsize=7.5)
    ax.set_yticks(range(3), reps)
    ax.set_xlabel("Calibration pool index")
    ax.set_title("Repair headroom (zero-label → full-Q regret drop)", loc="left")
    ax.set_xticks(np.arange(-0.5, 16, 1), minor=True); ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cbar.set_label("Headroom (MRL)", fontsize=8.4)
    cbar.ax.tick_params(labelsize=7.8); cbar.outline.set_visible(False)
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    budgets, curves = _parse_p3_curve()
    strat = ["Random", "Max-var", "Diversity", "EI", "Top-proxy", "Regret-analytic"]
    scolors = [PALETTE["neutral"], PALETTE["violet"], PALETTE["sand"], PALETTE["teal"], PALETTE["subink"], PALETTE["coral"]]
    for k in range(6):
        emph = strat[k] in ("EI", "Regret-analytic")
        ax.plot(budgets, curves[:, k], color=scolors[k], lw=2.4 if emph else 1.5,
                marker="o" if emph else None, ms=4.5, label=strat[k],
                alpha=1.0 if emph else 0.8, zorder=4 if emph else 3)
    ax.set_xlabel("Label budget")
    ax.set_ylabel("Pooled measured-MRL regret (5 kmer4 pools)")
    ax.set_title("kmer4: EI captures headroom; analytic not dominant", loc="left")
    light_grid(ax)
    ax.legend(fontsize=7.6, loc="upper right", ncol=2)
    panel_label(ax, "b")

    fig.suptitle("Paper 3 label-efficient repair: headroom exists, simple EI suffices",
                 x=0.055, y=1.02, ha="left", fontsize=13.5, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.055, -0.045,
             "Retrospective development/calibration diagnosis; permanent outer untouched. summary4 has zero headroom even at full-Q (0/16); "
             "kmer4 5/16 positive gates. Regret-analytic uses a diagonal-covariance independent-normal approximation. No acquisition superiority claimed.",
             ha="left", fontsize=7.5, color=PALETTE["subink"])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.17)
    save_figure(fig, "Figure8_v2_paper3_headroom")


def fig_schlusser():
    d = json.load(open(SRC / "schlusser2024_hepg2_te_confirmatory_v1" / "evaluator_round_1" / "summary.json"))
    dist = d["TE_distribution_after_frozen_unblinding"]
    paired16 = [p for p in d["paired_comparisons"] if p["proposal_cap"] == 16]

    fig = plt.figure(figsize=(11.4, 4.6))
    grid = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.32)

    ax = fig.add_subplot(grid[0, 0])
    keys = ["min", "q1", "median", "q3", "max"]
    vals = [dist[k] for k in keys]
    ax.barh([0], [dist["q3"] - dist["q1"]], left=[dist["q1"]], height=0.4, color=PALETTE["teal"], alpha=0.55)
    ax.scatter([dist["median"]], [0], s=90, color=PALETTE["blue"], zorder=4, edgecolor="white", linewidth=1)
    ax.plot([dist["min"], dist["q1"]], [0, 0], color=PALETTE["subink"], lw=1.2)
    ax.plot([dist["q3"], dist["max"]], [0, 0], color=PALETTE["subink"], lw=1.2)
    ax.scatter([dist["min"], dist["max"]], [0, 0], s=30, color=PALETTE["coral"], zorder=4)
    ax.set_xscale("log")
    ax.set_yticks([]); ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Translation efficiency (TE, log scale)")
    ax.set_title("HepG2 endogenous TE is extremely right-skewed", loc="left")
    for k, v in zip(keys, vals):
        ax.text(v, 0.32, f"{k}\n{v:.3g}", ha="center", fontsize=7.4, color=PALETTE["subink"])
    light_grid(ax, "x")
    ax.spines.left.set_visible(False)
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    names = {"bon_cnn-random": "BoN-CNN − Random", "bon_kmer-random": "BoN-4mer − Random",
             "lcb_cnn-random": "LCB-CNN − Random", "bon_cnn-bon_kmer": "BoN-CNN − BoN-4mer"}
    y = np.arange(len(paired16))[::-1]
    ax.axvline(0, color=PALETTE["zero"], lw=1.1, ls=(0, (4, 3)), zorder=1)
    for yi, p in zip(y, paired16):
        lo, hi = p["gene_cluster_bootstrap_95_interval"]
        est = p["mean_regret_difference_left_minus_right"]
        ax.plot([lo, hi], [yi, yi], color=PALETTE["violet"], lw=2.6, solid_capstyle="round", zorder=3)
        ax.scatter(est, yi, s=64, color=PALETTE["violet"], edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(10.6, yi, f"{est:+.3f}  [{lo:+.2f}, {hi:+.2f}]", va="center", ha="right", fontsize=7.6, color=PALETTE["ink"])
    ax.set_yticks(y, [names[p["comparison"]] for p in paired16])
    ax.set_xlim(-1.6, 11.0); ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("Paired TE-regret difference (cap 16, gene-cluster 95% CI)")
    ax.set_title("MRL proxy does not improve real TE", loc="left")
    light_grid(ax, "x")
    ax.spines.left.set_visible(False); ax.tick_params(axis="y", length=0)
    panel_label(ax, "b")

    fig.suptitle("Cross-endpoint OOD: Sample MRL proxy fails on HepG2 endogenous TE",
                 x=0.055, y=1.02, ha="left", fontsize=13.5, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.055, -0.045,
             "Confirmatory one-shot unblinding; frozen config. Paired differences cancel the shared in-pool oracle TE. "
             "All intervals cross zero; MRL–TE Spearman 0.072. Single study, not cross-cell generalization. TE from published RNA-seq/Ribo-seq table.",
             ha="left", fontsize=7.5, color=PALETTE["subink"])
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.17)
    save_figure(fig, "Figure9_v2_schlusser_te")


def fig_leppek():
    corr = pd.read_csv(SRC / "leppek2022_cds_mechanism_audit_v5" / "paired_correlations.tsv", sep="\t")
    rank = pd.read_csv(SRC / "leppek2022_cds_mechanism_audit_v5" / "rank_summary.tsv", sep="\t")
    nice = {"cds_in_solution_half_life": "solution half-life", "nluc_fluc_6h": "Nluc/Fluc 6h",
            "nluc_fluc_24h": "Nluc/Fluc 24h", "ribosome_load": "MRL", "in_cell_half_life": "in-cell half-life",
            "in_solution_half_life": "solution half-life"}

    fig = plt.figure(figsize=(11.6, 4.8))
    grid = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.34)

    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(len(corr))[::-1]
    ax.axvline(0, color=PALETTE["zero"], lw=1.1, ls=(0, (4, 3)), zorder=1)
    for yi, (_, r) in zip(y, corr.iterrows()):
        col = PALETTE["coral"] if r.source_family == "cds_24" else PALETTE["teal"]
        ax.scatter(r.pearson_r, yi + 0.13, s=46, color=col, edgecolor="white", linewidth=0.7, zorder=4)
        ax.scatter(r.spearman_rho, yi - 0.13, s=46, marker="D", facecolor="white", edgecolor=col, linewidth=1.3, zorder=4)
        lab = f"{nice[r.left_endpoint]} ↔ {nice[r.right_endpoint]}"
        ax.text(-0.42, yi, f"{lab}  (n={r.n})", va="center", ha="right", fontsize=7.6, color=PALETTE["ink"])
    ax.set_yticks([]); ax.set_xlim(-0.45, 0.62); ax.set_ylim(-0.7, len(corr) - 0.3)
    ax.set_xlabel("Correlation (● Pearson r, ◇ Spearman ρ)")
    ax.set_title("Endpoints are not rank-equivalent", loc="left")
    light_grid(ax, "x")
    ax.spines.left.set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color="w", markerfacecolor=PALETTE["coral"], markersize=8, label="CDS-24"),
                       Line2D([], [], marker="o", color="w", markerfacecolor=PALETTE["teal"], markersize=8, label="pooled-233")],
              loc="lower right", fontsize=7.8)
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    cds = rank[rank.source_family == "cds_24"].reset_index(drop=True)
    y = np.arange(len(cds))[::-1]
    for yi, (_, r) in zip(y, cds.iterrows()):
        ax.barh(yi, r.discordant_fraction_untied, height=0.5, color=PALETTE["coral"], alpha=0.85)
        ax.text(r.discordant_fraction_untied + 0.01, yi,
                f"{r.discordant_fraction_untied*100:.0f}% reversal · top-6 overlap {int(r.top_k_overlap_n)}/6",
                va="center", fontsize=7.8, color=PALETTE["ink"])
    ax.set_yticks(y, [f"{nice[r.left_endpoint]}\nvs {nice[r.right_endpoint]}" for _, r in cds.iterrows()], fontsize=7.6)
    ax.set_xlim(0, 0.95); ax.set_xlabel("Discordant pair fraction (rank reversal)")
    ax.set_title("CDS-24: stability ≠ expression ranking", loc="left")
    light_grid(ax, "x")
    ax.spines.left.set_visible(False); ax.tick_params(axis="y", length=0)
    panel_label(ax, "b")

    fig.suptitle("Leppek 2022: proxy contracts must specify the endpoint",
                 x=0.055, y=1.02, ha="left", fontsize=13.5, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.055, -0.045,
             "Exploratory descriptive audit (no p-values/CIs, no model selection). CDS-24 HBB-Nluc, PSU, HEK293T; complete pairs, no imputation. "
             "pooled Predicted expression never used as measured protein. Rank reversal shows endpoint ordering differs, not mechanism conflict.",
             ha="left", fontsize=7.5, color=PALETTE["subink"])
    fig.subplots_adjust(left=0.30, right=0.985, top=0.86, bottom=0.17)
    save_figure(fig, "Figure10_v2_leppek_endpoints")


if __name__ == "__main__":
    apply_style()
    fig_paper2(); fig_paper3(); fig_schlusser(); fig_leppek()
    print("done: Figure7-10 v2")