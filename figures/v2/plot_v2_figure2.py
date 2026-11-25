#!/usr/bin/env python3
from __future__ import annotations

import json

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from figure_common_v2 import (
    CONTEXT_COLORS, DATA, MULT_KEY, ORDER, PALETTE, SHORT,
    apply_style, format3, jitter, light_grid, load, panel_label, save_figure,
)

MULT = DATA.parent.parent / "reports" / "source_data" / "paper1_strict_h1_v3" / "analysis" / "strict_h1_multiplicity_sensitivity.json"
CAPS = [1, 2, 4, 8, 16]


def main() -> None:
    apply_style()
    data = load("strict_h1_figure2_paired_pressure.csv")
    pools = load("strict_h1_figure2_pool_effects.csv")
    mult = {(r["context"], r["proposal_cap"]): r for r in json.load(open(MULT))["rows"]}

    fig = plt.figure(figsize=(12.6, 7.0))
    grid = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.18], hspace=0.55, wspace=0.40)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(2)]

    for idx, (ax, ctx) in enumerate(zip(axes, ORDER)):
        rows = data[data.context == ctx].sort_values("proposal_cap")
        x = rows.proposal_cap.to_numpy()
        eff = rows.paired_bon_minus_random.to_numpy()
        lo = rows.ci_low.to_numpy(); hi = rows.ci_high.to_numpy()
        color = CONTEXT_COLORS[ctx]
        ax.axhline(0, color=PALETTE["zero"], lw=1.0, ls=(0, (4, 3)), zorder=1)
        # per-pool paired effects (the evidence distribution)
        for cap in CAPS:
            pr = pools[(pools.context == ctx) & (pools.proposal_cap == cap)]
            vals = pr.paired_bon_minus_random.to_numpy()
            ax.scatter(np.full(len(vals), cap) + jitter(len(vals), 100 + idx * 10 + cap, 0.16),
                       vals, s=9, color=color, alpha=0.28, linewidths=0, zorder=2)
        ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=3)
        ax.plot(x, eff, color=color, lw=2.2, zorder=4, solid_capstyle="round")
        # significance-aware markers
        for cap, e in zip(x, eff):
            surv = mult[(MULT_KEY[ctx], cap)]["bonferroni_20_comparisons_excludes_zero"]
            nom = mult[(MULT_KEY[ctx], cap)]["nominal_95_percent_excludes_zero"]
            if surv:
                ax.scatter(cap, e, s=52, color=color, edgecolor="white", linewidth=1.0, zorder=5)
            elif nom:
                ax.scatter(cap, e, s=42, marker="D", facecolor="white", edgecolor=color, linewidth=1.4, zorder=5)
            else:
                ax.scatter(cap, e, s=30, facecolor="white", edgecolor=color, linewidth=1.2, zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CAPS, labels=[str(c) for c in CAPS])
        ax.set_xlabel("Best-of-N proposal cap")
        if idx % 2 == 0:
            ax.set_ylabel("BoN − Random regret  (MRL)")
        ax.set_title(f"{SHORT[ctx]}   ·   n=16 pools", loc="left", color=PALETTE["ink"])
        light_grid(ax)
        bound = max(abs(lo.min()), abs(hi.max()), 0.6) * 1.20
        ax.set_ylim(-bound, bound)
        panel_label(ax, chr(ord("a") + idx))

    # e: forest cap16 (filled) + cap8 (open)
    ax = fig.add_subplot(grid[:, 2])
    y = np.arange(len(ORDER))[::-1]
    ax.axvline(0, color=PALETTE["zero"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    for yi, ctx in zip(y, ORDER):
        rows = data[data.context == ctx].set_index("proposal_cap")
        color = CONTEXT_COLORS[ctx]
        c8, c16 = rows.loc[8], rows.loc[16]
        ax.plot([c8.ci_low, c8.ci_high], [yi + 0.14, yi + 0.14], color=color, lw=1.4, alpha=0.55)
        ax.scatter(c8.paired_bon_minus_random, yi + 0.14, s=34, marker="D", facecolor="white",
                   edgecolor=color, linewidth=1.2, zorder=3)
        ax.plot([c16.ci_low, c16.ci_high], [yi - 0.11, yi - 0.11], color=color, lw=2.6, solid_capstyle="round")
        ax.scatter(c16.paired_bon_minus_random, yi - 0.11, s=64, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(1.62, yi - 0.11,
                f"{format3(c16.paired_bon_minus_random, True)}  [{format3(c16.ci_low, True)}, {format3(c16.ci_high, True)}]",
                va="center", ha="right", fontsize=7.8, color=PALETTE["ink"])
    ax.set_yticks(y, [SHORT[c] for c in ORDER])
    ax.set_xlim(-2.55, 1.70); ax.set_ylim(-0.65, 3.65)
    ax.set_xlabel("Paired BoN − Random regret")
    ax.set_title("Cap 16 effect (●); cap 8 (◇)", loc="left")
    light_grid(ax, "x")
    ax.spines.left.set_visible(False); ax.tick_params(axis="y", length=0)
    panel_label(ax, "e")

    # f: multiplicity robustness counts
    ax = fig.add_subplot(grid[1, 2])
    nom_counts, bon_counts = [], []
    for ctx in ORDER:
        nc = sum(mult[(MULT_KEY[ctx], c)]["nominal_95_percent_excludes_zero"] for c in CAPS)
        bc = sum(mult[(MULT_KEY[ctx], c)]["bonferroni_20_comparisons_excludes_zero"] for c in CAPS)
        nom_counts.append(nc); bon_counts.append(bc)
    yy = np.arange(len(ORDER))
    ax.barh(yy + 0.18, nom_counts, height=0.34, color=PALETTE["neutral"], label="Nominal 95% CI excludes 0")
    ax.barh(yy - 0.18, bon_counts, height=0.34, color=PALETTE["good"], label="Survives all-20 Bonferroni")
    for yi, (nc, bc) in zip(yy, zip(nom_counts, bon_counts)):
        ax.text(nc + 0.08, yi + 0.18, str(nc), va="center", fontsize=8.4, color=PALETTE["subink"])
        ax.text(bc + 0.08, yi - 0.18, str(bc), va="center", fontsize=8.4, color=PALETTE["good"], fontweight="bold")
    ax.set_yticks(yy, [SHORT[c] for c in ORDER])
    ax.set_xlim(0, 5.6); ax.set_xticks(range(0, 6))
    ax.set_xlabel("Caps (of 5) with interval excluding zero")
    ax.set_title("Multiplicity robustness", loc="left")
    ax.invert_yaxis()
    light_grid(ax, "x")
    ax.spines.left.set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=7.6)
    panel_label(ax, "f")

    fig.suptitle("Strict-H1 pressure gives reproducible, context-dependent gains",
                 x=0.055, y=0.995, ha="left", fontsize=14, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.055, 0.955,
             "Faint dots = 16 per-pool paired effects; line + ribbon = mean and nominal 95% pool-cluster CI. "
             "● Bonferroni-robust, ◇ nominal-only, ○ includes zero.",
             ha="left", fontsize=8.6, color=PALETTE["subink"])
    fig.text(0.055, 0.012,
             "T-cell caps 2–16 survive all-20 Bonferroni in both replicates; HepG2 cap 8 is exploratory; N50 shows no sustained gain. "
             "Retrospective protocol sensitivity, not a new blind test. Negative values favor BoN.",
             ha="left", fontsize=7.4, color=PALETTE["subink"])
    fig.subplots_adjust(left=0.065, right=0.985, top=0.885, bottom=0.115)
    save_figure(fig, "Figure2_v2_selection_pressure")


if __name__ == "__main__":
    main()