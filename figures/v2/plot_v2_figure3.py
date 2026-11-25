#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from figure_common_v2 import (
    CONTEXT_COLORS, ORDER, PALETTE, SHORT,
    apply_style, format3, jitter, light_grid, load, panel_label, save_figure,
)


def main() -> None:
    apply_style()
    summ = load("strict_h1_figure3_score_matching.csv").set_index("context").loc[ORDER]
    pools = load("strict_h1_figure3_pool_effects.csv")

    fig = plt.figure(figsize=(11.4, 4.7))
    grid = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.25], wspace=0.30)

    # a: forest of optimized - matched difference
    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(len(ORDER))[::-1]
    ax.axvline(0, color=PALETTE["zero"], lw=1.1, ls=(0, (4, 3)), zorder=1)
    for yi, ctx in zip(y, ORDER):
        r = summ.loc[ctx]; color = CONTEXT_COLORS[ctx]
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=color, lw=2.6, solid_capstyle="round", zorder=3)
        ax.scatter(r.difference, yi, s=70, color=color, edgecolor="white", linewidth=0.9, zorder=4)
        ax.text(1.62, yi, f"{format3(r.difference, True)}  [{format3(r.ci_low, True)}, {format3(r.ci_high, True)}]",
                va="center", ha="right", fontsize=7.8, color=PALETTE["ink"])
    ax.set_yticks(y, [SHORT[c] for c in ORDER])
    ax.set_xlim(-1.15, 1.72); ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("Optimized − matched absolute error")
    ax.set_title("Score-matched difference (cap 16)", loc="left")
    light_grid(ax, "x")
    ax.spines.left.set_visible(False); ax.tick_params(axis="y", length=0)
    panel_label(ax, "a")
    ax.text(0.02, -0.20, "All four 95% intervals include zero → no consistent optimizer-specific increment.",
            transform=ax.transAxes, fontsize=7.8, color=PALETTE["subink"])

    # b: per-pool paired difference strip
    ax = fig.add_subplot(grid[0, 1])
    ax.axvline(0, color=PALETTE["zero"], lw=1.1, ls=(0, (4, 3)), zorder=1)
    for yi, ctx in zip(y, ORDER):
        pr = pools[pools.context == ctx]
        vals = pr.paired_optimized_minus_matched.to_numpy()
        color = CONTEXT_COLORS[ctx]
        ax.scatter(vals, np.full(len(vals), yi) + jitter(len(vals), 7 + yi, 0.16),
                   s=16, color=color, alpha=0.45, linewidths=0, zorder=2)
        r = summ.loc[ctx]
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=PALETTE["ink"], lw=1.4, alpha=0.8, zorder=3)
        ax.scatter(r.difference, yi, s=58, marker="D", color=PALETTE["ink"], edgecolor="white", linewidth=0.8, zorder=4)
    ax.set_yticks(y, [SHORT[c] for c in ORDER])
    ax.set_xlabel("Per-pool optimized − matched absolute error")
    ax.set_title("Each point = one pool (n=16); ◆ mean", loc="left")
    light_grid(ax, "x")
    ax.spines.left.set_visible(False); ax.tick_params(axis="y", length=0)
    panel_label(ax, "b")

    fig.suptitle("Score matching does not support an optimizer-specific error increment",
                 x=0.055, y=1.02, ha="left", fontsize=13.5, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.055, -0.04,
             "Within-pool nearest-proxy-mean match (different sequence). Null result does not establish high-score extrapolation as the mechanism. "
             "Retrospective strict-H1 protocol sensitivity.",
             ha="left", fontsize=7.6, color=PALETTE["subink"])
    fig.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.16)
    save_figure(fig, "Figure3_v2_score_matching")


if __name__ == "__main__":
    main()