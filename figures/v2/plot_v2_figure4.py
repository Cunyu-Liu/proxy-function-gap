#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from figure_common_v2 import (
    CONTEXT_COLORS, ORDER, PALETTE, SHORT,
    apply_style, format3, light_grid, load, panel_label, save_figure,
)


def main() -> None:
    apply_style()
    summ = load("strict_h1_figure4_cross_architecture_summary.csv").set_index("context").loc[ORDER]

    fig = plt.figure(figsize=(11.4, 4.7))
    grid = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.15], wspace=0.30)
    y = np.arange(len(ORDER))[::-1]

    # a: error correlation lollipop
    ax = fig.add_subplot(grid[0, 0])
    ax.axvline(0, color=PALETTE["grid"], lw=1.0, zorder=0)
    for yi, ctx in zip(y, ORDER):
        r = summ.loc[ctx]; color = CONTEXT_COLORS[ctx]
        ax.plot([0, r.architecture_error_correlation], [yi, yi], color=color, lw=2.0, alpha=0.85, zorder=2)
        ax.plot([r.correlation_ci_low, r.correlation_ci_high], [yi, yi], color=color, lw=4.5,
                alpha=0.30, solid_capstyle="round", zorder=1)
        ax.scatter(r.architecture_error_correlation, yi, s=78, color=color, edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(1.005, yi, f"{r.architecture_error_correlation:.3f}", va="center", ha="left",
                fontsize=8.4, fontweight="bold", color=PALETTE["ink"])
    ax.set_yticks(y, [SHORT[c] for c in ORDER])
    ax.set_xlim(0, 1.10); ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("CNN ↔ 4-mer MLP absolute-error correlation")
    ax.set_title("Errors stay shared across architectures", loc="left")
    light_grid(ax, "x")
    ax.spines.left.set_visible(False); ax.tick_params(axis="y", length=0)
    panel_label(ax, "a")

    # b: CNN vs MLP MAE dumbbell
    ax = fig.add_subplot(grid[0, 1])
    for yi, ctx in zip(y, ORDER):
        r = summ.loc[ctx]
        cnn, kmer = r.cnn_mean_abs_error, r.kmer_mean_abs_error
        ax.plot([kmer, cnn], [yi, yi], color=PALETTE["neutral"], lw=2.0, alpha=0.7, zorder=2)
        ax.scatter(kmer, yi, s=64, marker="s", facecolor="white", edgecolor=PALETTE["bad"], linewidth=1.6, zorder=3, label="4-mer MLP" if yi == y[0] else None)
        ax.scatter(cnn, yi, s=64, color=PALETTE["blue"], edgecolor="white", linewidth=0.8, zorder=4, label="CNN ensemble" if yi == y[0] else None)
        ax.text(max(cnn, kmer) + 0.06, yi, f"Δ {format3(r.paired_cnn_minus_kmer, True)}",
                va="center", ha="left", fontsize=7.8, color=PALETTE["subink"])
    ax.set_yticks(y, [SHORT[c] for c in ORDER])
    ax.set_xlabel("Mean absolute error (MRL)")
    ax.set_title("CNN lower MAE in every context (paired CI excludes 0)", loc="left")
    light_grid(ax, "x")
    ax.spines.left.set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=8.0)
    panel_label(ax, "b")

    fig.suptitle("Changing architecture does not provide independent functional validation",
                 x=0.055, y=1.02, ha="left", fontsize=13.5, fontweight="bold", color=PALETTE["ink"])
    fig.text(0.055, -0.04,
             "Both models share Sample training labels; lower CNN error coexists with strongly correlated errors (0.878–0.962). "
             "Retrospective strict-H1 protocol sensitivity.",
             ha="left", fontsize=7.6, color=PALETTE["subink"])
    fig.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.16)
    save_figure(fig, "Figure4_v2_common_proxy_error")


if __name__ == "__main__":
    main()