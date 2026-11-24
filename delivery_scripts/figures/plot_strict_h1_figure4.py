#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figure_common import CONTEXT_COLORS, SHORT, apply_style, format3, load, panel_label, save_figure

ORDER = ["N50 HEK293T", "Defined HepG2 r1", "Defined T-cell r1", "Defined T-cell r2"]


def main() -> None:
    apply_style()
    summary = load("strict_h1_figure4_cross_architecture_summary.csv").set_index("context").loc[ORDER].reset_index()
    pools = load("strict_h1_figure4_pool_effects.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.65), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})
    y = np.arange(len(summary))[::-1]
    offsets = np.linspace(-0.12, 0.12, 16)
    for row, yi in zip(summary.itertuples(), y):
        color = CONTEXT_COLORS[row.context]
        pool_rows = pools[pools.context == row.context].sort_values("pool_index")
        ax.scatter(pool_rows.architecture_error_correlation, yi + offsets, s=10, color=color, alpha=0.22,
                   edgecolors="none", zorder=1)
        ax.plot([row.correlation_ci_low, row.correlation_ci_high], [yi, yi], color=color, lw=2.3,
                solid_capstyle="round", zorder=2)
        ax.scatter(row.architecture_error_correlation, yi, s=58, color=color, edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.text(0.998, yi,
                f"{row.architecture_error_correlation:.3f}  [{row.correlation_ci_low:.3f}, {row.correlation_ci_high:.3f}]",
                va="center", ha="right", fontsize=7.6)
    ax.set_yticks(y, [SHORT[c] for c in summary.context])
    ax.set_xlim(0.835, 1.005)
    ax.set_xticks([0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Within-pool CNN / 4-mer MLP absolute-error correlation")
    ax.set_title("Error correlation remains high", loc="left")
    ax.grid(axis="x", color="#E1E1E1", lw=0.55)
    ax.set_axisbelow(True)
    ax.spines.left.set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "a")

    ax2.axvline(0, color="#767676", lw=0.9, ls=(0, (3, 3)), zorder=0)
    for row, yi in zip(summary.itertuples(), y):
        color = CONTEXT_COLORS[row.context]
        pool_rows = pools[pools.context == row.context].sort_values("pool_index")
        ax2.scatter(pool_rows.paired_cnn_minus_kmer, yi + offsets, s=10, color=color, alpha=0.22,
                    edgecolors="none", zorder=1)
        ax2.plot([row.difference_ci_low, row.difference_ci_high], [yi, yi], color=color, lw=2.3,
                 solid_capstyle="round", zorder=2)
        ax2.scatter(row.paired_cnn_minus_kmer, yi, s=58, color=color, edgecolor="white",
                    linewidth=0.8, zorder=3)
        ax2.text(0.058, yi,
                 f"{format3(row.paired_cnn_minus_kmer, signed=True)}  "
                 f"[{format3(row.difference_ci_low, signed=True)}, {format3(row.difference_ci_high, signed=True)}]",
                 va="center", ha="right", fontsize=7.6)
    ax2.set_yticks(y, [SHORT[c] for c in summary.context])
    ax2.set_xlim(-0.36, 0.065)
    ax2.set_xlabel("CNN − 4-mer MLP mean absolute error")
    ax2.set_title("CNN has lower MAE, but errors remain shared", loc="left")
    ax2.grid(axis="x", color="#E1E1E1", lw=0.55)
    ax2.set_axisbelow(True)
    ax2.spines.left.set_visible(False)
    ax2.tick_params(axis="y", length=0)
    panel_label(ax2, "b")
    ax2.text(0.02, -0.22, "Negative values favor CNN  ·  All four paired nominal 95% intervals exclude zero",
             transform=ax2.transAxes, fontsize=7.5, color="#5F5F5F")

    fig.suptitle("Strict-H1 common proxy errors persist across architectures",
                 x=0.055, y=0.992, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.055, 0.905,
             "Large points and bars are context summaries with nominal pool-cluster bootstrap 95% intervals; faint points are 16 pool-level estimates.",
             fontsize=8.3, color="#5F5F5F", ha="left")
    fig.text(0.055, 0.02,
             "Retrospective protocol sensitivity, not a new blind test. Shared training data and endpoints preclude independent functional validation.",
             fontsize=7.2, color="#6A6A6A", ha="left")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.79, bottom=0.235)
    save_figure(fig, "Figure4_strict_h1_common_proxy_error")


if __name__ == "__main__":
    main()
