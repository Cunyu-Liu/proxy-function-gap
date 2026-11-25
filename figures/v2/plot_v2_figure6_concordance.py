#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figure_common_v2 import (
    CONTEXT_COLORS, ORDER, PALETTE, SHORT,
    apply_style, light_grid, load, panel_label, save_figure,
)


def main() -> None:
    apply_style()
    pools = load("strict_h1_figure4_pool_effects.csv")
    summ = load("strict_h1_figure4_cross_architecture_summary.csv").set_index("context")

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    allv = np.concatenate([pools.cnn_mean_abs_error.to_numpy(), pools.kmer_mean_abs_error.to_numpy()])
    lo, hi = 0, allv.max() * 1.08
    ax.plot([lo, hi], [lo, hi], color=PALETTE["zero"], lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=PALETTE["grid"], alpha=0.35, zorder=0)

    handles = []
    for ctx in ORDER:
        pr = pools[pools.context == ctx]
        color = CONTEXT_COLORS[ctx]
        ax.scatter(pr.cnn_mean_abs_error, pr.kmer_mean_abs_error, s=34, color=color,
                   alpha=0.72, edgecolor="white", linewidth=0.5, zorder=3, label=SHORT[ctx])
        r = summ.loc[ctx]
        handles.append(f"{SHORT[ctx]}: r = {r.architecture_error_correlation:.3f}")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("CNN ensemble mean absolute error (MRL)")
    ax.set_ylabel("4-mer MLP mean absolute error (MRL)")
    ax.set_title("Per-pool error concordance across architectures", loc="left",
                 fontsize=12, fontweight="bold", color=PALETTE["ink"])
    light_grid(ax, "both")
    ax.set_axisbelow(True)

    leg = ax.legend(loc="upper left", fontsize=8.6, title="Error correlation", title_fontsize=9)
    leg.get_title().set_fontweight("bold")
    ax.text(0.98, 0.04,
            "Each point = one pool (n=16 per context).\nPoints hug the identity line: architectures\nmake the same errors, so disagreement\nis not independent functional validation.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0, color=PALETTE["subink"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=PALETTE["grid"], alpha=0.9))
    fig.text(0.01, -0.05,
             "Both models trained on the same Sample labels. Retrospective strict-H1 protocol sensitivity; correlation is descriptive, not causal.",
             ha="left", fontsize=7.6, color=PALETTE["subink"])
    fig.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.12)
    save_figure(fig, "Figure6_v2_error_concordance")


if __name__ == "__main__":
    main()