#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figure_common import CONTEXT_COLORS, PALETTE, SHORT, apply_style, format3, load, panel_label, save_figure

ORDER = ["N50 HEK293T", "Defined HepG2 r1", "Defined T-cell r1", "Defined T-cell r2"]


def main() -> None:
    apply_style()
    data = load("strict_h1_figure3_score_matching.csv").set_index("context").loc[ORDER].reset_index()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.3), gridspec_kw={"width_ratios": [1.48, 1], "wspace": 0.35})
    y = np.arange(len(data))[::-1]
    ax.axvline(0, color="#767676", lw=0.9, ls=(0, (3, 3)), zorder=0)
    for row, yi in zip(data.itertuples(), y):
        color = CONTEXT_COLORS[row.context]
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=color, lw=2.4, solid_capstyle="round")
        ax.scatter(row.difference, yi, s=58, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(1.87, yi,
                f"{format3(row.difference, signed=True)}  "
                f"[{format3(row.ci_low, signed=True)}, {format3(row.ci_high, signed=True)}]",
                va="center", ha="right", fontsize=7.8)
    ax.set_yticks(y, [SHORT[c] for c in data.context])
    ax.set_xlim(-1.72, 1.93)
    ax.set_xlabel("BoN − score-matched absolute error")
    ax.set_title("Score-matched error difference", loc="left")
    ax.grid(axis="x", color="#E1E1E1", lw=0.55)
    ax.set_axisbelow(True)
    ax.spines.left.set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "a")
    ax.text(0.02, -0.22, "Positive: BoN has higher error  ·  All nominal pool-cluster 95% intervals include zero",
            transform=ax.transAxes, fontsize=7.5, color="#5F5F5F")

    x = np.arange(len(data))
    for index, row in enumerate(data.itertuples()):
        color = CONTEXT_COLORS[row.context]
        ax2.plot([index - 0.13, index + 0.13], [row.optimized_mean_abs_error, row.matched_mean_abs_error],
                 color=color, alpha=0.55, lw=1.4)
        ax2.scatter(index - 0.13, row.optimized_mean_abs_error, s=44, marker="o", color=color,
                    edgecolor="white", linewidth=0.6, zorder=3)
        ax2.scatter(index + 0.13, row.matched_mean_abs_error, s=48, marker="s", color="white",
                    edgecolor=color, linewidth=1.3, zorder=3)
    ax2.set_xticks(x, ["N50", "HepG2", "T-cell r1", "T-cell r2"])
    ax2.set_ylabel("Mean absolute error")
    ax2.set_title("BoN and within-pool score matches", loc="left")
    ax2.grid(axis="y", color="#E1E1E1", lw=0.55)
    ax2.set_axisbelow(True)
    ax2.set_ylim(1.25, 3.55)
    panel_label(ax2, "b")
    ax2.scatter([], [], s=40, marker="o", color=PALETTE["blue_main"], label="BoN cap 16")
    ax2.scatter([], [], s=42, marker="s", color="white", edgecolor=PALETTE["blue_main"], linewidth=1.2,
                label="Nearest proxy-score match")
    ax2.legend(loc="lower left", ncol=1, handlelength=1.2)

    fig.suptitle("Strict-H1 score matching finds no stable optimizer-specific increment",
                 x=0.055, y=0.995, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.055, 0.915,
             "Retrospective protocol sensitivity; 16 pools per context. Matching is descriptive and does not remove all distribution shift.",
             fontsize=8.4, color="#5F5F5F", ha="left")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.82, bottom=0.22)
    save_figure(fig, "Figure3_strict_h1_score_matching")


if __name__ == "__main__":
    main()
