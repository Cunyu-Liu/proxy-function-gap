#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from figure_common import CONTEXT_COLORS, SHORT, apply_style, format3, load, panel_label, save_figure

ORDER = ["N50 HEK293T", "Defined HepG2 r1", "Defined T-cell r1", "Defined T-cell r2"]


def main() -> None:
    apply_style()
    data = load("strict_h1_figure2_paired_pressure.csv")
    fig = plt.figure(figsize=(12.2, 6.35))
    grid = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1.23], hspace=0.52, wspace=0.38)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    for index, (ax, context) in enumerate(zip(axes, ORDER)):
        rows = data[data.context == context].sort_values("proposal_cap")
        x = rows.proposal_cap.to_numpy()
        effect = rows.paired_bon_minus_random.to_numpy()
        low = rows.ci_low.to_numpy()
        high = rows.ci_high.to_numpy()
        color = CONTEXT_COLORS[context]
        ax.axhline(0, color="#767676", lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
        ax.plot(x, effect, color=color, lw=1.9, marker="o", ms=4.5,
                markerfacecolor="white", markeredgewidth=1.4)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
        ax.set_xlabel("Best-of-N proposal cap")
        if index % 2 == 0:
            ax.set_ylabel("BoN − Random regret")
        ax.set_title(f"{SHORT[context]}  ·  n=16 pools", loc="left")
        ax.grid(axis="y", color="#D9D9D9", lw=0.55, alpha=0.65)
        ax.set_axisbelow(True)
        bound = max(abs(low.min()), abs(high.max()), 0.55) * 1.18
        ax.set_ylim(-bound, bound)
        panel_label(ax, chr(ord("a") + index))

    ax = fig.add_subplot(grid[:, 2])
    y = np.arange(len(ORDER))[::-1]
    ax.axvline(0, color="#767676", lw=0.9, ls=(0, (3, 3)), zorder=0)
    for yi, context in zip(y, ORDER):
        rows = data[data.context == context].set_index("proposal_cap")
        color = CONTEXT_COLORS[context]
        cap8 = rows.loc[8]
        cap16 = rows.loc[16]
        ax.plot([cap8.ci_low, cap8.ci_high], [yi + 0.13, yi + 0.13], color=color, lw=1.3, alpha=0.55)
        ax.scatter(cap8.paired_bon_minus_random, yi + 0.13, s=35, marker="D", color="white",
                   edgecolor=color, linewidth=1.1, zorder=3)
        ax.plot([cap16.ci_low, cap16.ci_high], [yi - 0.10, yi - 0.10], color=color, lw=2.1,
                solid_capstyle="round")
        ax.scatter(cap16.paired_bon_minus_random, yi - 0.10, s=50, color=color,
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(1.48, yi - 0.10,
                f"{format3(cap16.paired_bon_minus_random, signed=True)}  "
                f"[{format3(cap16.ci_low, signed=True)}, {format3(cap16.ci_high, signed=True)}]",
                va="center", ha="right", fontsize=7.5)
    ax.set_yticks(y, [SHORT[c] for c in ORDER])
    ax.set_xlim(-2.42, 1.55)
    ax.set_ylim(-0.65, 3.65)
    ax.set_xlabel("Paired BoN − Random regret")
    ax.set_title("Cap 16 effect; cap 8 shown as ◇", loc="left")
    ax.grid(axis="x", color="#E1E1E1", lw=0.55, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines.left.set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "e")
    ax.text(0.02, -0.14, "Negative values favor BoN over same-seed Random", transform=ax.transAxes,
            fontsize=7.5, color="#5F5F5F")

    fig.suptitle("Strict-H1 pressure gives reproducible but context-dependent gains",
                 x=0.055, y=0.995, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.055, 0.952,
             "Nominal 95% pool-cluster CIs; Random = one candidate, BoN cap varies (not an equal-candidate-budget comparison).",
             ha="left", fontsize=8.3, color="#5F5F5F")
    fig.text(0.055, 0.012,
             "T-cell cap 2–16 survives all-20 Bonferroni; HepG2 cap 8 crosses zero after correction and is exploratory. Retrospective, not a new blind test.",
             ha="left", fontsize=7.2, color="#6A6A6A")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.885, bottom=0.115)
    save_figure(fig, "Figure2_strict_h1_selection_pressure")


if __name__ == "__main__":
    main()
