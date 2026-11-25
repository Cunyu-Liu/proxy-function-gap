#!/usr/bin/env python3
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from figure_common_v2 import (
    DATA, MULT_KEY, ORDER, PALETTE, SHORT,
    apply_style, load, panel_label, save_figure,
)

MULT = DATA.parent.parent / "reports" / "source_data" / "paper1_strict_h1_v3" / "analysis" / "strict_h1_multiplicity_sensitivity.json"
CAPS = [1, 2, 4, 8, 16]


def main() -> None:
    apply_style()
    data = load("strict_h1_figure2_paired_pressure.csv")
    mult = {(r["context"], r["proposal_cap"]): r for r in json.load(open(MULT))["rows"]}

    mat = np.zeros((len(ORDER), len(CAPS)))
    surv = np.zeros((len(ORDER), len(CAPS)), dtype=bool)
    nom = np.zeros((len(ORDER), len(CAPS)), dtype=bool)
    for i, ctx in enumerate(ORDER):
        for j, cap in enumerate(CAPS):
            row = data[(data.context == ctx) & (data.proposal_cap == cap)].iloc[0]
            mat[i, j] = row.paired_bon_minus_random
            m = mult[(MULT_KEY[ctx], cap)]
            surv[i, j] = m["bonferroni_20_comparisons_excludes_zero"]
            nom[i, j] = m["nominal_95_percent_excludes_zero"]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    bound = max(abs(mat.min()), abs(mat.max())) * 1.02
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")

    for i in range(len(ORDER)):
        for j in range(len(CAPS)):
            v = mat[i, j]
            txt = f"{v:+.2f}"
            if surv[i, j]:
                txt += " ★"
            color = "white" if abs(v) > bound * 0.55 else PALETTE["ink"]
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.2,
                    fontweight="bold" if surv[i, j] else "normal", color=color)
            if surv[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=PALETTE["ink"], lw=2.0))
            elif nom[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=PALETTE["subink"], lw=1.0, ls=(0, (3, 2))))

    ax.set_xticks(range(len(CAPS)), [f"cap {c}" for c in CAPS])
    ax.set_yticks(range(len(ORDER)), [SHORT[c] for c in ORDER])
    ax.set_xlabel("Best-of-N proposal cap", labelpad=8)
    ax.set_title("Pressure–benefit landscape: paired BoN − Random regret (MRL)",
                 loc="left", fontsize=12, fontweight="bold", color=PALETTE["ink"], pad=12)
    ax.set_xticks(np.arange(-0.5, len(CAPS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.0)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("BoN − Random regret  (blue favors BoN)", fontsize=8.6)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    fig.text(0.01, -0.06,
             "★ = survives all-20 Bonferroni (99.75% CI excludes 0); dashed box = nominal-only; "
             "all eight T-cell caps 2–16 are robust. HepG2 cap 8 and all N50 cells are not multiplicity-robust. "
             "Retrospective strict-H1 protocol sensitivity.",
             ha="left", fontsize=7.6, color=PALETTE["subink"])
    fig.subplots_adjust(left=0.16, right=0.99, top=0.88, bottom=0.16)
    save_figure(fig, "Figure5_v2_pressure_heatmap")


if __name__ == "__main__":
    main()