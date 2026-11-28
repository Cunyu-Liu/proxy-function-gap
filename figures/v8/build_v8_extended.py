#!/usr/bin/env python3
"""Extended Data figures for V10 (v8): label distributions."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figure_style_v8 import C, apply_style, panel_label, clean, save

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "reports" / "source_data" / "cross_scenario_tail_audit_v2"
DIST = json.loads((SRC / "label_distributions.json").read_text())


def ed_distributions():
    apply_style()
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = GridSpec(3, 3, hspace=0.75, wspace=0.45, figure=fig)
    for idx, r in enumerate(DIST):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        q5, q25, med, q75, q95 = r["q5"], r["q25"], r["median"], r["q75"], r["q95"]
        ax.plot([q5, q95], [0, 0], color=C["gray"], lw=1.2, zorder=2)
        ax.plot([q5, q5], [-0.12, 0.12], color=C["gray"], lw=1.0)
        ax.plot([q95, q95], [-0.12, 0.12], color=C["gray"], lw=1.0)
        ax.add_patch(plt.Rectangle((q25, -0.22), q75 - q25, 0.44, facecolor=C["blue_light"],
                                   edgecolor=C["blue"], linewidth=1.0, zorder=3))
        ax.plot([med, med], [-0.22, 0.22], color=C["blue"], lw=1.8, zorder=4)
        ax.scatter([r["mean"]], [0], s=28, color=C["red"], marker="D", zorder=5, label="mean")
        ax.set_ylim(-0.6, 0.6); ax.set_yticks([])
        ax.set_xlabel(r["unit"], fontsize=7)
        ax.set_title(f"{r['scenario']} (n={r['n']:,})", fontsize=7.5, loc="left", color=C["ink"])
        ax.text(0.98, 0.92, f"skew {r['skew']:.1f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.8, color=C["sub"])
        ax.tick_params(labelsize=6.5); clean(ax, grid=False)
        panel_label(ax, chr(ord("a") + idx), dx=-0.18, dy=1.15)
    # legend panel
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    ax.plot([0.1, 0.5], [0.7, 0.7], color=C["gray"], lw=1.2)
    ax.add_patch(plt.Rectangle((0.2, 0.55), 0.2, 0.3, facecolor=C["blue_light"], edgecolor=C["blue"]))
    ax.plot([0.3, 0.3], [0.55, 0.85], color=C["blue"], lw=1.8)
    ax.scatter([0.6], [0.7], s=28, color=C["red"], marker="D")
    ax.text(0.1, 0.35, "box: IQR (q25–q75)\nline: median\nwhisker: q5–q95\ndiamond: mean",
            fontsize=7, va="top", color=C["sub"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "ED_Figure1_v8_label_distributions", HERE)


def main():
    ed_distributions()
    print("built ED label distributions v8")


if __name__ == "__main__":
    main()