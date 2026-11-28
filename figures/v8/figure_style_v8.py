"""Shared publication style for the V10 (v8) figure set.

House style per scientific-figure-making + plot-is-all-you-need:
compact multi-panel, lowercase bold panel letters, interpretation in captions
(not baked-in titles), semantic colorblind-safe palette, top/right spines off,
vector PDF/SVG + 300 dpi PNG, deterministic rebuild.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {
    "ink": "#1A1A1A", "sub": "#5D687A", "line": "#D9DEE6", "soft": "#F4F7FA",
    "blue": "#0F4D92", "blue2": "#3775BA", "blue_light": "#EAF1F8",
    "green": "#009E73", "green_light": "#DDF3DE",
    "red": "#D55E00", "red_light": "#F6CFCB",
    "gray": "#767676", "gray_light": "#CFCECE",
    "gold": "#E69F00", "violet": "#725A9C", "teal": "#238A7A",
}
FAMILY = {"mRNA": C["blue"], "protein": C["teal"], "gene therapy": C["violet"], "enzyme": C["gold"]}
STATUS_COLOR = {"positive": C["blue"], "nominal": C["blue2"], "condition-dependent": C["gold"],
                "null": C["gray"], "harmful": C["red"], "mixed/null": C["gray"]}
STATUS_MARKER = {"positive": "o", "nominal": "s", "condition-dependent": "D",
                 "null": "X", "harmful": "v", "mixed/null": "P"}


def apply_style():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": C["ink"], "axes.labelcolor": C["ink"], "axes.edgecolor": C["ink"],
        "xtick.color": C["ink"], "ytick.color": C["ink"],
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "svg.hashsalt": "v10-figures-v8",
        "pdf.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
        "lines.linewidth": 1.4, "figure.dpi": 150,
    })


def panel_label(ax, letter, dx=-0.14, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="left", color=C["ink"])


def clean(ax, grid=True):
    if grid:
        ax.grid(color=C["line"], lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def save(fig, basename, here):
    for suffix in ("png", "pdf", "svg"):
        kw = {"bbox_inches": "tight", "pad_inches": 0.06, "facecolor": "white"}
        if suffix == "png":
            kw.update(dpi=300, metadata={"Software": "v10-figures-v8"})
        elif suffix == "pdf":
            kw["metadata"] = {"CreationDate": None, "Creator": "v10-figures-v8"}
        else:
            kw["metadata"] = {"Date": None}
        fig.savefig(here / f"{basename}.{suffix}", **kw)
    plt.close(fig)