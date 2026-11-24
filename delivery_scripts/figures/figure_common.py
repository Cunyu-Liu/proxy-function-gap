from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#8BCF8B",
    "red": "#B64342",
    "neutral": "#767676",
    "light_neutral": "#CFCECE",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "ink": "#272727",
}
CONTEXT_COLORS = {
    "Sample designed": PALETTE["blue_main"],
    "Sample random": PALETTE["neutral"],
    "N50 HEK293T": PALETTE["teal"],
    "Defined HepG2 r1": PALETTE["violet"],
    "Defined T-cell r1": PALETTE["red"],
    "Defined T-cell r2": "#4E8B5B",
    "Designed outer": PALETTE["blue_main"],
    "Eligible random outer": PALETTE["neutral"],
}
CONTEXT_ORDER = [
    "Sample designed", "Sample random", "N50 HEK293T",
    "Defined HepG2 r1", "Defined T-cell r1", "Defined T-cell r2",
]
SHORT = {
    "Sample designed": "Sample designed",
    "Sample random": "Sample random",
    "N50 HEK293T": "N50 HEK293T",
    "Defined HepG2 r1": "HepG2 r1",
    "Defined T-cell r1": "T-cell r1",
    "Defined T-cell r2": "T-cell r2",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 7.8,
        "svg.fonttype": "none",
        "svg.hashsalt": "paper1-proxy-function-gap",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def format3(value, signed: bool = False) -> str:
    rounded = Decimal(str(float(value))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    sign = "+" if signed and rounded >= 0 else ""
    return f"{sign}{rounded:.3f}"


def panel_label(ax, label: str) -> None:
    ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color=PALETTE["ink"])


def save_figure(fig, basename: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.06, "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 300
        elif suffix == "svg":
            kwargs["metadata"] = {"Date": None}
        elif suffix == "pdf":
            kwargs["metadata"] = {"CreationDate": None}
        fig.savefig(HERE / f"{basename}.{suffix}", **kwargs)
    plt.close(fig)


def symmetric_limits(low, high, minimum_span=1.0):
    bound = max(abs(float(low)), abs(float(high)), minimum_span / 2) * 1.12
    return -bound, bound
