from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from delivery_scripts.publication_paths import DELIVERABLES, OUTPUT_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = OUTPUT_ROOT / "figures" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
DATA = DELIVERABLES / "figures" / "source_data"

# Modern, harmonious palette (Nature/Cell-style warm-cool family).
PALETTE = {
    "ink": "#1A1A2E",
    "subink": "#5C6378",
    "grid": "#E6E8EE",
    "zero": "#9AA0B0",
    "blue": "#264653",
    "teal": "#2A9D8F",
    "violet": "#6A4C93",
    "coral": "#E76F51",
    "sand": "#F4A261",
    "gold": "#E9C46A",
    "good": "#2A9D8F",
    "bad": "#C1443C",
    "neutral": "#8D99AE",
    "paper": "#FFFFFF",
}
CONTEXT_COLORS = {
    "N50 HEK293T": PALETTE["teal"],
    "Defined HepG2 r1": PALETTE["violet"],
    "Defined T-cell r1": PALETTE["coral"],
    "Defined T-cell r2": PALETTE["sand"],
    "Sample designed": PALETTE["blue"],
    "Sample random": PALETTE["neutral"],
}
ORDER = ["N50 HEK293T", "Defined HepG2 r1", "Defined T-cell r1", "Defined T-cell r2"]
SHORT = {
    "N50 HEK293T": "N50 HEK293T",
    "Defined HepG2 r1": "HepG2 r1",
    "Defined T-cell r1": "T-cell r1",
    "Defined T-cell r2": "T-cell r2",
    "Sample designed": "Sample designed",
    "Sample random": "Sample random",
}
MULT_KEY = {
    "N50 HEK293T": "n50_hek293t_r1",
    "Defined HepG2 r1": "defined_hepg2_r1",
    "Defined T-cell r1": "defined_tcell_r1",
    "Defined T-cell r2": "defined_tcell_r2",
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.6,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 8.6,
        "svg.fonttype": "none",
        "svg.hashsalt": "paper1-proxy-function-gap-v2",
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
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left", color=PALETTE["ink"])


def save_figure(fig, basename: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.07, "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 300
        elif suffix == "svg":
            kwargs["metadata"] = {"Date": None}
        elif suffix == "pdf":
            kwargs["metadata"] = {"CreationDate": None}
        fig.savefig(OUT / f"{basename}.{suffix}", **kwargs)
    plt.close(fig)


def jitter(n: int, seed: int, spread: float = 0.10) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-spread, spread, size=n)


def light_grid(ax, axis="y") -> None:
    ax.grid(axis=axis, color=PALETTE["grid"], lw=0.7, zorder=0)
    ax.set_axisbelow(True)