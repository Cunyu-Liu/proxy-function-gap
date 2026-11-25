from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from delivery_scripts.publication_paths import DELIVERABLES, EVIDENCE_ROOT, OUTPUT_ROOT, display_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = EVIDENCE_ROOT
FIG = DELIVERABLES / "figures"
OUT = OUTPUT_ROOT / "figures" / "v3"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS = DELIVERABLES / "reports"
DATA = FIG / "source_data"
SOURCE = REPORTS / "source_data"

PALETTE = {
    "ink": "#17233B",
    "subink": "#5D687A",
    "muted": "#8B95A7",
    "line": "#DDE3EB",
    "soft": "#F4F7FA",
    "paper": "#FFFFFF",
    "blue": "#0F4D92",
    "blue_light": "#EAF1F8",
    "teal": "#238A7A",
    "teal_light": "#E8F4F1",
    "coral": "#C65D4B",
    "coral_light": "#FAEEEB",
    "gold": "#B8862D",
    "gold_light": "#FBF4E7",
    "violet": "#725A9C",
    "violet_light": "#F0EDF6",
    "gray_light": "#EEF1F5",
}
STATUS = {
    "diagnostic": (PALETTE["gold"], PALETTE["gold_light"]),
    "confirmatory": (PALETTE["teal"], PALETTE["teal_light"]),
    "exploratory": (PALETTE["violet"], PALETTE["violet_light"]),
    "gap": (PALETTE["coral"], PALETTE["coral_light"]),
    "retrospective": (PALETTE["blue"], PALETTE["blue_light"]),
}


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "proxy-function-gap-v3",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def box(ax, x, y, w, h, facecolor="white", edgecolor=None, radius=0.025, lw=1.0, zorder=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=ax.transAxes, clip_on=False,
        facecolor=facecolor, edgecolor=edgecolor or facecolor,
        linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, color, facecolor, size=7.5, width=None):
    width = width or max(0.09, 0.0057 * len(text) + 0.022)
    box(ax, x, y - 0.022, width, 0.047, facecolor=facecolor, radius=0.018, zorder=4)
    ax.text(x + 0.012, y, text, transform=ax.transAxes, ha="left", va="center",
            fontsize=size, color=color, fontweight="bold", zorder=5)
    return width


def save_figure(fig, basename: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 300
            kwargs["metadata"] = {"Software": "proxy-function-gap-v3"}
        elif suffix == "svg":
            kwargs["metadata"] = {"Date": None}
        else:
            kwargs["metadata"] = {"CreationDate": None, "Creator": "proxy-function-gap-v3"}
        fig.savefig(OUT / f"{basename}.{suffix}", **kwargs)
    plt.close(fig)


def source_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
