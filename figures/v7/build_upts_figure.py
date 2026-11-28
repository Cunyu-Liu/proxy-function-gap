from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CANONICAL_PYTHON = Path.home() / "anaconda3" / "bin" / "python"
if Path(sys.executable).resolve() != CANONICAL_PYTHON.resolve():
    if not CANONICAL_PYTHON.exists():
        raise RuntimeError(f"Canonical figure renderer not found: {CANONICAL_PYTHON}")
    os.execv(str(CANONICAL_PYTHON), [str(CANONICAL_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "deliverables" / "reports" / "source_data" / "tcell_upts_v1" / "artifact" / "auxiliary"
C = {
    "ink": "#17233B", "subink": "#5D687A", "muted": "#8B95A7", "line": "#DDE3EB",
    "soft": "#F4F7FA", "paper": "#FFFFFF", "blue": "#0F4D92", "blue_light": "#EAF1F8",
    "teal": "#238A7A", "teal_light": "#E8F4F1", "coral": "#C65D4B", "coral_light": "#FAEEEB",
    "gold": "#B8862D", "gold_light": "#FBF4E7", "violet": "#725A9C", "violet_light": "#F0EDF6",
}
DIRS = [
    ("n50_hek293t_r1_to_defined_hepg2_r1", "N50→HepG2", C["blue"]),
    ("defined_hepg2_r1_to_n50_hek293t_r1", "HepG2→N50", C["violet"]),
]


def apply_style():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "text.color": C["ink"], "axes.labelcolor": C["ink"],
        "axes.edgecolor": C["ink"], "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "upts-v7", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white", "legend.frameon": False,
    })


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def heading(base, title, subtitle):
    base.text(0.035, 0.965, title, fontsize=20, fontweight="bold", va="top")
    base.text(0.035, 0.915, subtitle, fontsize=10.5, color=C["subink"], va="top")
    base.plot([0.035, 0.965], [0.865, 0.865], color=C["line"], lw=1.2)


def save_figure(fig, basename):
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": "white"}
        if suffix == "png":
            kwargs.update(dpi=300, metadata={"Software": "upts-v7"})
        elif suffix == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "Creator": "upts-v7"}
        else:
            kwargs["metadata"] = {"Date": None}
        fig.savefig(HERE / f"{basename}.{suffix}", **kwargs)
    plt.close(fig)


def load():
    out = {}
    for key, label, color in DIRS:
        out[key] = json.loads((SRC / key / "selection.json").read_text(encoding="utf-8"))
    return out


def main():
    apply_style()
    data = load()
    fig, base = canvas(14.0, 5.8)
    heading(base, "Uncertainty-penalized tail selection fails the pre-registered auxiliary gate",
            "UPTS v1: identical frozen proposal prefixes and scores; κ selected on auxiliary dev only. Retrospective method development; T-cell target labels never read.")

    kappas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    x = np.arange(len(kappas))

    axA = fig.add_axes([0.065, 0.20, 0.27, 0.58])
    for key, label, color in DIRS:
        s = data[key]
        dev = [c["dev_cap16_mean_regret"] for c in s["candidates"]]
        axA.plot(x, dev, "-o", color=color, ms=4.5, lw=1.4, label=label)
        ci = kappas.index(s["chosen"]["kappa"])
        axA.plot([ci], [dev[ci]], marker="*", ms=13, color=color, mec=C["ink"], mew=0.6, zorder=5)
    axA.set_xticks(x); axA.set_xticklabels([str(k) for k in kappas])
    axA.set_xlabel("κ (penalty on ensemble std)", fontsize=9.5)
    axA.set_ylabel("Auxiliary dev cap16 mean BoN regret (MRL)", fontsize=9.5)
    axA.set_title("A  Dev-only selection", fontsize=10, fontweight="bold", loc="left", color=C["ink"])
    axA.grid(color=C["line"], lw=0.6); axA.set_axisbelow(True); axA.tick_params(labelsize=8.5)
    axA.legend(fontsize=8.5, loc="upper left")
    axA.text(0.02, -0.24, "★ chosen κ (smaller-κ tie-break)", transform=axA.transAxes, fontsize=8, color=C["subink"])

    axB = fig.add_axes([0.415, 0.20, 0.24, 0.58])
    width = 0.34
    for i, (key, label, color) in enumerate(DIRS):
        s = data[key]
        base_v = s["baseline"]["test"]["cap16_mean_regret"]
        cho_v = s["test"]["cap16_mean_regret"]
        impr = s["test"]["cap16_regret_improvement"]
        xb = i + np.array([-width / 2, width / 2])
        axB.bar([xb[0]], [base_v], width=width, color=C["muted"], edgecolor=C["ink"], lw=0.6)
        axB.bar([xb[1]], [cho_v], width=width, color=color, edgecolor=C["ink"], lw=0.6)
        passed = s["gate_pass"]
        badge = "PASS" if passed else "FAIL"
        bcolor = C["teal"] if passed else C["coral"]
        axB.text(i, max(base_v, cho_v) + 0.06, f"{badge}  Δ={impr:+.4f}", ha="center", fontsize=8.5, fontweight="bold", color=bcolor)
    axB.set_xticks([0, 1]); axB.set_xticklabels([d[1] for d in DIRS], fontsize=9)
    axB.set_ylabel("Auxiliary test cap16 mean BoN regret (MRL)", fontsize=9.5)
    axB.set_title("B  Gate: both directions must improve", fontsize=10, fontweight="bold", loc="left", color=C["ink"])
    axB.grid(color=C["line"], lw=0.6, axis="y"); axB.set_axisbelow(True); axB.tick_params(labelsize=8.5)
    axB.set_ylim(3.8, 6.15)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C["muted"], edgecolor=C["ink"], lw=0.6, label="κ=0 (frozen BoN)"),
               plt.Rectangle((0, 0), 1, 1, facecolor=C["blue"], edgecolor=C["ink"], lw=0.6, label="chosen κ")]
    axB.legend(handles=handles, fontsize=8, loc="lower left")

    axC = fig.add_axes([0.735, 0.20, 0.225, 0.58])
    metrics = [("spearman_std_vs_signed_error", "std vs signed error"), ("spearman_std_vs_abs_error", "std vs |error|")]
    yc = np.arange(len(metrics))
    h = 0.16
    for j, (key, label, color) in enumerate(DIRS):
        vals = [data[key]["diagnostics"][m] for m, _ in metrics]
        axC.barh(yc + (j - 0.5) * 2 * h, vals, height=h * 1.6, color=color, edgecolor=C["ink"], lw=0.6, label=label)
        for yi, v in zip(yc + (j - 0.5) * 2 * h, vals):
            axC.text(v + 0.0006, yi, f"{v:.4f}", va="center", fontsize=8, color=C["ink"])
    axC.axvline(0, color=C["muted"], lw=1.0, ls="--")
    axC.set_yticks(yc); axC.set_yticklabels([lab for _, lab in metrics], fontsize=8.8)
    axC.set_xlim(-0.002, 0.024)
    axC.set_xlabel("Spearman ρ (auxiliary dev top16)", fontsize=9.5)
    axC.set_title("C  Disagreement carries no signal", fontsize=10, fontweight="bold", loc="left", color=C["ink"])
    axC.grid(color=C["line"], lw=0.6, axis="x"); axC.set_axisbelow(True); axC.tick_params(labelsize=8.5)
    axC.legend(fontsize=8, loc="lower right")

    base.add_patch(plt.Rectangle((0.035, 0.035), 0.93, 0.075, facecolor=C["coral_light"], edgecolor=C["coral"], lw=1.0))
    base.text(0.05, 0.072, "Auxiliary gate FAILED (HepG2→N50 worsened by 0.0296 MRL) → stopped before T-cell; target MRL not read.",
              fontsize=9.5, fontweight="bold", color=C["coral"], va="center")
    base.text(0.035, 0.135, "Evidence: retrospective · auxiliary-gated method development · κ=0 identity exact vs frozen baseline. Source: deliverables/reports/source_data/tcell_upts_v1/artifact/auxiliary/*/selection.json",
              fontsize=8, color=C["subink"])
    save_figure(fig, "Figure22_v7_upts_auxiliary_gate_falsification")
    config = {"figure": "Figure22_v7_upts_auxiliary_gate_falsification", "source": "tcell_upts_v1/artifact/auxiliary/*/selection.json", "retrospective": True, "gate_passed": False}
    (HERE / "upts_v7_build_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("built Figure22 v7")


if __name__ == "__main__":
    main()