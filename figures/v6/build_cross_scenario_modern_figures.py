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
SRC = ROOT / "deliverables" / "reports" / "source_data"
C = {
    "ink": "#17233B", "subink": "#5D687A", "muted": "#8B95A7", "line": "#DDE3EB",
    "soft": "#F4F7FA", "paper": "#FFFFFF", "blue": "#0F4D92", "blue_light": "#EAF1F8",
    "teal": "#238A7A", "teal_light": "#E8F4F1", "coral": "#C65D4B", "coral_light": "#FAEEEB",
    "gold": "#B8862D", "gold_light": "#FBF4E7", "violet": "#725A9C", "violet_light": "#F0EDF6",
}
FAMILY_COLOR = {"mRNA": C["blue"], "protein": C["teal"], "gene therapy": C["violet"], "enzyme": C["gold"]}
STATUS_MARKER = {"positive": "o", "nominal": "s", "condition-dependent": "D", "null": "X", "harmful": "v", "mixed/null": "P"}


def apply_style():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "text.color": C["ink"], "axes.labelcolor": C["ink"],
        "axes.edgecolor": C["ink"], "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "cross-scenario-v6", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white", "legend.frameon": False,
    })


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def heading(base, title, subtitle):
    base.text(0.035, 0.965, title, fontsize=22, fontweight="bold", va="top")
    base.text(0.035, 0.925, subtitle, fontsize=11, color=C["subink"], va="top")
    base.plot([0.035, 0.965], [0.885, 0.885], color=C["line"], lw=1.2)


def save_figure(fig, basename):
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": "white"}
        if suffix == "png":
            kwargs.update(dpi=300, metadata={"Software": "cross-scenario-v6"})
        elif suffix == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "Creator": "cross-scenario-v6"}
        else:
            kwargs["metadata"] = {"Date": None}
        fig.savefig(HERE / f"{basename}.{suffix}", **kwargs)
    plt.close(fig)


def load_audit():
    return json.loads((SRC / "cross_scenario_tail_audit_v1" / "tail_audit.json").read_text(encoding="utf-8"))["rows"]


def figure20(rows):
    fig, base = canvas(14.0, 8.8)
    heading(base, "Decision value is not a simple function of global prediction accuracy",
            "Each point is a task-specific cap16 equal-budget comparison. Relative regret reduction is descriptive; no cross-task pooled effect is computed.")
    ax = fig.add_axes([0.09, 0.16, 0.62, 0.66])
    plotted = []
    for r in rows:
        if r["relative_regret_reduction_pct"] is None:
            continue
        x = r["global_spearman"]; y = r["relative_regret_reduction_pct"]
        color = FAMILY_COLOR[r["family"]]
        marker = STATUS_MARKER[r["status"]]
        size = 110 if r["status"] == "positive" else 85
        ax.scatter(x, y, s=size, color=color, marker=marker, edgecolor=C["ink"], linewidth=0.8, zorder=3)
        plotted.append(r)
    ax.axhline(0, color=C["muted"], lw=1.0, ls="--")
    ax.axvline(0, color=C["muted"], lw=1.0, ls="--")
    ax.set_xlabel("Global Spearman ρ (task-specific test set)", fontsize=10)
    ax.set_ylabel("Relative regret reduction vs RandomN (%)", fontsize=10)
    ax.set_xlim(-0.9, 1.0); ax.set_ylim(-45, 100)
    ax.grid(color=C["line"], lw=0.6); ax.set_axisbelow(True); ax.tick_params(labelsize=9)
    offsets = {
        "T-cell MRL r1": (0.03, 4), "T-cell MRL r2": (0.03, -8), "GB1 binding": (0.03, 4),
        "GFP brightness": (0.03, 4), "AAV9 production": (-0.28, 6), "TEM-1 Amp→Amp": (0.03, 4),
        "TEM-1 Amp→cefotaxime": (0.03, -8), "HEK293T N50 MRL": (0.03, 4),
        "HepG2 defined-end MRL": (0.03, -8), "mRNA stability": (0.03, 4),
    }
    for r in plotted:
        dx, dy = offsets.get(r["scenario"], (0.03, 4))
        ax.text(r["global_spearman"] + dx, r["relative_regret_reduction_pct"] + dy, r["scenario"], fontsize=8.2, color=C["ink"], va="center")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor=C["ink"], markersize=8, label=family) for family, color in FAMILY_COLOR.items()]
    status_handles = [plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor=C["muted"], markeredgecolor=C["ink"], markersize=8, label=status) for status, marker in STATUS_MARKER.items()]
    leg = fig.add_axes([0.74, 0.50, 0.22, 0.30]); leg.axis("off")
    leg.legend(handles=handles, title="Task family", loc="upper left", fontsize=8.5, title_fontsize=9)
    leg2 = fig.add_axes([0.74, 0.20, 0.22, 0.28]); leg2.axis("off")
    leg2.legend(handles=status_handles, title="Evidence status", loc="upper left", fontsize=8.5, title_fontsize=9)
    base.text(0.09, 0.08, "High global ρ does not guarantee extreme-best recovery; negative or near-zero tail reliability marks null/harmful contexts. Endpoint units differ across tasks.", fontsize=8.5, color=C["subink"])
    save_figure(fig, "Figure20_v6_decision_value_map")


def figure21():
    fig, base = canvas(14.0, 9.0)
    heading(base, "A unified decision-focused protocol across heterogeneous biological tasks",
            "The shared object is not the molecule or endpoint, but the selection decision under a fixed candidate budget and explicit leakage control.")
    stages = [
        ("1  Define endpoint", "MRL · lnW · brightness · production · indel · stability · TE · resistance", "blue"),
        ("2  Freeze data roles", "train / dev / permanent test; exact/H1 or position/component isolation", "teal"),
        ("3  Fit proxy", "CNN, additive ridge, or condition-derived fitness; dev-only selection", "violet"),
        ("4  Equal-budget decision", "BoN(N) vs RandomN on identical pools/prefixes; functional regret", "gold"),
        ("5  Diagnose tail", "global ρ, top-tail recall/ρ, opportunity vs within-tail ranking loss", "coral"),
    ]
    y = 0.78
    for i, (title, body, color) in enumerate(stages):
        x = 0.055 + i * 0.185
        base.add_patch(plt.Rectangle((x, y - 0.16), 0.165, 0.20, facecolor=C[color + "_light"], edgecolor=C[color], linewidth=1.0))
        base.text(x + 0.012, y, title, fontsize=9.2, fontweight="bold", color=C[color], va="top")
        base.text(x + 0.012, y - 0.055, body, fontsize=7.6, color=C["subink"], va="top", wrap=True)
        if i < len(stages) - 1:
            base.annotate("", xy=(x + 0.185, y - 0.06), xytext=(x + 0.165, y - 0.06), arrowprops=dict(arrowstyle="->", color=C["muted"], lw=1.2))
    panel = fig.add_axes([0.075, 0.10, 0.875, 0.42]); panel.axis("off")
    panel.text(0, 0.98, "What is shared", fontsize=10, fontweight="bold", color=C["ink"], va="top")
    shared = [
        "A fixed candidate pool and a measured functional oracle.",
        "An equal-budget random control, so ranking value is separated from candidate opportunity.",
        "A pre-specified acceptance endpoint: regret, not MAE or global correlation.",
        "Explicit endpoint semantics and leakage boundaries; no cross-unit pooled effect.",
    ]
    for i, text in enumerate(shared):
        panel.text(0.02, 0.84 - i * 0.16, "• " + text, fontsize=8.8, color=C["subink"], va="top")
    panel.text(0.52, 0.98, "What is not shared", fontsize=10, fontweight="bold", color=C["coral"], va="top")
    not_shared = [
        "Molecule type, assay, endpoint unit, and biological meaning.",
        "Cluster unit: pool, position-pair, position, or UTR.",
        "RandomN implementation: exact expectation, fixed repeats, or shared prefix.",
        "Evidence level: retrospective grouped holdout, not prospective wet-lab validation.",
    ]
    for i, text in enumerate(not_shared):
        panel.text(0.54, 0.84 - i * 0.16, "• " + text, fontsize=8.8, color=C["subink"], va="top")
    save_figure(fig, "Figure21_v6_unified_protocol_governance")


def main():
    apply_style()
    rows = load_audit()
    figure20(rows)
    figure21()
    config = {"figures": ["Figure20_v6_decision_value_map", "Figure21_v6_unified_protocol_governance"], "source": "cross_scenario_tail_audit_v1/tail_audit.json", "retrospective": True}
    (HERE / "cross_scenario_v6_build_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("built Figure20/21 v6")


if __name__ == "__main__":
    main()