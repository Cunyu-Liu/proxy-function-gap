from __future__ import annotations

import csv
import hashlib
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


def apply_style():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "text.color": C["ink"], "axes.labelcolor": C["ink"],
        "axes.edgecolor": C["ink"], "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "cross-scenario-v5", "pdf.fonttype": 42,
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


def panel_label(base, x, y, letter, title, color="blue"):
    base.text(x, y, f"{letter}  {title}", fontsize=9.2, fontweight="bold", color=C[color], va="top")


def save_figure(fig, basename):
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": "white"}
        if suffix == "png":
            kwargs.update(dpi=300, metadata={"Software": "cross-scenario-v5"})
        elif suffix == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "Creator": "cross-scenario-v5"}
        else:
            kwargs["metadata"] = {"Date": None}
        fig.savefig(HERE / f"{basename}.{suffix}", **kwargs)
    plt.close(fig)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_evidence():
    budget = load_json(SRC / "paper1_budget_matched_v1" / "results.json")
    summaries = {(r["context"], r["cap"]): r for r in budget["context_summaries"]}
    fam = {}
    for r in budget["families"]["bon_minus_random_n"]:
        key = (r["context"], r["cap"])
        fam[key] = {**r, "random_n_regret": summaries[key]["random_n_regret"]}
    gb1 = load_json(SRC / "scenario_gb1_v1" / "aggregate" / "test_aggregate.json")
    gfp = load_json(SRC / "scenario_gfp_v1" / "results" / "primary.json")
    aav = load_json(SRC / "scenario_aav_v1" / "results" / "main_results.json")
    tem_dir = load_tsv(SRC / "scenario_tem1_v1" / "results" / "direction_results.tsv")
    mrna = load_tsv(SRC / "scenario_mrna_non_tcell_v1" / "artifact" / "results" / "decision_summary.tsv")
    metrics = load_json(SRC / "scenario_mrna_non_tcell_v1" / "artifact" / "results" / "unified_scenario_metrics.json")
    tcell_diag = load_json(SRC / "tcell_error_diagnosis_v1" / "analysis" / "summary.json")
    adapt = load_json(SRC / "tcell_target_adaptation_v1" / "target" / "primary.json")
    return budget, fam, gb1, gfp, aav, tem_dir, mrna, metrics, tcell_diag, adapt


def rel(delta, random):
    return -float(delta) / float(random) * 100.0


def figure18(fam, gb1, gfp, aav, tem_dir, mrna, metrics):
    fig, base = canvas(14.0, 9.2)
    heading(base, "Proxy ranking helps in several high-value tasks, but not universally",
            "Cap16 equal-budget functional regret. Relative reduction is descriptive within-task normalization; original units and CIs remain task-specific.")

    positive = []
    for ctx, label in [("defined_tcell_r1", "T-cell MRL r1"), ("defined_tcell_r2", "T-cell MRL r2")]:
        r = fam[(ctx, 16)]
        positive.append((label, rel(r["effect"], r["random_n_regret"]), [rel(r["nominal_95_ci"][1], r["random_n_regret"]), rel(r["nominal_95_ci"][0], r["random_n_regret"])], "teal" if r["holm_p"] < 0.05 else "gold"))
    b = gb1["budgets"]["16"]; ci = gb1["bootstrap"]["16"]["ci95"]
    positive.append(("GB1 IgG lnW", rel(b["delta_regret_bon_minus_random"], b["random_regret"]), [rel(ci[1], b["random_regret"]), rel(ci[0], b["random_regret"])], "teal"))
    methods = {m["method"]: m for m in gfp["methods"]}
    g = next(r for r in methods["additive"]["primary"] if r["budget"] == 16)
    positive.append(("GFP brightness (additive)", rel(g["mean_bon_minus_random_regret"], g["mean_random_regret"]), [rel(g["ci95_bon_minus_random_regret"]["high"], g["mean_random_regret"]), rel(g["ci95_bon_minus_random_regret"]["low"], g["mean_random_regret"])], "teal"))
    a = aav["caps"]["16"]
    positive.append(("AAV9 production", rel(a["paired_delta_bon_minus_random"], a["random_mean_regret"]), [rel(a["ci95_bootstrap_percentile"][1], a["random_mean_regret"]), rel(a["ci95_bootstrap_percentile"][0], a["random_mean_regret"])], "teal"))
    amp16 = [r for r in tem_dir if r["split"] == "test" and r["N"] == "16" and not r["target"].startswith("cefotaxime")]
    amp_delta = np.mean([float(r["delta_bon_minus_random"]) for r in amp16])
    amp_random = np.mean([float(r["random_n_regret"]) for r in amp16])
    positive.append(("TEM-1 Amp→Amp", rel(amp_delta, amp_random), None, "gold"))

    panel_label(base, 0.055, 0.855, "A", "TASKS WITH EQUAL-BUDGET FUNCTIONAL BENEFIT", "teal")
    ax = fig.add_axes([0.24, 0.50, 0.68, 0.30])
    y = np.arange(len(positive))[::-1]
    for yi, (label, value, ci, color) in zip(y, positive):
        ax.scatter(value, yi, s=70, color=C[color], edgecolor=C["ink"], linewidth=0.7, zorder=3)
        if ci is not None:
            ax.plot(ci, [yi, yi], color=C[color], lw=2.2, zorder=2)
        ax.text(value + 1.5, yi, f"{value:.1f}%", va="center", fontsize=8.5, fontweight="bold", color=C[color])
    ax.set_yticks(y, [p[0] for p in positive]); ax.set_xlim(-5, 100); ax.set_ylim(-0.6, len(positive)-0.4)
    ax.set_xlabel("Relative regret reduction vs RandomN (%)", fontsize=8.5)
    ax.axvline(0, color=C["muted"], lw=0.8, ls="--"); ax.grid(axis="x", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    base.text(0.24, 0.455, "TEM-1 Amp→Amp is a direction-level summary: 12/20 cap16 intervals favor BoN; no cross-direction pooled CI is claimed.", fontsize=8, color=C["subink"])

    boundary = []
    cef16 = [r for r in tem_dir if r["split"] == "test" and r["N"] == "16" and r["target"].startswith("cefotaxime")]
    cef_delta = np.mean([float(r["delta_bon_minus_random"]) for r in cef16])
    cef_random = np.mean([float(r["random_n_regret"]) for r in cef16])
    boundary.append(("TEM-1 Amp→cefotaxime", rel(cef_delta, cef_random), "gold", "15/15 CIs cross zero"))
    n50 = next(r for r in metrics if r["scenario"] == "n50_hek293t_r1" and r["cap"] == 16)
    boundary.append(("HEK293T N50 MRL", rel(n50["bon_minus_random_n"], n50["random_n_regret"]), "gold", "CI crosses zero"))
    hepg2 = next(r for r in metrics if r["scenario"] == "defined_hepg2_r1" and r["cap"] == 16)
    boundary.append(("HepG2 defined-end MRL", rel(hepg2["bon_minus_random_n"], hepg2["random_n_regret"]), "gold", "nominal only; Holm p=0.304"))
    stab = next(r for r in mrna if r["pool"] == "stability_kmer_27" and r["context"] == "primary|integrated_abundance" and r["cap"] == "16")
    boundary.append(("mRNA stability", rel(stab["bon_minus_random_n"], stab["random_n_regret"]), "coral", "significantly worse"))
    boundary.append(("Castillo indel", None, "gold", "0/4 corrected-positive contexts"))
    boundary.append(("Schlusser endogenous TE", None, "gold", "historical CI crosses zero"))

    panel_label(base, 0.055, 0.405, "B", "BOUNDARIES: NO ROBUST BENEFIT OR HARM", "coral")
    ax = fig.add_axes([0.24, 0.10, 0.68, 0.25])
    y = np.arange(len(boundary))[::-1]
    for yi, (label, value, color, note) in zip(y, boundary):
        if value is not None:
            ax.scatter(value, yi, s=70, color=C[color], edgecolor=C["ink"], linewidth=0.7, zorder=3)
            ax.text(value + 1.5, yi, f"{value:.1f}%", va="center", fontsize=8.5, fontweight="bold", color=C[color])
        else:
            ax.scatter(0, yi, s=70, marker="s", color=C[color], edgecolor=C["ink"], linewidth=0.7, zorder=3)
        ax.text(52, yi, note, va="center", fontsize=8, color=C["subink"])
    ax.set_yticks(y, [b[0] for b in boundary]); ax.set_xlim(-45, 100); ax.set_ylim(-0.6, len(boundary)-0.4)
    ax.set_xlabel("Relative regret reduction vs RandomN (%)", fontsize=8.5)
    ax.axvline(0, color=C["muted"], lw=0.8, ls="--"); ax.grid(axis="x", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    base.text(0.035, 0.035, "All analyses are retrospective. Endpoint units differ; no cross-task pooled effect size is computed. AAV production is not tropism or clinical efficacy; GB1 lnW is not Kd.", fontsize=8.2, color=C["subink"])
    save_figure(fig, "Figure18_v5_cross_scenario_functional_value")


def figure19(gfp, aav, tcell_diag, adapt):
    fig, base = canvas(14.0, 8.6)
    heading(base, "Regression quality and decision quality are different quantities",
            "Three independent demonstrations that MAE, global correlation, or top-tail recall cannot replace measured functional regret.")

    panel_label(base, 0.055, 0.855, "A", "T-CELL: LOWER MAE DID NOT IMPROVE CAP16 REGRET")
    ax = fig.add_axes([0.075, 0.56, 0.25, 0.24])
    repeats = ["r1", "r2"]
    mae_change = [-0.437653, -0.043556]
    regret_change = [0.038929, 0.012551]
    x = np.arange(2)
    ax.bar(x - 0.18, mae_change, width=0.34, color=C["teal"], edgecolor=C["ink"], linewidth=0.7, label="MAE change")
    ax.bar(x + 0.18, regret_change, width=0.34, color=C["coral"], edgecolor=C["ink"], linewidth=0.7, label="Regret change")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_xticks(x, ["T-cell r1", "T-cell r2"]); ax.set_ylabel("Adapted − original (MRL)", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="lower left"); ax.grid(axis="y", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    base.text(0.075, 0.50, "50k target labels reduced MAE, especially in r1, but cap16 regret increased in both repeats.", fontsize=8, color=C["subink"])

    panel_label(base, 0.385, 0.855, "B", "GFP: HIGH-ORDER CORRELATION DECAYS WHILE SELECTION STILL HELPS", "violet")
    ax = fig.add_axes([0.40, 0.56, 0.25, 0.24])
    methods = {m["method"]: m for m in gfp["methods"]}
    orders = methods["cnn_ensemble"]["point_metrics_by_order"]
    orders = [r for r in orders if 5 <= r["order"] <= 9]
    ax.plot([r["order"] for r in orders], [r["spearman"] for r in orders], marker="o", color=C["violet"], lw=2)
    ax.axhline(0, color=C["muted"], lw=0.8, ls="--")
    ax.set_xlabel("Mutation order", fontsize=8.5); ax.set_ylabel("CNN Spearman ρ", fontsize=8.5)
    ax.set_ylim(-0.1, 0.7); ax.grid(color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    g = next(r for r in methods["cnn_ensemble"]["primary"] if r["budget"] == 16)
    base.text(0.40, 0.50, f"Despite ρ≈−0.02 at order 9, cap16 BoN−RandomN = {g['mean_bon_minus_random_regret']:.3f} (CI excludes 0).", fontsize=8, color=C["subink"])

    panel_label(base, 0.715, 0.855, "C", "AAV9: HIGH GLOBAL ρ DOES NOT RECOVER THE EXTREME BEST", "blue")
    ax = fig.add_axes([0.73, 0.56, 0.22, 0.24])
    values = [0.820908, 0.1875]
    bars = ax.bar([0, 1], values, width=0.5, color=[C["blue"], C["gold"]], edgecolor=C["ink"], linewidth=0.7)
    ax.set_xticks([0, 1], ["Global Spearman", "True top1% recall"], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Fraction / correlation", fontsize=8.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.03, f"{value:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.grid(axis="y", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    base.text(0.73, 0.50, "Production ranking improves average choice, but most extreme-best variants remain missed.", fontsize=8, color=C["subink"])

    panel_label(base, 0.055, 0.42, "D", "WHAT THE CROSS-SCENARIO EVIDENCE SUPPORTS", "teal")
    box = fig.add_axes([0.075, 0.10, 0.875, 0.26]); box.axis("off")
    messages = [
        ("Use functional regret as the acceptance endpoint", "MAE and global correlation are diagnostics, not substitutes for the selected sequence's measured function."),
        ("Keep equal-budget RandomN and simple baselines", "GFP additive outperformed CNN; GB1 additive ridge was sufficient; complexity is not automatically valuable."),
        ("State the task and endpoint contract", "MRL, lnW, brightness, production, indel, stability, TE and relative resistance are not interchangeable."),
        ("Do not claim universal proxy validity", "Benefits coexist with null and harmful contexts; retrospective grouped holdouts are not prospective wet-lab validation."),
    ]
    for i, (title, body) in enumerate(messages):
        x = 0.02 + (i % 2) * 0.50; y = 0.72 - (i // 2) * 0.42
        box.add_patch(plt.Rectangle((x, y - 0.30), 0.46, 0.34, facecolor=C["soft"], edgecolor=C["line"], linewidth=0.8))
        box.text(x + 0.02, y, title, fontsize=9, fontweight="bold", color=C["teal"], va="top")
        box.text(x + 0.02, y - 0.10, body, fontsize=8, color=C["subink"], va="top", wrap=True)
    save_figure(fig, "Figure19_v5_regression_decision_dissociation")


def main():
    apply_style()
    budget, fam, gb1, gfp, aav, tem_dir, mrna, metrics, tcell_diag, adapt = load_evidence()
    figure18(fam, gb1, gfp, aav, tem_dir, mrna, metrics)
    figure19(gfp, aav, tcell_diag, adapt)
    config = {
        "figures": ["Figure18_v5_cross_scenario_functional_value", "Figure19_v5_regression_decision_dissociation"],
        "sources": [
            "paper1_budget_matched_v1/results.json",
            "scenario_gb1_v1/aggregate/test_aggregate.json",
            "scenario_gfp_v1/results/primary.json",
            "scenario_aav_v1/results/main_results.json",
            "scenario_tem1_v1/results/direction_results.tsv",
            "scenario_mrna_non_tcell_v1/artifact/results/decision_summary.tsv",
            "scenario_mrna_non_tcell_v1/artifact/results/unified_scenario_metrics.json",
        ],
        "cross_task_pooled_effect": False,
        "retrospective": True,
    }
    (HERE / "cross_scenario_v5_build_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("built Figure18/19 v5")


if __name__ == "__main__":
    main()