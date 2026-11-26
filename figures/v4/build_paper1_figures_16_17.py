from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from delivery_scripts.publication_paths import DELIVERABLES, EVIDENCE_ROOT, OUTPUT_ROOT

CANONICAL_PYTHON = Path(os.environ.get("PFG_CANONICAL_PYTHON", sys.executable)).expanduser().resolve()
if Path(sys.executable).resolve() != CANONICAL_PYTHON:
    if not CANONICAL_PYTHON.exists():
        raise RuntimeError(f"Canonical figure renderer not found: {CANONICAL_PYTHON}")
    os.execv(str(CANONICAL_PYTHON), [str(CANONICAL_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = EVIDENCE_ROOT
SRC = DELIVERABLES / "reports" / "source_data"
OUT = OUTPUT_ROOT / "figures" / "v4"
OUT.mkdir(parents=True, exist_ok=True)
DIAG = SRC / "tcell_error_diagnosis_v1" / "analysis"
V1 = SRC / "tcell_target_adaptation_v1" / "target"
V2 = SRC / "tcell_listnet_adaptation_v2" / "target_v2"
ANCHOR = SRC / "tcell_cross_cell_anchor_v1" / "target"
RANK = SRC / "tcell_rank_bstrict_v2" / "artifacts" / "auxiliary_gate.json"
BACKBONE = SRC / "tcell_multicontext_backbone_v1" / "auxiliary_gate.json"
C = {
    "ink": "#17233B", "subink": "#5D687A", "muted": "#8B95A7", "line": "#DDE3EB",
    "soft": "#F4F7FA", "paper": "#FFFFFF", "blue": "#0F4D92", "blue_light": "#EAF1F8",
    "teal": "#238A7A", "teal_light": "#E8F4F1", "coral": "#C65D4B", "coral_light": "#FAEEEB",
    "gold": "#B8862D", "gold_light": "#FBF4E7", "violet": "#725A9C", "violet_light": "#F0EDF6",
}
CONTEXTS = ["defined_tcell_r1", "defined_tcell_r2"]


def apply_style():
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9, "text.color": C["ink"], "axes.labelcolor": C["ink"],
        "axes.edgecolor": C["ink"], "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "svg.hashsalt": "paper1-v4", "pdf.fonttype": 42,
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
            kwargs.update(dpi=300, metadata={"Software": "paper1-v4"})
        elif suffix == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "Creator": "paper1-v4"}
        else:
            kwargs["metadata"] = {"Date": None}
        fig.savefig(OUT / f"{basename}.{suffix}", **kwargs)
    plt.close(fig)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_evidence():
    point = pd.read_csv(DIAG / "pointwise_metrics.tsv", sep="\t")
    selection = pd.read_csv(DIAG / "selection_summary.tsv", sep="\t")
    tail = pd.read_csv(DIAG / "top_tail_summary.tsv", sep="\t")
    effects = pd.DataFrame(load_json(DIAG / "summary.json")["selection_effects"])
    v1_primary = load_json(V1 / "primary.json")
    v1_results = load_json(V1 / "results.json")
    v1_secondary = load_json(V1 / "secondary.json")
    v2_primary = load_json(V2 / "primary.json")
    v2_results = load_json(V2 / "results.json")
    anchor_primary = load_json(ANCHOR / "primary.json")
    anchor_secondary = load_json(ANCHOR / "secondary.json")
    rank = load_json(RANK)
    backbone = load_json(BACKBONE)
    assert not load_json(DIAG / "summary.json")["new_blind_test"]
    assert not v1_primary["success"] and not v2_primary["success"] and not anchor_primary["success"]
    assert not rank["target_mrl_read"] and not rank["target_evaluation_permitted"]
    assert backbone["target_evaluation_count"] == 0 and not backbone["target_mrl_read"]
    return point, selection, tail, effects, v1_primary, v1_results, v1_secondary, v2_primary, v2_results, anchor_primary, anchor_secondary, rank, backbone


def figure16(point, selection, tail, effects):
    fig, base = canvas(14.0, 9.0)
    heading(base, "A useful global proxy can still fail at the high-scoring tail",
            "T-cell strict-H1 diagnostic: pointwise error, global ranking, equal-budget value and top-16 regret are distinct quantities.")
    panel_label(base, 0.055, 0.855, "A", "GLOBAL SIGNAL DESPITE HIGH POINTWISE ERROR")
    ax = fig.add_axes([0.075, 0.565, 0.355, 0.235])
    cnn = point[point.model == "cnn"].set_index("context").loc[CONTEXTS]
    y = np.array([1.0, 0.0])
    bars = ax.barh(y, cnn.spearman, height=0.42, color=C["blue"], edgecolor=C["ink"], linewidth=0.8)
    ax.set_yticks(y, ["T-cell r1", "T-cell r2"]); ax.set_xlim(0, 0.72); ax.set_ylim(-0.65, 1.65)
    ax.set_xlabel("Global Spearman ρ (all 1,024 candidates)", fontsize=8.5)
    ax.grid(axis="x", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    for bar, (_, row) in zip(bars, cnn.iterrows()):
        ax.text(row.spearman + 0.015, bar.get_y() + bar.get_height() / 2, f"ρ = {row.spearman:.3f}", va="center", fontsize=9, fontweight="bold", color=C["blue"])
        ax.text(0.02, bar.get_y() + bar.get_height() / 2, f"MAE {row.mae:.3f} MRL  ·  bias {row.bias:+.3f}", va="center", fontsize=8, color="white", fontweight="bold")
    base.text(0.075, 0.505, "High MAE does not imply zero ranking information; a constant monotone shift cannot change BoN selection or regret.", fontsize=8.2, color=C["subink"])

    panel_label(base, 0.540, 0.855, "B", "EQUAL-BUDGET RANKING VALUE", "teal")
    ax = fig.add_axes([0.555, 0.565, 0.380, 0.235])
    for context, color, marker in zip(CONTEXTS, ("blue", "teal"), ("o", "s")):
        rows = effects[(effects.context == context) & (effects.model == "cnn") & (effects.contrast == "bon_minus_random_n")].sort_values("cap")
        x = np.arange(5)
        ax.vlines(x, rows.ci_low, rows.ci_high, color=C[color], lw=1.0, alpha=0.45)
        ax.plot(x, rows.effect, color=C[color], lw=1.8, marker=marker, markersize=4.2, label=context.replace("defined_tcell_", "T-cell "))
    ax.axhline(0, color=C["line"], lw=0.9); ax.set_xticks(np.arange(5), [1, 2, 4, 8, 16]); ax.set_ylim(-2.35, 0.45)
    ax.set_xlabel("Shared proposal budget N", fontsize=8.5); ax.set_ylabel("BoN(N) − Random(N) regret", fontsize=8.5)
    ax.grid(axis="y", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8); ax.legend(fontsize=8, loc="lower left")
    base.text(0.555, 0.505, "Cap 16: r1 −1.706 [−2.169, −1.211]; r2 −1.087 [−1.812, −0.336]. All cap 2–16 point estimates favor BoN.", fontsize=8.2, color=C["subink"])

    panel_label(base, 0.055, 0.455, "C", "TOP-16 REGRET: COVERAGE vs WITHIN-TAIL ORDER", "coral")
    ax = fig.add_axes([0.075, 0.175, 0.355, 0.225])
    rows = tail[(tail.model == "cnn") & (tail.cap == 16)].set_index("context").loc[CONTEXTS]
    x = np.arange(2)
    opportunity = rows.opportunity_loss.to_numpy(); within = rows.within_tail_ranking_loss.to_numpy()
    ax.bar(x, opportunity, width=0.52, color=C["gold"], edgecolor=C["ink"], linewidth=0.8, label="Opportunity loss: tail misses pool oracle")
    ax.bar(x, within, width=0.52, bottom=opportunity, color=C["coral"], edgecolor=C["ink"], linewidth=0.8, label="Within-tail ranking loss")
    for i, (_, row) in enumerate(rows.iterrows()):
        ax.text(i, row.regret + 0.10, f"total {row.regret:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i, row.opportunity_loss + row.within_tail_ranking_loss / 2, f"{row.within_tail_ranking_loss:.3f}", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.set_xticks(x, ["T-cell r1", "T-cell r2"]); ax.set_ylim(0, 4.75); ax.set_ylabel("Top-1 regret (MRL)", fontsize=8.5)
    ax.grid(axis="y", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8); ax.legend(fontsize=7.3, loc="upper left")
    base.text(0.075, 0.135, "Within-tail loss exceeds opportunity loss in both repeats (74.8% and 52.0% of total regret).", fontsize=8.2, color=C["subink"])

    panel_label(base, 0.540, 0.455, "D", "THE BOTTLENECK IS LOCAL ORDER, NOT GLOBAL USELESSNESS", "violet")
    ax = fig.add_axes([0.555, 0.175, 0.380, 0.225])
    global_rho = cnn.spearman.to_numpy(); tail_rho = rows.spearman.to_numpy(); xx = np.arange(2); width = 0.30
    ax.bar(xx - width / 2, global_rho, width, color=C["blue"], edgecolor=C["ink"], linewidth=0.8, label="Global")
    ax.bar(xx + width / 2, tail_rho, width, color=C["violet"], edgecolor=C["ink"], linewidth=0.8, label="Within predicted top 16")
    for i, value in enumerate(tail_rho):
        ax.text(i + width / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=8.5, color=C["violet"], fontweight="bold")
        ax.text(i, 0.68, "15/16 pools\nwinner flipped", ha="center", fontsize=8.2, color=C["coral"], fontweight="bold")
    ax.set_xticks(xx, ["T-cell r1", "T-cell r2"]); ax.set_ylim(0, 0.82); ax.set_ylabel("Spearman ρ", fontsize=8.5)
    ax.grid(axis="y", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8); ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    base.text(0.555, 0.135, "Design implication: optimize and evaluate high-tail decisions directly; do not use global calibration as a surrogate repair.", fontsize=8.2, color=C["subink"])
    base.text(0.035, 0.040, "Retrospective diagnostic of previously inspected strict-H1 pools (16 pools × 5 proposal seeds per repeat). Negative Δregret favors BoN. CIs are nominal 95% pool-cluster bootstrap intervals; not a new blind test or a validated repair.", fontsize=7.5, color=C["subink"])
    save_figure(fig, "Figure16_v4_tcell_global_value_tail_bottleneck")


def figure17(v1p, v1r, v1s, v2p, v2r, anchorp, anchors, rank, backbone):
    fig, base = canvas(14.0, 9.4)
    heading(base, "Regression improvement does not guarantee a better design decision",
            "Target adaptation, listwise learning and cross-cell anchoring are separated by test set and by whether T-cell evaluation was ever opened.")
    panel_label(base, 0.045, 0.855, "A", "TEST-V1 · 50k TARGET ADAPTATION: MAE DECREASES")
    ax = fig.add_axes([0.065, 0.580, 0.250, 0.220])
    x = np.arange(2); width = 0.32
    original = [v1s["pointwise_errors_mrl_units"][c]["original_cnn"]["mae"] for c in CONTEXTS]
    main = [v1s["pointwise_errors_mrl_units"][c]["main_50k"]["mae"] for c in CONTEXTS]
    ax.bar(x - width / 2, original, width, color=C["muted"], edgecolor=C["ink"], linewidth=0.8, label="Original CNN")
    ax.bar(x + width / 2, main, width, color=C["blue"], edgecolor=C["ink"], linewidth=0.8, label="50k main")
    for i, (a, b) in enumerate(zip(original, main)):
        ax.text(i, max(a, b) + 0.07, f"Δ {b-a:+.3f}", ha="center", fontsize=8.3, color=C["blue"], fontweight="bold")
    ax.set_xticks(x, ["r1", "r2"]); ax.set_ylim(0, 1.65); ax.set_ylabel("MAE (MRL; lower is better)", fontsize=8)
    ax.grid(axis="y", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8); ax.legend(fontsize=7.3)

    panel_label(base, 0.365, 0.855, "B", "TEST-V1 · CAP16 REGRET DOES NOT IMPROVE", "coral")
    ax = fig.add_axes([0.385, 0.580, 0.250, 0.220])
    values = [v1p["repeat_main_minus_original_effects"][c] for c in CONTEXTS]
    bars = ax.bar(x, values, 0.48, color=C["coral"], edgecolor=C["ink"], linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.007, f"{value:+.4f}", ha="center", fontsize=8.5, color=C["coral"], fontweight="bold")
    ax.axhline(0, color=C["ink"], lw=0.8); ax.set_xticks(x, ["r1", "r2"]); ax.set_ylim(-0.015, 0.065)
    ax.set_ylabel("50k main − original regret", fontsize=8); ax.tick_params(labelsize=8)
    base.text(0.385, 0.525, f"Combined +0.025740 [−0.072019, +0.140994]\nRequired reduction ≥ {v1p['minimum_required_absolute_reduction']:.6f}. Primary FAIL.", fontsize=7.9, color=C["coral"], fontweight="bold")

    panel_label(base, 0.685, 0.855, "C", "TEST-V2 · LISTNET: SMALL ESTIMATES, JOINT GATE FAILS", "violet")
    ax = fig.add_axes([0.705, 0.580, 0.250, 0.220])
    labels = ["vs original CNN", "vs v1 main"]
    comps = [v2p["comparisons"]["original_cnn"], v2p["comparisons"]["v1_main"]]
    yy = np.array([1.0, 0.0]); effects = [c["effect"]["effect"] for c in comps]
    for y0, comp, color in zip(yy, comps, ("violet", "blue")):
        e = comp["effect"]
        ax.plot([e["ci_low"], e["ci_high"]], [y0, y0], color=C[color], lw=2)
        ax.scatter(e["effect"], y0, s=48, color=C[color], zorder=3)
        ax.text(0.115, y0, f"{e['effect']:+.4f}", va="center", fontsize=8.2, color=C[color], fontweight="bold")
    ax.axvline(0, color=C["ink"], lw=0.8); ax.axvline(-0.1, color=C["coral"], lw=1.0, ls="--")
    ax.text(-0.098, 1.43, "−0.1 minimum", fontsize=7.2, color=C["coral"], va="top")
    ax.set_yticks(yy, labels); ax.set_xlim(-0.22, 0.17); ax.set_ylim(-0.6, 1.55); ax.set_xlabel("ListNet − comparator regret", fontsize=8)
    ax.grid(axis="x", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5)
    base.text(0.705, 0.505, "Both 95% CIs cross zero; both reductions are <0.1 MRL.\nCo-primary IUT FAIL. Test-v2 is component-disjoint from test-v1.", fontsize=7.8, color=C["subink"])

    panel_label(base, 0.055, 0.455, "D", "PAIRED CROSS-CELL TEST · HepG2 ANCHOR REVERSES", "coral")
    ax = fig.add_axes([0.075, 0.175, 0.365, 0.220])
    vals = [anchorp["repeat_anchor_minus_cnn_effects"][c] for c in CONTEXTS]
    bars = ax.bar(x, vals, 0.48, color=C["coral"], edgecolor=C["ink"], linewidth=0.8)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.018, f"+{value:.3f}", ha="center", fontsize=8.5, color=C["coral"], fontweight="bold")
    ax.set_xticks(x, ["T-cell r1", "T-cell r2"]); ax.set_ylim(0, 0.58); ax.set_ylabel("HepG2-anchor − CNN regret", fontsize=8)
    ax.grid(axis="y", color=C["line"], lw=0.55); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
    base.text(0.075, 0.115, "Combined +0.428897 [+0.245460, +0.616474].\nHepG2 global ρ is higher (0.730/0.750 vs CNN 0.722/0.725), yet T-cell selection is worse.", fontsize=8.0, color=C["subink"])

    panel_label(base, 0.535, 0.455, "E", "AUXILIARY GATES FAILED CLOSED · NO T-CELL RESULT", "gold")
    base.add_patch(plt.Rectangle((0.545, 0.185), 0.185, 0.205, transform=base.transAxes, facecolor=C["gold_light"], edgecolor="none"))
    base.text(0.560, 0.355, "Pairwise B-strict v2", fontsize=10, fontweight="bold", color=C["gold"])
    base.text(0.560, 0.315, "N50→HepG2: −0.00654\n[−0.01420, +0.00123]", fontsize=8.5, color=C["ink"], va="top")
    base.text(0.560, 0.245, "HepG2→N50: +0.02119\n[−0.00772, +0.05010]", fontsize=8.5, color=C["ink"], va="top")
    base.text(0.560, 0.198, "Both gates FAIL", fontsize=8.5, color=C["coral"], fontweight="bold")
    base.add_patch(plt.Rectangle((0.750, 0.185), 0.190, 0.205, transform=base.transAxes, facecolor=C["soft"], edgecolor="none"))
    base.text(0.765, 0.355, "Multicontext backbone v1", fontsize=10, fontweight="bold", color=C["blue"])
    base.text(0.765, 0.315, "N50→HepG2: +0.04463\n[+0.02184, +0.06835]", fontsize=8.5, color=C["ink"], va="top")
    base.text(0.765, 0.245, "HepG2→N50: +0.02805\n[−0.04912, +0.10461]", fontsize=8.5, color=C["ink"], va="top")
    base.text(0.765, 0.198, "Both gates FAIL", fontsize=8.5, color=C["coral"], fontweight="bold")
    base.text(0.545, 0.115, "Target evaluation count = 0 for both methods.\nAuxiliary failure is not a negative T-cell effect estimate.", fontsize=8.0, color=C["subink"])
    base.text(0.035, 0.040, "All target tests are retrospective grouped holdouts from previously inspected studies, not prospective or independent biological validation. Negative Δregret favors the proposed method; positive values favor the comparator. No failed method is claimed as a performance improvement.", fontsize=7.5, color=C["subink"])
    save_figure(fig, "Figure17_v4_tcell_regression_decision_decoupling")


def main():
    apply_style()
    evidence = load_evidence()
    figure16(*evidence[:4])
    figure17(*evidence[4:])
    source_paths = [
        DIAG / "summary.json", DIAG / "pointwise_metrics.tsv", DIAG / "selection_summary.tsv", DIAG / "top_tail_summary.tsv",
        V1 / "primary.json", V1 / "results.json", V1 / "secondary.json",
        V2 / "primary.json", V2 / "results.json", ANCHOR / "primary.json", ANCHOR / "secondary.json", RANK, BACKBONE,
    ]
    figures = []
    for stem in ("Figure16_v4_tcell_global_value_tail_bottleneck", "Figure17_v4_tcell_regression_decision_decoupling"):
        for suffix in ("png", "pdf", "svg"):
            path = OUT / f"{stem}.{suffix}"
            figures.append({"path": f"deliverables/figures/v4/{path.name}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    config = {
        "schema": "paper1-v4-figures-1", "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "evidence_boundary": "retrospective T-cell analyses; test-v1/test-v2 separated; auxiliary gate failures have target evaluation count zero; no failed method is presented as improvement",
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths],
        "figures": figures,
    }
    (OUT / "paper1_figures_16_17_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "figures": len(figures), "sources": len(source_paths)}, indent=2))


if __name__ == "__main__":
    main()
