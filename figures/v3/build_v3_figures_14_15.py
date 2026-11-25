from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from delivery_scripts.publication_paths import OUTPUT_ROOT

CANONICAL_PYTHON = Path(os.environ.get("PFG_CANONICAL_PYTHON", sys.executable)).expanduser()
if Path(sys.executable).resolve() != CANONICAL_PYTHON.resolve():
    if not CANONICAL_PYTHON.exists():
        raise RuntimeError(f"Configured V3 renderer not found: {CANONICAL_PYTHON}")
    os.execv(str(CANONICAL_PYTHON), [str(CANONICAL_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_common_v3 import HERE, OUT, PALETTE, SOURCE, apply_style, box, save_figure, source_record, write_json

BUDGET = SOURCE / "paper1_budget_matched_v1"
SEQUENCE = SOURCE / "paper1_sequence_ablation_v1" / "analysis" / "results"
CAPS = np.array([1, 2, 4, 8, 16])
DISPLAY_CAPS = np.array([2, 4, 8, 16])
CONTEXTS = ["n50_hek293t_r1", "defined_hepg2_r1", "defined_tcell_r1", "defined_tcell_r2"]
TITLES = {
    "n50_hek293t_r1": "N50 HEK293T",
    "defined_hepg2_r1": "Defined HepG2 r1",
    "defined_tcell_r1": "Defined T-cell r1",
    "defined_tcell_r2": "Defined T-cell r2",
}


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def load_evidence():
    budget = json.loads((BUDGET / "results.json").read_text())
    sequence = json.loads((SEQUENCE / "summary.json").read_text())
    assert not budget["new_blind_test"] and budget["execution"] == "CPU_only_no_training_no_inference"
    assert not sequence["new_blind_test"] and not sequence["causal_claim"]
    assert budget["grid"]["pools_per_context"] == 16 and budget["bootstrap"]["replicates"] == 10000
    assert sequence["poly_u_q75"] == 3 and sequence["primary_comparisons"] == 20
    equal = pd.DataFrame(budget["families"]["bon_minus_random_n"])
    random = pd.DataFrame(budget["families"]["random_n_minus_random1"])
    primary = pd.DataFrame(sequence["primary"])
    family = pd.DataFrame(sequence["sequence_family_sensitivity"])
    assert len(equal) == len(random) == len(primary) == 20 and len(family) == 40
    r1 = equal[(equal.context == "defined_tcell_r1") & (equal.cap > 1)]
    r2 = equal[(equal.context == "defined_tcell_r2") & (equal.cap > 1)]
    assert (r1.holm_p < 0.05).all()
    assert r2.loc[r2.holm_p < 0.05, "cap"].tolist() == [4]
    assert not (random.holm_p < 0.05).any()
    assert not (primary.adjusted_holm_p < 0.05).any()
    assert not (family[family.family == "no_uAUG"].holm_p < 0.05).any()
    poly_sig = family[(family.family == "no_polyU_upper_quartile") & (family.holm_p < 0.05)]
    assert set(zip(poly_sig.context, poly_sig.cap)) == {("defined_tcell_r1", 4), ("defined_tcell_r2", 8)}
    return budget, sequence, equal, random, primary, family


def effect_curve(ax, rows, effect, low, high, color, filled=None, linestyle="-", marker="o", size=30, lw=1.8):
    rows = rows.sort_values("cap")
    x = np.arange(len(rows))
    y = rows[effect].to_numpy(float)
    lo = rows[low].to_numpy(float)
    hi = rows[high].to_numpy(float)
    ax.vlines(x, lo, hi, color=PALETTE[color], lw=1.0, alpha=0.48, zorder=2)
    ax.plot(x, y, color=PALETTE[color], lw=lw, ls=linestyle, zorder=3)
    if filled is None:
        filled = np.ones(len(rows), dtype=bool)
    for xx, yy, is_filled in zip(x, y, filled):
        ax.scatter(xx, yy, s=size, marker=marker, facecolor=PALETTE[color] if is_filled else "white",
                   edgecolor=PALETTE[color], linewidth=1.2, zorder=4)


def fig_budget(equal, random):
    fig, base = canvas(14.0, 8.7)
    base.text(0.035, 0.955, "Equal-budget ranking value is supported in T-cell r1—not universal",
              fontsize=22, fontweight="bold", va="top")
    base.text(0.035, 0.915,
              "BoN(N) and Random(N) draw from the same proposal prefix; negative regret difference favors BoN at the same N.",
              fontsize=11.2, color=PALETTE["subink"], va="top")
    base.plot([0.035, 0.965], [0.880, 0.880], color=PALETTE["line"], lw=1.2)
    base.text(0.045, 0.850, "A  FOUR STRICT-H1 CONTEXTS · NOMINAL 95% POOL-CLUSTER CI",
              fontsize=9.2, fontweight="bold", color=PALETTE["blue"])
    handles = [
        plt.Line2D([], [], color=PALETTE["blue"], marker="o", lw=1.8, label="BoN(N) − Random(N)"),
        plt.Line2D([], [], color=PALETTE["muted"], marker="o", markerfacecolor="white", lw=1.3,
                   ls="--", label="Random(N) − Random(1)"),
        plt.Line2D([], [], color=PALETTE["blue"], marker="o", markerfacecolor="white", lw=0,
                   label="Open blue: Holm p ≥ 0.05"),
        plt.Line2D([], [], color=PALETTE["blue"], marker="o", lw=0, label="Filled blue: Holm p < 0.05"),
    ]
    base.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.340, 0.866), ncol=2,
                fontsize=8.0, handletextpad=0.5, columnspacing=1.4)
    positions = [(0.070, 0.515), (0.555, 0.515), (0.070, 0.205), (0.555, 0.205)]
    for context, (x, y) in zip(CONTEXTS, positions):
        ax = fig.add_axes([x, y, 0.375, 0.255])
        rows = equal[equal.context == context].sort_values("cap")
        lo = np.array([v[0] for v in rows.nominal_95_ci])
        hi = np.array([v[1] for v in rows.nominal_95_ci])
        xx = np.arange(5)
        ax.vlines(xx, lo, hi, color=PALETTE["blue"], lw=1.0, alpha=0.45, zorder=2)
        ax.plot(xx, rows.effect, color=PALETTE["blue"], lw=1.9, zorder=3)
        for xi, (_, row) in zip(xx, rows.iterrows()):
            significant = row.cap > 1 and row.holm_p < 0.05
            ax.scatter(xi, row.effect, s=38 if significant else 31,
                       facecolor=PALETTE["blue"] if significant else "white",
                       edgecolor=PALETTE["blue"], linewidth=1.25, zorder=4)
        rr = random[random.context == context].sort_values("cap")
        ax.plot(xx, rr.effect, color=PALETTE["muted"], lw=1.25, ls="--", zorder=2)
        ax.scatter(xx, rr.effect, s=22, facecolor="white", edgecolor=PALETTE["muted"], linewidth=1.0, zorder=3)
        ax.axhline(0, color=PALETTE["line"], lw=0.9, zorder=0)
        ax.set_xlim(-0.3, 4.3)
        ax.set_ylim(-2.55, 1.80)
        ax.set_xticks(xx, CAPS)
        ax.set_xlabel("Candidate budget N", fontsize=8.5, labelpad=4)
        ax.set_title(TITLES[context], loc="left", fontsize=11.0, fontweight="bold", pad=7)
        ax.grid(axis="y", color=PALETTE["line"], lw=0.55, alpha=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
        if x < 0.1:
            ax.set_ylabel("Δ measured-MRL regret", fontsize=8.5, labelpad=5)
        if context == "defined_tcell_r1":
            ax.text(0.98, 0.06, "4/4 caps > 1\nHolm significant", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.1, color=PALETTE["blue"], fontweight="bold")
        elif context == "defined_tcell_r2":
            ax.text(0.98, 0.06, "Only cap 4\nHolm significant", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.1, color=PALETTE["coral"], fontweight="bold")
        elif context == "defined_hepg2_r1":
            ax.text(0.98, 0.06, "Only cap 2\nHolm significant", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.1, color=PALETTE["subink"], fontweight="bold")
        else:
            ax.text(0.98, 0.06, "No Holm-significant\nequal-budget effect", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.1, color=PALETTE["subink"], fontweight="bold")
    box(base, 0.035, 0.066, 0.930, 0.078, facecolor=PALETTE["soft"], edgecolor=PALETTE["soft"])
    base.text(0.055, 0.116, "RANDOMIZATION CONTROL", fontsize=8.0, fontweight="bold", color=PALETTE["coral"])
    base.text(0.235, 0.116, "Random(N) − Random(1): 0/20 comparisons survive the separate all-20 Holm correction.",
              fontsize=10.0, fontweight="bold", color=PALETTE["ink"])
    base.text(0.055, 0.084,
              "The old BoN(N)−Random(1) gain is not merely a larger-N effect; nonzero Random contrasts are paired randomization realizations, not systematic budget effects.",
              fontsize=8.1, color=PALETTE["subink"])
    base.text(0.035, 0.026,
              "Evidence boundary: retrospective sensitivity on four previously inspected outer contexts; 16 pools × 5 seeds per context; 10,000 bootstrap replicates. Not a new blind test or universal proxy-ranking claim.",
              fontsize=7.4, color=PALETTE["subink"])
    save_figure(fig, "Figure14_v3_equal_budget_value")


def fig_sequence(primary, family):
    fig, base = canvas(14.0, 9.7)
    base.text(0.035, 0.962, "Sequence composition delimits—not causally explains—the T-cell benefit",
              fontsize=22, fontweight="bold", va="top")
    base.text(0.035, 0.927,
              "Paired BoN−Random(1) effects before and after GC / uAUG / longest-polyU / train-support adjustment; negative favors BoN.",
              fontsize=11.0, color=PALETTE["subink"], va="top")
    base.plot([0.035, 0.965], [0.895, 0.895], color=PALETTE["line"], lw=1.2)
    base.text(0.045, 0.868, "A  FOUR-COVARIATE OLS ADJUSTMENT · ZERO FEATURE CONTRAST",
              fontsize=9.2, fontweight="bold", color=PALETTE["coral"])
    handles_a = [
        plt.Line2D([], [], color=PALETTE["blue"], marker="o", lw=1.8, label="Unadjusted"),
        plt.Line2D([], [], color=PALETTE["coral"], marker="s", lw=1.8, ls="--", label="Adjusted"),
    ]
    base.legend(handles=handles_a, loc="upper left", bbox_to_anchor=(0.735, 0.880), ncol=2, fontsize=8.3)
    contexts = ["defined_tcell_r1", "defined_tcell_r2", "defined_hepg2_r1"]
    xs = [0.070, 0.385, 0.700]
    for context, x in zip(contexts, xs):
        ax = fig.add_axes([x, 0.565, 0.255, 0.235])
        rows = primary[(primary.context == context) & (primary.cap > 1)].sort_values("cap")
        effect_curve(ax, rows, "unadjusted_effect", "unadjusted_ci_low", "unadjusted_ci_high", "blue", size=31)
        effect_curve(ax, rows, "adjusted_effect", "adjusted_ci_low", "adjusted_ci_high", "coral",
                     linestyle="--", marker="s", size=29)
        ax.axhline(0, color=PALETTE["line"], lw=0.9, zorder=0)
        ax.set_xticks(np.arange(4), DISPLAY_CAPS)
        ax.set_xlim(-0.3, 3.3)
        ax.set_ylim(-2.40, 1.85)
        ax.set_title(TITLES[context], loc="left", fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xlabel("Proposal cap N", fontsize=8.2, labelpad=3)
        ax.grid(axis="y", color=PALETTE["line"], lw=0.55, alpha=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.7)
        if x == xs[0]:
            ax.set_ylabel("Δ measured-MRL regret", fontsize=8.3)
    base.text(0.070, 0.528,
              "Adjusted effects attenuate in both T-cell replicates; 0/20 adjusted comparisons survive Holm. T-cell r2 cap 4 reverses sign; HepG2 cap 4 becomes positive (+0.879; Holm p=0.524).",
              fontsize=8.0, color=PALETTE["subink"])
    base.text(0.045, 0.480, "B  COMPLETE-PAIR EXCLUSION SENSITIVITY · UNADJUSTED EFFECTS",
              fontsize=9.2, fontweight="bold", color=PALETTE["violet"])
    handles_b = [
        plt.Line2D([], [], color=PALETTE["violet"], marker="o", lw=1.7, label="Exclude uAUG-positive pairs"),
        plt.Line2D([], [], color=PALETTE["teal"], marker="s", lw=1.7, ls="--", label="Exclude polyU upper quartile (≥3)"),
        plt.Line2D([], [], color=PALETTE["teal"], marker="*", lw=0, markersize=8, label="Family-specific Holm p < 0.05"),
    ]
    base.legend(handles=handles_b, loc="upper left", bbox_to_anchor=(0.405, 0.494), ncol=3, fontsize=7.8,
                handletextpad=0.4, columnspacing=1.0)
    for context, x in zip(contexts, xs):
        ax = fig.add_axes([x, 0.190, 0.255, 0.220])
        for fam, color, marker, ls in (("no_uAUG", "violet", "o", "-"),
                                       ("no_polyU_upper_quartile", "teal", "s", "--")):
            rows = family[(family.context == context) & (family.family == fam) & (family.cap > 1)].sort_values("cap")
            effect_curve(ax, rows, "effect", "ci_low", "ci_high", color, linestyle=ls, marker=marker, size=29)
            for xi, (_, row) in enumerate(rows.iterrows()):
                if row.holm_p < 0.05:
                    ax.scatter(xi, row.effect - 0.34, marker="*", s=64, color=PALETTE[color], zorder=5)
        ax.axhline(0, color=PALETTE["line"], lw=0.9, zorder=0)
        ax.set_xticks(np.arange(4), DISPLAY_CAPS)
        ax.set_xlim(-0.3, 3.3)
        ax.set_ylim(-4.20, 2.90)
        ax.set_title(TITLES[context], loc="left", fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xlabel("Proposal cap N", fontsize=8.2, labelpad=3)
        ax.grid(axis="y", color=PALETTE["line"], lw=0.55, alpha=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.7)
        if x == xs[0]:
            ax.set_ylabel("Δ measured-MRL regret", fontsize=8.3)
    base.text(0.070, 0.155,
              "uAUG exclusion: 0/20 Holm-significant. polyU exclusion: only T-cell r1 cap 4 and r2 cap 8 survive; r1 cap 8 is borderline (Holm p=0.061), and both cap 16 estimates remain non-significant after correction.",
              fontsize=7.9, color=PALETTE["subink"])
    box(base, 0.035, 0.060, 0.930, 0.062, facecolor=PALETTE["coral_light"], edgecolor=PALETTE["coral_light"])
    base.text(0.055, 0.099, "SENSITIVITY ≠ MEDIATION", fontsize=8.2, fontweight="bold", color=PALETTE["coral"])
    base.text(0.235, 0.099,
              "Pairs were removed without reranking or replacement; retained pairs/pools vary and intervals widen. These results show composition sensitivity, not causal mediation.",
              fontsize=8.6, fontweight="bold", color=PALETTE["ink"])
    base.text(0.035, 0.025,
              "Retrospective/exploratory analysis of previously inspected outer contexts; no new blind test. Adjustment uses 16 pool indicators and four selected-sequence contrasts; all CIs are nominal 95% pool-cluster bootstrap intervals (10,000 replicates), with cap 1 retained in each all-20 Holm family.",
              fontsize=7.25, color=PALETTE["subink"])
    save_figure(fig, "Figure15_v3_sequence_composition_sensitivity")


def main():
    apply_style()
    budget, sequence, equal, random, primary, family = load_evidence()
    fig_budget(equal, random)
    fig_sequence(primary, family)
    paths = [BUDGET / "results.json", BUDGET / "pool_effects.tsv", BUDGET / "family_effects.tsv",
             SEQUENCE / "summary.json", SEQUENCE / "primary_paired.tsv", SEQUENCE / "pool_effects.tsv",
             SEQUENCE / "sequence_family_sensitivity.tsv"]
    write_json(OUT / "v3_figures_14_15_build_config.json", {
        "version": "v3-figure14-15",
        "date": "2026-11-26",
        "runtime": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__},
        "figures": ["Figure14_v3_equal_budget_value", "Figure15_v3_sequence_composition_sensitivity"],
        "exports": {"png_dpi": 300, "pdf": "vector", "svg": "editable text"},
        "execution": "CPU rendering and frozen-result statistics only; no training, inference, or new outer analysis",
        "evidence_boundary": "retrospective sensitivity on previously inspected historical outer contexts; not a new blind test; sequence adjustment and exclusions are not causal analyses",
        "key_results": {
            "tcell_r1_equal_budget_holm_caps": [2, 4, 8, 16],
            "tcell_r2_equal_budget_holm_caps": [4],
            "random_n_minus_random1_holm_significant": 0,
            "adjusted_holm_significant": 0,
            "no_uaug_holm_significant": 0,
            "no_polyu_holm_significant": [["defined_tcell_r1", 4], ["defined_tcell_r2", 8]],
        },
        "sources": [source_record(p) for p in paths],
    })
    print("saved Figure14–15 v3 in", OUT)


if __name__ == "__main__":
    main()
