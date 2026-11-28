#!/usr/bin/env python3
"""Build the V10 (v8) main figure set (Fig 1-6) from frozen-artifact data."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figure_style_v8 import C, FAMILY, STATUS_COLOR, STATUS_MARKER, apply_style, panel_label, clean, save

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "figures_v8_data.json").read_text())


def fig1():
    apply_style()
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.5], wspace=0.32, figure=fig)
    tb = DATA["tcell_equal_budget"]
    # a: cap16 dumbbell RandomN -> BoN
    ax = fig.add_subplot(gs[0, 0])
    labels = ["T-cell r1", "T-cell r2"]
    keys = ["defined_tcell_r1", "defined_tcell_r2"]
    for i, (lab, k) in enumerate(zip(labels, keys)):
        c16 = next(c for c in tb[k] if c["cap"] == 16)
        rnd, bon = c16["random_n_regret"], c16["bon_regret"]
        ax.plot([rnd, bon], [i, i], color=C["gray_light"], lw=2.5, zorder=2)
        ax.scatter([rnd], [i], s=42, color=C["gray"], zorder=3, label="RandomN" if i == 0 else None)
        ax.scatter([bon], [i], s=42, color=C["blue"], zorder=3, label="BoN (proxy)" if i == 0 else None)
        ax.text(rnd + 0.08, i + 0.16, f"{rnd:.2f}", fontsize=7, color=C["gray"])
        ax.text(bon - 0.08, i - 0.26, f"{bon:.2f}", fontsize=7, color=C["blue"], ha="right")
    ax.set_yticks(range(2)); ax.set_yticklabels(labels)
    ax.set_xlabel("cap16 functional regret (MRL)")
    ax.set_ylim(-0.6, 1.6); ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=7)
    clean(ax); panel_label(ax, "a")
    ax.set_title("Equal budget, lower regret", fontsize=8.5, loc="left", color=C["sub"])
    # b: forest cap2-16
    ax = fig.add_subplot(gs[0, 1])
    caps = [2, 4, 8, 16]
    y = np.arange(len(caps))
    for off, k, col, lab in [(-0.16, "defined_tcell_r1", C["blue"], "r1"),
                             (0.16, "defined_tcell_r2", C["blue2"], "r2")]:
        eff = [next(c for c in tb[k] if c["cap"] == cp)["effect"] for cp in caps]
        ci = [next(c for c in tb[k] if c["cap"] == cp)["ci"] for cp in caps]
        holm = [next(c for c in tb[k] if c["cap"] == cp)["holm_p"] for cp in caps]
        for j, cp in enumerate(caps):
            lo, hi = ci[j]
            ax.plot([lo, hi], [y[j] + off] * 2, color=col, lw=1.6, zorder=2)
            sig = holm[j] < 0.05
            ax.scatter([eff[j]], [y[j] + off], s=34, color=col if sig else "white",
                       edgecolor=col, linewidth=1.2, zorder=3)
        ax.plot([], [], color=col, lw=1.6, label=f"{lab} (filled = Holm<0.05)")
    ax.axvline(0, color=C["ink"], lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([f"cap {c}" for c in caps])
    ax.set_xlabel("BoN − RandomN regret (MRL); negative favours proxy")
    ax.invert_yaxis(); ax.legend(loc="lower left", fontsize=6.8)
    clean(ax); panel_label(ax, "b")
    save(fig, "Figure1_v8_equal_budget_value", HERE)


def fig2():
    apply_style()
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.2], wspace=0.34, figure=fig)
    bn = DATA["tcell_bottleneck"]
    keys = ["defined_tcell_r1", "defined_tcell_r2"]; labels = ["r1", "r2"]
    # a: global vs top16 spearman
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(2); w = 0.36
    g = [bn[k]["global_spearman"] for k in keys]; t = [bn[k]["top16_spearman"] for k in keys]
    ax.bar(x - w / 2, g, w, color=C["blue"], label="global Spearman")
    ax.bar(x + w / 2, t, w, color=C["red"], label="top-16 Spearman")
    for j in range(2):
        ax.text(x[j] - w / 2, g[j] + 0.01, f"{g[j]:.2f}", ha="center", fontsize=7)
        ax.text(x[j] + w / 2, t[j] + 0.01, f"{t[j]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Spearman ρ")
    ax.set_ylim(0, 0.72); ax.legend(fontsize=7, loc="upper right")
    clean(ax); panel_label(ax, "a")
    ax.set_title("Ranking collapses in the tail", fontsize=8.5, loc="left", color=C["sub"])
    # b: stacked decomposition
    ax = fig.add_subplot(gs[0, 1])
    opp = [bn[k]["opportunity_loss"] for k in keys]; wit = [bn[k]["within_tail_ranking_loss"] for k in keys]
    ax.bar(x, opp, 0.5, color=C["gray_light"], edgecolor=C["gray"], linewidth=0.6, label="opportunity loss")
    ax.bar(x, wit, 0.5, bottom=opp, color=C["red"], label="within-tail ranking loss")
    for j in range(2):
        tot = opp[j] + wit[j]
        ax.text(x[j], tot + 0.06, f"{tot:.2f}", ha="center", fontsize=7.5, fontweight="bold")
        ax.text(x[j], opp[j] + wit[j] / 2, f"{wit[j] / tot * 100:.0f}%", ha="center", va="center",
                fontsize=7, color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("cap16 regret decomposition (MRL)")
    ax.set_ylim(0, 4.6); ax.legend(fontsize=7, loc="upper right")
    clean(ax); panel_label(ax, "b")
    save(fig, "Figure2_v8_tail_bottleneck", HERE)


def fig3():
    apply_style()
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.3], wspace=0.34, figure=fig)
    ad = DATA["adaptation"]; keys = ["defined_tcell_r1", "defined_tcell_r2"]; labels = ["r1", "r2"]
    # a: MAE original vs adapted (grouped bars) -- regression improves
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(2); w = 0.36
    m0 = [ad["mae"][k]["original_mae"] for k in keys]; m1 = [ad["mae"][k]["adapted_mae"] for k in keys]
    ax.bar(x - w / 2, m0, w, color=C["gray_light"], edgecolor=C["gray"], linewidth=0.6, label="original CNN")
    ax.bar(x + w / 2, m1, w, color=C["green"], label="50k adapted")
    for j in range(2):
        ax.text(x[j] - w / 2, m0[j] + 0.02, f"{m0[j]:.2f}", ha="center", fontsize=7)
        ax.text(x[j] + w / 2, m1[j] + 0.02, f"{m1[j]:.2f}", ha="center", fontsize=7, color=C["green"])
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("MAE (MRL)")
    ax.set_ylim(0, 1.6); ax.legend(fontsize=7, loc="upper right")
    clean(ax); panel_label(ax, "a")
    ax.set_title("Regression improves (MAE ↓)", fontsize=8.5, loc="left", color=C["sub"])
    # b: forest of repair combined effects -- decision does not
    ax = fig.add_subplot(gs[0, 1])
    rows = [
        ("50k adaptation", ad["combined"]["effect"], [ad["combined"]["ci_low"], ad["combined"]["ci_high"]], "null"),
        ("ListNet vs CNN", DATA["listnet"]["original_cnn"]["effect"]["effect"],
         [DATA["listnet"]["original_cnn"]["effect"]["ci_low"], DATA["listnet"]["original_cnn"]["effect"]["ci_high"]], "null"),
        ("ListNet vs 50k", DATA["listnet"]["v1_main"]["effect"]["effect"],
         [DATA["listnet"]["v1_main"]["effect"]["ci_low"], DATA["listnet"]["v1_main"]["effect"]["ci_high"]], "null"),
        ("HepG2 anchor", DATA["anchor"]["combined"]["effect"],
         [DATA["anchor"]["combined"]["ci_low"], DATA["anchor"]["combined"]["ci_high"]], "harmful"),
    ]
    y = np.arange(len(rows))
    for j, (lab, eff, ci, st) in enumerate(rows):
        col = STATUS_COLOR[st]
        ax.plot(ci, [y[j]] * 2, color=col, lw=1.8, zorder=2)
        ax.scatter([eff], [y[j]], s=40, color=col, zorder=3)
        ax.text(ci[1] + 0.02, y[j], f"{eff:+.3f}", va="center", fontsize=7, color=col)
    ax.axvline(0, color=C["ink"], lw=0.8, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("combined cap16 Δregret (MRL); positive = worse")
    ax.set_xlim(-0.25, 0.78); clean(ax); panel_label(ax, "b")
    ax.set_title("Decision does not (regret ↔ / ↑)", fontsize=8.5, loc="left", color=C["sub"])
    save(fig, "Figure3_v8_regression_decision_dissociation", HERE)


def fig4():
    apply_style()
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = GridSpec(1, 3, width_ratios=[1.2, 0.9, 1.0], wspace=0.42, figure=fig)
    u = DATA["upts"]; sel = DATA["upts_selection"]
    dirs = ["n50_hek293t_r1_to_defined_hepg2_r1", "defined_hepg2_r1_to_n50_hek293t_r1"]
    short = ["N50→HepG2", "HepG2→N50"]
    # a: kappa dev curves
    ax = fig.add_subplot(gs[0, 0])
    for d, s, col in zip(dirs, short, [C["blue"], C["violet"]]):
        cand = sel[d]["candidates"]
        ks = [c["kappa"] for c in cand]; rs = [c["dev_cap16_mean_regret"] for c in cand]
        ax.plot(ks, rs, "-o", color=col, ms=3.5, label=s)
        ch = sel[d]["chosen"]
        ax.scatter([ch["kappa"]], [ch["dev_cap16_mean_regret"]], s=70, facecolor="white",
                   edgecolor=col, linewidth=1.6, zorder=4)
    ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("κ (penalty on proxy std)")
    ax.set_ylabel("auxiliary dev cap16 regret (MRL)"); ax.legend(fontsize=6.8)
    clean(ax); panel_label(ax, "a")
    # b: gate bars
    ax = fig.add_subplot(gs[0, 1])
    imp = [u["directions"][i]["test"]["cap16_regret_improvement"] for i in range(2)]
    cols = [C["green"] if u["directions"][i]["gate_pass"] else C["red"] for i in range(2)]
    ax.bar(range(2), imp, 0.55, color=cols)
    for i in range(2):
        ax.text(i, imp[i] + (0.002 if imp[i] >= 0 else -0.004), f"{imp[i]:+.3f}",
                ha="center", fontsize=7, color=cols[i])
        ax.text(i, -0.036, "PASS" if u["directions"][i]["gate_pass"] else "FAIL",
                ha="center", fontsize=7, fontweight="bold", color=cols[i])
    ax.axhline(0, color=C["ink"], lw=0.8)
    ax.set_xticks(range(2)); ax.set_xticklabels(short, fontsize=7)
    ax.set_ylabel("test cap16 improvement (MRL)"); ax.set_ylim(-0.045, 0.025)
    clean(ax); panel_label(ax, "b")
    # c: disagreement-error spearman
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(2); w = 0.36
    signed = [u["directions"][i]["diagnostics"]["spearman_std_vs_signed_error"] for i in range(2)]
    abserr = [u["directions"][i]["diagnostics"]["spearman_std_vs_abs_error"] for i in range(2)]
    ax.bar(x - w / 2, signed, w, color=C["blue"], label="vs signed error")
    ax.bar(x + w / 2, abserr, w, color=C["gray_light"], edgecolor=C["gray"], linewidth=0.6, label="vs |error|")
    ax.axhline(0, color=C["ink"], lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7); ax.set_ylabel("Spearman(std, error)")
    ax.set_ylim(0, 0.025); ax.legend(fontsize=6.5); clean(ax); panel_label(ax, "c")
    save(fig, "Figure4_v8_upts_falsification", HERE)


def fig5():
    apply_style()
    rows = DATA["tail_audit_v2"]
    groups = [
        ("measured MRL", ["T-cell MRL r1", "T-cell MRL r2", "HEK293T N50 MRL", "HepG2 defined-end MRL"], "MRL"),
        ("IgG-Fc lnW", ["GB1 binding"], "lnW"),
        ("log brightness", ["GFP brightness"], "log brightness"),
        ("Production", ["AAV9 production"], "Production enrichment"),
        ("relative resistance", ["TEM-1 Amp→Amp", "TEM-1 Amp→cefotaxime"], "relative resistance"),
        ("integrated abundance", ["mRNA stability"], "integrated abundance"),
    ]
    by = {r["scenario"]: r for r in rows}
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = GridSpec(2, 3, hspace=0.62, wspace=0.42, figure=fig)
    for idx, (title, names, unit) in enumerate(groups):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        y = np.arange(len(names))
        for j, nm in enumerate(names):
            r = by[nm]; col = STATUS_COLOR[r["status"]]
            eff = r["cap16_delta"]; ci = r["ci95"]
            short = nm.replace(" MRL", "").replace("defined-end ", "").replace("TEM-1 ", "")
            if ci is not None:
                ax.plot(ci, [y[j]] * 2, color=col, lw=1.7, zorder=2)
            ax.scatter([eff], [y[j]], s=34, color=col, marker=STATUS_MARKER[r["status"]],
                       edgecolor="white", linewidth=0.4, zorder=3)
        ax.axvline(0, color=C["ink"], lw=0.7, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([n.replace(" MRL", "").replace("defined-end ", "").replace("TEM-1 ", "") for n in names], fontsize=6.8)
        ax.invert_yaxis(); ax.set_xlabel(f"Δregret ({unit})", fontsize=7)
        ax.set_title(title, fontsize=8, loc="left", color=C["sub"])
        ax.tick_params(labelsize=6.5); clean(ax)
        panel_label(ax, chr(ord("a") + idx), dx=-0.22, dy=1.12)
    save(fig, "Figure5_v8_cross_scenario_forest", HERE)


def fig6():
    apply_style()
    rows = DATA["tail_audit_v2"]; assoc = DATA["association"]
    inc = [r for r in rows if r["relative_regret_reduction_pct"] is not None]
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.4, figure=fig)
    # a: global spearman vs reduction
    ax = fig.add_subplot(gs[0, 0])
    for r in inc:
        ax.scatter(r["global_spearman"], r["relative_regret_reduction_pct"],
                   s=40, color=FAMILY[r["family"]], marker=STATUS_MARKER[r["status"]],
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.axhline(0, color=C["gray"], lw=0.7, ls="--"); ax.axvline(0, color=C["gray"], lw=0.7, ls="--")
    ax.set_xlabel("global Spearman ρ"); ax.set_ylabel("relative regret reduction (%)")
    ax.text(0.02, 0.95, f"ρ = {assoc['rho_global_spearman']:.2f}", transform=ax.transAxes,
            fontsize=8, va="top", color=C["sub"])
    clean(ax); panel_label(ax, "a")
    # b: tail recall vs reduction
    ax = fig.add_subplot(gs[0, 1])
    xs = [r["tail_recall_decile"] for r in inc]; ys = [r["relative_regret_reduction_pct"] for r in inc]
    for r in inc:
        ax.scatter(r["tail_recall_decile"], r["relative_regret_reduction_pct"],
                   s=40, color=FAMILY[r["family"]], marker=STATUS_MARKER[r["status"]],
                   edgecolor="white", linewidth=0.4, zorder=3)
    z = np.polyfit(xs, ys, 1); xx = np.linspace(min(xs), max(xs), 50)
    ax.plot(xx, np.polyval(z, xx), color=C["blue"], lw=1.2, ls="-", zorder=2)
    ax.axhline(0, color=C["gray"], lw=0.7, ls="--")
    ax.set_xlabel("unified tail recall (top decile)"); ax.set_ylabel("relative regret reduction (%)")
    ax.text(0.02, 0.95, f"ρ = {assoc['rho_tail_recall']:.2f}\np = {assoc['permutation_p_tail']:.4f}",
            transform=ax.transAxes, fontsize=8, va="top", color=C["blue"])
    clean(ax); panel_label(ax, "b")
    # c: LOO robustness
    ax = fig.add_subplot(gs[0, 2])
    loo = assoc["leave_one_out"]
    vals = [l["rho_tail"] for l in loo]
    ax.barh(np.arange(len(vals)), vals, 0.6, color=C["blue2"])
    ax.axvline(assoc["rho_tail_recall"], color=C["blue"], lw=1.2, ls="--", label="full ρ")
    ax.set_yticks(np.arange(len(vals))); ax.set_yticklabels([l["dropped"].replace(" MRL", "").replace("TEM-1 ", "")[:14] for l in loo], fontsize=6)
    ax.set_xlabel("leave-one-out ρ"); ax.set_xlim(0, 1); ax.invert_yaxis()
    ax.legend(fontsize=6.5, loc="lower right"); clean(ax); panel_label(ax, "c")
    save(fig, "Figure6_v8_unified_tail_diagnostic", HERE)


def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    cfg = {"figures": [f"Figure{i}_v8" for i in range(1, 7)],
           "source": "figures_v8_data.json (frozen artifacts)", "retrospective": True,
           "style": "scientific-figure-making + plot-is-all-you-need house style"}
    (HERE / "v8_build_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("built Figure1-6 v8")


if __name__ == "__main__":
    main()