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
        raise RuntimeError(f"Configured V3/V4 renderer not found: {CANONICAL_PYTHON}")
    os.execv(str(CANONICAL_PYTHON), [str(CANONICAL_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_common_v3 import (
    DATA, HERE, OUT, PALETTE, REPORTS, SOURCE, STATUS, apply_style, box, label,
    save_figure, source_record, write_json,
)

PRESSURE = DATA / "strict_h1_figure2_paired_pressure.csv"
COMMON = DATA / "strict_h1_figure4_cross_architecture_summary.csv"
SCORE = DATA / "strict_h1_figure3_score_matching.csv"
RISK_FROZEN = SOURCE / "paper2_risk_frozen_v1" / "frozen_protocol.json"
PAPER3 = SOURCE / "paper3_headroom_v4" / "headroom_v4_round1" / "summary.json"
P3CURVE = REPORTS / "paper3_headroom_v4_20261124.md"
SCHLUSSER = SOURCE / "schlusser2024_hepg2_te_confirmatory_v1" / "evaluator_round_1" / "summary.json"
LEPPEK = SOURCE / "leppek2022_cds_mechanism_audit_v5" / "rank_summary.tsv"
H1MANIFEST = SOURCE / "paper1_strict_h1_v3" / "pools" / "build_manifest.json"
SCHLUSSER_ROLE = SOURCE / "schlusser2024_gate0" / "role_manifest.json"
AUDIT = REPORTS / "docs" / "FULL_AUDIT_20261124.md"
PERSIST = SOURCE / "persist_mrl_external_v1"
PERSIST_SUMMARY = PERSIST / "results" / "summary.json"
PERSIST_PREDICTIONS = PERSIST / "predictions" / "predictions.tsv"
PERSIST_LABELS = PERSIST / "protocol" / "evaluator_labels.tsv"
PERSIST_SEEDS = PERSIST / "results" / "per_seed.tsv"
PERSIST_BOOTSTRAP = PERSIST / "results" / "bootstrap_replicates.npy"
PERSIST_AUDIT = PERSIST / "protocol" / "inclusion_audit.json"
PERSIST_RAW = PERSIST / "raw" / "GSE173083_Table_S1_-_Attributes_for_pooled_233_sequences.xlsx"


def evidence() -> dict:
    pressure = pd.read_csv(PRESSURE)
    common = pd.read_csv(COMMON)
    score = pd.read_csv(SCORE)
    risk_frozen = json.loads(RISK_FROZEN.read_text())
    p3 = json.loads(PAPER3.read_text())
    sch = json.loads(SCHLUSSER.read_text())
    lep = pd.read_csv(LEPPEK, sep="\t")
    h1 = json.loads(H1MANIFEST.read_text())
    role = json.loads(SCHLUSSER_ROLE.read_text())
    tcell = pressure[(pressure.context == "Defined T-cell r1") & (pressure.proposal_cap == 16)].iloc[0]
    calibration = risk_frozen["calibration"]
    rank = lep[(lep.source_family == "cds_24") & (lep.left_endpoint == "cds_in_solution_half_life") & (lep.right_endpoint == "nluc_fluc_6h")].iloc[0]
    block = P3CURVE.read_text().split("### kmer4：固定 5 池")[1].split("###")[0]
    rows = []
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0].isdigit():
            rows.append([float(v) for v in cells])
    curves = pd.DataFrame(rows, columns=["budget", "random", "maxvar", "diversity", "ei", "topproxy", "regret"])
    ei16 = curves[curves.budget == 16].iloc[0]
    gates = pd.DataFrame(p3["cases"]).groupby("representation").gate_pass.sum().astype(int).to_dict()
    assert len(pressure) == 20 and len(common) == 4 and len(score) == 4
    assert np.allclose(pressure.bon_mean_regret - pressure.random_mean_regret, pressure.paired_bon_minus_random)
    assert ((score.ci_low < 0) & (score.ci_high > 0)).all()
    assert h1["pass"] and all(c["pass"] for c in h1["contexts"])
    assert not h1["old_directories_overwritten"]
    assert p3["case_count"] == 48 and not p3["outer_labels_read"] and not p3["outer_metrics_computed"]
    assert role["frozen_outer"]["frozen_before_label_inspection"] and not role["labels_returned_to_learner"]
    assert rank.discordant_pairs + rank.concordant_pairs == rank.n_unordered_pairs
    assert risk_frozen["threshold_locked"] and not risk_frozen["calibration_gate"]["pass"]
    assert not risk_frozen["outer_scoring_authorized"] and not risk_frozen["outer_scored"]
    persist = json.loads(PERSIST_SUMMARY.read_text())
    persist_pred = pd.read_csv(PERSIST_PREDICTIONS, sep="\t")
    persist_labels = pd.read_csv(PERSIST_LABELS, sep="\t")
    persist_seeds = pd.read_csv(PERSIST_SEEDS, sep="\t")
    persist_boot = np.load(PERSIST_BOOTSTRAP)
    persist_audit = json.loads(PERSIST_AUDIT.read_text())
    joined = persist_pred.merge(persist_labels, on=["sequence_id", "utr"], validate="one_to_one")
    assert len(joined) == persist["constructs"] == 59
    assert joined.utr.nunique() == persist["utr_clusters"] == 49
    assert len(persist_seeds) == persist["algorithm_seeds"] == 200
    assert persist_boot.shape == (10000, 8)
    assert np.isclose(joined.prediction.rank().corr(joined.mrl.rank()), persist["secondary"]["spearman"]["value"])
    assert np.isclose(persist_seeds.delta.mean(), persist["primary"]["bon_minus_random_regret"])
    assert np.allclose(persist_seeds.bon_regret - persist_seeds.random_regret, persist_seeds.delta)
    assert persist["primary"]["ci95"][0] < 0 < persist["primary"]["ci95"][1]
    assert not persist["primary"]["significant_benefit"] and persist["negative_results_retained"]
    assert persist_audit["conflicts"] == {} and persist_audit["excluded_constructs"] == 0
    return {
        "tcell_cap16": float(tcell.paired_bon_minus_random),
        "tcell_ci": [float(tcell.ci_low), float(tcell.ci_high)],
        "correlation_min": float(common.architecture_error_correlation.min()),
        "correlation_max": float(common.architecture_error_correlation.max()),
        "schlusser_rho": float(sch["endpoint_alignment_descriptive"]["spearman_cnn_MRL_vs_TE"]),
        "leppek_reversal": float(rank.discordant_fraction_untied),
        "leppek_pairs": [int(rank.discordant_pairs), int(rank.n_unordered_pairs)],
        "risk_coverage": float(calibration["coverage"]),
        "risk_accepted": f"{calibration['accepted_decisions']}/{calibration['decisions']}",
        "risk_pools": f"{calibration['accepted_pools']}/{calibration['pools']}",
        "risk_reduction": float(calibration["regret_reduction"]),
        "risk_ci": [float(v) for v in calibration["regret_reduction_95ci"]],
        "risk_threshold": float(calibration["threshold"]),
        "risk_gate_pass": risk_frozen["calibration_gate"]["pass"],
        "risk_outer_scored": risk_frozen["outer_scored"],
        "ei16": float(ei16.ei),
        "random16": float(ei16.random),
        "gates": gates,
        "h2_cross_sample": [c["h2_sensitivity_diagnostic"]["cross_sample_pairs"] for c in h1["contexts"]],
        "strict_pools_per_context": sorted(pressure.outer_pools.unique().astype(int).tolist()),
        "strict_candidates_per_context": sorted(common.candidate_count.unique().astype(int).tolist()),
        "paper3_outer_labels_read": p3["outer_labels_read"],
        "persist": {
            "constructs": persist["constructs"],
            "utr_clusters": persist["utr_clusters"],
            "singleton_utrs": int((persist_pred.groupby("utr").size() == 1).sum()),
            "multi_construct_utrs": int((persist_pred.groupby("utr").size() > 1).sum()),
            "algorithm_seeds": persist["algorithm_seeds"],
            "spearman": persist["secondary"]["spearman"],
            "random_top1_regret": persist["secondary"]["random_top1_regret"],
            "bon_top1_regret": persist["secondary"]["bon_top1_regret"],
            "primary": persist["primary"],
            "exact_h1_conflicts": len(persist_audit["conflicts"]),
            "retrospective_nonblind": True,
        },
    }


def canvas(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def fig_evidence_chain(e):
    fig, ax = canvas(14.0, 7.3)
    ax.text(0.035, 0.945, "A proxy is a contract with an operating domain", fontsize=22,
            fontweight="bold", va="top", color=PALETTE["ink"])
    ax.text(0.035, 0.892, "Observed benefit  →  shared-error mechanism  →  constrained risk rule  →  targeted label repair",
            fontsize=11.5, va="top", color=PALETTE["subink"])
    ax.plot([0.035, 0.965], [0.855, 0.855], color=PALETTE["line"], lw=1.2, transform=ax.transAxes)
    cards = [
        {
            "n": "01  OBSERVE", "title": "Benefit is\ncontext-dependent", "color": "blue", "light": "blue_light",
            "metric": f"Δ regret {e['tcell_cap16']:.3f}",
            "unit": "T-cell replicate 1 · cap 16\nnegative favors BoN pressure",
            "support": "Replicated direction in T-cell r2;\nN50 and HepG2 delimit the boundary.",
            "status": "RETROSPECTIVE · STRICT-H1", "status_kind": "retrospective",
        },
        {
            "n": "02  ATTRIBUTE", "title": "Errors are shared;\nendpoints are not\ninterchangeable", "color": "teal", "light": "teal_light", "title_size": 12.8,
            "metric": f"r {e['correlation_min']:.3f}–{e['correlation_max']:.3f}",
            "unit": "CNN vs 4-mer absolute error\nacross four strict-H1 contexts",
            "support": f"MRL–TE ρ {e['schlusser_rho']:.3f} · confirmatory\nRank reversal {e['leppek_reversal']*100:.1f}% · exploratory",
            "status": "MIXED EVIDENCE", "status_kind": "confirmatory",
        },
        {
            "n": "03  CONSTRAIN", "title": "Frozen risk rule\nfails calibration", "color": "coral", "light": "coral_light",
            "metric": f"coverage {e['risk_coverage']:.3f}",
            "unit": f"{e['risk_accepted']} decisions · {e['risk_pools']} pools\nΔ regret reduction {e['risk_reduction']:.6f}".replace("-", "−"),
            "support": f"95% CI [{e['risk_ci'][0]:.6f}, {e['risk_ci'][1]:+.6f}] crosses 0.\nGate failed; outer scoring not authorized.".replace("-", "−"),
            "status": "FROZEN CALIBRATION · GATE FAILED", "status_kind": "gap",
        },
        {
            "n": "04  REPAIR", "title": "Spend labels only\nwhere repair has headroom", "color": "violet", "light": "violet_light",
            "metric": f"EI {e['ei16']:.3f}  |  random {e['random16']:.3f}",
            "unit": "pooled regret at 16 labels\nfixed five-pool kmer4 cohort",
            "support": f"Positive gates: kmer4 {e['gates']['kmer4']}/16; summary4 {e['gates']['summary4']}/16.\nNo universal acquisition superiority claimed.",
            "status": "DIAGNOSTIC · OUTER UNTOUCHED", "status_kind": "diagnostic",
        },
    ]
    xs = [0.035, 0.2775, 0.520, 0.7625]
    for i, (x, c) in enumerate(zip(xs, cards)):
        box(ax, x, 0.275, 0.2025, 0.525, facecolor=PALETTE["paper"], edgecolor=PALETTE["line"], lw=1.1)
        box(ax, x, 0.715, 0.2025, 0.085, facecolor=PALETTE[c["light"]], radius=0.022)
        ax.text(x + 0.016, 0.757, c["n"], fontsize=8.2, fontweight="bold", color=PALETTE[c["color"]], va="center")
        ax.text(x + 0.016, 0.670, c["title"], fontsize=c.get("title_size", 13.5), fontweight="bold", va="top", linespacing=1.20)
        ax.text(x + 0.016, 0.535, c["metric"], fontsize=16.2 if i != 3 else 12.7,
                fontweight="bold", color=PALETTE[c["color"]], va="center")
        ax.text(x + 0.016, 0.480, c["unit"], fontsize=8.4, color=PALETTE["subink"], va="top", linespacing=1.35)
        ax.plot([x + 0.016, x + 0.1865], [0.413, 0.413], color=PALETTE["line"], lw=0.8)
        ax.text(x + 0.016, 0.382, c["support"], fontsize=8.4, color=PALETTE["ink"], va="top", linespacing=1.45)
        sc, sf = STATUS[c["status_kind"]]
        label(ax, x + 0.016, 0.307, c["status"], sc, sf, size=6.7, width=0.170)
        if i < 3:
            ax.annotate("", xy=(x + 0.239, 0.545), xytext=(x + 0.207, 0.545), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", lw=1.5, color=PALETTE["muted"], shrinkA=0, shrinkB=0))
    box(ax, 0.035, 0.085, 0.930, 0.125, facecolor=PALETTE["soft"], edgecolor=PALETTE["soft"])
    ax.text(0.055, 0.168, "CONSTRAINED DECISION RULE", fontsize=8.3, fontweight="bold", color=PALETTE["coral"], va="center")
    ax.text(0.055, 0.127, "Specify the context and endpoint; do not deploy ensemble-std abstention after the failed gate; acquire labels only where repair has headroom.",
            fontsize=10.7, fontweight="bold", va="center")
    ax.text(0.035, 0.035,
            "Evidence boundary: pressure, mechanism and repair are retrospective. Frozen calibration is a held-out negative gate; outer was not scored. Schlusser TE is confirmatory; Leppek CDS is exploratory. Lower regret is better.",
            fontsize=7.5, color=PALETTE["subink"], va="center")
    save_figure(fig, "Figure11_v3_evidence_chain")


def fig_governance(e):
    fig, ax = canvas(14.0, 8.1)
    ax.text(0.035, 0.955, "Data governance makes the OOD claim auditable", fontsize=22, fontweight="bold", va="top")
    ax.text(0.035, 0.910, "Frozen roles and sequence isolation define what the evidence can—and cannot—support.",
            fontsize=11.5, color=PALETTE["subink"], va="top")
    ax.plot([0.035, 0.965], [0.875, 0.875], color=PALETTE["line"], lw=1.2)
    ax.text(0.045, 0.835, "A  GOVERNANCE CHAIN", fontsize=9.2, fontweight="bold", color=PALETTE["blue"])
    ax.text(0.535, 0.835, "B  OOD COVERAGE", fontsize=9.2, fontweight="bold", color=PALETTE["teal"])
    nodes = [
        ("Frozen study roles", "train / dev / calibration / outer\nSchlusser allocated at study level", "blue", "blue_light"),
        ("Sequence isolation", f"4 strict-H1 contexts · {e['strict_pools_per_context'][0]} pools × 64 sequences\nexact + Hamming-1 exclusion gates passed", "teal", "teal_light"),
        ("Sensitivity, not selection", "H2 cross-Sample pairs: " + " / ".join(map(str, e["h2_cross_sample"])) + "\npredeclared diagnostic; never a pass gate", "gold", "gold_light"),
        ("Learner–evaluator firewall", "selection frozen before labels are read\nSchlusser labels not returned to learner", "violet", "violet_light"),
        ("Locked evaluation", "Schlusser one-shot confirmatory unblinding\nPaper 3 outer labels unread; metrics uncomputed", "coral", "coral_light"),
    ]
    ys = [0.710, 0.575, 0.440, 0.305, 0.170]
    for i, ((title, body, color, light), y) in enumerate(zip(nodes, ys)):
        box(ax, 0.045, y, 0.420, 0.105, facecolor=PALETTE[light], edgecolor=PALETTE[light], radius=0.020)
        box(ax, 0.045, y, 0.010, 0.105, facecolor=PALETTE[color], radius=0.005)
        ax.text(0.072, y + 0.074, title, fontsize=11.2, fontweight="bold", color=PALETTE[color], va="center")
        ax.text(0.072, y + 0.032, body, fontsize=8.7, color=PALETTE["ink"], va="center", linespacing=1.35)
        if i < 4:
            ax.plot([0.255, 0.255], [y - 0.023, y - 0.005], color=PALETTE["muted"], lw=1.0, transform=ax.transAxes)
            ax.scatter([0.255], [y - 0.023], s=10, color=PALETTE["muted"], transform=ax.transAxes, zorder=4)
    ax.text(0.545, 0.782, "Dimension", fontsize=8, fontweight="bold", color=PALETTE["subink"])
    ax.text(0.742, 0.782, "Evidence", fontsize=8, fontweight="bold", color=PALETTE["subink"])
    ax.text(0.906, 0.782, "Status", fontsize=8, fontweight="bold", color=PALETTE["subink"])
    rows = [
        ("Cross-cell", "HEK293T · HepG2 · T-cell×2\nsame-study external contexts", "COVERED", "retrospective"),
        ("Cross-construct", "N50 / defined-end\nLeppek pooled / CDS", "COVERED", "retrospective"),
        ("Cross-endpoint", "MRL → endogenous TE · confirmatory\nhalf-life → expression · exploratory", "STRONGEST", "confirmatory"),
        ("Cross-study, same MRL", "PERSIST: 59 constructs / 49 UTRs\nretrospective; no BoN benefit", "NO BENEFIT", "gap"),
    ]
    for i, (dim, body, status, kind) in enumerate(rows):
        y = 0.660 - i * 0.145
        box(ax, 0.535, y, 0.430, 0.118, facecolor=PALETTE["paper"], edgecolor=PALETTE["line"], radius=0.016)
        ax.text(0.550, y + 0.083, dim, fontsize=10.3, fontweight="bold", color=PALETTE["ink"], va="center")
        ax.text(0.742, y + 0.058, body, fontsize=8.2, color=PALETTE["subink"], va="center", linespacing=1.35)
        color, light = STATUS[kind]
        label(ax, 0.876, y + 0.060, status, color, light, size=7.0, width=0.076)
    box(ax, 0.535, 0.092, 0.430, 0.072, facecolor=PALETTE["coral_light"], edgecolor=PALETTE["coral_light"], radius=0.016)
    ax.text(0.550, 0.128, "CLAIM BOUNDARY", fontsize=7.8, fontweight="bold", color=PALETTE["coral"], va="center")
    ax.text(0.655, 0.128, "PERSIST same-MRL transfer: negative, retrospective, non-blind.",
            fontsize=8.6, fontweight="bold", color=PALETTE["ink"], va="center")
    ax.text(0.035, 0.040,
            "Component overlap is not claimed to be absent for Leppek. PERSIST is an independent-study, same-endpoint retrospective evaluation—not a blind test. Governance supports provenance and isolation, not automatic biological generalization.",
            fontsize=7.5, color=PALETTE["subink"], va="center")
    save_figure(fig, "Figure12_v3_data_governance_ood")


def fig_persist(e):
    p = e["persist"]
    pred = pd.read_csv(PERSIST_PREDICTIONS, sep="\t")
    labels = pd.read_csv(PERSIST_LABELS, sep="\t")
    seeds = pd.read_csv(PERSIST_SEEDS, sep="\t")
    d = pred.merge(labels, on=["sequence_id", "utr"], validate="one_to_one")
    sizes = d.groupby("utr").size()
    d["multi"] = d.utr.map(sizes) > 1

    fig, canvas_ax = canvas(14.0, 7.8)
    canvas_ax.text(0.035, 0.955, "Internal gains do not guarantee cross-study transfer",
                   fontsize=22, fontweight="bold", va="top")
    canvas_ax.text(0.035, 0.913,
                   "PERSIST-seq · independent study, same MRL endpoint · frozen 4-mer MLP on native variable-length 5′UTRs",
                   fontsize=11.2, color=PALETTE["subink"], va="top")
    canvas_ax.plot([0.035, 0.965], [0.875, 0.875], color=PALETTE["line"], lw=1.2)
    canvas_ax.text(0.045, 0.838, "A  NATIVE-UTR PREDICTION VS MEASURED MRL",
                   fontsize=9.2, fontweight="bold", color=PALETTE["blue"])
    canvas_ax.text(0.380, 0.838, "B  PAIRED TOP-1 REGRET AND PROPOSAL VARIATION",
                   fontsize=9.2, fontweight="bold", color=PALETTE["coral"])
    canvas_ax.text(0.745, 0.838, "C  DESIGN & EVIDENCE BOUNDARY",
                   fontsize=9.2, fontweight="bold", color=PALETTE["violet"])

    ax = fig.add_axes([0.060, 0.260, 0.270, 0.500])
    for utr, group in d[d.multi].groupby("utr"):
        order = group.sort_values("prediction")
        ax.plot(order.prediction, order.mrl, color=PALETTE["coral"], lw=0.8,
                alpha=0.45, zorder=1)
    for multi, color, marker, size in ((False, "blue", "o", 31), (True, "coral", "s", 38)):
        group = d[d.multi == multi]
        ax.scatter(group.prediction, group.mrl, s=size, c=PALETTE[color], marker=marker,
                   edgecolors="white", linewidths=0.55, alpha=0.88, zorder=3)
    ax.set_xlabel("Frozen 4-mer MLP prediction (native full UTR)", labelpad=7)
    ax.set_ylabel("Measured MRL (ribosome load)", labelpad=7)
    ax.set_xlim(3.35, 8.30); ax.set_ylim(1.17, 2.20)
    ax.grid(axis="y", color=PALETTE["line"], lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.text(0.035, 0.965,
            f"Spearman ρ = {p['spearman']['value']:+.6f}\n95% UTR-cluster CI [{p['spearman']['ci95'][0]:+.6f}, {p['spearman']['ci95'][1]:+.6f}]".replace("-", "−"),
            transform=ax.transAxes, va="top", fontsize=8.4, color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=0.40", facecolor="white", edgecolor=PALETTE["line"], alpha=0.95))
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=5.5, markerfacecolor=PALETTE["blue"],
                   markeredgecolor="white", label=f"Single-construct UTRs: {p['singleton_utrs']}"),
        plt.Line2D([], [], marker="s", linestyle="-", markersize=5.5, color=PALETTE["coral"],
                   markerfacecolor=PALETTE["coral"], markeredgecolor="white", label=f"Multi-construct UTRs: {p['multi_construct_utrs']} (linked)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7.4,
              handletextpad=0.4, borderaxespad=0.2)
    ax.text(0.0, -0.205, "Each point is one construct (n=59); linked points share a UTR.",
            transform=ax.transAxes, fontsize=7.4, color=PALETTE["subink"])

    forest = fig.add_axes([0.405, 0.470, 0.290, 0.285])
    rows = [
        ("Random", p["random_top1_regret"], "blue"),
        ("BoN · cap16", p["bon_top1_regret"], "gold"),
        ("BoN − Random", {"value": p["primary"]["bon_minus_random_regret"], "ci95": p["primary"]["ci95"]}, "coral"),
    ]
    for y, (name, row, color) in zip([2, 1, 0], rows):
        forest.plot(row["ci95"], [y, y], color=PALETTE[color], lw=2.2, solid_capstyle="round")
        forest.scatter(row["value"], y, s=62, color=PALETTE[color], edgecolor="white", linewidth=0.8, zorder=4)
        forest.text(0.885, y, f"{row['value']:+.6f}" if y == 0 else f"{row['value']:.6f}",
                    va="center", ha="right", fontsize=8.0, color=PALETTE[color], fontweight="bold")
    forest.axvline(0, color=PALETTE["muted"], lw=0.9, ls="--", zorder=0)
    forest.set_yticks([2, 1, 0], [r[0] for r in rows])
    forest.set_xlim(-0.20, 0.90); forest.set_ylim(-0.55, 2.55)
    forest.set_xlabel("Measured-MRL regret · lower is better", fontsize=8.2, labelpad=5)
    forest.tick_params(axis="y", length=0, labelsize=8.5)
    forest.grid(axis="x", color=PALETTE["line"], lw=0.6, alpha=0.75)
    forest.set_axisbelow(True)
    forest.text(0.0, -0.37,
                f"Paired effect 95% UTR-cluster CI [{p['primary']['ci95'][0]:+.6f}, {p['primary']['ci95'][1]:+.6f}]".replace("-", "−"),
                transform=forest.transAxes, fontsize=7.8, color=PALETTE["coral"], fontweight="bold")

    dist = fig.add_axes([0.405, 0.225, 0.290, 0.125])
    dist.hist(seeds.delta, bins=24, color=PALETTE["blue_light"], edgecolor=PALETTE["blue"], lw=0.55)
    dist.axvline(0, color=PALETTE["muted"], lw=0.9, ls="--")
    dist.axvline(seeds.delta.mean(), color=PALETTE["coral"], lw=1.7)
    dist.set_yticks([]); dist.set_xlim(-0.9, 0.9)
    dist.set_xlabel("Same-seed BoN − Random regret (200 repeats)", fontsize=7.8, labelpad=4)
    dist.tick_params(axis="x", labelsize=7.2)
    dist.text(0.0, -0.70,
              f"Sign p={p['primary']['p_two_sided_sign']:.6f}; negative favors BoN.\nSeeds are repeated proposals, not independent biological samples.",
              transform=dist.transAxes, fontsize=7.5, color=PALETTE["subink"], va="top", linespacing=1.35)

    cards = [
        ("59 constructs · 49 UTR clusters", "39 singletons + 10 two-construct UTRs\nall native sequences retained", "blue", "blue_light"),
        ("200 algorithm seeds · cap16", "fixed 59-construct pool\ncluster bootstrap: 10,000 replicates", "gold", "gold_light"),
        ("exact / equal-length H1 = 0", "Sample train + dev + calibration\nno UTR cluster excluded", "teal", "teal_light"),
        ("RETROSPECTIVE · NON-BLIND", "independent study, same MRL endpoint\nhistorical label contact; not prospective", "violet", "violet_light"),
    ]
    for i, (head, body, color, light) in enumerate(cards):
        y = 0.666 - i * 0.145
        box(canvas_ax, 0.745, y, 0.220, 0.118, facecolor=PALETTE[light], edgecolor=PALETTE[light], radius=0.016)
        canvas_ax.text(0.760, y + 0.079, head, fontsize=9.3, fontweight="bold", color=PALETTE[color], va="center")
        canvas_ax.text(0.760, y + 0.034, body, fontsize=7.6, color=PALETTE["subink"], va="center", linespacing=1.35)

    box(canvas_ax, 0.035, 0.075, 0.930, 0.092, facecolor=PALETTE["coral_light"], edgecolor=PALETTE["coral_light"])
    canvas_ax.text(0.055, 0.134, "INTERNAL GAIN ≠ CROSS-STUDY TRANSFER", fontsize=9.0,
                   fontweight="bold", color=PALETTE["coral"], va="center")
    canvas_ax.text(0.055, 0.098,
                   "Point estimate favors Random; the UTR-cluster CI crosses 0. No BoN benefit or general failure is claimed.",
                   fontsize=10.0, fontweight="bold", color=PALETTE["ink"], va="center")
    canvas_ax.text(0.035, 0.032,
                   "Evidence boundary: one independent study and a small 5′UTR-only cohort; frozen model, no truncation, padding, recalibration or new training. Non-significance does not prove no effect.",
                   fontsize=7.4, color=PALETTE["subink"], va="center")
    save_figure(fig, "Figure13_v3_persist_external")


def main():
    apply_style()
    e = evidence()
    fig_evidence_chain(e)
    fig_governance(e)
    fig_persist(e)
    paths = [PRESSURE, COMMON, SCORE, RISK_FROZEN, PAPER3, P3CURVE, SCHLUSSER, LEPPEK, H1MANIFEST, SCHLUSSER_ROLE, AUDIT,
             PERSIST_SUMMARY, PERSIST_PREDICTIONS, PERSIST_LABELS, PERSIST_SEEDS, PERSIST_BOOTSTRAP, PERSIST_AUDIT, PERSIST_RAW]
    write_json(OUT / "v3_build_config.json", {
        "version": "v3",
        "date": "2026-11-25",
        "runtime": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "figures": ["Figure11_v3_evidence_chain", "Figure12_v3_data_governance_ood", "Figure13_v3_persist_external"],
        "exports": {"png_dpi": 300, "pdf": "vector", "svg": "editable text"},
        "execution": "CPU rendering and statistics only; no training or inference",
        "evidence": e,
        "status_policy": {
            "paper1": "retrospective strict-H1 protocol sensitivity; not a new blind test",
            "schlusser": "confirmatory one-shot unblinding",
            "leppek": "exploratory descriptive audit",
            "paper2_abstention": "frozen calibration gate failed; outer scoring unauthorized and not performed; no deployment claim",
            "paper3": "diagnostic development/calibration; outer untouched",
            "persist": "independent-study same-endpoint retrospective external evaluation; non-blind; no BoN benefit; UTR-cluster uncertainty; seeds are not biological replicates",
        },
        "sources": [source_record(p) for p in paths],
    })
    print(json.dumps(e, indent=2))
    print("saved Figure11–13 v3 in", OUT)


if __name__ == "__main__":
    main()
