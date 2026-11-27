#!/usr/bin/env python3
"""Render publication figures from accepted local RARE-Seq results."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = PROJECT_ROOT.parent
FIGURE_ROOT = PROJECT_ROOT / "figures"
DATA_ROOT = FIGURE_ROOT / "data"
VERIFY_ROOT = FIGURE_ROOT / "verification"
AUDIT_HISTORY = FIGURE_ROOT / "audit_history"

E0 = STAGE_ROOT / "E0_数据审计" / "results"
E1 = PROJECT_ROOT / "E1_强基线"
E6 = PROJECT_ROOT / "E6_蛋白迁移"
E5_REPORT = PROJECT_ROOT / "E5_OOD" / "PAR-CLIP跨协议评测报告_2026-11-25.md"
E5_HOLDOUT = PROJECT_ROOT / "E5_OOD" / "results" / "holdout_convergence_final_summary"
E5_HOLDOUT_REPORT = PROJECT_ROOT / "E5_OOD" / "E5_整RBP留出三seed双轨最终报告_2026-11-26.md"
RNACOMPETE = DATA_ROOT / "rnacompete_ray2013_final"
RNACOMPETE_REPORT = E1 / "RNAcompete_Ray2013_RNAFM五折最终报告_2026-11-27.md"

RUNS = {
    "fig1": [
        "E0-leakage-quant-20261122-102238-ZDLg6Y",
        "E0-window-identity-20261122-104517-ZDLg6Y",
        "E1-main-independent-audit-20261126-032635-ZDLg6Y",
        "E1-main-independent-recalculation-20261126-105611-ZDLg6Y",
    ],
    "fig2": [
        "E1-main-independent-audit-20261126-032635-ZDLg6Y",
        "E1-cv-prediction-combine-20261126-064253-ZDLg6Y",
        "E1-cv-summary-20261126-065042-ZDLg6Y",
        "E1-paired-bootstrap-20261126-065044-ZDLg6Y",
        "E1-main-independent-recalculation-20261126-105611-ZDLg6Y",
    ],
    "fig3": [
        "E5-parclip-cross-protocol-contract-p20-20261125-065228-ZDLg6Y",
        "E5-parclip-cross-protocol-contract-p20-20261125-065619-ZDLg6Y",
        "E5-parclip-cross-protocol-contract-p20-20261125-065953-ZDLg6Y",
    ],
    "fig4": [
        "E6-grimm-frozen-baselines-20261124-085329-ZDLg6Y",
        "E6-grimm-frozen-baselines-20261124-070400-ZDLg6Y",
        "E6-grimm-frozen-baselines-20261124-104344-ZDLg6Y",
        "E6-grimm-frozen-baselines-20261124-123513-ZDLg6Y",
        "E6-grimm-frozen-baselines-20261124-171811-ZDLg6Y",
        "E6-ood-metric-recompute-v2-20261125-072246-ZDLg6Y",
        "E6-five-split-summary-20261125-080318-ZDLg6Y",
        "E6-five-split-independent-verification-20261125-080348-ZDLg6Y",
    ],
    "fig5": [
        "E5-holdout-convergence-summary-20261126-112101-ZDLg6Y",
        "E5-rbp-holdout-convergence-s0-neg1c-20261125-142458-ZDLg6Y",
        "E5-rbp-holdout-convergence-s0-neg2-20261125-142458-ZDLg6Y",
        "E5-rbp-holdout-convergence-s0-neg2-20261125-172933-ZDLg6Y",
        "E5-rbp-holdout-convergence-s1-neg1c-20261125-185026-ZDLg6Y",
        "E5-rbp-holdout-convergence-s1-neg2-20261125-205610-ZDLg6Y",
        "E5-rbp-holdout-convergence-s1-neg2-20261126-002502-ZDLg6Y",
        "E5-rbp-holdout-convergence-s2-neg1c-20261126-014523-ZDLg6Y",
        "E5-rbp-holdout-convergence-s2-neg2-20261126-064623-ZDLg6Y",
    ],
    "fig6": [
        "rnacompete-verify-final-all-all-f0-s0-2026-11-27T021412.509921Z0000-4099546",
        "rnacompete-summarize-all-all-f0-s0-2026-11-27T024426.072076Z0000-4101649",
    ],
}

ARCHIVED_FIG2_RUNS = [
    "E1-cv-prediction-combine-20261123-085050-ZDLg6Y",
    "E1-cv-summary-20261123-085240-ZDLg6Y",
    "E1-paired-bootstrap-20261123-085242-ZDLg6Y",
]

BLUE = "#0F4D92"
TEAL = "#168C88"
ORANGE = "#D97925"
PALE_BLUE = "#A9C7E4"
PALE_TEAL = "#A8D8D3"
PALE_ORANGE = "#F2C9A8"
INK = "#272727"
MID = "#6E6E6E"
LIGHT = "#D8D8D8"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_tsv(path: Path, required: list[str], numeric: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    missing = set(required) - set(frame.columns)
    assert not missing, f"{path}: missing columns {sorted(missing)}"
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        assert np.isfinite(values).all(), f"{path}: non-finite values in {column}"
    return frame


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict) and value
    return value


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    return float(((2 * np.arange(1, n + 1) - n - 1) * x).sum() / (n * x.sum()))


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.text(
        -0.16,
        1.055,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(fig: plt.Figure, stem: str, output_root: Path = FIGURE_ROOT, layout: str = "constrained") -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if layout == "manual":
        fig.set_layout_engine("none")
    elif layout == "tight":
        fig.set_layout_engine("none")
        fig.tight_layout(pad=1.2)
    else:
        fig.set_layout_engine("constrained")
    fig.savefig(output_root / f"{stem}_300dpi.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output_root / f"{stem}_600dpi.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output_root / f"{stem}.svg", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_plot_data(figure: str, frames: dict[str, pd.DataFrame]) -> list[dict]:
    out = VERIFY_ROOT / "data" / figure
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, frame in frames.items():
        path = out / f"{name}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        records.append(source(path))
    return records


def write_audit(figure: str, audit: dict) -> None:
    VERIFY_ROOT.mkdir(parents=True, exist_ok=True)
    audit["run_ids"] = RUNS[figure]
    path = VERIFY_ROOT / f"{figure}_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")


def completed_run(log_path: Path, run_id: str) -> dict:
    matches = []
    with log_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("run_id") == run_id and record.get("status") == "completed":
                matches.append(record)
    assert len(matches) == 1, f"{run_id}: expected one completed record, got {len(matches)}"
    return matches[0]


def figure1() -> dict:
    positives_path = E0 / "positives_audit.tsv"
    dual_path = E0 / "dual_cellline_consistency.tsv"
    identity_path = E0 / "identity_summary.json"
    leakage_path = E0 / "leakage_summary.json"
    manifest_path = E0 / "split_manifest.id90.json"
    distribution_path = E0 / "rbp_distribution.json"
    contract_path = E1 / "results" / "strict3train_p20_v2" / "verification" / "final_verification.json"
    positives = read_tsv(
        positives_path,
        ["rbp", "cell", "n_sites", "n_unique"],
        ["n_sites", "n_unique"],
    )
    dual = read_tsv(
        dual_path,
        ["rbp", "overlap50_K562_frac", "overlap50_HepG2_frac"],
        ["overlap50_K562_frac", "overlap50_HepG2_frac"],
    )
    identity = read_json(identity_path)
    leakage = read_json(leakage_path)
    manifest = read_json(manifest_path)
    distribution = read_json(distribution_path)
    contract = read_json(contract_path)
    assert contract["accepted"] and contract["data_version"] == "e0-v2-id90-strict3train-main"
    assert contract["pass_2_protocol_alignment"]["n_experiments"] == 146
    assert contract["pass_2_protocol_alignment"]["all_train_strict_three_folds"]
    assert contract["pass_2_protocol_alignment"]["no_fold_minus1_in_train"]
    assert len(positives) == 223 and positives["rbp"].nunique() == 150
    assert len(dual) == 73 and dual["rbp"].is_unique
    assert manifest["identity"] == 0.9 and manifest["seed"] == 20261122 and manifest["folds"] == 5
    assert identity["n_windows"] == sum(identity["bucket_counts"].values())
    assert leakage["channel5_implementation"]["fixed_seeds"] == {"split": 20261122, "neg": 42}

    per_rbp = positives.groupby("rbp", as_index=False)["n_sites"].sum().sort_values("n_sites")
    support = per_rbp["n_sites"].to_numpy(dtype=float)
    g = gini(support)
    cumulative = np.r_[0.0, np.cumsum(support) / support.sum()]
    fraction = np.r_[0.0, np.arange(1, len(support) + 1) / len(support)]
    top20_share = float(support[-int(np.ceil(0.2 * len(support))) :].sum() / support.sum())
    assert round(g, 4) == distribution["gini_per_rbp_total"] == 0.3523

    bucket_order = ["0-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1.0001"]
    thresholds = np.array([0.0, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    counts = np.array([identity["bucket_counts"][key] for key in bucket_order])
    homology_cdf = np.r_[0.0, np.cumsum(counts) / counts.sum()]
    below07 = float(homology_cdf[2])
    assert round(below07 * 100, 1) == 94.7
    cluster_leak = leakage["channel1_homology"]["cluster_leak_frac_mean"]
    assert cluster_leak == 0.000225
    global_neff = manifest["n_clusters"] / manifest["n_windows_clustered"]
    assert round(global_neff, 3) == 0.515

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=1.1, ls="--", label="Equality")
    ax.plot(fraction, cumulative, color=BLUE, lw=2.2)
    ax.fill_between(fraction, cumulative, fraction, color=PALE_BLUE, alpha=0.45)
    ax.axvline(0.8, color=MID, lw=0.8, ls=":")
    top_y = float(1 - top20_share)
    ax.axhline(top_y, color=MID, lw=0.8, ls=":")
    ax.scatter([0.8], [top_y], s=32, color=ORANGE, zorder=4)
    ax.text(0.80, top_y - 0.09, f"Top 20% RBPs\n= {top20_share:.1%} of sites", ha="center", va="top", fontsize=8)
    ax.text(0.05, 0.86, f"Gini = {g:.3f}\n150 RBPs", fontsize=9, fontweight="bold")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="RBPs, cumulative fraction", ylabel="Crosslink sites, cumulative share")
    panel(ax, "a", "RBP support is uneven, not uniformly capped")

    ax = axes[1]
    k562 = dual["overlap50_K562_frac"].to_numpy()
    hepg2 = dual["overlap50_HepG2_frac"].to_numpy()
    for left, right in zip(k562, hepg2):
        ax.plot([0, 1], [left, right], color=LIGHT, lw=0.55, alpha=0.65, zorder=1)
    rng = np.random.default_rng(20261125)
    for x, values, color, name in [(0, k562, BLUE, "K562"), (1, hepg2, TEAL, "HepG2")]:
        jitter = rng.uniform(-0.055, 0.055, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=10, color=color, alpha=0.45, edgecolors="none")
        median = float(np.median(values))
        ax.scatter([x], [median], s=72, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(x, median + 0.035, f"Median {median:.3f}", ha="center", color=color, fontweight="bold")
    ax.set(xlim=(-0.35, 1.35), ylim=(0, 0.58), xticks=[0, 1], xticklabels=["K562", "HepG2"], ylabel="Cross-cell site overlap (±50 nt)")
    panel(ax, "b", "Cross-cell site overlap is limited")

    ax = axes[2]
    ax.plot(thresholds, homology_cdf * 100, color=TEAL, lw=2.2, marker="o", ms=4)
    ax.fill_between(thresholds, homology_cdf * 100, color=PALE_TEAL, alpha=0.55)
    ax.axvline(0.7, color=ORANGE, lw=1.0, ls="--")
    ax.scatter([0.7], [below07 * 100], s=44, color=ORANGE, zorder=4)
    ax.annotate(
        f"{below07:.1%} below 0.70",
        xy=(0.7, below07 * 100),
        xytext=(0.42, 72),
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.9},
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.98,
        8,
        "Current E1 strict3train contract:\nfold=-1 excluded from train | 146 experiments accepted",
        ha="right",
        va="bottom",
        fontsize=8,
        color=TEAL,
        fontweight="bold",
        bbox={"facecolor": "#EAF6F4", "edgecolor": TEAL, "boxstyle": "round,pad=0.35"},
    )
    ax.text(
        0.98,
        38,
        f"Historical loader issue remediated; legacy cluster check* = {cluster_leak:.4%}\nGlobal $n_{{eff}}$/windows = {global_neff:.3f}",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.set(xlim=(0, 1), ylim=(0, 105), xlabel="Nearest training identity ($d_{hom}$)", ylabel="Test windows, cumulative (%)")
    panel(ax, "c", "Current data contract repaired and independently accepted")

    stem = "Fig1_E0_data_and_leakage_audit"
    save_figure(fig, stem)
    cdf_frame = pd.DataFrame({"identity_threshold": thresholds, "cumulative_fraction": homology_cdf})
    plot_files = write_plot_data("fig1", {"rbp_support": per_rbp, "dual_cell_overlap": dual, "identity_cdf": cdf_frame})
    return {
        "evidence_level": "E0 design and historical audit; current E1 strict3train data contract repaired and accepted",
        "sources": [source(p) for p in [positives_path, dual_path, identity_path, leakage_path, manifest_path, distribution_path, contract_path]],
        "pass1_schema_finite": True,
        "pass2_configuration_alignment": {"identity": 0.9, "split_seed": 20261122, "folds": 5, "negative_seed": 42, "current_data_version": contract["data_version"], "current_contract_accepted": True, "current_n_experiments": 146, "current_all_train_strict_three_folds": True, "current_no_fold_minus1_in_train": True, "legacy_loader_fold_minus_one_training_gap": "historical; remediated", "strict_zero_leakage_claim": False, "legacy_cluster_check_caveat": "historical only; includes validation and counts each positive in two tracks; not a strict three-fold train rate"},
        "pass3_report_recalculation": {
            "n_rbps": len(per_rbp),
            "gini": g,
            "top20_site_share": top20_share,
            "median_overlap_K562": float(np.median(k562)),
            "median_overlap_HepG2": float(np.median(hepg2)),
            "identity_below_0.7": below07,
            "cross_fold_cluster_leak": cluster_leak,
            "global_neff_ratio": global_neff,
        },
        "plot_data": plot_files,
        "figure_stem": stem,
        "excluded_ambiguous_number": "45.98 is an experiment-level max/min ratio, not the RBP-summed ratio; it is not plotted.",
    }


def figure2() -> dict:
    result_dir = E1 / "results" / "strict3train_p20_v2"
    bootstrap_path = result_dir / "cv_bootstrap" / "paired_bootstrap.tsv"
    folds_path = result_dir / "cv_summary" / "cv_by_fold.tsv"
    metrics_path = result_dir / "cv_summary" / "cv_metrics.tsv"
    verification_path = result_dir / "verification" / "final_verification.json"
    manifest_path = result_dir / "verification" / "final_run_manifest.tsv"
    report_path = E1 / "E1_主集严格三折冻结基线最终报告_2026-11-26.md"
    bootstrap = read_tsv(
        bootstrap_path,
        ["track", "slice", "metric", "n_experiments", "n_seeds", "n_boot", "observed_delta", "ci_low", "ci_high", "p_two_sided"],
        ["n_experiments", "n_seeds", "n_boot", "observed_delta", "ci_low", "ci_high", "p_two_sided"],
    )
    folds = read_tsv(
        folds_path,
        ["fold", "track", "split", "tier", "auroc_modern", "auprc_modern", "auroc_knn", "auprc_knn", "d_auroc", "d_auprc"],
        ["fold", "auroc_modern", "auprc_modern", "auroc_knn", "auprc_knn", "d_auroc", "d_auprc"],
    )
    metrics = read_tsv(
        metrics_path,
        ["track", "split", "tier", "d_auroc_mean", "d_auprc_mean", "n_pairs", "n_folds", "n_seeds"],
        ["d_auroc_mean", "d_auprc_mean", "n_pairs", "n_folds", "n_seeds"],
    )
    manifest = read_tsv(manifest_path, ["stage", "name", "run_id", "status", "data_version"], [])
    verification = read_json(verification_path)
    assert verification["accepted"] and verification["data_version"] == "e0-v2-id90-strict3train-main"
    assert sha256(report_path) == verification["report_sha256"]
    assert verification["pass_2_protocol_alignment"]["n_experiments"] == 146
    assert verification["pass_2_protocol_alignment"]["all_train_strict_three_folds"]
    assert verification["pass_2_protocol_alignment"]["no_fold_minus1_in_train"]
    assert verification["pass_3_numeric_recalculation"]["n_cv_pairs"] == 8760
    assert verification["pass_3_numeric_recalculation"]["bootstrap_observed_ci_p_recalculated"]
    assert len(manifest) == 26 and manifest["status"].eq("completed").all()
    assert manifest["data_version"].eq("e0-v2-id90-strict3train-main").all()
    assert set(RUNS["fig2"]).issubset(set(manifest["run_id"]))
    assert len(bootstrap) == 12 and set(bootstrap["n_seeds"]) == {3} and set(bootstrap["n_boot"]) == {10000}
    assert set(bootstrap["slice"]) == {"head", "mid", "tail"} and set(bootstrap["track"]) == {"neg1c", "neg2"}
    expected_counts = {"head": 100, "mid": 28, "tail": 18}
    assert all(row["n_experiments"] == expected_counts[row["slice"]] for _, row in bootstrap.iterrows())
    assert ((bootstrap["observed_delta"] > 0) & (bootstrap["ci_low"] > 0)).all()
    test = folds[folds["split"] == "test"].copy()
    assert len(test) == 30 and set(test["fold"]) == set(range(5)) and set(test["tier"]) == set(expected_counts)
    assert np.allclose(test["d_auroc"], test["auroc_modern"] - test["auroc_knn"])
    assert np.allclose(test["d_auprc"], test["auprc_modern"] - test["auprc_knn"])
    assert (test[["d_auroc", "d_auprc"]] > 0).all().all()

    ci_recalculated = {}
    for track in ["neg1c", "neg2"]:
        for slice_name in ["head", "mid", "tail"]:
            npz_path = result_dir / "cv_bootstrap" / f"bootstrap_{track}_{slice_name}.npz"
            deltas = np.load(npz_path)["deltas"]
            assert deltas.shape == (10000, 2) and np.isfinite(deltas).all()
            interval = np.quantile(deltas, [0.025, 0.975], axis=0)
            for index, metric in enumerate(["auroc", "auprc"]):
                row = bootstrap[(bootstrap["track"] == track) & (bootstrap["slice"] == slice_name) & (bootstrap["metric"] == metric)].iloc[0]
                assert np.allclose(interval[:, index], [row["ci_low"], row["ci_high"]], atol=1e-12)
                negative = (1 + np.sum(deltas[:, index] <= 0)) / 10001
                positive = (1 + np.sum(deltas[:, index] >= 0)) / 10001
                assert np.isclose(min(1.0, 2 * min(negative, positive)), row["p_two_sided"], atol=1e-12)
                ci_recalculated[f"{track}_{slice_name}_{metric}"] = interval[:, index].tolist()

    fold_means = test.groupby(["track", "tier"])[["d_auroc", "d_auprc"]].mean().sort_index()
    published_metrics = metrics[metrics["split"] == "test"].set_index(["track", "tier"])[["d_auroc_mean", "d_auprc_mean"]].sort_index()
    assert np.allclose(fold_means.to_numpy(), published_metrics.to_numpy())
    assert metrics["n_pairs"].sum() == 8760 and metrics[metrics["split"] == "test"]["n_pairs"].sum() == 4380

    fig = plt.figure(figsize=(13.0, 7.1))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.70, 0.78, 0.78], height_ratios=[1, 1], wspace=0.38, hspace=0.65)
    ax = fig.add_subplot(grid[:, 0])
    ordered = pd.concat(
        [
            bootstrap[(bootstrap["slice"] == slice_name) & (bootstrap["track"] == track) & (bootstrap["metric"] == metric)]
            for slice_name in ["head", "mid", "tail"]
            for track in ["neg1c", "neg2"]
            for metric in ["auroc", "auprc"]
        ],
        ignore_index=True,
    )
    y = np.arange(len(ordered))[::-1]
    for pos, (_, row) in zip(y, ordered.iterrows()):
        color = BLUE if row["metric"] == "auroc" else TEAL
        marker = "o" if row["metric"] == "auroc" else "D"
        ax.plot([row["ci_low"], row["ci_high"]], [pos, pos], color=color, lw=2.0, solid_capstyle="round")
        ax.scatter(row["observed_delta"], pos, s=48, marker=marker, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(0.055, pos, f"{row['observed_delta']:.3f}  [{row['ci_low']:.3f}, {row['ci_high']:.3f}]", va="center", fontsize=7.2)
    ax.axvline(0, color=MID, lw=0.9, ls="--")
    ax.axhline(7.5, color=LIGHT, lw=0.7)
    ax.axhline(3.5, color=LIGHT, lw=0.7)
    labels = [f"{r['slice'].capitalize()} · {r['track']} · {r['metric'].upper()}" for _, r in ordered.iterrows()]
    ax.set_yticks(y, labels)
    ax.set(xlim=(-0.002, 0.077), ylim=(-0.7, 11.7), xlabel="Δ BCE − kNN (pooled test predictions; 95% CI)\nCV panels: fold × experiment × seed macro deltas; descriptive, not bootstrap intervals")
    ax.xaxis.label.set_fontsize(8)
    panel(ax, "a", "Accepted 146-experiment main set: all 12 effects favor BCE")

    tier_styles = {"head": (BLUE, "o", "Head"), "mid": (ORANGE, "s", "Mid"), "tail": (TEAL, "D", "Tail")}
    letters = [["b", "c"], ["d", "e"]]
    for row_index, track in enumerate(["neg1c", "neg2"]):
        for col_index, metric in enumerate(["auroc", "auprc"]):
            ax = fig.add_subplot(grid[row_index, col_index + 1])
            for tier, (color, marker, label) in tier_styles.items():
                subset = test[(test["track"] == track) & (test["tier"] == tier)].sort_values("fold")
                ax.plot(subset["fold"], subset[f"d_{metric}"], color=color, marker=marker, ms=4.5, lw=1.35, label=label)
            ax.axhline(0, color=MID, lw=0.8, ls="--")
            ax.set(xticks=range(5), xlabel="Test fold", ylim=(0.015, 0.060), ylabel=f"Δ{metric.upper()} (BCE − kNN)" if col_index == 0 else "")
            panel(ax, letters[row_index][col_index], f"CV by fold · {track} · {metric.upper()}")
            if row_index == 0 and col_index == 1:
                ax.legend(loc="upper right", fontsize=7, ncol=3, columnspacing=0.8, handlelength=1.4)
    stem = "Fig2_E1_main_strict3train_five_fold_three_seed"
    save_figure(fig, stem)
    plot_files = write_plot_data("fig2", {"paired_bootstrap": bootstrap, "cv_test_by_fold": test, "cv_test_metrics": published_metrics.reset_index()})
    return {
        "evidence_level": "ACCEPTED E1 main strict3train; 146 experiments; five folds; three seeds; frozen RNA-FM BCE versus cluster-deduplicated cosine kNN",
        "sources": [source(p) for p in [bootstrap_path, folds_path, metrics_path, verification_path, manifest_path, report_path]],
        "pass1_schema_finite": {"bootstrap_rows": len(bootstrap), "cv_test_rows": len(test), "bootstrap_npz": 6, "manifest_rows": len(manifest), "upstream_accepted": verification["accepted"]},
        "pass2_configuration_alignment": {"data_version": verification["data_version"], "n_experiments": 146, "tier_experiments": expected_counts, "folds": [0, 1, 2, 3, 4], "model_seeds": [0, 1, 2], "n_boot": 10000, "strict_three_fold_train": True, "no_fold_minus1_in_train": True, "submission_acceptance": True},
        "pass3_report_recalculation": {"bootstrap_ci_and_p_from_six_new_npz": ci_recalculated, "cv_metrics_exact_match": True, "n_cv_pairs": 8760, "all_12_bootstrap_effects_positive_ci_exclude_zero": True, "all_60_fold_track_tier_metric_deltas_positive": True},
        "plot_data": plot_files,
        "figure_stem": stem,
    }


def figure3() -> dict:
    snapshot = DATA_ROOT / "parclip_p20"
    result_paths = [snapshot / f"seed{seed}_cross_protocol_results.tsv" for seed in range(3)]
    macro_by_seed_path = snapshot / "three_seed_macro_by_seed.tsv"
    macro_summary_path = snapshot / "three_seed_macro_summary.tsv"
    per_rbp_summary_path = snapshot / "three_seed_per_rbp_summary.tsv"
    verification_path = snapshot / "three_pass_verification.json"
    required = ["rbp", "train_track", "eval_protocol", "n_parclip_pos", "n_neg", "best_epoch", "stopped_epoch", "auroc", "auprc"]
    numeric = ["n_parclip_pos", "n_neg", "best_epoch", "stopped_epoch", "auroc", "auprc"]
    frames = []
    for seed, path in enumerate(result_paths):
        frame = read_tsv(path, required, numeric)
        assert len(frame) == 27 and frame["rbp"].nunique() == 9
        assert set(frame["eval_protocol"]) == {
            "eCLIP-neg1c-fold0",
            "eCLIP-neg2-fold0",
            "PARCLIP-otherRBP-geneMatched-excl300",
        }
        assert ((frame["stopped_epoch"] - frame["best_epoch"]) == 20).all()
        assert (frame["stopped_epoch"] < 1000).all()
        frame.insert(0, "seed", seed)
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    macro_by_seed = read_tsv(macro_by_seed_path, ["eval_protocol", "seed", "auroc", "auprc"], ["seed", "auroc", "auprc"])
    verification = read_json(verification_path)
    report = E5_REPORT.read_text()
    assert verification["status"] == "passed"
    assert [verification["pass2_alignment"][f"seed{seed}"]["run_id"] for seed in range(3)] == RUNS["fig3"]
    assert all(verification["pass2_alignment"][f"seed{seed}"]["split_seed_history_checkpoint_aligned"] for seed in range(3))
    assert all(verification["pass3_metric_recalculation"][f"seed{seed}"]["macro_recalculated"] for seed in range(3))
    assert all(run_id in report for run_id in RUNS["fig3"])

    calculated_by_seed = results.groupby(["eval_protocol", "seed"], as_index=False)[["auroc", "auprc"]].mean()
    aligned = macro_by_seed.set_index(["eval_protocol", "seed"])[["auroc", "auprc"]]
    assert np.allclose(calculated_by_seed.set_index(["eval_protocol", "seed"]), aligned)
    calculated_summary = calculated_by_seed.groupby("eval_protocol")[["auroc", "auprc"]].agg(["mean", "std", "min", "max"])
    published_summary = pd.read_csv(macro_summary_path, sep="\t", header=[0, 1], index_col=0)
    assert np.allclose(calculated_summary.sort_index(), published_summary.sort_index())
    calculated_per_rbp = results.groupby(["eval_protocol", "rbp"])[["auroc", "auprc"]].agg(["mean", "std"])
    published_per_rbp = pd.read_csv(per_rbp_summary_path, sep="\t", header=[0, 1], index_col=[0, 1])
    assert np.allclose(calculated_per_rbp.sort_index(), published_per_rbp.sort_index())

    protocols = ["eCLIP-neg1c-fold0", "eCLIP-neg2-fold0", "PARCLIP-otherRBP-geneMatched-excl300"]
    short = {
        "eCLIP-neg1c-fold0": "eCLIP neg1c",
        "eCLIP-neg2-fold0": "eCLIP neg2",
        "PARCLIP-otherRBP-geneMatched-excl300": "PAR-CLIP matched neg2",
    }
    styles = {
        "eCLIP-neg1c-fold0": (BLUE, "o"),
        "eCLIP-neg2-fold0": (TEAL, "D"),
        "PARCLIP-otherRBP-geneMatched-excl300": (ORANGE, "s"),
    }
    means = calculated_per_rbp.xs("mean", level=1, axis=1)["auroc"].unstack(0)
    order = means["eCLIP-neg1c-fold0"].sort_values().index

    fig = plt.figure(figsize=(12.2, 5.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.75, 0.85], wspace=0.43)
    for column, metric in [(0, "auroc"), (2, "auprc")]:
        ax = fig.add_subplot(grid[0, column])
        for index, protocol in enumerate(protocols):
            color, marker = styles[protocol]
            subset = macro_by_seed[macro_by_seed["eval_protocol"] == protocol].sort_values("seed")
            jitter = np.array([-0.055, 0.0, 0.055])
            ax.scatter(np.full(3, index) + jitter, subset[metric], s=32, color=color, marker=marker, alpha=0.75)
            mean = subset[metric].mean()
            sd = subset[metric].std()
            ax.errorbar(index, mean, yerr=sd, color=color, lw=2.0, capsize=4, marker=marker, ms=7, zorder=4)
            ax.text(index, mean + 0.055 if metric == "auroc" else mean + 0.075, f"{mean:.3f} ± {sd:.3f}", ha="center", fontsize=7.4, color=color, fontweight="bold")
        if metric == "auroc":
            ax.axhline(0.5, color=MID, lw=0.9, ls="--")
            ax.set_ylim(0.43, 0.72)
            panel(ax, "a", "Three-seed macro AUROC by negative reference")
        else:
            ax.set_ylim(0.55, 0.75)
            panel(ax, "c", "Macro AUPRC is secondary and prevalence dependent")
        ax.set(xlim=(-0.45, 2.45), xticks=range(3), xticklabels=[short[p].replace(" ", "\n", 1) for p in protocols], ylabel=metric.upper())
        ax.text(0.5, -0.21, "points: seeds 0/1/2; bars: sample SD", transform=ax.transAxes, ha="center", fontsize=7.5, color=MID)

    ax = fig.add_subplot(grid[0, 1])
    y = np.arange(len(order))
    offsets = [-0.22, 0.0, 0.22]
    for protocol, offset in zip(protocols, offsets):
        color, marker = styles[protocol]
        subset = calculated_per_rbp.xs(protocol, level=0)
        values = subset.loc[order, ("auroc", "mean")].to_numpy()
        errors = subset.loc[order, ("auroc", "std")].to_numpy()
        ax.errorbar(values, y + offset, xerr=errors, fmt=marker, ms=5.5, color=color, lw=1.0, capsize=2, label=short[protocol])
    ax.axvline(0.5, color=MID, lw=0.9, ls="--")
    ax.set_yticks(y, order)
    ax.set(xlim=(0.28, 0.99), ylim=(-0.65, len(order) - 0.35), xlabel="Per-RBP AUROC, mean ± sample SD across 3 seeds")
    ax.legend(loc="lower right", fontsize=7)
    panel(ax, "b", "Cross-protocol discrimination remains RBP and negative-reference dependent")

    stem = "Fig3_E5_PARCLIP_cross_protocol"
    save_figure(fig, stem)
    plot_files = write_plot_data(
        "fig3",
        {"cross_protocol_three_seed": results, "macro_by_seed": calculated_by_seed, "per_rbp_summary": calculated_per_rbp.reset_index()},
    )
    return {
        "evidence_level": "E5 PAR-CLIP hg38 v2 external descriptive; 9 RBPs; 3 seeds; 3 negative references; no significance claim",
        "sources": [source(p) for p in result_paths + [macro_by_seed_path, macro_summary_path, per_rbp_summary_path, verification_path, E5_REPORT]],
        "pass1_schema_finite": {"rows_per_seed": 27, "n_rbp": 9, "upstream_verification": verification["status"]},
        "pass2_configuration_alignment": {"seeds": [0, 1, 2], "protocols": protocols, "patience": 20, "min_delta": 0.0001, "all_stopped_at_best_plus_20": True, "selection": "eCLIP validation fold1; PAR-CLIP excluded from selection"},
        "pass3_report_recalculation": {"macro_by_seed_exact_match": True, "macro_summary_exact_match": True, "per_rbp_summary_exact_match": True, "macro_auroc_mean_sd": {protocol: [float(macro_by_seed[macro_by_seed["eval_protocol"] == protocol]["auroc"].mean()), float(macro_by_seed[macro_by_seed["eval_protocol"] == protocol]["auroc"].std())] for protocol in protocols}},
        "plot_data": plot_files,
        "figure_stem": stem,
    }


def figure4() -> dict:
    result_dir = E6 / "results" / "five_split_agg"
    buckets_path = result_dir / "five_split_buckets.tsv"
    bucket_summary_path = result_dir / "five_split_bucket_summary.tsv"
    ood_path = result_dir / "five_split_ood.tsv"
    ood_summary_path = result_dir / "five_split_ood_summary.tsv"
    convergence_path = result_dir / "five_split_convergence.tsv"
    summary_path = result_dir / "five_split_summary.json"
    verification_path = result_dir / "independent_verification.json"
    report_path = E6 / "E6_五划分冻结基线报告_2026-11-25.md"
    buckets = read_tsv(
        buckets_path,
        ["split", "bucket", "n_labels", "delta_macro_auprc", "ci_low", "ci_high", "p_two_sided", "n_bootstrap", "run_id"],
        ["split", "n_labels", "delta_macro_auprc", "ci_low", "ci_high", "p_two_sided", "n_bootstrap"],
    )
    ood = read_tsv(
        ood_path,
        ["split", "method", "auroc_ood", "fpr95", "aurc_ood_error", "metric_definition", "score_sha256", "ood_recompute_run_id", "run_id"],
        ["split", "auroc_ood", "fpr95", "aurc_ood_error"],
    )
    convergence = read_tsv(convergence_path, ["split", "seed", "best_epoch", "last_epoch"], ["split", "seed", "best_epoch", "last_epoch"])
    summary = read_json(summary_path)
    verification = read_json(verification_path)
    report = report_path.read_text()
    assert summary["complete"] and summary["accepted"] and summary["ood_semantics_verified"] and summary["missing_splits"] == []
    assert verification["accepted"] and verification["ood_semantics_verified"] and verification["synthetic_counterexamples_passed"]
    assert summary["aggregation_run_id"] in report and verification["verification_run_id"] in report
    assert set(buckets["split"]) == set(range(1, 6)) and len(buckets) == 25
    assert buckets["n_bootstrap"].eq(10000).all() and (buckets["delta_macro_auprc"] > 0).all()
    assert ((buckets["ci_low"] > 0) & (buckets["ci_high"] > 0)).all()
    assert set(convergence["seed"]) == {0, 1, 2} and len(convergence) == 15
    assert ((convergence["last_epoch"] - convergence["best_epoch"]) == 20).all()
    assert [summary["per_split_meta"][str(split)]["run_id"] for split in range(1, 6)] == RUNS["fig4"][:5]
    assert set(ood["split"]) == set(range(1, 6)) and len(ood) == 15
    assert ood["metric_definition"].eq("e6-ood-v2:fpr95=id_at_ood_tpr95;aurc=ascending_ood_score").all()
    assert ood["ood_recompute_run_id"].eq(RUNS["fig4"][5]).all()
    assert ood["score_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    bucket_order = ["tail_lt10", "tail_10_29", "mid_30_99", "head_ge100", "all"]
    calculated_bucket = buckets.groupby("bucket")["delta_macro_auprc"].agg(split_mean="mean", split_sd="std", split_min="min", split_max="max").loc[bucket_order]
    published_bucket = read_tsv(bucket_summary_path, ["bucket", "split_mean", "split_sd", "split_min", "split_max"], ["split_mean", "split_sd", "split_min", "split_max"]).set_index("bucket").loc[bucket_order]
    assert np.allclose(calculated_bucket, published_bucket[["split_mean", "split_sd", "split_min", "split_max"]])
    metric_order = ["auroc_ood", "fpr95", "aurc_ood_error"]
    calculated_ood = ood.melt(id_vars=["split", "method"], value_vars=metric_order, var_name="metric", value_name="value").groupby(["method", "metric"])["value"].agg(split_mean="mean", split_sd="std", split_min="min", split_max="max")
    published_ood = read_tsv(ood_summary_path, ["method", "metric", "split_mean", "split_sd", "split_min", "split_max"], ["split_mean", "split_sd", "split_min", "split_max"]).set_index(["method", "metric"])
    assert np.allclose(calculated_ood.sort_index(), published_ood[["split_mean", "split_sd", "split_min", "split_max"]].sort_index())
    for split in range(1, 6):
        subset = ood[ood["split"] == split].set_index("method")
        for method in ["msp", "energy"]:
            assert subset.loc[method, "auroc_ood"] > subset.loc["deep_knn", "auroc_ood"]
            assert subset.loc[method, "fpr95"] < subset.loc["deep_knn", "fpr95"]
            assert subset.loc[method, "aurc_ood_error"] < subset.loc["deep_knn", "aurc_ood_error"]

    labels = {"tail_lt10": "<10", "tail_10_29": "10–29", "mid_30_99": "30–99", "head_ge100": "≥100", "all": "All"}
    bucket_style = {
        "tail_lt10": (BLUE, "o"),
        "tail_10_29": (TEAL, "D"),
        "mid_30_99": (ORANGE, "s"),
        "head_ge100": (MID, "^"),
    }
    method_style = {"msp": (BLUE, "o", "MSP"), "energy": (TEAL, "D", "Energy"), "deep_knn": (ORANGE, "s", "deep-kNN")}
    fig = plt.figure(figsize=(12.2, 7.3))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.05, 1], wspace=0.36, hspace=0.64)

    ax = fig.add_subplot(grid[0, :2])
    for bucket, (color, marker) in bucket_style.items():
        subset = buckets[buckets["bucket"] == bucket].sort_values("split")
        ax.errorbar(subset["split"], subset["delta_macro_auprc"], yerr=[subset["delta_macro_auprc"] - subset["ci_low"], subset["ci_high"] - subset["delta_macro_auprc"]], color=color, marker=marker, ms=5.5, lw=1.4, capsize=2.5, label=labels[bucket])
    ax.axhline(0, color=MID, lw=0.9, ls="--")
    ax.set(xticks=range(1, 6), xlabel="Official GRIMM split", ylabel="Δmacro AUPRC (BCE − deep-kNN)", ylim=(0, 0.43))
    ax.legend(ncol=4, loc="upper center", title="Training support per EC label")
    panel(ax, "a", "BCE exceeds deep-kNN in all 20 split × frequency-bucket cells")

    ax = fig.add_subplot(grid[0, 2])
    x = np.arange(4)
    summary_four = published_bucket.loc[bucket_order[:4]]
    ax.errorbar(x, summary_four["split_mean"], yerr=summary_four["split_sd"], fmt="o", ms=6, color=BLUE, lw=1.8, capsize=4)
    for pos, bucket in enumerate(bucket_order[:4]):
        values = buckets[buckets["bucket"] == bucket]["delta_macro_auprc"]
        ax.scatter(np.full(5, pos) + np.linspace(-0.09, 0.09, 5), values, s=18, color=PALE_BLUE, edgecolor=BLUE, linewidth=0.4, zorder=3)
        ax.text(pos, summary_four.loc[bucket, "split_mean"] + 0.024, f"{summary_four.loc[bucket, 'split_mean']:.3f}", ha="center", fontsize=7.5, fontweight="bold")
    ax.set(xticks=x, xticklabels=[labels[b] for b in bucket_order[:4]], xlabel="Training support per EC label", ylabel="Split mean ± sample SD", ylim=(0.12, 0.40))
    panel(ax, "b", "Equal-weight split summary; no pooled inference")

    metric_titles = {
        "auroc_ood": "AUROC-OOD ↑",
        "fpr95": "FPR95 ↓ (ID at OOD TPR95)",
        "aurc_ood_error": "AURC ↓ (ascending OOD score)",
    }
    for index, metric in enumerate(metric_order):
        ax = fig.add_subplot(grid[1, index])
        for method, (color, marker, name) in method_style.items():
            subset = ood[(ood["metric_definition"].str.len() > 0) & (ood["method"] == method)].sort_values("split")
            ax.plot(subset["split"], subset[metric], color=color, marker=marker, ms=5, lw=1.6, label=name)
        ax.set(xticks=range(1, 6), xlabel="Official split", ylabel=metric_titles[metric])
        panel(ax, chr(99 + index), f"Test-2 pseudo-OOD v2: {metric_titles[metric]}")
        if index == 0:
            ax.legend(loc="center right", fontsize=7)
        if metric == "aurc_ood_error":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    stem = "Fig4_E6_GRIMM_five_split_frequency_and_OOD_v2"
    save_figure(fig, stem)
    plot_files = write_plot_data(
        "fig4",
        {"function_by_split": buckets, "function_split_summary": published_bucket.reset_index(), "ood_v2_by_split": ood, "ood_v2_split_summary": published_ood.reset_index(), "convergence": convergence},
    )
    return {
        "evidence_level": "E6 GRIMM EC_v2 five official splits; Test-1; Test-2 pseudo-OOD v2; cross-split descriptive only",
        "sources": [source(p) for p in [buckets_path, bucket_summary_path, ood_path, ood_summary_path, convergence_path, summary_path, verification_path, report_path]],
        "pass1_schema_finite": {"function_rows": len(buckets), "ood_rows": len(ood), "convergence_rows": len(convergence), "independent_verification_accepted": verification["accepted"]},
        "pass2_configuration_alignment": {"splits": [1, 2, 3, 4, 5], "seeds": [0, 1, 2], "n_bootstrap": 10000, "patience": 20, "ood_definition": summary["ood_metric_definition"], "ood_semantics_verified": True, "selection": "validation checkpoint, tau and Fmax threshold; Test-2 pseudo-OOD only"},
        "pass3_report_recalculation": {"all_20_frequency_cells_positive_and_ci_exclude_zero": True, "bucket_mean_sd_range_exact_match": True, "ood_mean_sd_range_exact_match": True, "msp_energy_better_than_deep_knn_all_metrics_all_splits_descriptive": True, "bucket_split_mean_sd": calculated_bucket.reset_index().to_dict(orient="records")},
        "plot_data": plot_files,
        "figure_stem": stem,
    }


def figure5() -> dict:
    raw_path = E5_HOLDOUT / "holdout_per_rbp.tsv"
    per_seed_path = E5_HOLDOUT / "holdout_per_seed.tsv"
    multiseed_path = E5_HOLDOUT / "holdout_multiseed.tsv"
    conflicts_path = E5_HOLDOUT / "metric_winner_conflicts.tsv"
    convergence_path = E5_HOLDOUT / "convergence_per_pair.tsv"
    independent_path = E5_HOLDOUT / "independent_window_recalculation.tsv"
    verification_path = E5_HOLDOUT / "verification.json"
    independent_verification_path = E5_HOLDOUT / "independent_verification.json"
    config_path = E5_HOLDOUT / "config.json"
    metric_names = ["auroc_ood", "fpr95", "aurc"]
    raw = read_tsv(
        raw_path,
        ["run_id", "holdout", "track", "seed", "stable_seed", "method", "metric_version", *metric_names, "stop_reason", "max_epochs_failsafe"],
        ["seed", "stable_seed", "metric_version", *metric_names, "max_epochs_failsafe"],
    )
    per_seed_published = read_tsv(per_seed_path, ["seed", "track", "method", *metric_names], ["seed", *metric_names])
    multiseed_published = read_tsv(
        multiseed_path,
        ["track", "method", *[f"{metric}_{stat}" for metric in metric_names for stat in ["mean", "std"]], "n_seeds"],
        [* [f"{metric}_{stat}" for metric in metric_names for stat in ["mean", "std"]], "n_seeds"],
    )
    independent = read_tsv(independent_path, ["seed", "track", "holdout", "method", *metric_names, "manual_passed"], ["seed", *metric_names])
    convergence = read_tsv(convergence_path, ["holdout", "track", "seed", "stop_reason", "best_epoch", "epochs_run"], ["seed", "best_epoch", "epochs_run"])
    conflicts_published = read_tsv(conflicts_path, ["track", "auroc_ood_winner", "fpr95_winner", "aurc_winner", "winner_conflict"], [])
    verification = read_json(verification_path)
    independent_verification = read_json(independent_verification_path)
    config = read_json(config_path)
    holdout_order = config["args"]["holdouts"].split(",")
    methods = ["msp", "energy", "deep_knn"]
    tracks = ["neg1c", "neg2"]
    assert verification["status"] == independent_verification["status"] == "complete"
    assert verification["n_pairs_completed"] == verification["n_pairs_expected"] == 72
    assert independent_verification["n_pairs_completed"] == independent_verification["n_pairs_expected"] == 72
    assert verification["run_id"] == config["run_id"] == RUNS["fig5"][0]
    assert verification["source_run_ids"] == RUNS["fig5"][1:]
    assert verification["train_fold_policy"] == independent_verification["train_fold_policy"] == "assigned-folds-only-v1"
    assert independent_verification["metric_version"] == 2 and independent_verification["pass2_independent_vector_recalculation"]
    assert independent_verification["pass3_independent_rank_loop_and_history_replay"]
    assert verification["cross_seed_window_npz_alignment"]["comparisons"] == 48
    assert not verification["test_used_for_selection"] and verification["missing"] == []
    assert len(raw) == len(independent) == 216 and len(convergence) == 72
    assert raw.groupby(["holdout", "track", "seed"]).ngroups == 72
    assert set(raw["holdout"]) == set(holdout_order) and set(raw["track"]) == set(tracks)
    assert set(raw["seed"]) == {0, 1, 2} and set(raw["method"]) == set(methods)
    assert raw["metric_version"].eq(2).all() and raw["stop_reason"].eq("validation_early_stop").all()
    assert raw["max_epochs_failsafe"].eq(1000).all() and convergence["stop_reason"].eq("validation_early_stop").all()
    assert (convergence["epochs_run"] - convergence["best_epoch"] == 20).all()

    keys = ["seed", "track", "holdout", "method"]
    aligned = raw[keys + metric_names].merge(independent, on=keys, suffixes=("_published", "_independent"), validate="one_to_one")
    assert len(aligned) == 216 and independent["manual_passed"].all()
    for metric in metric_names:
        assert np.allclose(aligned[f"{metric}_published"], aligned[f"{metric}_independent"], rtol=0, atol=1e-12)

    per_seed = raw.groupby(["seed", "track", "method"], as_index=False)[metric_names].mean()
    published_seed = per_seed_published.set_index(["seed", "track", "method"])[metric_names].sort_index()
    assert np.allclose(per_seed.set_index(["seed", "track", "method"])[metric_names].sort_index(), published_seed, rtol=0, atol=1e-12)
    multiseed = per_seed.groupby(["track", "method"])[metric_names].agg(["mean", "std"])
    published_multi = multiseed_published.set_index(["track", "method"])
    for metric in metric_names:
        assert np.allclose(multiseed[(metric, "mean")], published_multi[f"{metric}_mean"], rtol=0, atol=1e-12)
        assert np.allclose(multiseed[(metric, "std")], published_multi[f"{metric}_std"], rtol=0, atol=1e-12)
    assert multiseed_published["n_seeds"].eq(3).all()

    rbp_means = raw.groupby(["track", "holdout", "method"])[metric_names].mean()
    rbp_sd = raw.groupby(["track", "holdout", "method"])[metric_names].std()
    conflict_records = []
    effect_records = []
    for track in tracks:
        for holdout in holdout_order:
            values = rbp_means.loc[(track, holdout)]
            winners = {
                "auroc_ood": values["auroc_ood"].idxmax(),
                "fpr95": values["fpr95"].idxmin(),
                "aurc": values["aurc"].idxmin(),
            }
            conflict = len(set(winners.values())) > 1
            conflict_records.append({"track": track, "holdout": holdout, **{f"{metric}_winner": winner for metric, winner in winners.items()}, "winner_conflict": conflict})
            effect_records.append(
                {
                    "track": track,
                    "holdout": holdout,
                    "auroc_ood_delta": values.loc["deep_knn", "auroc_ood"] - values.loc["msp", "auroc_ood"],
                    "fpr95_delta": values.loc["msp", "fpr95"] - values.loc["deep_knn", "fpr95"],
                    "aurc_delta": values.loc["msp", "aurc"] - values.loc["deep_knn", "aurc"],
                    "winner_conflict": conflict,
                    **{f"{metric}_winner": winner for metric, winner in winners.items()},
                }
            )
    conflicts = pd.DataFrame(conflict_records)
    effects = pd.DataFrame(effect_records)
    assert len(conflicts) == 24 and conflicts["winner_conflict"].sum() == 23
    no_conflict = conflicts[~conflicts["winner_conflict"]]
    assert len(no_conflict) == 1 and no_conflict.iloc[0]["track"] == "neg1c" and no_conflict.iloc[0]["holdout"] == "IGF2BP1"
    assert no_conflict.iloc[0][["auroc_ood_winner", "fpr95_winner", "aurc_winner"]].eq("energy").all()
    macro_conflicts = {}
    for track in tracks:
        values = multiseed.loc[track]
        macro_conflicts[track] = {
            "auroc_ood": values["auroc_ood"]["mean"].idxmax(),
            "fpr95": values["fpr95"]["mean"].idxmin(),
            "aurc": values["aurc"]["mean"].idxmin(),
        }
        published = conflicts_published[conflicts_published["track"] == track].iloc[0]
        assert published["winner_conflict"]
        for metric in metric_names:
            assert published[f"{metric}_winner"] == macro_conflicts[track][metric]
            assert np.isclose(published[f"{metric}_value"], multiseed.loc[(track, macro_conflicts[track][metric]), (metric, "mean")], rtol=0, atol=1e-12)

    method_styles = {"msp": (BLUE, "o", "MSP"), "energy": (TEAL, "D", "Energy"), "deep_knn": (ORANGE, "s", "deep-kNN")}
    metric_titles = {"auroc_ood": "AUROC-OOD ↑ higher is better", "fpr95": "FPR95 ↓ lower is better", "aurc": "AURC ↓ lower is better"}
    fig = plt.figure(figsize=(14.2, 11.0))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.18], height_ratios=[0.88, 1.55], wspace=0.48, hspace=0.32)
    for metric_index, metric in enumerate(metric_names):
        ax = fig.add_subplot(grid[0, metric_index])
        for method_index, method in enumerate(methods):
            color, marker, label = method_styles[method]
            offset = (method_index - 1) * 0.20
            for track_index, track in enumerate(tracks):
                subset = per_seed[(per_seed["track"] == track) & (per_seed["method"] == method)].sort_values("seed")
                center = track_index + offset
                ax.scatter(center + np.linspace(-0.035, 0.035, 3), subset[metric], s=20, color=color, marker=marker, alpha=0.60, zorder=3)
                mean = multiseed.loc[(track, method), (metric, "mean")]
                sd = multiseed.loc[(track, method), (metric, "std")]
                ax.errorbar(center, mean, yerr=sd, fmt=marker, ms=6, color=color, lw=1.5, capsize=3, label=label if track_index == 0 and metric_index == 0 else None, zorder=4)
        if metric == "auroc_ood":
            ax.axhline(0.5, color=MID, lw=0.8, ls="--")
            ax.set_ylim(0.42, 0.525)
            ax.legend(loc="lower left", fontsize=7, ncol=3, columnspacing=0.8)
        elif metric == "fpr95":
            ax.set_ylim(0.935, 0.970)
        else:
            ax.set_ylim(0.13, 0.51)
        ax.set(xlim=(-0.48, 1.48), xticks=[0, 1], xticklabels=tracks, ylabel=metric.upper().replace("_OOD", "-OOD"))
        panel(ax, chr(97 + metric_index), metric_titles[metric])
        ax.text(0.5, -0.18, "points: seeds; bars: mean ± sample SD", transform=ax.transAxes, ha="center", fontsize=7, color=MID)

    ax = fig.add_subplot(grid[1, :3])
    effect_columns = ["auroc_ood_delta", "fpr95_delta", "aurc_delta"]
    matrix = effects[effect_columns].to_numpy()
    limit = float(np.abs(matrix).max())
    cmap = LinearSegmentedColormap.from_list("ood_effect", [ORANGE, "#FFFFFF", BLUE])
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    for row in range(len(effects)):
        for column in range(3):
            value = matrix[row, column]
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=6.8, color="white" if abs(value) > 0.60 * limit else INK)
        ax.text(3, row, "Yes" if effects.iloc[row]["winner_conflict"] else "No", ha="center", va="center", fontsize=7.2, color=ORANGE if effects.iloc[row]["winner_conflict"] else TEAL, fontweight="bold")
    labels = [f"{row['track']} · {row['holdout']}" for _, row in effects.iterrows()]
    ax.set_yticks(np.arange(len(effects)), labels, fontsize=7.3)
    ax.set_xticks([0, 1, 2, 3], ["AUROC-OOD ↑\nΔ=kNN−MSP", "FPR95 ↓\nΔ=MSP−kNN", "AURC ↓\nΔ=MSP−kNN", "Winner\nconflict"])
    ax.set_xlim(-0.5, 3.5)
    ax.axhline(11.5, color=INK, lw=1.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar_axis = ax.inset_axes([1.025, 0.08, 0.022, 0.84])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Direction-aligned Δ; positive favors deep-kNN", fontsize=8)
    panel(ax, "d", "24 RBP×track effects: 23/24 groups have metric-winner conflicts")

    ax = fig.add_subplot(grid[:, 3])
    for metric_index, metric in enumerate(metric_names):
        y = len(metric_names) - 1 - metric_index
        for method_index, method in enumerate(methods):
            color, marker, label = method_styles[method]
            values = rbp_sd.xs(method, level="method")[metric].to_numpy()
            pos = y + (method_index - 1) * 0.20
            ax.scatter(values, np.full(len(values), pos), s=18, color=color, marker=marker, alpha=0.38, edgecolors="none")
            median = float(np.median(values))
            ax.plot([median, median], [pos - 0.105, pos + 0.105], color=color, lw=2.2)
    ax.set_xscale("log")
    ax.set(xlim=(1e-4, 5e-2), ylim=(-0.55, 2.55), yticks=[2, 1, 0], yticklabels=["AUROC-OOD", "FPR95", "AURC"], xlabel="Within-RBP SD across 3 seeds (log scale) ↓")
    panel(ax, "e", "Seed stability varies across heldout RBPs")
    ax.text(0.5, -0.09, "points: 24 RBP×track groups per method; bars: median", transform=ax.transAxes, ha="center", fontsize=7.2, color=MID)
    ax.text(0.5, -0.145, "Descriptive only; no bootstrap or unified winner", transform=ax.transAxes, ha="center", fontsize=8, color=ORANGE, fontweight="bold")

    fig.subplots_adjust(left=0.065, right=0.965, bottom=0.08, top=0.94, wspace=0.48, hspace=0.38)
    stem = "Fig5_E5_whole_RBP_holdout_three_seed_OOD"
    save_figure(fig, stem, layout="manual")
    plot_files = write_plot_data(
        "fig5",
        {
            "holdout_metrics_216": raw,
            "macro_by_seed": per_seed,
            "macro_three_seed": multiseed_published,
            "rbp_direction_aligned_effects": effects,
            "seed_stability": rbp_sd.reset_index(),
        },
    )
    return {
        "evidence_level": "ACCEPTED E5 whole-RBP holdout pilot fold0 test; 12 RBPs × 2 tracks × 3 seeds; descriptive only; no unified winner or significance claim",
        "sources": [source(p) for p in [raw_path, per_seed_path, multiseed_path, conflicts_path, convergence_path, independent_path, verification_path, independent_verification_path, config_path, E5_HOLDOUT_REPORT]],
        "pass1_schema_finite": {"raw_rows": len(raw), "independent_rows": len(independent), "pairs": 72, "rbp_track_groups": 24, "methods": methods, "convergence_rows": len(convergence)},
        "pass2_configuration_alignment": {"data_version": config["args"]["data_version"], "train_fold_policy": verification["train_fold_policy"], "metric_version": 2, "seeds": [0, 1, 2], "tracks": tracks, "holdouts": holdout_order, "cross_seed_alignment_comparisons": 48, "test_used_for_selection": False, "all_validation_early_stop": True},
        "pass3_report_recalculation": {"independent_window_metrics_exact_match_216": True, "per_seed_macro_exact_match_18": True, "three_seed_mean_sample_sd_exact_match_6": True, "winner_conflicts": 23, "rbp_track_groups": 24, "sole_no_conflict_group": "neg1c/IGF2BP1 (energy)", "macro_winners": macro_conflicts, "unified_winner": False},
        "plot_data": plot_files,
        "figure_stem": stem,
    }


def figure6() -> dict:
    runs_path = RNACOMPETE / "final_independent_run_metrics.tsv.gz"
    summary_path = RNACOMPETE / "final_independent_summary.tsv"
    verification_path = RNACOMPETE / "final_independent_verification.json"
    gate_path = RNACOMPETE / "final_gate.json"
    metrics = ["spearman", "pearson", "mse"]
    keys = ["track", "arch", "probe_set", "tier"]
    runs = read_tsv(
        runs_path,
        [*keys, "fold", "seed", "n_experiments", *metrics, "group_spearman", "group_pearson", "group_mse", "n_groups"],
        ["fold", "n_experiments", *metrics],
    )
    published = read_tsv(summary_path, [*keys, "n_runs", "spearman_mean", "spearman_sd", "pearson_mean", "pearson_sd", "mse_mean", "mse_sd"], ["n_runs", "spearman_mean", "spearman_sd", "pearson_mean", "pearson_sd", "mse_mean", "mse_sd"])
    verification = read_json(verification_path)
    gate = read_json(gate_path)
    assert verification["status"] == "passed" and verification["scope"] == "independent_five_fold_three_seed_aggregate"
    assert verification["n_evaluations"] == 70 and verification["n_neural_checkpoint_replays"] == 60
    assert verification["n_run_metric_rows"] == 525 and verification["n_summary_rows"] == 45
    assert not verification["selection_test_used"]
    assert sha256(gate_path) == verification["final_gate_sha256"] == "83ec4cfef1c25c91839e8e51d40dc34272bb901f76d681d61ad6c199b41fe15b"
    assert sha256(runs_path) == verification["run_metrics_sha256"] == "a276339a28b02a862a24c1edd48fed9cc3115c863a6b6f6e7fd3143acabba05c"
    assert sha256(summary_path) == verification["summary_sha256"] == "71a4b708120d8831acc7362da49a2a2f9159e4e486ad1cd82a4c610890b546f1"
    assert gate["status"] == "passed" and gate["scope"] == "five_fold_three_seed"
    assert gate["n_evaluations"] == 70 and len(gate["runs"]) == 70
    assert gate["selection_audit"]["test_used_for_selection"] is False
    assert all(record["replay_passed"] and record.get("checkpoint_replay_passed", True) for record in gate["runs"])
    assert len(runs) == 525 and len(published) == 45 and set(runs["fold"]) == set(range(5))
    assert set(runs["track"]) == {"sequence", "rbp_ood"} and set(runs["arch"]) == {"linear", "mlp", "knn"}

    calculated_records = []
    for group_keys, frame in runs.groupby(keys):
        row = dict(zip(keys, group_keys))
        row["n_runs"] = len(frame)
        expected_runs = 5 if group_keys[1] == "knn" else 15
        assert len(frame) == expected_runs and set(frame["fold"]) == set(range(5))
        fold_counts = frame.groupby("fold").size()
        assert (fold_counts == (1 if group_keys[1] == "knn" else 3)).all()
        if group_keys[0] == "sequence":
            assert frame["n_experiments"].nunique() == 1
            row["n_experiments"] = int(frame["n_experiments"].iloc[0])
        else:
            row["n_experiments_min"] = int(frame["n_experiments"].min())
            row["n_experiments_max"] = int(frame["n_experiments"].max())
            row["n_groups_min"] = int(frame["n_groups"].min())
            row["n_groups_max"] = int(frame["n_groups"].max())
        for metric in metrics:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_sd"] = frame[metric].std(ddof=1)
        if group_keys[0] == "rbp_ood":
            for metric in ["group_spearman", "group_pearson", "group_mse"]:
                row[f"{metric}_mean"] = frame[metric].mean()
                row[f"{metric}_sd"] = frame[metric].std(ddof=1)
        calculated_records.append(row)
    calculated = pd.DataFrame(calculated_records).sort_values(keys).reset_index(drop=True)
    published_index = published.set_index(keys)
    assert len(calculated) == 45 and not calculated.duplicated(keys).any()
    for _, row in calculated.iterrows():
        index = tuple(row[key] for key in keys)
        expected = published_index.loc[index]
        assert row["n_runs"] == expected["n_runs"]
        for metric in metrics:
            assert np.isclose(row[f"{metric}_mean"], expected[f"{metric}_mean"], rtol=0, atol=5e-8)
            assert np.isclose(row[f"{metric}_sd"], expected[f"{metric}_sd"], rtol=0, atol=5e-8)
        if row["track"] == "sequence":
            assert row["n_experiments"] == expected["n_experiments"]
        else:
            for field in ["n_experiments_min", "n_experiments_max", "n_groups_min", "n_groups_max"]:
                assert row[field] == expected[field]
            for metric in ["group_spearman", "group_pearson", "group_mse"]:
                assert np.isclose(row[f"{metric}_mean"], expected[f"{metric}_mean"], rtol=0, atol=5e-8)
                assert np.isclose(row[f"{metric}_sd"], expected[f"{metric}_sd"], rtol=0, atol=5e-8)
    sequence_overall = calculated[(calculated["track"] == "sequence") & (calculated["probe_set"] == "All") & (calculated["tier"] == "Overall")]
    sequence_tail = calculated[(calculated["track"] == "sequence") & (calculated["probe_set"] == "All") & (calculated["tier"] == "tail")]
    assert sequence_overall["n_experiments"].eq(244).all() and sequence_tail["n_experiments"].eq(80).all()
    assert calculated[calculated["track"] == "rbp_ood"]["n_experiments_min"].eq(48).all()
    assert calculated[calculated["track"] == "rbp_ood"]["n_experiments_max"].eq(49).all()
    assert calculated[calculated["track"] == "rbp_ood"]["n_groups_min"].eq(40).all()
    assert calculated[calculated["track"] == "rbp_ood"]["n_groups_max"].eq(42).all()

    effect_records = []
    for tier in ["Overall", "tail"]:
        subset = runs[(runs["track"] == "sequence") & (runs["probe_set"] == "All") & (runs["tier"] == tier)]
        mlp = subset[subset["arch"] == "mlp"].set_index(["fold", "seed"])
        linear = subset[subset["arch"] == "linear"].set_index(["fold", "seed"])
        knn = subset[subset["arch"] == "knn"].set_index("fold")
        for metric in ["spearman", "pearson"]:
            neural_delta = mlp[metric] - linear[metric]
            knn_delta = mlp[metric].groupby("fold").mean() - knn[metric]
            assert len(neural_delta) == 15 and len(knn_delta) == 5
            for comparator, values in [("linear", neural_delta), ("knn", knn_delta)]:
                effect_records.append({"tier": tier, "metric": metric, "effect": f"mlp_minus_{comparator}", "delta_mean": values.mean(), "delta_sd": values.std(ddof=1), "delta_min": values.min(), "delta_max": values.max(), "n_comparisons": len(values)})
    effects = pd.DataFrame(effect_records)
    assert effects["delta_mean"].gt(0).all()
    summary_index = calculated.set_index(keys)
    rbp = calculated[(calculated["track"] == "rbp_ood") & (calculated["probe_set"] == "All")].set_index("arch")
    for metric in ["spearman_mean", "pearson_mean", "group_spearman_mean"]:
        assert rbp.loc["knn", metric] > rbp.loc["mlp", metric] > rbp.loc["linear", metric]
    assert rbp["spearman_mean"].max() < 0.20 and rbp["group_spearman_mean"].max() < 0.18

    styles = {"linear": (MID, "^", "Linear"), "mlp": (BLUE, "o", "MLP"), "knn": (ORANGE, "s", "kNN")}
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.7))
    for column, tier in enumerate(["Overall", "tail"]):
        ax = axes[0, column]
        subset = runs[(runs["track"] == "sequence") & (runs["probe_set"] == "All") & (runs["tier"] == tier)]
        for arch_index, arch in enumerate(["linear", "mlp", "knn"]):
            color, marker, label = styles[arch]
            for metric_index, metric in enumerate(["spearman", "pearson"]):
                values = subset[subset["arch"] == arch][metric].to_numpy()
                center = metric_index + (arch_index - 1) * 0.22
                ax.scatter(center + np.linspace(-0.035, 0.035, len(values)), values, s=14, color=color, marker=marker, alpha=0.38, edgecolors="none")
                mean = summary_index.loc[("sequence", arch, "All", tier), f"{metric}_mean"]
                sd = summary_index.loc[("sequence", arch, "All", tier), f"{metric}_sd"]
                ax.errorbar(center, mean, yerr=sd, fmt=marker, ms=6, color=color, lw=1.5, capsize=3, label=label if metric_index == 0 and column == 0 else None)
        for metric_index, metric in enumerate(["spearman", "pearson"]):
            deltas = effects[(effects["tier"] == tier) & (effects["metric"] == metric)].set_index("effect")["delta_mean"]
            ax.text(metric_index, 0.505, f"MLP−linear {deltas['mlp_minus_linear']:+.3f}\nMLP−kNN {deltas['mlp_minus_knn']:+.3f}", ha="center", va="top", fontsize=7.5, fontweight="bold")
        ax.set(xlim=(-0.52, 1.52), ylim=(0.20, 0.515), xticks=[0, 1], xticklabels=["Spearman", "Pearson"], ylabel="Correlation")
        panel(ax, "a" if column == 0 else "b", f"Sequence {tier}: MLP advantage over both baselines")
        if column == 0:
            ax.legend(loc="lower left", fontsize=7, ncol=3, columnspacing=0.8)

    ax = axes[0, 2]
    rbp_runs = runs[(runs["track"] == "rbp_ood") & (runs["probe_set"] == "All")]
    rbp_metrics = [("spearman", "Experiment\nSpearman", "spearman_mean", "spearman_sd"), ("pearson", "Experiment\nPearson", "pearson_mean", "pearson_sd"), ("group_spearman", "Gene-group\nSpearman", "group_spearman_mean", "group_spearman_sd")]
    for arch_index, arch in enumerate(["linear", "mlp", "knn"]):
        color, marker, label = styles[arch]
        for metric_index, (run_metric, _, mean_field, sd_field) in enumerate(rbp_metrics):
            values = rbp_runs[rbp_runs["arch"] == arch][run_metric].to_numpy()
            center = metric_index + (arch_index - 1) * 0.22
            ax.scatter(center + np.linspace(-0.035, 0.035, len(values)), values, s=14, color=color, marker=marker, alpha=0.38, edgecolors="none")
            ax.errorbar(center, rbp.loc[arch, mean_field], yerr=rbp.loc[arch, sd_field], fmt=marker, ms=6, color=color, lw=1.5, capsize=3)
    ax.text(0.97, 0.04, "relative gain ≠ reliable OOD", transform=ax.transAxes, ha="right", fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.set(xlim=(-0.52, 2.52), ylim=(0, 0.235), xticks=[0, 1, 2], xticklabels=[item[1] for item in rbp_metrics], ylabel="Correlation")
    panel(ax, "c", "RBP-OOD: kNN relative gain, low absolute performance")

    def fold_panel(ax: plt.Axes, track: str, tier: str, letter: str, title: str, ylim: tuple[float, float]) -> None:
        subset = runs[(runs["track"] == track) & (runs["probe_set"] == "All") & (runs["tier"] == tier)]
        for arch_index, arch in enumerate(["linear", "mlp", "knn"]):
            color, marker, _ = styles[arch]
            offset = (arch_index - 1) * 0.20
            for metric in ["spearman", "pearson"]:
                values = subset[subset["arch"] == arch]
                centers = values["fold"].to_numpy() + offset
                jitter = np.linspace(-0.025, 0.025, len(values))
                ax.scatter(centers + jitter, values[metric], s=10, marker=marker, facecolors=color if metric == "spearman" else "none", edgecolors=color, linewidths=0.45, alpha=0.48)
                fold_means = values.groupby("fold")[metric].mean()
                ax.plot(fold_means.index + offset, fold_means, color=color, lw=1.25, ls="-" if metric == "spearman" else ":", marker=marker, ms=3.8)
        ax.plot([], [], color=INK, lw=1.25, ls="-", label="Spearman")
        ax.plot([], [], color=INK, lw=1.25, ls=":", label="Pearson")
        ax.set(xticks=range(5), xlabel="Fold", ylabel="Correlation", ylim=ylim)
        panel(ax, letter, title)
        ax.legend(loc="lower left", fontsize=7, ncol=2)

    fold_panel(axes[1, 0], "sequence", "Overall", "d", "Sequence Overall: fold/seed distribution", (0.20, 0.52))
    fold_panel(axes[1, 1], "sequence", "tail", "e", "Sequence tail: fold/seed distribution", (0.20, 0.52))
    fold_panel(axes[1, 2], "rbp_ood", "Overall", "f", "RBP-OOD: fold/seed distribution", (0, 0.31))
    stem = "Fig6_RNAcompete_Ray2013_five_fold_sequence_and_RBP_OOD"
    save_figure(fig, stem)
    fold_distribution = runs[(runs["probe_set"] == "All") & (((runs["track"] == "sequence") & runs["tier"].isin(["Overall", "tail"])) | ((runs["track"] == "rbp_ood") & (runs["tier"] == "Overall")))].copy()
    plot_files = write_plot_data("fig6", {"run_metrics_525": runs, "summary_45_recalculated": calculated, "sequence_effects": effects, "fold_seed_distribution": fold_distribution})
    return {
        "evidence_level": "ACCEPTED RNAcompete Ray2013 frozen RNA-FM five-fold; sequence ID90 and whole-RBP-OOD; descriptive only; low absolute OOD performance",
        "sources": [source(p) for p in [runs_path, summary_path, verification_path, gate_path, RNACOMPETE_REPORT]],
        "pass1_schema_finite": {"run_rows": len(runs), "summary_rows": len(calculated), "evaluations": 70, "neural_checkpoint_replays": 60, "source_sha256_match": True},
        "pass2_configuration_alignment": {"folds": [0, 1, 2, 3, 4], "neural_seeds": [0, 1, 2], "neural_runs_per_group": 15, "knn_runs_per_group": 5, "selection_test_used": False, "sequence_all_experiments": 244, "sequence_tail_experiments": 80, "rbp_experiments_per_fold": [48, 49], "rbp_groups_per_fold": [40, 42]},
        "pass3_report_recalculation": {"all_45_summary_rows_recalculated": True, "all_45_groups_fold0_4_support_consistent": True, "sequence_effects": effects.to_dict(orient="records"), "rbp_knn_relative_spearman_advantage_over_mlp": float(rbp.loc["knn", "spearman_mean"] - rbp.loc["mlp", "spearman_mean"]), "rbp_max_experiment_spearman": float(rbp["spearman_mean"].max()), "reliable_ood_claim": False, "significance_claim": False},
        "plot_data": plot_files,
        "figure_stem": stem,
    }


def make_contact_sheets(stems: list[str]) -> None:
    images = []
    for stem in stems:
        path = FIGURE_ROOT / f"{stem}_300dpi.png"
        image = Image.open(path).convert("RGB")
        image.thumbnail((1500, 850), Image.Resampling.LANCZOS)
        images.append((stem, image.copy()))
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    title_font = ImageFont.truetype(font_path, 34)
    label_font = ImageFont.truetype(font_path, 24)
    margin, gap, label_h = 42, 34, 58
    cell_w = max(image.width for _, image in images)
    cell_h = max(image.height for _, image in images) + label_h
    rows = (len(images) + 1) // 2
    sheet = Image.new("RGB", (margin * 2 + cell_w * 2 + gap, margin * 2 + cell_h * rows + gap * (rows - 1) + 64), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 22), "RARE-Seq | publication figure contact sheet", fill=INK, font=title_font)
    for index, (stem, image) in enumerate(images):
        col, row = index % 2, index // 2
        x = margin + col * (cell_w + gap)
        y = margin + 64 + row * (cell_h + gap)
        draw.text((x, y), chr(97 + index) + " | " + stem, fill=INK, font=label_font)
        sheet.paste(image, (x, y + label_h))
    sheet.save(FIGURE_ROOT / "contact_sheet.png", dpi=(200, 200))
    sheet.convert("L").save(FIGURE_ROOT / "contact_sheet_grayscale.png", dpi=(200, 200))

    reference_dir = FIGURE_ROOT / "_work" / "gallery_refs"
    reference_ids = ["oa-tech-pone-0258309-f05", "oa-bio-plos-pbio-3002856-f06", "oa-geo-hess-30-629-2026-f12"]
    refs = []
    for reference_id in reference_ids:
        path = reference_dir / f"{reference_id}.png"
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((900, 620), Image.Resampling.LANCZOS)
            refs.append((reference_id, image.copy()))
    if refs:
        chosen = images[0][1].copy()
        chosen.thumbnail((900, 620), Image.Resampling.LANCZOS)
        refs.append(("Rendition: Fig1_E0_data_and_leakage_audit", chosen))
        w = max(image.width for _, image in refs)
        h = max(image.height for _, image in refs) + 56
        comparison = Image.new("RGB", (margin * 2 + 2 * w + gap, margin * 2 + 2 * h + gap + 64), "white")
        draw = ImageDraw.Draw(comparison)
        draw.text((margin, 22), "Gallery references and rendition", fill=INK, font=title_font)
        for index, (label, image) in enumerate(refs):
            col, row = index % 2, index // 2
            x = margin + col * (w + gap)
            y = margin + 64 + row * (h + gap)
            draw.text((x, y), label, fill=INK, font=label_font)
            comparison.paste(image, (x, y + 48))
        comparison.save(FIGURE_ROOT / "gallery_comparison.png", dpi=(200, 200))


def validate_rendered_outputs() -> dict:
    expected = {
        "Fig1_E0_data_and_leakage_audit": ["Current E1 strict3train contract", "146 experiments accepted"],
        "Fig2_E1_main_strict3train_five_fold_three_seed": ["Accepted 146-experiment main set", "CV by fold", "neg2"],
        "Fig3_E5_PARCLIP_cross_protocol": ["Three-seed macro AUROC", "PAR-CLIP matched neg2"],
        "Fig4_E6_GRIMM_five_split_frequency_and_OOD_v2": ["all 20 split", "FPR95", "ascending OOD score"],
        "Fig5_E5_whole_RBP_holdout_three_seed_OOD": ["23/24 groups have metric-winner conflicts", "Seed stability", "no bootstrap or unified winner"],
        "Fig6_RNAcompete_Ray2013_five_fold_sequence_and_RBP_OOD": ["Sequence Overall", "Sequence tail", "relative gain ≠ reliable OOD"],
    }
    records = {}
    for stem, tokens in expected.items():
        paths = {suffix: FIGURE_ROOT / f"{stem}{suffix}" for suffix in ["_300dpi.png", "_600dpi.png", ".pdf", ".svg"]}
        assert all(path.is_file() and path.stat().st_size > 1000 for path in paths.values())
        image300 = Image.open(paths["_300dpi.png"])
        image600 = Image.open(paths["_600dpi.png"])
        assert abs(image300.info["dpi"][0] - 300) < 1 and abs(image600.info["dpi"][0] - 600) < 1
        ratio = image600.width / image300.width
        assert 1.99 < ratio < 2.01
        ET.parse(paths[".svg"])
        info = subprocess.check_output(["pdfinfo", str(paths[".pdf"])], text=True)
        assert "Pages:           1" in info
        text = subprocess.check_output(["pdftotext", str(paths[".pdf"]), "-"], text=True)
        assert all(token in text for token in tokens)
        records[stem] = {"png300_size": list(image300.size), "png600_size": list(image600.size), "pdf_pages": 1, "svg_xml": True, "required_pdf_text": tokens}

    archived = AUDIT_HISTORY / "Fig2_E1_pilot_five_fold_baselines.pdf"
    archived_text = subprocess.check_output(["pdftotext", str(archived), "-"], text=True)
    assert "HISTORICAL LOADER GAP" in archived_text and "fold=-1 positives" in archived_text
    assert not list(FIGURE_ROOT.glob("Fig2_E1_pilot_five_fold_baselines*"))
    assert not list(FIGURE_ROOT.glob("Fig4_E6_GRIMM_split1_frequency_and_OOD*"))
    color = Image.open(FIGURE_ROOT / "contact_sheet.png")
    gray = Image.open(FIGURE_ROOT / "contact_sheet_grayscale.png")
    gray_array = np.asarray(gray)
    assert color.size == gray.size and gray.mode == "L"
    assert gray_array.min() < 40 and gray_array.max() > 250 and gray_array.std() > 20 and len(np.unique(gray_array)) > 100
    result = {
        "status": "passed",
        "figures": records,
        "archived_fig2_warning_verified": True,
        "contact_sheet": {"size": list(color.size), "grayscale_mode": gray.mode, "grayscale_std": float(gray_array.std()), "grayscale_unique_values": len(np.unique(gray_array)), "fig2_included": True, "fig5_included": True, "fig6_included": True},
    }
    (VERIFY_ROOT / "output_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", "fig1", "fig2", "fig3", "fig4", "fig5", "fig6"], default="all")
    parser.add_argument("--contact-sheet", action="store_true")
    args = parser.parse_args()
    functions = {"fig1": figure1, "fig2": figure2, "fig3": figure3, "fig4": figure4, "fig5": figure5, "fig6": figure6}
    selected = list(functions) if args.figure == "all" else [args.figure]
    audits = {}
    for name in selected:
        audit = functions[name]()
        write_audit(name, audit)
        audits[name] = audit
        print(f"{name}: rendered and verified {audit['figure_stem']}")
    if args.contact_sheet:
        make_contact_sheets([audits[name]["figure_stem"] for name in selected])
        print("contact sheets: rendered; selected current figures included")
    if args.figure == "all":
        if args.contact_sheet:
            validate_rendered_outputs()
            print("output validation: numeric/vector/DPI/grayscale checks passed")
        summary = {
            "status": "rendering_and_numeric_verification_passed",
            "submission_status": {"fig1": "descriptive_current_contract_accepted", "fig2": "accepted_strict3train_main", "fig3": "external_descriptive", "fig4": "five_split_descriptive", "fig5": "accepted_whole_rbp_holdout_descriptive", "fig6": "accepted_rnacompete_five_fold_descriptive"},
            "figures": {name: {"run_ids": RUNS[name], "evidence_level": audits[name]["evidence_level"], "figure_stem": audits[name]["figure_stem"]} for name in selected},
            "audit_history": {"fig2": {"run_ids": ARCHIVED_FIG2_RUNS, "evidence_level": "Historical E1 pilot with fold=-1 training-use gap; retained only for audit", "figure_stem": "Fig2_E1_pilot_five_fold_baselines", "excluded_from_contact_sheet": True}},
        }
        (VERIFY_ROOT / "verification_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
