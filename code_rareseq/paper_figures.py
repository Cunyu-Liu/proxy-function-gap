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
E1 = STAGE_ROOT / "E1_强基线"
E6 = PROJECT_ROOT / "E6_蛋白迁移"
E5_REPORT = PROJECT_ROOT / "E5_OOD" / "PAR-CLIP跨协议评测报告_2026-11-25.md"

RUNS = {
    "fig1": [
        "E0-leakage-quant-20261122-102238-ZDLg6Y",
        "E0-window-identity-20261122-104517-ZDLg6Y",
    ],
    "fig2": [
        "E1-cv-prediction-combine-20261123-085050-ZDLg6Y",
        "E1-cv-summary-20261123-085240-ZDLg6Y",
        "E1-paired-bootstrap-20261123-085242-ZDLg6Y",
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
}

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


def save_figure(fig: plt.Figure, stem: str, output_root: Path = FIGURE_ROOT) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
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
        "Legacy E1 loader gap: fold=-1 positives\nentered every training fold; foldfix pending",
        ha="right",
        va="bottom",
        fontsize=8,
        color=ORANGE,
        fontweight="bold",
        bbox={"facecolor": "#FFF4E8", "edgecolor": ORANGE, "boxstyle": "round,pad=0.35"},
    )
    ax.text(
        0.98,
        38,
        f"Legacy cluster check* = {cluster_leak:.4%}\nGlobal $n_{{eff}}$/windows = {global_neff:.3f}",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.set(xlim=(0, 1), ylim=(0, 105), xlabel="Nearest training identity ($d_{hom}$)", ylabel="Test windows, cumulative (%)")
    panel(ax, "c", "Split design and audit reveal a training-use gap")

    stem = "Fig1_E0_data_and_leakage_audit"
    save_figure(fig, stem)
    cdf_frame = pd.DataFrame({"identity_threshold": thresholds, "cumulative_fraction": homology_cdf})
    plot_files = write_plot_data("fig1", {"rbp_support": per_rbp, "dual_cell_overlap": dual, "identity_cdf": cdf_frame})
    return {
        "evidence_level": "E0 design and audit; descriptive; legacy E1 fold=-1 training-use gap disclosed",
        "sources": [source(p) for p in [positives_path, dual_path, identity_path, leakage_path, manifest_path, distribution_path]],
        "pass1_schema_finite": True,
        "pass2_configuration_alignment": {"identity": 0.9, "split_seed": 20261122, "folds": 5, "negative_seed": 42, "legacy_loader_fold_minus_one_training_gap": True, "strict_zero_leakage_claim": False, "legacy_cluster_check_caveat": "includes validation and counts each positive in two tracks; not a strict three-fold train rate"},
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
    bootstrap_path = E1 / "results" / "cv_all_bootstrap" / "paired_bootstrap.tsv"
    folds_path = E1 / "results" / "cv_all_summary" / "cv_by_fold.tsv"
    config_path = E1 / "results" / "pilot_f0_modern_bce_s0" / "modern_config.json"
    log_path = STAGE_ROOT / "实验日志" / "experiments.jsonl"
    bootstrap = read_tsv(
        bootstrap_path,
        ["track", "slice", "metric", "n_experiments", "n_seeds", "n_boot", "observed_delta", "ci_low", "ci_high"],
        ["observed_delta", "ci_low", "ci_high", "p_two_sided"],
    )
    folds = read_tsv(
        folds_path,
        ["fold", "track", "split", "tier", "auroc_modern", "auroc_knn", "d_auroc"],
        ["auroc_modern", "auroc_knn", "d_auroc"],
    )
    config = read_json(config_path)
    run = completed_run(log_path, RUNS["fig2"][-1])
    assert len(bootstrap) == 8 and set(bootstrap["n_seeds"]) == {3} and set(bootstrap["n_boot"]) == {10000}
    assert run["config"]["folds"] if "folds" in run["config"] else True
    assert run["config"]["n_boot"] == 10000 and run["config"]["seed"] == 20261123
    assert config["data_version"] == "e0-v1-id90-frozen" and config["seed"] == 0
    assert config["selection"] == "mean validation AUPRC over tail RBPs; test excluded"
    assert set(folds["fold"]) == set(range(5)) and set(folds["split"]) == {"test", "val"}
    tail = folds[(folds["split"] == "test") & (folds["tier"] == "tail")].copy()
    assert len(tail) == 10 and (tail["d_auroc"] > 0).all()
    assert np.allclose(tail["d_auroc"], tail["auroc_modern"] - tail["auroc_knn"])

    ci_recalculated = {}
    for track in ["neg1c", "neg2"]:
        for slice_name in ["tail", "head"]:
            npz_path = E1 / "results" / "cv_all_bootstrap" / f"bootstrap_{track}_{slice_name}.npz"
            deltas = np.load(npz_path)["deltas"]
            assert deltas.shape == (10000, 2) and np.isfinite(deltas).all()
            interval = np.quantile(deltas, [0.025, 0.975], axis=0)
            for index, metric in enumerate(["auroc", "auprc"]):
                row = bootstrap[(bootstrap["track"] == track) & (bootstrap["slice"] == slice_name) & (bootstrap["metric"] == metric)].iloc[0]
                assert np.allclose(interval[:, index], [row["ci_low"], row["ci_high"]], atol=1e-12)
                ci_recalculated[f"{track}_{slice_name}_{metric}"] = interval[:, index].tolist()

    fold_means = tail.groupby("track")["d_auroc"].mean()
    assert round(fold_means["neg1c"], 5) == 0.02446
    assert round(fold_means["neg2"], 5) == 0.02888

    fig = plt.figure(figsize=(12.2, 5.8))
    fig.suptitle(
        "HISTORICAL LOADER GAP: fold=-1 positives in every train fold | replace after foldfix",
        color=ORANGE,
        fontsize=12,
        fontweight="bold",
    )
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1], wspace=0.30, hspace=0.65)
    ax = fig.add_subplot(grid[:, 0])
    ordered = pd.concat(
        [
            bootstrap[(bootstrap["slice"] == slice_name) & (bootstrap["track"] == track) & (bootstrap["metric"] == metric)]
            for slice_name in ["tail", "head"]
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
        ax.scatter(row["observed_delta"], pos, s=52, marker=marker, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        ax.text(0.057, pos, f"{row['observed_delta']:.3f}  [{row['ci_low']:.3f}, {row['ci_high']:.3f}]", va="center", fontsize=7.7)
    ax.axvline(0, color=MID, lw=0.9, ls="--")
    labels = [f"{r['slice'].capitalize()} · {r['track']} · {r['metric'].upper()}" for _, r in ordered.iterrows()]
    ax.set_yticks(y, labels)
    ax.set(xlim=(0, 0.084), xlabel="Δ performance (BCE − cluster-deduplicated kNN), 95% CI")
    panel(ax, "a", "Frozen BCE improves head and tail performance")

    for row_index, track in enumerate(["neg1c", "neg2"]):
        ax = fig.add_subplot(grid[row_index, 1])
        subset = tail[tail["track"] == track].sort_values("fold")
        for _, record in subset.iterrows():
            ax.plot([0, 1], [record["auroc_knn"], record["auroc_modern"]], color=LIGHT, lw=1.25, zorder=1)
        ax.scatter(np.zeros(len(subset)), subset["auroc_knn"], s=42, color=ORANGE, label="kNN", zorder=3)
        ax.scatter(np.ones(len(subset)), subset["auroc_modern"], s=42, color=BLUE, label="BCE", zorder=3)
        delta_min = subset["d_auroc"].min()
        delta_max = subset["d_auroc"].max()
        ax.text(0.5, 0.05, f"5/5 positive\nΔ range {delta_min:.3f} to {delta_max:.3f}", transform=ax.transAxes, ha="center", fontsize=8, fontweight="bold")
        ax.set(xlim=(-0.30, 1.43), xticks=[0, 1], xticklabels=["kNN", "BCE"], ylabel="Tail AUROC")
        panel(ax, "b" if row_index == 0 else "c", f"Every fold favors BCE in the tail ({track})")

    stem = "Fig2_E1_pilot_five_fold_baselines"
    save_figure(fig, stem, AUDIT_HISTORY)
    plot_files = write_plot_data("fig2", {"paired_bootstrap": bootstrap, "tail_by_fold": tail})
    return {
        "evidence_level": "ARCHIVED historical E1 pilot; fold=-1 training-use gap; do not submit; replace after foldfix",
        "sources": [source(p) for p in [bootstrap_path, folds_path, config_path, log_path]],
        "pass1_schema_finite": True,
        "pass2_configuration_alignment": {"folds": [0, 1, 2, 3, 4], "model_seeds": [0, 1, 2], "bootstrap_seed": 20261123, "n_boot": 10000, "data_version": config["data_version"], "selection": config["selection"], "legacy_loader_fold_minus_one_training_gap": True, "submission_acceptance": False},
        "pass3_report_recalculation": {"bootstrap_ci_from_npz": ci_recalculated, "fold_macro_delta_neg1c": float(fold_means["neg1c"]), "fold_macro_delta_neg2": float(fold_means["neg2"]), "all_tail_fold_deltas_positive": True},
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
        "Fig1_E0_data_and_leakage_audit": ["fold=-1 positives", "foldfix pending"],
        "Fig3_E5_PARCLIP_cross_protocol": ["Three-seed macro AUROC", "PAR-CLIP matched neg2"],
        "Fig4_E6_GRIMM_five_split_frequency_and_OOD_v2": ["all 20 split", "FPR95", "ascending OOD score"],
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
        "contact_sheet": {"size": list(color.size), "grayscale_mode": gray.mode, "grayscale_std": float(gray_array.std()), "grayscale_unique_values": len(np.unique(gray_array)), "fig2_excluded": True},
    }
    (VERIFY_ROOT / "output_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", "fig1", "fig2", "fig3", "fig4"], default="all")
    parser.add_argument("--contact-sheet", action="store_true")
    args = parser.parse_args()
    functions = {"fig1": figure1, "fig2": figure2, "fig3": figure3, "fig4": figure4}
    selected = list(functions) if args.figure == "all" else [args.figure]
    audits = {}
    for name in selected:
        audit = functions[name]()
        write_audit(name, audit)
        audits[name] = audit
        print(f"{name}: rendered and verified {audit['figure_stem']}")
    if args.contact_sheet:
        make_contact_sheets([audits[name]["figure_stem"] for name in selected if name != "fig2"])
        print("contact sheets: rendered; archived Fig2 excluded")
    if args.figure == "all":
        if args.contact_sheet:
            validate_rendered_outputs()
            print("output validation: numeric/vector/DPI/grayscale checks passed")
        summary = {
            "status": "rendering_and_numeric_verification_passed",
            "submission_status": {"fig1": "descriptive_with_disclosed_training_gap", "fig2": "archived_do_not_submit", "fig3": "external_descriptive", "fig4": "five_split_descriptive"},
            "figures": {name: {"run_ids": RUNS[name], "evidence_level": audits[name]["evidence_level"], "figure_stem": audits[name]["figure_stem"]} for name in selected if name != "fig2"},
            "audit_history": {"fig2": {"run_ids": RUNS["fig2"], "evidence_level": audits["fig2"]["evidence_level"], "figure_stem": audits["fig2"]["figure_stem"], "excluded_from_contact_sheet": True}},
        }
        (VERIFY_ROOT / "verification_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
