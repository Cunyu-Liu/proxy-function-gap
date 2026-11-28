#!/usr/bin/env python3
"""Fig8 cross-encoder tail comparison (RNA-FM vs RiNALMo), run after RiNALMo bootstrap accepted.

Reads figures/data_20261128/paired_bootstrap.tsv (RNA-FM, accepted) and
figures/data_20261128/rinalmo_paired_bootstrap.tsv (RiNALMo, fetched after acceptance).
Descriptive comparison of tail pooled bootstrap deltas; no cross-encoder significance test.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "figures" / "data_20261128"
OUT = ROOT / "figures"
PALETTE = {"rnafm": "#0F4D92", "rinalmo": "#42949E", "ci": "#ABDDF1"}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 1.1, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def main():
    rnafm = pd.read_csv(DATA / "paired_bootstrap.tsv", sep="\t")
    rinalmo_path = DATA / "rinalmo_paired_bootstrap.tsv"
    if not rinalmo_path.exists():
        raise SystemExit("RiNALMo bootstrap not yet fetched; run after acceptance.")
    rinalmo = pd.read_csv(rinalmo_path, sep="\t")
    rows = [("neg1c", "auroc"), ("neg1c", "auprc"), ("neg2", "auroc"), ("neg2", "auprc")]
    labels = ["neg1c\nAUROC", "neg1c\nAUPRC", "neg2\nAUROC", "neg2\nAUPRC"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(rows))
    for offset, (df, name, col) in enumerate([(rnafm, "RNA-FM", PALETTE["rnafm"]), (rinalmo, "RiNALMo", PALETTE["rinalmo"])]):
        means, los, his = [], [], []
        for track, metric in rows:
            r = df[(df.track == track) & (df.slice == "tail") & (df.metric == metric)].iloc[0]
            means.append(r.observed_delta); los.append(r.observed_delta - r.ci_low); his.append(r.ci_high - r.observed_delta)
        ax.bar(x + (offset - 0.5) * 0.36, means, 0.36, color=col, alpha=0.88, label=name,
               yerr=[los, his], error_kw=dict(ecolor="#333333", elinewidth=1.1, capsize=3))
        for xi, m in zip(x + (offset - 0.5) * 0.36, means):
            ax.text(xi, m + 0.0015, f"{m:+.4f}", ha="center", fontsize=8.5)
    ax.axhline(0, color="#888888", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Tail pooled Δ (frozen BCE − cluster-dedup cosine kNN)")
    ax.set_title("Cross-encoder tail advantage under identical strict3train protocol (descriptive)", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(axis="y", color="#EEEEEE", lw=0.7); ax.set_axisbelow(True)
    ax.text(0.0, -0.16, "Both encoders beat kNN; magnitudes differ descriptively. No cross-encoder significance test; not a universal encoder ranking.",
            transform=ax.transAxes, fontsize=8.5, color="#666666")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Fig8_cross_encoder_tail_v2_20261128.{ext}", dpi=300, bbox_inches="tight")
    print("FIG8_SAVED")


if __name__ == "__main__":
    main()