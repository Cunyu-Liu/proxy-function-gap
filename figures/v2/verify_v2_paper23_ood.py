#!/usr/bin/env python3
"""Three-round verification of Figure7-10 v2 (Paper2/Paper3/Schlusser/Leppek)."""
from __future__ import annotations
import hashlib, json, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from delivery_scripts.publication_paths import DELIVERABLES, OUTPUT_ROOT

import fitz
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = OUTPUT_ROOT / "figures" / "v2"
SRC = DELIVERABLES / "reports" / "source_data"
P3MD = DELIVERABLES / "reports" / "paper3_headroom_v4_20261124.md"
BASE = ["Figure7_v2_paper2_baselines", "Figure8_v2_paper3_headroom",
        "Figure9_v2_schlusser_te", "Figure10_v2_leppek_endpoints"]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a, b, t=1e-9): assert np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=t), (a, b)


def round1():
    # Paper2
    d = json.load(open(SRC / "paper2_baselines_v1" / "outer_designed_summary.json"))
    c16 = {c["method"]: c for c in d["configurations"] if c["proposal_cap"] == 16}
    for m in ["bon", "lcb", "distance", "com", "diversity_lcb"]:
        assert c16[m]["mean_cnn_member_item_forwards"] == 48.0  # budget matched
    paired = {p["method"]: p for p in d["paired_comparisons"] if p["reference"] == "bon" and p["metric"] == "regret"}
    for m in ["lcb", "distance", "com", "diversity_lcb"]:
        lo, hi = paired[m]["pool_cluster_bootstrap_95_percent_interval"]
        assert lo <= 0 <= hi, (m, lo, hi)  # all cross zero
    # Paper3 heatmap + curve
    p3 = json.load(open(SRC / "paper3_headroom_v4" / "headroom_v4_round1" / "summary.json"))
    reps = ["summary4", "kmer4", "cnn64"]
    pos = {r: sum(1 for c in p3["cases"] if c["representation"] == r and c["repair_headroom"] > 0) for r in reps}
    assert pos == {"summary4": 0, "kmer4": 5, "cnn64": 1}, pos
    block = P3MD.read_text().split("### kmer4：固定 5 池")[1].split("###")[0]
    rows = [[float(c) for c in ln.strip().strip("|").split("|")] for ln in block.splitlines()
            if len(ln.strip().strip("|").split("|")) == 7 and ln.strip().strip("|").split("|")[0].strip().isdigit()]
    arr = np.array(rows)
    close(arr[0, 1:], [3.706713] * 6, 1e-6)  # budget 0 all equal
    close(arr[-1, 1:], [2.131689] * 6, 1e-6)  # budget 64 all equal
    assert arr[3, 4] < arr[3, 1]  # EI@16 < Random@16
    # Schlusser
    s = json.load(open(SRC / "schlusser2024_hepg2_te_confirmatory_v1" / "evaluator_round_1" / "summary.json"))
    dist = s["TE_distribution_after_frozen_unblinding"]
    assert dist["max"] > 4000 and dist["median"] < 1  # extreme skew
    p16 = [p for p in s["paired_comparisons"] if p["proposal_cap"] == 16]
    for p in p16:
        lo, hi = p["gene_cluster_bootstrap_95_interval"]
        assert lo <= 0 <= hi, (p["comparison"], lo, hi)
    # Leppek
    corr = pd.read_csv(SRC / "leppek2022_cds_mechanism_audit_v5" / "paired_correlations.tsv", sep="\t")
    rank = pd.read_csv(SRC / "leppek2022_cds_mechanism_audit_v5" / "rank_summary.tsv", sep="\t")
    cds = rank[rank.source_family == "cds_24"]
    hl6 = cds[(cds.left_endpoint == "cds_in_solution_half_life") & (cds.right_endpoint == "nluc_fluc_6h")].iloc[0]
    close(hl6.discordant_fraction_untied, 171 / 276, 1e-9)
    assert int(hl6.top_k_overlap_n) == 1
    return {"status": "PASS", "p3_positive_gates": pos, "paper2_all_cross_zero": True,
            "schlusser_all_cross_zero": True, "leppek_reversal_171_276": True}


def round2():
    before = {b: digest(OUT / f"{b}.svg") for b in BASE}
    subprocess.run([sys.executable, str(HERE / "plot_v2_paper23_ood.py")], cwd=HERE, check=True, capture_output=True)
    after = {b: digest(OUT / f"{b}.svg") for b in BASE}
    for b in BASE:
        assert before[b] == after[b], b
    return {"deterministic_svg": len(BASE), "status": "PASS"}


def round3():
    out = {}
    for b in BASE:
        im = Image.open(OUT / f"{b}.png"); im.verify()
        w, h = Image.open(OUT / f"{b}.png").size
        assert w > 1000 and h > 800
        doc = fitz.open(OUT / f"{b}.pdf"); assert doc.page_count == 1; doc.close()
        ET.parse(OUT / f"{b}.svg")
        out[b] = {"png_px": [w, h]}
    return {"figures": out, "status": "PASS"}


if __name__ == "__main__":
    rep = {"round1": round1(), "round2": round2(), "round3": round3(), "overall": "PASS"}
    (OUT / "v2_paper23_ood_validation_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))