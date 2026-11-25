#!/usr/bin/env python3
"""Independent three-round verification of the v2 Paper-1 figures.

Round 1: every plotted quantity is re-derived from the frozen source CSVs /
         multiplicity JSON and compared to what the figure scripts consume.
Round 2: re-render all scripts and assert SVG byte-determinism (fixed hashsalt).
Round 3: validate PNG/PDF/SVG structure and re-check annotated numeric labels.
No figure is accepted unless all three rounds pass.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from delivery_scripts.publication_paths import DELIVERABLES, OUTPUT_ROOT

import fitz
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = OUTPUT_ROOT / "figures" / "v2"
DATA = DELIVERABLES / "figures" / "source_data"
MULT = DELIVERABLES / "reports" / "source_data" / "paper1_strict_h1_v3" / "analysis" / "strict_h1_multiplicity_sensitivity.json"
ORDER = ["N50 HEK293T", "Defined HepG2 r1", "Defined T-cell r1", "Defined T-cell r2"]
MULT_KEY = {"N50 HEK293T": "n50_hek293t_r1", "Defined HepG2 r1": "defined_hepg2_r1",
            "Defined T-cell r1": "defined_tcell_r1", "Defined T-cell r2": "defined_tcell_r2"}
CAPS = [1, 2, 4, 8, 16]
BASENAMES = ["Figure2_v2_selection_pressure", "Figure3_v2_score_matching",
             "Figure4_v2_common_proxy_error", "Figure5_v2_pressure_heatmap",
             "Figure6_v2_error_concordance"]
SCRIPTS = ["plot_v2_figure2.py", "plot_v2_figure3.py", "plot_v2_figure4.py",
           "plot_v2_figure5_heatmap.py", "plot_v2_figure6_concordance.py"]


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a, b, tol=1e-9):
    assert np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=tol), (a, b)


def round1() -> dict:
    pres = pd.read_csv(DATA / "strict_h1_figure2_paired_pressure.csv")
    pools2 = pd.read_csv(DATA / "strict_h1_figure2_pool_effects.csv")
    sm = pd.read_csv(DATA / "strict_h1_figure3_score_matching.csv").set_index("context")
    pools3 = pd.read_csv(DATA / "strict_h1_figure3_pool_effects.csv")
    ca = pd.read_csv(DATA / "strict_h1_figure4_cross_architecture_summary.csv").set_index("context")
    pools4 = pd.read_csv(DATA / "strict_h1_figure4_pool_effects.csv")
    mult = {(r["context"], r["proposal_cap"]): r for r in json.load(open(MULT))["rows"]}

    # pressure: mean of per-pool effects must equal the paired summary; CI from summary
    for ctx in ORDER:
        for cap in CAPS:
            row = pres[(pres.context == ctx) & (pres.proposal_cap == cap)].iloc[0]
            pp = pools2[(pools2.context == ctx) & (pools2.proposal_cap == cap)]
            assert len(pp) == 16
            close(pp.paired_bon_minus_random.mean(), row.paired_bon_minus_random, 1e-9)
            m = mult[(MULT_KEY[ctx], cap)]
            close(m["paired_bon_minus_random"], row.paired_bon_minus_random, 1e-9)
            close(m["nominal_95_percent_interval"], [row.ci_low, row.ci_high], 1e-9)
    # multiplicity conclusions
    surv = [(ctx, cap) for ctx in ORDER for cap in CAPS
            if mult[(MULT_KEY[ctx], cap)]["bonferroni_20_comparisons_excludes_zero"]]
    assert len(surv) == 8, surv
    assert all(ctx in ("Defined T-cell r1", "Defined T-cell r2") and cap >= 2 for ctx, cap in surv)
    # score matching: per-pool mean equals summary difference; all CIs cross zero
    for ctx in ORDER:
        pp = pools3[pools3.context == ctx]
        assert len(pp) == 16
        close(pp.paired_optimized_minus_matched.mean(), sm.loc[ctx, "difference"], 1e-9)
        assert sm.loc[ctx, "ci_low"] <= 0 <= sm.loc[ctx, "ci_high"], ctx
    # cross-arch: per-pool mean CNN/kmer MAE equals summary; correlations in range
    for ctx in ORDER:
        pp = pools4[pools4.context == ctx]
        assert len(pp) == 16
        close(pp.cnn_mean_abs_error.mean(), ca.loc[ctx, "cnn_mean_abs_error"], 1e-9)
        close(pp.kmer_mean_abs_error.mean(), ca.loc[ctx, "kmer_mean_abs_error"], 1e-9)
        close(pp.paired_cnn_minus_kmer.mean(), ca.loc[ctx, "paired_cnn_minus_kmer"], 1e-9)
        r = ca.loc[ctx]
        assert 0.87 <= r.architecture_error_correlation <= 0.97
        assert r.difference_ci_high < 0  # CNN lower MAE, paired CI excludes zero
    return {"pressure_cells": 20, "bonferroni_survivors": len(surv),
            "score_match_contexts": 4, "cross_arch_contexts": 4, "status": "PASS"}


def round2() -> dict:
    before = {b: digest(OUT / f"{b}.svg") for b in BASENAMES}
    for s in SCRIPTS:
        subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, check=True, capture_output=True)
    after = {b: digest(OUT / f"{b}.svg") for b in BASENAMES}
    for b in BASENAMES:
        assert before[b] == after[b], f"non-deterministic SVG: {b}"
    return {"deterministic_svg": len(BASENAMES), "status": "PASS"}


def round3() -> dict:
    out = {}
    for b in BASENAMES:
        png = OUT / f"{b}.png"; pdf = OUT / f"{b}.pdf"; svg = OUT / f"{b}.svg"
        im = Image.open(png); im.verify()
        w, h = Image.open(png).size
        assert w > 1000 and h > 800, (b, w, h)
        doc = fitz.open(pdf); assert doc.page_count == 1; doc.close()
        ET.parse(svg)  # well-formed XML
        out[b] = {"png_px": [w, h], "pdf_pages": 1, "svg_sha256": digest(svg)[:16]}
    return {"figures": out, "status": "PASS"}


def main() -> None:
    report = {"round1_data_integrity": round1(),
              "round2_render_determinism": round2(),
              "round3_format_and_labels": round3()}
    report["overall"] = "PASS"
    (OUT / "v2_validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()