#!/usr/bin/env python3
"""Paper 1 V10 deck (2026-11-28): seven-question chain, v8 figures, quantitative diagnostic. DRAFT_NOT_SUBMITTED."""
from __future__ import annotations
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "deliverables" / "figures" / "v8"
OUT = ROOT / "deliverables" / "ppt" / "proxy_function_gap_paper1_v10_20261128.pptx"
W, H = 13.333, 7.5
FONT = "Arial Unicode MS"
C = {"ink": "1A1A1A", "sub": "5D687A", "muted": "8B95A7", "line": "D9DEE6",
     "blue": "0F4D92", "blue2": "3775BA", "green": "009E73", "red": "D55E00",
     "gray": "767676", "violet": "725A9C", "gold": "E69F00", "bg": "FFFFFF", "soft": "F4F7FA",
     "blue_l": "EAF1F8", "green_l": "DDF3DE", "red_l": "F6CFCB", "violet_l": "F0EDF6", "gold_l": "FBF4E7"}


def rgb(k): return RGBColor.from_string(C[k])


def txt(slide, x, y, w, h, text, size=14, color="ink", bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, ls=1.08):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = ls
        r = p.add_run(); r.text = ln
        r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = rgb(color)
    return tb


def bar(slide, x, y, w, h, color):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = rgb(color); sp.line.fill.background()
    return sp


def slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); bar(s, 0, 0, W, H, "bg"); return s


def header(s, kicker, title):
    bar(s, 0.55, 0.5, 0.12, 0.62, "blue")
    txt(s, 0.8, 0.46, 11.8, 0.35, kicker, size=12, color="sub", bold=True)
    txt(s, 0.8, 0.78, 11.9, 0.7, title, size=24, color="ink", bold=True)
    bar(s, 0.8, 1.5, 11.73, 0.018, "line")


def chip(s, x, y, text, color):
    w = 0.42 + 0.085 * len(text)
    bar(s, x, y, w, 0.34, color + "_l")
    txt(s, x + 0.12, y + 0.03, w - 0.2, 0.28, text, size=11, color=color, bold=True)
    return x + w + 0.18


def picture(s, path, x, y, w):
    img = Image.open(path); ar = img.height / img.width
    s.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(w * ar))


def main():
    prs = Presentation(); prs.slide_width = Inches(W); prs.slide_height = Inches(H)

    s = slide(prs); bar(s, 0, 0, W, 0.18, "blue")
    txt(s, 0.9, 1.9, 11.5, 1.4, "Decision value of sequence proxies tracks\nhigh-score tail reliability", size=33, color="ink", bold=True, ls=1.05)
    txt(s, 0.9, 3.4, 11.3, 0.8, "A connected question-to-evidence chain across biological design tasks", size=17, color="sub")
    txt(s, 0.9, 4.3, 11.0, 0.5, "Paper 1 · V10 · 2026-11-28", size=14, color="blue", bold=True)
    x = 0.9; x = chip(s, x, 5.0, "retrospective", "violet"); x = chip(s, x, 5.0, "grouped holdout", "gold"); chip(s, x, 5.0, "DRAFT_NOT_SUBMITTED", "red")
    txt(s, 0.9, 6.2, 11.5, 0.6, "T-cell · GB1 · GFP · AAV9 production · TEM-1 · mRNA stability/indel/TE", size=13, color="sub")

    s = slide(prs); header(s, "PRINCIPLE", "Prediction accuracy ≠ selection correctness")
    txt(s, 0.8, 1.85, 11.7, 1.0, "A proxy is trained to minimize average error but deployed to choose a few candidates. The acceptance endpoint must be measured functional regret under an equal-budget control.", size=16, color="ink")
    bar(s, 0.8, 3.0, 11.73, 1.5, "soft")
    txt(s, 1.0, 3.2, 11.3, 1.1, "Regret  R = max Y(pool) − Y(selected)\nBoN(N) vs RandomN on the identical nested proposal prefix → isolates ranking value from candidate opportunity.", size=15, color="blue", bold=True, ls=1.2)
    txt(s, 0.8, 4.9, 11.7, 1.6, "Seven questions: equal-budget value → tail bottleneck → regression repair? → selection repair? → cross-task value → unifying diagnostic → independent study?\nMAE and global Spearman are diagnostics, not substitutes.", size=14.5, color="sub", ls=1.25)

    s = slide(prs); header(s, "Q1–Q2 · POSITIVE + BOTTLENECK", "Equal-budget value coexists with a tail bottleneck")
    picture(s, FIG / "Figure1_v8_equal_budget_value.png", 0.8, 1.75, 6.0)
    picture(s, FIG / "Figure2_v8_tail_bottleneck.png", 6.9, 1.75, 5.6)
    txt(s, 0.8, 4.5, 11.7, 2.2, "T-cell r1 cap16 −1.705760 [−2.163445, −1.211914], all-20 Holm; r2 only cap4 Holm-robust.\nGlobal Spearman 0.591/0.529 falls to 0.206/0.144 in the predicted top16; winner flips in 15/16 pools.\nWithin-tail ranking loss = 2.480 of 3.316 (r1); 2.117 of 4.071 (r2) → the bottleneck is tail ordering.", size=13.5, color="ink", ls=1.25)

    s = slide(prs); header(s, "Q3–Q4 · FALSIFICATION", "Neither regression nor selection reweighting repairs the tail")
    picture(s, FIG / "Figure3_v8_regression_decision_dissociation.png", 0.8, 1.75, 6.0)
    picture(s, FIG / "Figure4_v8_upts_falsification.png", 6.9, 1.75, 5.6)
    txt(s, 0.8, 4.5, 11.7, 2.2, "50k adaptation: MAE 1.352→0.915 but cap16 regret +0.038929; ListNet gate-failing; HepG2 anchor +0.428897 worse.\nUPTS (mean − κ·std), fixed pool/scores/prefix: dual-direction auxiliary gate FAILED (HepG2→N50 −0.029558); T-cell labels never read.\nDisagreement ρ≈0.004–0.017 → 8th pre-registered falsification; residual tail loss needs information outside the frozen proxy.", size=13, color="ink", ls=1.22)

    s = slide(prs); header(s, "Q5 · CROSS-TASK", "When proxy ranking has value — and where it fails")
    picture(s, FIG / "Figure5_v8_cross_scenario_forest.png", 0.8, 1.7, 7.4)
    txt(s, 8.5, 1.85, 4.0, 4.8, "Positive: GB1 lnW −0.160894; GFP additive −0.2800 (beats CNN); AAV9 production −4.206431 (top1% recall only 18.75%).\nBoundary: TEM-1 Amp→cef 15/15 cross zero; stability +0.2838 harmful; indel 0/4; HEK null.\nFaceted by endpoint unit — no cross-unit pooling. lnW≠Kd; production≠tropism; brightness≠expression.", size=12.5, color="ink", ls=1.25)

    s = slide(prs); header(s, "Q6 · NEW IN V10 · QUANTITATIVE DIAGNOSTIC", "Unified tail reliability tracks decision benefit")
    picture(s, FIG / "Figure6_v8_unified_tail_diagnostic.png", 0.8, 1.75, 8.6)
    txt(s, 9.6, 1.85, 2.95, 4.8, "Single top-decile recall metric recomputed on every frozen evaluation population (AAV top1% recall 0.1875 reproduced).\nρ = 0.879, permutation p = 0.00076; LOO ρ 0.83–0.95.\nGlobal ρ = 0.770; difference CI [0.00, 0.46] touches 0 → superiority suggestive, not confirmed.\nTail enrichment > within-tail ordering for fixed-pool selection. Associational, n=10.", size=12, color="ink", ls=1.2)
    chip(s, 9.6, 6.5, "associational", "violet")

    s = slide(prs); header(s, "Q7 · BOUNDARIES & INDEPENDENT DATA", "What this paper does not claim")
    txt(s, 0.8, 1.8, 11.7, 2.7, "• All retrospective; grouped holdouts ≠ prospective blind tests; no clinical/wet-lab efficacy.\n• No official COM / generative MBO equivalence on the fixed-pool measured-regret surface (disclosed).\n• Cross-scenario diagnostic is associational, not causal; n=10 limits power.\n• Independent candidates qualified but NOT evaluated (access-barred): Cao 2021 endogenous 5′UTR TE (corrected from 'protein production'; GSE176581 synthetic library needs reprocessing); Ogden 2019 AAV2 liver tropism (NCBI blocked).", size=13.5, color="ink", ls=1.28)
    bar(s, 0.8, 4.75, 11.73, 1.85, "soft")
    txt(s, 1.0, 4.9, 11.3, 1.55, "Pre-submission manual items: authors · affiliations · COI · licenses · target journal · data/code DOI · T3PO reference verification.\nContribution: an evaluation principle + a quantitatively supported cross-task tail diagnostic + an evidence map with explicit boundaries — not a universal design algorithm.", size=13, color="sub", ls=1.25)

    prs.save(OUT)
    print(f"saved {OUT} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()