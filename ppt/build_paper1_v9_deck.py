#!/usr/bin/env python3
"""Paper 1 V9 update deck (2026-11-28): concise, modern, evidence-labeled. DRAFT_NOT_SUBMITTED."""
from __future__ import annotations
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "deliverables" / "figures"
OUT = ROOT / "deliverables" / "ppt" / "proxy_function_gap_paper1_v9_20261128.pptx"
W, H = 13.333, 7.5
FONT = "Arial Unicode MS"
C = {"ink": "17233B", "sub": "5D687A", "muted": "8B95A7", "line": "DDE3EB",
     "blue": "0F4D92", "teal": "238A7A", "coral": "C65D4B", "violet": "725A9C",
     "gold": "B8862D", "bg": "FFFFFF", "soft": "F4F7FA",
     "blue_l": "EAF1F8", "teal_l": "E8F4F1", "coral_l": "FAEEEB", "violet_l": "F0EDF6", "gold_l": "FBF4E7"}


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
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar(s, 0, 0, W, H, "bg")
    return s


def header(s, kicker, title):
    bar(s, 0.55, 0.5, 0.12, 0.62, "blue")
    txt(s, 0.8, 0.46, 11.8, 0.35, kicker, size=12, color="sub", bold=True)
    txt(s, 0.8, 0.78, 11.9, 0.7, title, size=25, color="ink", bold=True)
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

    s = slide(prs)
    bar(s, 0, 0, W, 0.18, "blue")
    txt(s, 0.9, 2.0, 11.5, 1.4, "Decision value of sequence proxies tracks\nhigh-score tail reliability", size=34, color="ink", bold=True, ls=1.05)
    txt(s, 0.9, 3.5, 11.3, 0.8, "Equal-budget functional regret reveals when global prediction quality is insufficient for molecular selection", size=17, color="sub")
    txt(s, 0.9, 4.5, 11.0, 0.5, "Paper 1 · V9 · 2026-11-28", size=14, color="blue", bold=True)
    x = 0.9; x = chip(s, x, 5.2, "retrospective", "violet"); x = chip(s, x, 5.2, "grouped holdout", "gold"); chip(s, x, 5.2, "DRAFT_NOT_SUBMITTED", "coral")
    txt(s, 0.9, 6.4, 11.5, 0.6, "T-cell · GB1 · GFP · AAV9 production · TEM-1 · mRNA stability/indel/TE", size=13, color="sub")

    s = slide(prs); header(s, "PRINCIPLE", "Prediction accuracy ≠ selection correctness")
    txt(s, 0.8, 1.85, 11.7, 1.0, "A proxy is trained to minimize average error but deployed to choose a few candidates. The acceptance endpoint must be measured functional regret under an equal-budget control.", size=16, color="ink")
    bar(s, 0.8, 3.0, 11.73, 1.5, "soft")
    txt(s, 1.0, 3.2, 11.3, 1.1, "Regret  R = max Y(pool) − Y(selected)\nBoN(N) vs RandomN on the identical nested proposal prefix → isolates ranking value from candidate opportunity.", size=15, color="blue", bold=True, ls=1.2)
    txt(s, 0.8, 4.9, 11.7, 1.4, "MAE and global Spearman are diagnostics, not substitutes.\nA model can improve average error yet leave the selected candidate unchanged or worse.", size=15, color="sub", ls=1.25)

    s = slide(prs); header(s, "POSITIVE EVIDENCE", "Proxy ranking reduces measured regret at equal budget")
    rows = [
        ("T-cell MRL r1", "cap16 −1.705760  [−2.163445, −1.211914]", "all-20 Holm (cap2–16)", "blue"),
        ("GB1 IgG-Fc lnW", "cap16 −0.160894  [−0.282853, −0.050515]", "31/55 pools improve", "teal"),
        ("GFP brightness", "additive −0.2800  [−0.3095, −0.2521]", "low-order train → high-order test", "teal"),
        ("AAV9 production", "cap16 −4.206431  [−4.564111, −3.931098]", "production ≠ tropism/clinical", "violet"),
    ]
    y = 1.85
    for name, val, note, col in rows:
        bar(s, 0.8, y, 0.1, 0.78, col)
        txt(s, 1.05, y + 0.02, 3.0, 0.7, name, size=15, color="ink", bold=True)
        txt(s, 4.1, y + 0.02, 5.6, 0.7, val, size=14, color=col, bold=True)
        txt(s, 9.8, y + 0.02, 2.9, 0.7, note, size=11.5, color="sub")
        y += 0.92
    txt(s, 0.8, 5.7, 11.7, 1.2, "Endpoint semantics are binding: lnW ≠ Kd; production ≠ tropism; brightness ≠ pure expression; relative resistance ≠ enzyme kinetics. Heterogeneity retained (GB1 24/55 pools worsen).", size=13.5, color="sub", ls=1.2)

    s = slide(prs); header(s, "MECHANISM", "Useful global ranking, unreliable high-score tail")
    txt(s, 0.8, 1.8, 5.7, 2.6, "T-cell top16 Spearman 0.206 / 0.144\nPredicted winner not measured optimum in 15/16 pools\nWithin-tail loss 2.480 of 3.316 (r1); 2.117 of 4.071 (r2)\n→ remaining bottleneck is tail ordering, not global information", size=14.5, color="ink", ls=1.3)
    bar(s, 6.8, 1.8, 5.73, 2.6, "coral_l")
    txt(s, 7.0, 1.95, 5.3, 2.3, "Regression–decision decoupling\n50k adaptation: MAE 1.352→0.915 (r1) but cap16 regret +0.038929\nHepG2 anchor: global ρ comparable, T-cell cap16 +0.428897 worse\nListNet: gate-failing point improvements only", size=13.5, color="coral", bold=True, ls=1.25)
    picture(s, FIG / "v6" / "Figure20_v6_decision_value_map.png", 0.8, 4.55, 7.6)
    txt(s, 8.7, 4.7, 3.9, 2.2, "Decision value map: positive tasks retain tail enrichment; failures show weak/negative/endpoint-mismatched tail ordering. Associational, not causal.", size=12.5, color="sub", ls=1.2)

    s = slide(prs); header(s, "NEW IN V9 · FALSIFICATION", "A selection-side intervention fails its auxiliary gate")
    picture(s, FIG / "v7" / "Figure22_v7_upts_auxiliary_gate_falsification.png", 0.8, 1.75, 8.4)
    txt(s, 9.45, 1.85, 3.1, 4.6, "UPTS v1: fixed pool, scores, prefix; only the within-tail choice rule changes (mean − κ·std).\nκ selected on auxiliary dev; dual-direction gate.\nHepG2→N50 worsened 0.0296 MRL → STOP, T-cell labels never read.\nDisagreement ρ≈0.004–0.017: common error dominates.\n8th pre-registered falsification; first selection-side.", size=12.5, color="ink", ls=1.22)
    chip(s, 9.45, 6.5, "retrospective", "violet")

    s = slide(prs); header(s, "BOUNDARIES", "Where proxy ranking does not help")
    rows = [
        ("TEM-1 Amp→cefotaxime", "15/15 intervals cross zero — null", "coral"),
        ("mRNA stability", "cap16 +0.2838 [+0.0824, +0.4435] — harmful", "coral"),
        ("Castillo indel", "0/4 corrected-positive — mixed/null", "gold"),
        ("HEK293T high-cap MRL", "no sustained benefit — null", "gold"),
        ("Leppek MRL / Schlusser TE", "no robust benefit", "gold"),
    ]
    y = 1.9
    for name, val, col in rows:
        bar(s, 0.8, y, 0.1, 0.66, col)
        txt(s, 1.05, y, 4.6, 0.66, name, size=14.5, color="ink", bold=True)
        txt(s, 5.8, y, 6.7, 0.66, val, size=13.5, color=col)
        y += 0.82
    txt(s, 0.8, 6.2, 11.7, 0.8, "These prevent a universal proxy-validity claim. Different endpoint units are never pooled into one effect.", size=13.5, color="sub")

    s = slide(prs); header(s, "LIMITATIONS & NEXT", "What this paper does not claim")
    txt(s, 0.8, 1.8, 11.7, 2.6, "• All retrospective; grouped holdouts ≠ prospective blind tests; no clinical/wet-lab efficacy.\n• No official COM / generative MBO equivalence on the fixed-pool measured-regret surface (Design-Bench uses learned oracle + Sample leakage; GABO/T3PO/mRNAutilus generate pool-external sequences; Reddy is promoters).\n• Tail audit is associational, not a causal intervention proof.\n• Independent candidates identified but NOT evaluated: Cao 2021 5′UTR protein production (GSE176581); Ogden 2019 AAV2 liver tropism (GSE139657).", size=14, color="ink", ls=1.3)
    bar(s, 0.8, 4.7, 11.73, 1.9, "soft")
    txt(s, 1.0, 4.85, 11.3, 1.6, "Pre-submission manual items: authors · affiliations · COI · licenses · target journal · data/code DOI · T3PO reference verification · PPT finalization.\nContribution: an evaluation principle + cross-task evidence map + mechanism boundaries, not a universal design algorithm.", size=13.5, color="sub", ls=1.25)

    prs.save(OUT)
    print(f"saved {OUT} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()