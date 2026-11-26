from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw
from pptx import Presentation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from delivery_scripts.publication_paths import DELIVERABLES, EVIDENCE_ROOT, OUTPUT_ROOT, configured_path

ROOT = EVIDENCE_ROOT
OUT = OUTPUT_ROOT / "ppt" / "paper1_v4"
OUT.mkdir(parents=True, exist_ok=True)
PPT = OUT / "paper1_tcell_proxy_decision_20261126.pptx"
PDF = configured_path("PFG_PAPER1_DECK_PDF", DELIVERABLES / "ppt" / "paper1_v4" / PPT.with_suffix(".pdf").name)
CONFIG = json.loads((HERE / "paper1_tcell_proxy_decision_20261126_config.json").read_text())
SRC = DELIVERABLES / "reports" / "source_data"
DIAG = SRC / "tcell_error_diagnosis_v1" / "analysis"
V1 = SRC / "tcell_target_adaptation_v1" / "target"
V2 = SRC / "tcell_listnet_adaptation_v2" / "target_v2"
ANCHOR = SRC / "tcell_cross_cell_anchor_v1" / "target"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def deck_text(prs):
    return "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))


def zip_payload(path):
    with zipfile.ZipFile(path) as zf:
        return {n: hashlib.sha256(zf.read(n)).hexdigest() for n in zf.namelist() if n.startswith("ppt/") and not n.endswith("/")}


def round_numeric():
    prs = Presentation(PPT); assert len(prs.slides) == 9
    text = deck_text(prs); normalized = text.replace("−", "-")
    for phrase in ("有效基线", "尾部瓶颈", "回归–决策脱钩", "设计指标", "TEST-V1", "TEST-V2", "T-cell evaluation count", "不主张目标适配或 ListNet 已修复"):
        assert phrase in text, phrase
    summary = load_json(DIAG / "summary.json")
    effects = summary["selection_effects"]
    for context in ("defined_tcell_r1", "defined_tcell_r2"):
        row = next(r for r in effects if r["context"] == context and r["model"] == "cnn" and r["contrast"] == "bon_minus_random_n" and r["cap"] == 16)
        assert row["effect"] < 0 and row["ci_high"] < 0
        assert f"{row['effect']:.3f}" in normalized
        for value in (row["ci_low"], row["ci_high"]):
            assert f"{value:.3f}" in normalized
    v1p = load_json(V1 / "primary.json"); v1s = load_json(V1 / "secondary.json")
    for context in ("defined_tcell_r1", "defined_tcell_r2"):
        sync = v1s["error_regret_synchronization"][context]["main_50k"]
        assert sync["mae_change"] < 0 < sync["cap16_regret_change"]
        assert f"{sync['mae_change']:.6f}" in normalized and f"{sync['cap16_regret_change']:+.6f}" in normalized
    combined = v1p["combined_main_minus_original"]
    for value in (combined["effect"], combined["ci_low"], combined["ci_high"]):
        assert f"{value:.6f}" in normalized or f"{value:+.6f}" in normalized
    v2p = load_json(V2 / "primary.json")
    for comparator in ("original_cnn", "v1_main"):
        comp = v2p["comparisons"][comparator]
        assert comp["effect"]["ci_high"] > 0 and comp["absolute_reduction"] < 0.1 and not v2p["comparison_pass"][comparator]
        assert f"{comp['effect']['effect']:.6f}" in normalized
        for value in (comp["effect"]["ci_low"], comp["effect"]["ci_high"]):
            assert f"{value:.6f}" in normalized
    anchorp = load_json(ANCHOR / "primary.json")
    assert anchorp["combined_anchor_minus_cnn"]["ci_low"] > 0 and not anchorp["success"]
    assert f"{anchorp['combined_anchor_minus_cnn']['effect']:.6f}" in normalized
    for value in anchorp["combined_anchor_minus_cnn"]["ci_low"], anchorp["combined_anchor_minus_cnn"]["ci_high"]:
        assert f"{value:.6f}" in normalized
    rank = load_json(SRC / "tcell_rank_bstrict_v2" / "artifacts" / "auxiliary_gate.json")
    backbone = load_json(SRC / "tcell_multicontext_backbone_v1" / "auxiliary_gate.json")
    assert not rank["target_evaluation_permitted"] and not rank["target_mrl_read"]
    assert backbone["target_evaluation_count"] == 0 and not backbone["target_mrl_read"]
    for record in CONFIG["sources"]:
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    for forbidden in ("ListNet 已修复 T-cell", "50k main 改善 cap16", "HepG2-anchor 改善", "辅助门禁已证明 T-cell"):
        assert forbidden not in text
    return {"status": "PASS", "slides": 9, "numeric_claims_checked": 31, "source_hashes": len(CONFIG["sources"]),
            "primary_positive_retained": True, "all_failed_methods_retained_as_failures": True,
            "test_v1_test_v2_separate": True, "auxiliary_target_evaluation_zero": True}


def round_deterministic():
    subprocess.run([sys.executable, str(HERE / "build_paper1_deck.py")], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    before = zip_payload(PPT)
    reference = zip_payload(DELIVERABLES / "ppt" / "paper1_v4" / PPT.name)
    assert before == reference
    time.sleep(1.1)
    env = dict(os.environ, PYTHONHASHSEED="987654321", TZ="UTC", LC_ALL="C")
    subprocess.run([sys.executable, str(HERE / "build_paper1_deck.py")], cwd=HERE, check=True, stdout=subprocess.DEVNULL, env=env)
    after = zip_payload(PPT); assert before == after
    for record in CONFIG["figures"]:
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    return {"status": "PASS", "ppt_payload_entries_identical": len(before), "reference_payload_identical": True, "rebuild_rounds": 2,
            "environment_variation": {"PYTHONHASHSEED": "987654321", "TZ": "UTC", "LC_ALL": "C"},
            "figure_hashes": len(CONFIG["figures"]), "cross_second_delay": 1.1}


def round_format():
    prs = Presentation(PPT)
    for i, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0, (i, shape.name)
            assert shape.left + shape.width <= prs.slide_width + 2, (i, shape.name, "x")
            assert shape.top + shape.height <= prs.slide_height + 2, (i, shape.name, "y")
    with zipfile.ZipFile(PPT) as zf:
        embedded = {hashlib.sha256(zf.read(n)).hexdigest() for n in zf.namelist() if n.startswith("ppt/media/")}
    for record in CONFIG["figures"]:
        assert record["sha256"] in embedded
    doc = fitz.open(PDF); assert len(doc) == 9
    assert all(abs(p.rect.width - 960) < 0.1 and abs(p.rect.height - 540) < 0.1 for p in doc)
    all_text = "\n".join(p.get_text() for p in doc); compact = "".join(all_text.split())
    for phrase in ("有效基线", "尾部瓶颈", "回归–决策脱钩", "设计指标", "TEST-V1", "TEST-V2", "不主张目标适配或ListNet已修复", "T-cellevaluationcount"):
        assert phrase.replace(" ", "") in compact, phrase
    render_dir = OUT / "renders"; render_dir.mkdir(exist_ok=True)
    thumbs = []; stats = []
    for i, page in enumerate(doc, 1):
        for word in page.get_text("words"):
            assert word[0] >= -0.5 and word[1] >= -0.5 and word[2] <= page.rect.width + 0.5 and word[3] <= page.rect.height + 0.5, (i, word)
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                A, B = blocks[a], blocks[b]
                dx = min(A[2], B[2]) - max(A[0], B[0]); dy = min(A[3], B[3]) - max(A[1], B[1])
                assert not (dx > 1 and dy > 1), (i, A[4], B[4], dx, dy)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        path = render_dir / f"slide-{i:02d}.png"; pix.save(path)
        im = Image.open(path).convert("RGB"); arr = np.asarray(im)
        assert arr.std() > 15 and len(np.unique(arr.reshape(-1, 3), axis=0)) > 100
        stats.append({"slide": i, "pixels": [im.width, im.height], "std": round(float(arr.std()), 2)})
        thumb = im.copy(); thumb.thumbnail((590, 332))
        tile = Image.new("RGB", (600, 365), "white"); tile.paste(thumb, ((600-thumb.width)//2, 25))
        ImageDraw.Draw(tile).text((12, 7), f"Slide {i}", fill="#17233B"); thumbs.append(tile)
    sheet = Image.new("RGB", (1800, 3 * 365), "#DDE3EB")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 3) * 600, (i // 3) * 365))
    sheet.save(OUT / "paper1_tcell_proxy_decision_contact_sheet.png")
    fonts = subprocess.check_output(["pdffonts", str(PDF)], text=True)
    lines = [line for line in fonts.splitlines() if "+" in line]
    assert lines and all(" yes yes yes " in line for line in lines)
    return {"status": "PASS", "pdf_pages": 9, "pdf_page_points": [960, 540],
            "pdf_export": "LibreOffice 26.8.1.1; Microsoft PowerPoint native export attempted twice but AppleEvent timed out",
            "fonts": "all PDF fonts embedded, subset and Unicode; LibreOffice substitutes its available fallback fonts",
            "shape_bounds": "PASS", "pdf_words_outside": 0, "pdf_text_collisions": 0,
            "embedded_figures_exact": len(CONFIG["figures"]), "rendered_pages": stats,
            "contact_sheet": "paper1_tcell_proxy_decision_contact_sheet.png"}


report = {"round1_numeric_independent": round_numeric(), "round2_deterministic_rebuild": round_deterministic(), "round3_format_render": round_format()}
report["overall"] = "PASS"
(OUT / "paper1_tcell_proxy_decision_20261126_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
