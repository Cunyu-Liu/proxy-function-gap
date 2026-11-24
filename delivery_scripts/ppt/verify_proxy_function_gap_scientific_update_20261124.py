from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

ROOT = Path(__file__).resolve().parents[2]
PPT = ROOT / "deliverables" / "ppt" / "proxy_function_gap_scientific_update_20261124.pptx"
PDF = PPT.with_suffix(".pdf")
BUILD = ROOT / "deliverables" / "ppt" / "build_proxy_function_gap_scientific_update_20261124.py"
CONFIG = ROOT / "deliverables" / "ppt" / "proxy_function_gap_scientific_update_20261124_config.json"
REPORT = ROOT / "deliverables" / "ppt" / "proxy_function_gap_scientific_update_20261124_verification.json"
FIG = ROOT / "deliverables" / "figures"
DATA = FIG / "source_data"
AUDIT = ROOT / "deliverables" / "reports" / "docs" / "GATE0_READ_ONLY_AUDIT_20261124.md"
MULTIPLICITY = ROOT / "deliverables" / "reports" / "source_data" / "paper1_strict_h1_v3" / "analysis" / "strict_h1_multiplicity_sensitivity.json"
STAGING = ROOT / ".pool_staging" / "ppt_scientific_update_20261124"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slide_texts(prs: Presentation) -> list[str]:
    return ["\n".join(shape.text for shape in slide.shapes if hasattr(shape, "text_frame")) for slide in prs.slides]


def media_hashes(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return sorted(hashlib.sha256(zf.read(name)).hexdigest() for name in zf.namelist() if name.startswith("ppt/media/"))


def round_numeric(prs: Presentation, texts: list[str]) -> dict:
    assert len(prs.slides) == 14
    assert CONFIG.exists() and AUDIT.exists()
    all_text = "\n".join(texts)
    audit = AUDIT.read_text(encoding="utf-8")
    pressure = pd.read_csv(DATA / "strict_h1_figure2_paired_pressure.csv")
    assert len(pressure) == 20
    assert np.allclose(pressure.bon_mean_regret - pressure.random_mean_regret, pressure.paired_bon_minus_random)
    for context in ("Defined T-cell r1", "Defined T-cell r2"):
        rows = pressure[(pressure.context == context) & (pressure.proposal_cap >= 2)]
        assert (rows.ci_high < 0).all()
        cap16 = rows[rows.proposal_cap == 16].iloc[0]
        assert f"{cap16.paired_bon_minus_random:.3f}" in all_text.replace("−", "-")
    hepg2 = pressure[(pressure.context == "Defined HepG2 r1") & (pressure.proposal_cap >= 2)]
    assert (hepg2.ci_high < 0).sum() == 1 and hepg2[hepg2.proposal_cap == 8].ci_high.iloc[0] < 0
    n50 = pressure[pressure.context == "N50 HEK293T"]
    assert ((n50.ci_low <= 0) & (n50.ci_high >= 0)).all()
    multiplicity = json.loads(MULTIPLICITY.read_text())
    assert multiplicity["status"] == "PASS" and len(multiplicity["rows"]) == 20
    tcell_adjusted = [
        row for row in multiplicity["rows"]
        if row["context"] in {"defined_tcell_r1", "defined_tcell_r2"} and row["proposal_cap"] >= 2
    ]
    assert len(tcell_adjusted) == 8
    assert all(row["bonferroni_20_comparisons_99_75_percent_interval"][1] < 0 for row in tcell_adjusted)
    hepg2_cap8 = next(
        row for row in multiplicity["rows"]
        if row["context"] == "defined_hepg2_r1" and row["proposal_cap"] == 8
    )
    for key in (
        "bonferroni_20_comparisons_99_75_percent_interval",
        "bonferroni_within_context_5_comparisons_99_percent_interval",
    ):
        assert hepg2_cap8[key][0] < 0 < hepg2_cap8[key][1]
    paired = pd.read_csv(DATA / "figure2_paired_effects.csv").set_index("context")
    designed = paired.loc["Sample designed"]
    assert designed.ci_high < 0
    for value in (f"{designed.cap16_minus_cap1:+.3f}", f"{designed.ci_low:+.3f}", f"{designed.ci_high:+.3f}"):
        assert value in all_text
    mechanism = pd.read_csv(DATA / "strict_h1_figure3_score_matching.csv")
    assert len(mechanism) == 4 and ((mechanism.ci_low < 0) & (mechanism.ci_high > 0)).all()
    assert np.allclose(mechanism.optimized_mean_abs_error - mechanism.matched_mean_abs_error, mechanism.difference)
    common = pd.read_csv(DATA / "strict_h1_figure4_cross_architecture_summary.csv")
    assert len(common) == 4 and common.candidate_count.sum() == 4096 and (common.difference_ci_high < 0).all()
    assert f"{common.architecture_error_correlation.min():.3f}–{common.architecture_error_correlation.max():.3f}" in all_text
    noise = pd.read_csv(DATA / "extended_data_noise.csv")
    assert len(noise) == 15 and noise.noise_sigma.nunique() == 5 and set(noise.proposal_cap) == {1, 4, 16}
    assert np.allclose(noise[noise.proposal_cap == 1].difference_from_sigma0, 0)
    vgsc = pd.read_csv(DATA / "extended_data_vgsc.csv").set_index("context")
    assert vgsc.loc["Designed outer", "ci_low"] > 0
    assert vgsc.loc["Eligible random outer", "ci_low"] < 0 < vgsc.loc["Eligible random outer", "ci_high"]
    row = vgsc.loc["Designed outer"]
    for value in (f"{row.difference:+.3f}", f"{row.ci_low:+.3f}", f"{row.ci_high:+.3f}"):
        assert value in all_text
    audit_claims = [
        "2,753,885", "913,643", "3,193", "5,507,770", "776,514", "58,424", "3,163",
        "2,695", "96,043", "5,710", "1,857,006", "1,581,588",
    ]
    for value in audit_claims:
        assert value.replace(",", "") in audit.replace(",", "") and value in all_text, value
    required = [
        "不是“全量 Gate 0 PASS”", "不得包装成新盲测", "retrained outer",
        "严格 H1 重建通过", "实际 CUDA", "禁止 CPU 静默降级", "未启动", "待数据",
        "H2 仅作预声明诊断", "不能据此建议注入噪声", "3.4041", "3.2983",
        "STOP_NO_REPAIR_HEADROOM", "full-Q", "未运行 acquisition 对比，不声称优势",
        "cap 2–16 名义及全20对比区间均负", "HepG2 cap8 仅探索性", "全20对比 Bonferroni",
        "Random 始终为单候选基线", "不是等候选预算比较", "Bonferroni 仅保守敏感性",
        "不是新确证分析", "名义 95% CI",
    ]
    for phrase in required:
        assert phrase in all_text, phrase
    forbidden = [
        "四项独立研究验证已完成", "VGSC PASS", "retrained outer PASS", "新盲测已完成",
        "H2-clean", "已完成严格近邻外部验证", "全量 Gate 0 PASS。", "强 H1 失败",
        "强近邻 OOD 仍受阻", "Q 在所有池上 abstain", "外部情境｜更高 BoN 压力没有稳定跨情境获益",
    ]
    for phrase in forbidden:
        assert phrase not in all_text, phrase
    return {
        "status": "PASS",
        "strict_pressure_effects_recomputed": len(pressure),
        "tcell_negative_intervals_caps_2_to_16": 8,
        "tcell_all_20_bonferroni_negative_intervals": 8,
        "hepg2_cap8_multiplicity_adjusted_intervals_include_zero": True,
        "random_single_candidate_and_unequal_budget_boundary": True,
        "designed_effect": designed.cap16_minus_cap1,
        "score_matching_intervals_cross_zero": int(((mechanism.ci_low < 0) & (mechanism.ci_high > 0)).sum()),
        "common_error_pool_estimates": int(common.outer_pools.sum()),
        "noise_rows": len(noise),
        "historical_vgsc_comparisons": len(vgsc),
        "q_recalibration_boundary": "PASS",
        "paper3_no_repair_headroom_boundary": "PASS",
        "audit_claims_cross_checked": len(audit_claims),
        "scientific_boundary": "PASS",
        "pending_results_claimed_complete": False,
    }


def round_artifact(prs: Presentation, texts: list[str]) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["main_slides"] == 10 and config["backup_slides"] == 4
    assert "HepG2 cap8 crosses zero" in config["scientific_boundary"]["multiplicity"]
    assert "not an equal-candidate-budget comparison" in config["scientific_boundary"]["comparison_budget"]
    multiplicity_source = config["sources"]["multiplicity"]
    assert sha256(ROOT / multiplicity_source["path"]) == multiplicity_source["sha256"]
    embedded = media_hashes(PPT)
    figure_results = {}
    for key, source in config["sources"]["figures"].items():
        path = ROOT / source["path"]
        current_hash = sha256(path)
        assert current_hash == source["sha256"]
        assert current_hash in embedded, key
        with Image.open(path) as im:
            im.verify()
        figure_results[key] = {"sha256": current_hash, "embedded_exactly": True}
    picture_slides = []
    for index, slide in enumerate(prs.slides, start=1):
        count = sum(shape.shape_type == MSO_SHAPE_TYPE.PICTURE for shape in slide.shapes)
        if count:
            picture_slides.append(index)
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0, (index, shape.name)
            assert shape.left + shape.width <= prs.slide_width + 2, (index, shape.name, "x")
            assert shape.top + shape.height <= prs.slide_height + 2, (index, shape.name, "y")
    assert picture_slides == [4, 5, 6, 7, 8]
    doc = fitz.open(PDF)
    assert len(doc) == 14
    pdf_text = "\n".join(page.get_text() for page in doc)
    compact_pdf_text = "".join(pdf_text.split())
    for phrase in ("全20对比 Bonferroni", "Random 始终为单候选基线", "不是等候选预算比较", "不是新确证分析"):
        assert phrase.replace(" ", "") in compact_pdf_text, phrase
    assert "HepG2仅cap8不跨0" not in compact_pdf_text
    outside = []
    collisions = []
    render_stats = []
    for index, page in enumerate(doc, start=1):
        assert abs(page.rect.width - 960) < 0.1 and abs(page.rect.height - 540) < 0.1
        for word in page.get_text("words"):
            if word[0] < -0.5 or word[1] < -0.5 or word[2] > page.rect.width + 0.5 or word[3] > page.rect.height + 0.5:
                outside.append([index, word[4]])
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                A, B = blocks[a], blocks[b]
                dx = min(A[2], B[2]) - max(A[0], B[0])
                dy = min(A[3], B[3]) - max(A[1], B[1])
                if dx > 1 and dy > 1:
                    collisions.append([index, A[4][:40], B[4][:40]])
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        array = np.asarray(image)
        assert array.std() > 15 and len(np.unique(array.reshape(-1, 3), axis=0)) > 100
        render_stats.append({"slide": index, "pixels": [image.width, image.height], "std": round(float(array.std()), 2)})
        assert texts[index - 1].split("\n")[0] in page.get_text()
    assert not outside, outside
    assert not collisions, collisions
    fonts = subprocess.check_output(["pdffonts", str(PDF)], text=True)
    assert "ArialUnicodeMS" in fonts
    assert "FrankRuhlHofshi" not in fonts and "LinuxLibertineG" not in fonts
    return {
        "status": "PASS",
        "figures": figure_results,
        "figure_slides": picture_slides,
        "shape_bounds": "PASS",
        "pdf_pages": len(doc),
        "pdf_page_points": [960, 540],
        "pdf_words_outside_page": 0,
        "pdf_text_block_collisions": 0,
        "rendered_pages": len(render_stats),
        "render_stats": render_stats,
        "embedded_cjk_font": "ArialUnicodeMS",
        "pdf_text_extraction_chinese": "PASS: slide titles and body text extracted from every delivered page",
        "rejected_libreoffice_preview": "CJK glyph failure; not delivered",
    }


def round_zip_rebuild(prs: Presentation, texts: list[str]) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with zipfile.ZipFile(PPT) as zf:
        assert zf.testzip() is None
        names = set(zf.namelist())
        assert "[Content_Types].xml" in names and "ppt/presentation.xml" in names
        slide_xml = [name for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
        assert len(slide_xml) == 14
        assert len([name for name in names if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")]) == 0
    old_after = {name: sha256(PPT.parent / name) for name in config["old_deck_sha256_before"]}
    assert old_after == config["old_deck_sha256_before"]
    target_hash = sha256(PPT)
    STAGING.mkdir(parents=True, exist_ok=True)
    rebuilt = STAGING / "clean_rebuild.pptx"
    rebuilt_config = STAGING / "clean_rebuild_config.json"
    subprocess.run([
        sys.executable, str(BUILD), "--output", str(rebuilt), "--config", str(rebuilt_config),
    ], check=True, cwd=ROOT, capture_output=True, text=True)
    rebuilt_prs = Presentation(rebuilt)
    assert slide_texts(rebuilt_prs) == texts
    assert media_hashes(rebuilt) == media_hashes(PPT)
    assert len(rebuilt_prs.slides) == 14
    assert sha256(PPT) == target_hash
    assert PDF.exists() and PDF.stat().st_size > 1_000_000
    rebuilt.unlink()
    rebuilt_config.unlink()
    return {
        "status": "PASS",
        "zip_crc": "PASS",
        "slide_xml_parts": 14,
        "clean_rebuild_text_match": True,
        "clean_rebuild_media_match": True,
        "target_unchanged_by_rebuild": True,
        "old_decks_unchanged": True,
        "old_deck_sha256_after": old_after,
        "remote_connection": "read-only source inspection; no remote deliverable generation",
        "gpu_training_or_inference": False,
    }


def main():
    prs = Presentation(PPT)
    texts = slide_texts(prs)
    results = {
        "artifact": str(PPT.relative_to(ROOT)),
        "preview_pdf": str(PDF.relative_to(ROOT)),
        "slides": len(prs.slides),
        "main_slides": 10,
        "backup_slides": 4,
        "round_1_numeric_and_scientific_boundary": round_numeric(prs, texts),
        "round_2_images_artifacts_and_layout": round_artifact(prs, texts),
        "round_3_zip_clean_rebuild_and_preservation": round_zip_rebuild(prs, texts),
        "conclusions": {
            "scientific_results": "Strict-H1 retrospective protocol sensitivity: reproducible but context-dependent gains; no new blind test claimed.",
            "gate0": "Not a full Gate 0 PASS; strict-H1 exact/H1 checks pass, H2 remains diagnostic, and retrained outer remains NO-GO.",
            "vgsc_q": "Original pilot failed; Q recalibration improved log loss but not selection regret and added cost without outer rejection.",
            "paper3": "STOP_NO_REPAIR_HEADROOM; no acquisition comparison or advantage claimed.",
            "delivery": "PPTX and PowerPoint/Quartz preview PDF validated; LibreOffice preview rejected for CJK rendering.",
        },
    }
    results["status"] = "PASS"
    results["pptx_bytes"] = PPT.stat().st_size
    results["pptx_sha256"] = sha256(PPT)
    results["pdf_bytes"] = PDF.stat().st_size
    results["pdf_sha256"] = sha256(PDF)
    REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in results.items() if key in ("status", "slides", "pptx_bytes", "pptx_sha256", "pdf_bytes", "pdf_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
