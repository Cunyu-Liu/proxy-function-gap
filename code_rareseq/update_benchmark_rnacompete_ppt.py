from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pptx import Presentation

from build_benchmark_update_ppt import (
    INK, MUTED, NAVY, ORANGE, PALE_BLUE, PALE_ORANGE, PALE_TEAL, TEAL,
    add_footer, add_note, add_picture_fit, add_rect, add_text, add_title, sha256, slide_signature,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "生物序列长尾与OOD_基准证据更新_2026-11-26.pptx"
BASELINE = ROOT / ".pptx_build/pre_rnacompete_update.pptx"
WORK = ROOT / ".pptx_build/rnacompete_update_work.pptx"
AUDIT = ROOT / ".pptx_build/ppt_rnacompete_build_audit.json"


def clear_slide(slide) -> None:
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)


def build_mainline(slide) -> None:
    clear_slide(slide)
    add_title(slide, "当前科学主线：E1、E5 与 RNAcompete 五折均已完成", 31, 28)
    add_text(slide, "泄漏受控基准 + 强基线 + 误差分层；PAR-CLIP 与 E6 保持各自证据边界",
             0.55, 0.91, 12.23, 0.36, size=18, color=MUTED, name="Mainline subtitle")
    cards = [
        ("01", "数据合同已验收", "146 实验严格三折；fold=-1 不进入 CV train。", PALE_TEAL, TEAL),
        ("02", "主集与五折完成", "E1 12/12 CI 排除 0；RNAcompete 五折三 seed 完成。", PALE_BLUE, NAVY),
        ("03", "外部与蛋白分层", "PAR-CLIP 外部描述；E6 分类与 pseudo-OOD 保持独立边界。", PALE_TEAL, TEAL),
        ("04", "OOD 排序分歧", "E5 23/24 冲突；RNAcompete kNN 相对优势仍弱。", PALE_ORANGE, ORANGE),
    ]
    for i, (number, heading, body, fill, accent) in enumerate(cards):
        x = 0.55 + i * 3.07
        add_rect(slide, x, 1.52, 2.72, 3.55, fill, line="DCE6ED", name=f"Mainline card {number}")
        add_text(slide, number, x + 0.20, 1.76, 0.55, 0.34, size=18, color=accent, bold=True,
                 name=f"Mainline number {number}")
        add_text(slide, heading, x + 0.20, 2.24, 2.30, 0.62, size=21, color=NAVY, bold=True,
                 valign=3, name=f"Mainline heading {number}")
        add_text(slide, body, x + 0.20, 3.08, 2.30, 1.54, size=18, color=INK,
                 name=f"Mainline body {number}", margin=0)
        if i < 3:
            add_text(slide, "→", x + 2.68, 2.91, 0.38, 0.50, size=25, color=ORANGE, bold=True,
                     align=2, valign=3, name=f"Mainline arrow {i + 1}", margin=0)
    notes = [
        "E1：146 实验，五折三 seed，strict3train 最终验收。",
        "E5：72/72 完成；23/24 组指标冲突。",
        "RNAcompete：MLP 尾部占优；kNN OOD 相对优势，0.185 仍弱。",
    ]
    for i, text in enumerate(notes):
        add_note(slide, text, 0.55 + i * 4.11, 5.48, 3.93, 0.98, i + 1,
                 (PALE_BLUE, PALE_ORANGE, PALE_TEAL)[i], (NAVY, ORANGE, TEAL)[i])
    add_footer(
        slide,
        "证据等级：E1 M-test；E5 pilot fold0 三 seed 描述性；RNAcompete 五折三 seed 描述性；PAR-CLIP 与 E6 边界不变。",
        "Runs: E1-main-independent-recalculation-20261126-105611-ZDLg6Y · E5-holdout-convergence-summary-20261126-112101-ZDLg6Y · rnacompete-summarize-all-all-f0-s0-2026-11-27T024426.072076Z0000-4101649",
    )


def build_fig6(slide, path: Path) -> None:
    add_title(slide, "Fig. 6｜序列尾部与整RBP-OOD的方法排序不同", 37, 28)
    add_picture_fit(slide, path, 0.45, 1.00, 9.25)
    notes = [
        "序列尾部：MLP 占优；Spearman=0.475605，MLP−kNN=+0.123046。",
        "整 RBP-OOD：kNN 相对占优，但实验宏 Spearman=0.185360，仍弱。",
        "五折 × 三 seed；70/70 评价完成；描述性结果，非 SOTA 或可靠 OOD。",
    ]
    positions = [(1.08, 1.45), (2.75, 1.60), (4.54, 1.65)]
    for i, (text, (y, h)) in enumerate(zip(notes, positions)):
        add_note(slide, text, 9.91, y, 2.87, h, i + 1,
                 (PALE_BLUE, PALE_ORANGE, PALE_TEAL)[i], (NAVY, ORANGE, TEAL)[i])
    add_footer(
        slide,
        "证据等级：RNAcompete Ray2013 frozen RNA-FM 五折三 seed 描述性；序列 ID90 与整 RBP-OOD 分轨；非显著性、SOTA 或可靠 OOD。",
        "Runs: rnacompete-verify-final-all-all-f0-s0-2026-11-27T021412.509921Z0000-4099546 · rnacompete-summarize-all-all-f0-s0-2026-11-27T024426.072076Z0000-4101649",
    )


def main() -> None:
    baseline = Presentation(BASELINE)
    assert len(baseline.slides) == 36
    shutil.copy2(BASELINE, WORK)
    prs = Presentation(WORK)
    fig6 = ROOT / "figures/Fig6_RNAcompete_Ray2013_five_fold_sequence_and_RBP_OOD_300dpi.png"
    build_mainline(prs.slides[30])
    build_fig6(prs.slides.add_slide(prs.slide_layouts[0]), fig6)
    prs.core_properties.subject = "E1、E5 与 RNAcompete 五折最终验收；新增 Fig.6，保留其余页面"
    prs.core_properties.comments = "第31页加入RNAcompete五折完成，第37页新增Fig.6；其余35页保持原位原样。"
    prs.save(WORK)

    check = Presentation(WORK)
    assert len(check.slides) == 37
    unchanged = list(range(1, 31)) + list(range(32, 37))
    for number in unchanged:
        assert slide_signature(check.slides[number - 1]) == slide_signature(baseline.slides[number - 1]), number
    required = {
        31: ["RNAcompete 五折", "五折三 seed", "0.185", "MLP 尾部占优"],
        37: ["序列尾部与整RBP-OOD的方法排序不同", "0.475605", "+0.123046", "0.185360", "五折 × 三 seed", "70/70", "非 SOTA 或可靠 OOD"],
    }
    records = []
    for number in (31, 37):
        slide = check.slides[number - 1]
        text = "\n".join(shape.text for shape in slide.shapes if hasattr(shape, "text"))
        assert all(token in text for token in required[number]), (number, "missing token")
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= check.slide_width
            assert shape.top + shape.height <= check.slide_height
            if getattr(shape, "has_text_frame", False) and not shape.name.startswith("Footer"):
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        if run.text.strip():
                            assert run.font.size is not None and run.font.size.pt >= 18
        pictures = [shape for shape in slide.shapes if int(shape.shape_type) == 13]
        if number == 37:
            assert len(pictures) == 1
            assert hashlib.sha256(pictures[0].image.blob).hexdigest() == sha256(fig6)
        records.append({"slide": number, "required_tokens": required[number], "pictures": len(pictures),
                        "shape_count": len(slide.shapes)})
    shutil.copy2(WORK, TARGET)
    audit = {
        "status": "passed", "target": str(TARGET), "target_sha256": sha256(TARGET), "slides": 37,
        "modified_slides": [31], "added_slides": [37], "unchanged_slide_shape_text_geometry": unchanged,
        "new_slide_checks": records, "verified_png_sha256": sha256(fig6),
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "slides": 37, "sha256": audit["target_sha256"],
                      "target": str(TARGET), "audit": str(AUDIT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
