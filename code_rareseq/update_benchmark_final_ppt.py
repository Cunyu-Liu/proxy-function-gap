from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pptx import Presentation

from build_benchmark_update_ppt import (
    INK,
    MUTED,
    NAVY,
    ORANGE,
    PALE_BLUE,
    PALE_ORANGE,
    PALE_TEAL,
    TEAL,
    add_footer,
    add_note,
    add_picture_fit,
    add_rect,
    add_text,
    add_title,
    sha256,
    slide_signature,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "生物序列长尾与OOD_基准证据更新_2026-11-26.pptx"
BASELINE = ROOT / ".pptx_build/pre_final_update.pptx"
WORK = ROOT / ".pptx_build/final_update_work.pptx"
AUDIT = ROOT / ".pptx_build/ppt_final_build_audit.json"


def clear_slide(slide) -> None:
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)


def build_mainline(slide) -> None:
    clear_slide(slide)
    add_title(slide, "当前科学主线：E1 strict3train 与 E5 72/72 已完成", 31, 28)
    add_text(slide, "泄漏受控基准 + 强基线 + 误差分层；PAR-CLIP 与 E6 保持各自证据边界",
             0.55, 0.91, 12.23, 0.36, size=18, color=MUTED, name="Mainline subtitle")
    cards = [
        ("01", "数据合同已验收", "146 实验严格三折；fold=-1 不进入 CV train。", PALE_TEAL, TEAL),
        ("02", "主集基线已验收", "12/12 切片 CI 排除 0；仅比较 frozen BCE 与簇去重 kNN。", PALE_BLUE, NAVY),
        ("03", "外部分层保留", "PAR-CLIP 外部描述；E6 分类与 pseudo-OOD 保持独立边界。", PALE_TEAL, TEAL),
        ("04", "OOD 无统一赢家", "72/72 完成；23/24 组指标冲突，不能称可靠统一拒识。", PALE_ORANGE, ORANGE),
    ]
    for i, (number, heading, body, fill, accent) in enumerate(cards):
        x = 0.55 + i * 3.07
        add_rect(slide, x, 1.52, 2.72, 3.55, fill, line="DCE6ED", name=f"Mainline card {number}")
        add_text(slide, number, x + 0.20, 1.76, 0.55, 0.34, size=18, color=accent, bold=True,
                 name=f"Mainline number {number}")
        add_text(slide, heading, x + 0.20, 2.24, 2.30, 0.62, size=21, color=NAVY, bold=True,
                 name=f"Mainline heading {number}")
        add_text(slide, body, x + 0.20, 3.08, 2.30, 1.54, size=18, color=INK,
                 name=f"Mainline body {number}", margin=0)
        if i < 3:
            add_text(slide, "→", x + 2.68, 2.91, 0.38, 0.50, size=25, color=ORANGE, bold=True,
                     align=2, valign=3, name=f"Mainline arrow {i + 1}", margin=0)
    notes = [
        "E1：146 实验，五折三 seed，strict3train 最终验收。",
        "E5：72/72 完成；23/24 组指标冲突。",
        "仅限既定基线与描述性分层；非 SOTA 或统一拒识。",
    ]
    for i, text in enumerate(notes):
        add_note(slide, text, 0.55 + i * 4.11, 5.48, 3.93, 0.98, i + 1,
                 (PALE_BLUE, PALE_ORANGE, PALE_TEAL)[i], (NAVY, ORANGE, TEAL)[i])
    add_footer(
        slide,
        "证据等级：E1 M-test；E5 pilot fold0 test 三 seed 描述性；PAR-CLIP M-desc；E6 5S-test 与 pseudo-OOD v2。",
        "Runs: E1-main-independent-audit-20261126-032635-ZDLg6Y · E1-main-independent-recalculation-20261126-105611-ZDLg6Y · E5-holdout-convergence-summary-20261126-112101-ZDLg6Y",
    )


def build_fig1(slide, path: Path) -> None:
    clear_slide(slide)
    add_title(slide, "Fig. 1｜strict3train 数据合同已修复并独立验收", 32, 28)
    add_picture_fit(slide, path, 0.55, 1.02, 12.23)
    notes = [
        "146 实验、5 fold、3 seed 完整；train 只含三个显式折。",
        "fold=-1 不进入 CV train；预测键与原始 split 对齐。",
        "历史加载器问题已修复；不主张预训练语料无污染。",
    ]
    for i, text in enumerate(notes):
        add_note(slide, text, 0.55 + i * 4.11, 5.12, 3.93, 1.34, i + 1,
                 (PALE_BLUE, PALE_TEAL, PALE_ORANGE)[i], (NAVY, TEAL, ORANGE)[i])
    add_footer(
        slide,
        "证据等级：E0 设计与历史审计；当前 E1 strict3train 数据合同独立验收。旧簇检查值不代表当前严格三折泄漏率。",
        "Runs: E0-leakage-quant-20261122-102238-ZDLg6Y · E0-window-identity-20261122-104517-ZDLg6Y · E1-main-independent-audit-20261126-032635-ZDLg6Y · E1-main-independent-recalculation-20261126-105611-ZDLg6Y",
    )


def build_fig2(slide, path: Path) -> None:
    add_title(slide, "Fig. 2｜strict3train 主集 12 切片均为 BCE 相对 kNN 优势", 35, 27)
    add_picture_fit(slide, path, 0.45, 1.02, 9.25)
    notes = [
        "12/12 切片差值为正，95% CI 均排除 0；10,000 次逐窗配对 bootstrap。",
        "尾部 ΔAUROC：+0.031365 / +0.024114；ΔAUPRC：+0.043649 / +0.034854（neg1c/neg2）。",
        "146 实验、5 fold × 3 seed；仅相对簇去重余弦 kNN，非 SOTA。",
    ]
    positions = [(1.08, 1.40), (2.62, 2.28), (5.05, 1.20)]
    for i, (text, (y, h)) in enumerate(zip(notes, positions)):
        add_note(slide, text, 9.91, y, 2.87, h, i + 1,
                 (PALE_BLUE, PALE_TEAL, PALE_ORANGE)[i], (NAVY, TEAL, ORANGE)[i])
    add_footer(
        slide,
        "证据等级：M-test｜146 实验、五折三 seed、预指定逐窗配对 bootstrap；三 seed 为统计重复，不是独立生物重复。",
        "Runs: E1-main-independent-audit-20261126-032635-ZDLg6Y · E1-paired-bootstrap-20261126-065044-ZDLg6Y · E1-main-independent-recalculation-20261126-105611-ZDLg6Y",
    )


def build_fig5(slide, path: Path) -> None:
    add_title(slide, "Fig. 5｜23/24 指标冲突：OOD 近随机且无统一赢家", 36, 28)
    add_picture_fit(slide, path, 0.45, 1.00, 7.05)
    notes = [
        "72/72 pair、216 行完成；23/24 RBP×轨组存在指标赢家冲突。",
        "deep-kNN AUROC 仅 0.507697 / 0.508286，FPR95 达 0.947266 / 0.945827。",
        "AURC 由 energy / MSP 分居双轨最佳；pilot fold0 三 seed 描述性，无统一赢家。",
    ]
    positions = [(1.12, 1.45), (2.80, 1.55), (4.58, 1.45)]
    for i, (text, (y, h)) in enumerate(zip(notes, positions)):
        add_note(slide, text, 7.80, y, 4.98, h, i + 1,
                 (PALE_BLUE, PALE_ORANGE, PALE_TEAL)[i], (NAVY, ORANGE, TEAL)[i])
    add_footer(
        slide,
        "证据等级：pilot fold0 test｜12 RBP × 2 轨 × 3 seed；描述性宏平均，无 OOD bootstrap 或显著性检验，非独立生物重复。",
        "Run: E5-holdout-convergence-summary-20261126-112101-ZDLg6Y（72 pair、216 行、missing=[]；独立三遍复核 complete）",
    )


def main() -> None:
    assert TARGET.exists() and BASELINE.exists()
    baseline = Presentation(BASELINE)
    assert len(baseline.slides) == 34
    shutil.copy2(BASELINE, WORK)
    prs = Presentation(WORK)
    fig1 = ROOT / "figures/Fig1_E0_data_and_leakage_audit_300dpi.png"
    fig2 = ROOT / "figures/Fig2_E1_main_strict3train_five_fold_three_seed_300dpi.png"
    fig5 = ROOT / "figures/Fig5_E5_whole_RBP_holdout_three_seed_OOD_300dpi.png"
    build_mainline(prs.slides[30])
    build_fig1(prs.slides[31], fig1)
    build_fig2(prs.slides.add_slide(prs.slide_layouts[0]), fig2)
    build_fig5(prs.slides.add_slide(prs.slide_layouts[0]), fig5)
    prs.core_properties.subject = "E1 strict3train 与 E5 72/72 最终验收；保留 PAR-CLIP、E6 与历史页"
    prs.core_properties.comments = "第31/32页更新，第35/36页新增；其余页保持。只用2026-11-26最终验收报告与PNG。"
    prs.save(WORK)

    check = Presentation(WORK)
    assert len(check.slides) == 36
    unchanged = list(range(1, 31)) + [33, 34]
    for number in unchanged:
        assert slide_signature(check.slides[number - 1]) == slide_signature(baseline.slides[number - 1]), number
    required = {
        31: ["E1 strict3train", "E5 72/72", "23/24", "非 SOTA"],
        32: ["146 实验", "fold=-1 不进入 CV train", "独立验收", "预训练语料无污染"],
        35: ["12/12", "+0.031365", "+0.024114", "+0.043649", "+0.034854", "非 SOTA"],
        36: ["72/72", "23/24", "0.507697", "0.508286", "0.947266", "0.945827", "无统一赢家"],
    }
    forbidden = ["foldfix 重跑中", "0.02515", "0.02789", "E5-rbp-holdout-20261123"]
    records = []
    for number in (31, 32, 35, 36):
        slide = check.slides[number - 1]
        text = "\n".join(shape.text for shape in slide.shapes if hasattr(shape, "text"))
        assert all(token in text for token in required[number]), (number, "missing token")
        assert not any(token in text for token in forbidden), (number, "forbidden token")
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
        if number in (32, 35, 36):
            assert len(pictures) == 1
            expected = {32: fig1, 35: fig2, 36: fig5}[number]
            assert hashlib.sha256(pictures[0].image.blob).hexdigest() == sha256(expected)
        records.append({"slide": number, "required_tokens": required[number], "pictures": len(pictures),
                        "shape_count": len(slide.shapes)})
    shutil.copy2(WORK, TARGET)
    audit = {
        "status": "passed",
        "target": str(TARGET),
        "target_sha256": sha256(TARGET),
        "slides": len(check.slides),
        "modified_slides": [31, 32],
        "added_slides": [35, 36],
        "unchanged_slide_shape_text_geometry": unchanged,
        "new_slide_checks": records,
        "verified_png_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (fig1, fig2, fig5)},
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "target": str(TARGET), "sha256": audit["target_sha256"],
                      "slides": 36, "audit": str(AUDIT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
