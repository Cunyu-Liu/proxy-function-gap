from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "生物序列长尾与OOD_三大会调研_RARE-Seq架构_M1更新_2026-11-24.pptx"
TARGET = ROOT / "生物序列长尾与OOD_基准证据更新_2026-11-26.pptx"
BUILD = ROOT / ".pptx_build"
AUDIT = BUILD / "ppt_build_audit.json"

NAVY = "0F4D92"
TEAL = "168C88"
ORANGE = "D97925"
INK = "20364B"
MUTED = "536B7A"
PALE_BLUE = "F3F7FB"
PALE_TEAL = "F1F8F7"
PALE_ORANGE = "FFF6ED"
WHITE = "FFFFFF"
FONT = "PingFang SC"
LATIN_FONT = "Arial"


def rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_run_font(run, size: float, color: str, bold: bool = False) -> None:
    run.font.name = LATIN_FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    rpr = run._r.get_or_add_rPr()
    rpr.set("lang", "zh-CN")
    for tag, typeface in (("a:latin", LATIN_FONT), ("a:ea", FONT), ("a:cs", FONT)):
        element = rpr.find(tag, rpr.nsmap)
        if element is None:
            element = OxmlElement(tag)
            rpr.append(element)
        element.set("typeface", typeface)


def add_text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 18,
    color: str = INK,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    name: str | None = None,
    margin: float = 0.02,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        shape.name = name
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.line_spacing = 1.0
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    set_run_font(run, size, color, bold)
    return shape


def add_rect(slide, x, y, w, h, fill, *, radius=True, line=None, name=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        shape.name = name
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if line:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(0.8)
    else:
        shape.line.fill.background()
    if radius:
        try:
            shape.adjustments[0] = 0.08
        except (IndexError, ValueError):
            pass
    return shape


def add_title(slide, title: str, page: int, size: float = 28) -> None:
    add_rect(slide, 0, 0, 13.333, 0.065, NAVY, radius=False, name="Top navy rule")
    add_rect(slide, 0, 0, 2.25, 0.065, TEAL, radius=False, name="Top teal rule")
    add_text(slide, title, 0.55, 0.25, 11.75, 0.58, size=size, color=NAVY, bold=True,
             valign=MSO_ANCHOR.MIDDLE, name="Conclusion title")
    add_text(slide, f"{page:02d}", 12.42, 0.29, 0.38, 0.36, size=18, color=TEAL,
             bold=True, align=PP_ALIGN.RIGHT, valign=MSO_ANCHOR.MIDDLE, name="Page number")


def add_footer(slide, evidence: str, runs: str) -> None:
    add_rect(slide, 0.55, 6.76, 12.23, 0.012, "D8E2EA", radius=False, name="Footer rule")
    add_text(slide, evidence, 0.55, 6.84, 12.23, 0.20, size=9.5, color=MUTED,
             valign=MSO_ANCHOR.MIDDLE, name="Footer Evidence", margin=0)
    add_text(slide, runs, 0.55, 7.07, 12.23, 0.28, size=8.5, color=MUTED,
             valign=MSO_ANCHOR.TOP, name="Footer Runs", margin=0)


def add_note(slide, text: str, x: float, y: float, w: float, h: float, index: int, fill: str, accent: str) -> None:
    add_rect(slide, x, y, w, h, fill, name=f"Note {index} background")
    add_rect(slide, x, y, 0.055, h, accent, radius=False, name=f"Note {index} accent")
    add_text(slide, text, x + 0.18, y + 0.10, w - 0.31, h - 0.18, size=18, color=INK,
             valign=MSO_ANCHOR.MIDDLE, name=f"Note {index} text", margin=0)


def add_picture_fit(slide, path: Path, x: float, y: float, w: float):
    with Image.open(path) as image:
        height = w * image.height / image.width
    picture = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(height))
    picture.name = f"Verified main figure: {path.stem}"
    return picture


def slide_signature(slide) -> list:
    signature = []
    for shape in slide.shapes:
        signature.append({
            "type": int(shape.shape_type),
            "name": shape.name,
            "left": shape.left,
            "top": shape.top,
            "width": shape.width,
            "height": shape.height,
            "text": shape.text if hasattr(shape, "text") else None,
        })
    return signature


def build() -> None:
    BUILD.mkdir(exist_ok=True)
    shutil.copy2(SOURCE, TARGET)
    source_prs = Presentation(SOURCE)
    prs = Presentation(TARGET)
    assert len(prs.slides) == 30

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_title(slide, "当前科学主线：先控制测量合同，再限定外部与蛋白结论", 31, 28)
    add_text(slide, "泄漏受控基准（设计目标） → 强基线 → 误差分层；不是统一新架构的性能宣言",
             0.55, 0.91, 12.23, 0.36, size=18, color=MUTED, name="Mainline subtitle")
    cards = [
        ("01", "先审计测量合同", "id90 与双负例轨是设计；fold=-1 缺口阻断旧“零泄漏”表述。", PALE_ORANGE, ORANGE),
        ("02", "再看外部分层", "PAR-CLIP p20：9 RBP、3 seed；只作外部描述性。", PALE_TEAL, TEAL),
        ("03", "蛋白分类验收", "五 split、三 seed；BCE 相对 deep-kNN 全频率桶方向一致。", PALE_BLUE, NAVY),
        ("04", "拒识限定边界", "OOD v2 三指标同向；Test-2 只是 pseudo-OOD。", PALE_TEAL, TEAL),
    ]
    for i, (number, heading, body, fill, accent) in enumerate(cards):
        x = 0.55 + i * 3.07
        add_rect(slide, x, 1.52, 2.72, 3.55, fill, line="DCE6ED", name=f"Mainline card {number}")
        add_text(slide, number, x + 0.20, 1.76, 0.55, 0.34, size=18, color=accent, bold=True,
                 name=f"Mainline number {number}")
        add_text(slide, heading, x + 0.20, 2.24, 2.30, 0.62, size=21, color=NAVY, bold=True,
                 valign=MSO_ANCHOR.MIDDLE, name=f"Mainline heading {number}")
        add_text(slide, body, x + 0.20, 3.08, 2.30, 1.54, size=18, color=INK,
                 name=f"Mainline body {number}", margin=0)
        if i < 3:
            add_text(slide, "→", x + 2.68, 2.91, 0.38, 0.50, size=25, color=ORANGE, bold=True,
                     align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, name=f"Mainline arrow {i + 1}", margin=0)
    mainline_notes = [
        "已验收：E0 审计、E5 p20、E6 分类与 OOD v2。",
        "旧 E1 主集撤回；foldfix 重跑中，不展示旧性能。",
        "不主张 SOTA、统一机制或真实未知功能发现。",
    ]
    for i, text in enumerate(mainline_notes):
        add_note(slide, text, 0.55 + i * 4.11, 5.48, 3.93, 0.98, i + 1,
                 (PALE_BLUE, PALE_ORANGE, PALE_TEAL)[i], (NAVY, ORANGE, TEAL)[i])
    add_footer(
        slide,
        "证据等级：E0 设计/审计；E5 M-desc（外部描述性）；E6 5S-test 与 pseudo-OOD v2。当前无 foldfix 主集最终性能。",
        "Runs: E0-leakage-quant-20261122-102238-ZDLg6Y · E0-window-identity-20261122-104517-ZDLg6Y · E6-five-split-summary-20261125-080318-ZDLg6Y",
    )

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_title(slide, "Fig. 1｜fold=-1 审计揭示旧训练使用缺口", 32, 28)
    fig1 = ROOT / "figures/Fig1_E0_data_and_leakage_audit_300dpi.png"
    add_picture_fit(slide, fig1, 0.55, 1.02, 12.23)
    fig1_notes = [
        "150 RBP；73 个双细胞系 RBP；2,147,319 个有 fold 归属窗口。",
        "21,420 / 2,168,739 正例为 fold=-1；旧加载器将其带入每个 train 折。",
        "不主张零泄漏；旧 E1 主集撤回，foldfix 重跑中，最终性能留空。",
    ]
    for i, text in enumerate(fig1_notes):
        add_note(slide, text, 0.55 + i * 4.11, 5.20, 3.93, 1.30, i + 1,
                 (PALE_BLUE, PALE_ORANGE, PALE_TEAL)[i], (NAVY, ORANGE, TEAL)[i])
    add_footer(
        slide,
        "证据等级：E0 设计与数据审计（描述性）；fold=-1 缺口已确认。id90 是设计目标，不等于旧训练实现严格无泄漏。",
        "Runs: E0-leakage-quant-20261122-102238-ZDLg6Y · E0-window-identity-20261122-104517-ZDLg6Y",
    )

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_title(slide, "Fig. 3｜跨协议判别随 RBP 与负例定义而变", 33, 28)
    fig3 = ROOT / "figures/Fig3_E5_PARCLIP_cross_protocol_300dpi.png"
    add_picture_fit(slide, fig3, 1.00, 1.00, 11.33)
    fig3_notes = [
        "宏 AUROC（左→右）：0.646192 / 0.607869 / 0.655701。",
        "9 RBP；p20 / min_delta=1e-4；PAR-CLIP 不参与训练或选参。",
        "外部描述性：同协议 neg2 非真非位点；切片差≠提升，亦非 OOD 拒识。",
    ]
    for i, text in enumerate(fig3_notes):
        add_note(slide, text, 0.55 + i * 4.11, 5.82, 3.93, 0.80, i + 1,
                 (PALE_BLUE, PALE_TEAL, PALE_ORANGE)[i], (NAVY, TEAL, ORANGE)[i])
    add_footer(
        slide,
        "证据等级：M-desc｜PAR-CLIP hg38 v2 外部描述性；9 RBP × 3 seed；mean±样本 SD，不是生物重复或显著性检验。",
        "Runs: E5-parclip-cross-protocol-contract-p20-20261125-065228-ZDLg6Y · E5-parclip-cross-protocol-contract-p20-20261125-065619-ZDLg6Y · E5-parclip-cross-protocol-contract-p20-20261125-065953-ZDLg6Y",
    )

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    add_title(slide, "Fig. 4｜BCE 五划分全桶占优；拒识仅属 pseudo-OOD", 34, 27)
    fig4 = ROOT / "figures/Fig4_E6_GRIMM_five_split_frequency_and_OOD_v2_300dpi.png"
    add_picture_fit(slide, fig4, 0.45, 1.02, 9.25)
    fig4_notes = [
        "Test-1：20/20 频率单元格 Δmacro AUPRC>0，且折内 95% CI 排除 0。",
        "四桶均值：+0.270534 / +0.344729 / +0.313893 / +0.203189。",
        "pseudo-OOD v2：MSP/energy 三指标 5/5 描述性优于 deep-kNN；非真实未知 EC。",
    ]
    positions = [(1.08, 1.48), (2.75, 1.60), (4.54, 1.72)]
    for i, (text, (y, h)) in enumerate(zip(fig4_notes, positions)):
        add_note(slide, text, 9.91, y, 2.87, h, i + 1,
                 (PALE_BLUE, PALE_TEAL, PALE_ORANGE)[i], (NAVY, TEAL, ORANGE)[i])
    add_footer(
        slide,
        "证据等级：5S-test 分类；跨 split 仅等权描述；Test-2 pseudo-OOD v2；无 FPR95/AURC 逐 seed 区间或跨 split 显著性。",
        "Runs: E6-ood-metric-recompute-v2-20261125-072246-ZDLg6Y · E6-five-split-summary-20261125-080318-ZDLg6Y · E6-five-split-independent-verification-20261125-080348-ZDLg6Y",
    )

    prs.core_properties.title = "生物序列长尾与 OOD：基准证据更新"
    prs.core_properties.subject = "保留历史调研页；新增当前科学主线与已验收 Fig.1/3/4"
    prs.core_properties.author = source_prs.core_properties.author or "RARE-Seq"
    prs.core_properties.comments = "新增页仅使用 2026-11-25 已核验报告和 PNG；旧 E1 性能与未完成 E5 矩阵未加入。"
    prs.save(TARGET)

    check = Presentation(TARGET)
    assert len(check.slides) == 34
    assert [slide_signature(s) for s in list(check.slides)[:30]] == [slide_signature(s) for s in source_prs.slides]
    forbidden = ["0.02515", "0.02789", "E5-rbp-holdout", "2/72", "M1a", "nnPU"]
    required = {
        31: ["fold=-1", "foldfix", "pseudo-OOD"],
        32: ["21,420", "2,168,739", "2,147,319", "foldfix"],
        33: ["0.646192", "0.607869", "0.655701", "p20", "外部描述性"],
        34: ["20/20", "+0.270534", "+0.344729", "+0.313893", "+0.203189", "pseudo-OOD v2"],
    }
    new_slide_records = []
    for number in range(31, 35):
        slide = check.slides[number - 1]
        texts = [shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text]
        joined = "\n".join(texts)
        assert not any(token in joined for token in forbidden), (number, "forbidden token")
        assert all(token in joined for token in required[number]), (number, "missing required token")
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= check.slide_width
            assert shape.top + shape.height <= check.slide_height
            if hasattr(shape, "text_frame") and shape.has_text_frame and not shape.name.startswith("Footer"):
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        if run.text.strip():
                            assert run.font.size is not None and run.font.size.pt >= 18, (number, shape.name, run.text)
        pictures = [shape for shape in slide.shapes if int(shape.shape_type) == 13]
        if number >= 32:
            assert len(pictures) == 1
            embedded_hash = hashlib.sha256(pictures[0].image.blob).hexdigest()
            expected_path = {32: fig1, 33: fig3, 34: fig4}[number]
            assert embedded_hash == sha256(expected_path)
        new_slide_records.append({
            "slide": number,
            "title": texts[0],
            "shape_count": len(slide.shapes),
            "pictures": len(pictures),
            "required_tokens": required[number],
        })

    audit = {
        "status": "passed",
        "source": str(SOURCE),
        "target": str(TARGET),
        "target_sha256": sha256(TARGET),
        "source_slide_count": len(source_prs.slides),
        "target_slide_count": len(check.slides),
        "historical_slides_shape_text_geometry_preserved": True,
        "new_slides": new_slide_records,
        "forbidden_old_E1_or_unfinished_E5_tokens_absent": True,
        "new_slide_non_footer_text_min_pt": 18,
        "verified_png_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in (fig1, fig3, fig4)},
    }
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"target": str(TARGET), "slides": len(check.slides), "audit": str(AUDIT), "status": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    build()
