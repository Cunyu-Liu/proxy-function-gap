from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "deliverables" / "figures"
DATA = FIG / "source_data"
AUDIT = ROOT / "deliverables" / "reports" / "docs" / "GATE0_READ_ONLY_AUDIT_20261124.md"
STRICT_REPORT = ROOT / "deliverables" / "reports" / "docs" / "PAPER1_STRICT_H1_RESULT.md"
MULTIPLICITY = ROOT / "deliverables" / "reports" / "source_data" / "paper1_strict_h1_v3" / "analysis" / "strict_h1_multiplicity_sensitivity.json"
Q_REPORT = ROOT / "deliverables" / "reports" / "q_actual_cost_recalibration_20261124.md"
PAPER3_REPORT = ROOT / "deliverables" / "reports" / "paper3_e0_repair_headroom_20261124.md"
OUT = ROOT / "deliverables" / "ppt" / "proxy_function_gap_scientific_update_20261124.pptx"
CONFIG = ROOT / "deliverables" / "ppt" / "proxy_function_gap_scientific_update_20261124_config.json"

W, H = 13.333, 7.5
FONT_LATIN = "Arial Unicode MS"
FONT_CJK = "Arial Unicode MS"
C = {
    "navy": "0B1F33", "blue": "0F4D92", "blue2": "3775BA", "teal": "42949E",
    "green": "4E8B5B", "green_light": "DDF3DE", "red": "B64342", "red_light": "F6CFCB",
    "violet": "9A4D8E", "ink": "272727", "gray": "5F6B76", "mid": "8A96A3",
    "line": "DCE3EA", "bg": "F5F7FA", "white": "FFFFFF", "blue_light": "E8F0F8", "gray_light": "E8EDF2",
    "teal_light": "E8F4F4", "amber": "9A6700", "amber_light": "FFF3D6",
}
FIGURES = {
    "pressure": FIG / "Figure2_strict_h1_selection_pressure.png",
    "mechanism": FIG / "Figure3_strict_h1_score_matching.png",
    "common": FIG / "Figure4_strict_h1_common_proxy_error.png",
    "noise": FIG / "ExtendedData_proxy_noise.png",
    "vgsc": FIG / "ExtendedData_VGSC_paired_difference.png",
}


def rgb(key: str) -> RGBColor:
    return RGBColor.from_string(C.get(key, key))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_run(run, size=12, color="ink", bold=False, italic=False):
    run.font.name = FONT_LATIN
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = rgb(color)
    rpr = run._r.get_or_add_rPr()
    rpr.set("lang", "zh-CN")
    for tag, face in (("a:latin", FONT_LATIN), ("a:ea", FONT_CJK), ("a:cs", FONT_LATIN)):
        node = rpr.find(qn(tag))
        if node is None:
            node = OxmlElement(tag)
            rpr.append(node)
        node.set("typeface", face)


def text(slide, value, x, y, w, h, size=12, color="ink", bold=False,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, margin=0.0,
         line_spacing=1.08, italic=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    lines = str(value).split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        p.space_after = Pt(0)
        run = p.add_run()
        run.text = line
        set_run(run, size, color, bold, italic)
    return box


def rect(slide, x, y, w, h, fill="white", line="line", radius=True, line_width=0.8):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(line_width)
    return shape


def line(slide, x, y, w, color="line", height=0.012):
    return rect(slide, x, y, w, height, color, None, False)


def pill(slide, label, x, y, w, fill="blue_light", color="blue", size=8.5):
    rect(slide, x, y, w, 0.30, fill, None)
    text(slide, label, x + 0.08, y + 0.045, w - 0.16, 0.20, size, color, True,
         PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)


def base_slide(prs, section, title_value, number, subtitle=None, backup=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb("bg")
    rect(slide, 0, 0, W, 0.075, "blue", None, False)
    text(slide, section, 0.60, 0.28, 8.5, 0.22, 8.5, "blue", True)
    text(slide, title_value, 0.60, 0.54, 12.1, 0.48, 25, "navy", True)
    if subtitle:
        text(slide, subtitle, 0.62, 1.04, 12.0, 0.30, 10.5, "gray")
    footer(slide, number, backup)
    return slide


def footer(slide, number, backup=False, source="严格 H1 三轮验证 + Gate 0 历史审计 · 2026-11-24"):
    line(slide, 0.60, 7.12, 12.12, "line")
    text(slide, source, 0.62, 7.19, 10.8, 0.16, 6.8, "gray")
    label = f"B{number}" if backup else f"{number:02d} / 10"
    text(slide, label, 11.75, 7.17, 0.95, 0.18, 7.5, "blue", True, PP_ALIGN.RIGHT)


def picture_contain(slide, path: Path, x, y, w, h):
    with Image.open(path) as im:
        ratio = im.width / im.height
    box_ratio = w / h
    if ratio > box_ratio:
        pw, ph = w, w / ratio
        px, py = x, y + (h - ph) / 2
    else:
        ph, pw = h, h * ratio
        py, px = y, x + (w - pw) / 2
    return slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))


def card_title(slide, label, x, y, w, color="blue"):
    text(slide, label, x, y, w, 0.27, 11, color, True)


def status_row(slide, y, status, status_fill, status_color, title_value, detail, x=0.65, w=12.0):
    rect(slide, x, y, w, 0.72, "white", "line")
    pill(slide, status, x + 0.16, y + 0.20, 1.10, status_fill, status_color, 8)
    text(slide, title_value, x + 1.48, y + 0.13, 3.30, 0.24, 11.5, "navy", True)
    text(slide, detail, x + 4.72, y + 0.12, w - 4.92, 0.47, 9.4, "gray")


def add_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb("navy")
    rect(slide, 0, 0, 0.13, H, "teal", None, False)
    text(slide, "PROXY–FUNCTION GAP  /  SCIENTIFIC UPDATE", 0.70, 0.52, 8.8, 0.28, 10, "8BCF8B", True)
    text(slide, "Proxy 高分，为什么仍可能选错？", 0.70, 1.05, 11.8, 0.65, 30, "white", True)
    text(slide, "科学问题与 regret：优化分数不等于优化功能", 0.72, 1.80, 10.8, 0.35, 15, "CFCECE")
    rect(slide, 0.72, 2.42, 5.65, 1.42, "12344F", None)
    text(slide, "regret(x) = f(x*) − f(x)", 1.02, 2.72, 5.05, 0.45, 23, "white", True, PP_ALIGN.CENTER)
    text(slide, "f：measured MRL  ·  x*：池内真实最优  ·  x：规则选中序列", 0.98, 3.30, 5.12, 0.24, 9, "CFCECE", False, PP_ALIGN.CENTER)
    rect(slide, 6.68, 2.42, 5.92, 1.42, "white", None)
    text(slide, "当前证据一句话", 6.98, 2.66, 5.3, 0.25, 10, "blue", True)
    text(slide, "固定池压力可产生真实收益，但高度依赖情境；\n严格 H1 重建仍是回顾性证据，不是新盲测。", 6.98, 3.02, 5.25, 0.58, 13, "navy", True)
    items = [
        ("01", "严格 H1 · T-cell", "cap 2–16 名义及全20对比区间均负\ncap16：−1.243 / −1.355；n=16 pools"),
        ("02", "证据边界", "HepG2 cap8 仅探索性；N50 无持续改善\n历史已测 outer 不得包装成新盲测"),
        ("03", "当前决策", "Q 重校准未改善选择 regret\nPaper3 因无修复空间按门禁停止"),
    ]
    for i, (num, title_value, body) in enumerate(items):
        x = 0.72 + i * 4.03
        rect(slide, x, 4.35, 3.75, 1.63, "12344F", None)
        text(slide, num, x + 0.22, 4.56, 0.45, 0.25, 10, "8BCF8B", True)
        text(slide, title_value, x + 0.72, 4.53, 2.78, 0.28, 12, "white", True)
        text(slide, body, x + 0.22, 5.03, 3.30, 0.65, 9.5, "CFCECE")
    text(slide, "2026-11-24  ·  本地整合真实结果  ·  未启动训练", 0.72, 6.72, 9.2, 0.25, 8.5, "8A96A3")
    text(slide, "01 / 10", 11.52, 6.72, 1.05, 0.25, 8.5, "8BCF8B", True, PP_ALIGN.RIGHT)


def add_contract(prs):
    slide = base_slide(prs, "02  DATA CONTRACT", "数据契约｜先冻结角色与边界，再谈功能提升", 2,
                       "数据齐备、角色守恒、近邻 OOD 是三道不同门禁；任何一道都不能由结果反推。")
    steps = [
        ("01", "原始与终点", "来源哈希、长度、有限 rl\n缺失标签不填 0 / 预测值"),
        ("02", "角色与隔离", "train / dev / calibration\n与 outer 按 exact、H1 组隔离"),
        ("03", "固定池与规则", "pool、cap、阈值与分析先冻结\nLearner 不含 measured label / regret"),
        ("04", "outer 解释", "historical measured-state 仅回顾性\nnever-evaluated 才能支撑新盲测"),
    ]
    for i, (num, title_value, body) in enumerate(steps):
        x = 0.62 + i * 3.08
        rect(slide, x, 1.48, 2.84, 1.40, "white", "line")
        pill(slide, num, x + 0.16, 1.66, 0.48, "blue_light", "blue", 8)
        text(slide, title_value, x + 0.76, 1.65, 1.85, 0.26, 11.5, "navy", True)
        text(slide, body, x + 0.16, 2.12, 2.50, 0.52, 9.3, "gray")
    text(slide, "研究级证据就绪度", 0.65, 3.18, 4.0, 0.28, 12, "navy", True)
    rows = [
        ("Sample 2019", "限定范围可复核", "original 非 never-evaluated；retrained NO-GO", "green"),
        ("Castillo-Hair 2024", "严格 H1 重建通过 / 回顾性", "双构建与跨 Sample exact/H1 检查通过；H2 仅诊断", "green"),
        ("Leppek 2022", "未就绪", "未找到可审计原始数据、manifest 或划分", "gray"),
        ("Schlusser 2024", "未就绪", "未找到可审计原始数据、manifest 或划分", "gray"),
    ]
    for i, (study, status, detail, color) in enumerate(rows):
        y = 3.56 + i * 0.57
        rect(slide, 0.65, y, 12.0, 0.49, "white", None)
        rect(slide, 0.65, y, 0.055, 0.49, color, None, False)
        text(slide, study, 0.86, y + 0.11, 2.55, 0.22, 10, "navy", True)
        text(slide, status, 3.48, y + 0.11, 2.85, 0.22, 9.5, color, True)
        text(slide, detail, 6.42, y + 0.10, 5.92, 0.25, 9.2, "gray")
    rect(slide, 0.65, 6.05, 12.0, 0.70, "amber_light", None)
    text(slide, "门禁原则", 0.88, 6.25, 1.05, 0.24, 10, "amber", True)
    text(slide, "不是“全量 Gate 0 PASS”；H2 仅作新增诊断，不得倒改冻结协议。", 2.05, 6.22, 10.15, 0.28, 11.5, "amber", True)


def add_sample(prs):
    paired = pd.read_csv(DATA / "figure2_paired_effects.csv").set_index("context").loc["Sample designed"]
    slide = base_slide(prs, "03  SAMPLE 2019", "Sample｜选择压力信号清晰，但权限边界同样清晰", 3,
                       "designed 的固定池配对效应支持提高 cap；该证据仍是历史已测 outer 的回顾性分析。")
    rect(slide, 0.62, 1.47, 4.00, 2.03, "blue", None)
    text(slide, "SAMPLE DESIGNED · N=16 POOLS", 0.90, 1.73, 3.45, 0.24, 8.5, "DDF3DE", True)
    text(slide, f"Δregret  {paired.cap16_minus_cap1:+.3f}", 0.90, 2.13, 3.40, 0.43, 23, "white", True)
    text(slide, f"cap 16 − cap 1  ·  95% CI [{paired.ci_low:+.3f}, {paired.ci_high:+.3f}]", 0.90, 2.72, 3.42, 0.28, 9.5, "white")
    text(slide, "负值有利于更高选择压力", 0.90, 3.10, 3.30, 0.22, 9, "DDF3DE")
    rect(slide, 0.62, 3.72, 4.00, 2.80, "white", "line")
    card_title(slide, "原始—清洗—终点守恒", 0.88, 3.98, 3.40)
    stats = [("2,753,885", "原始行"), ("913,643", "规范化唯一序列"), ("3,193", "空 rl 排除；保持缺失")]
    for i, (value, label) in enumerate(stats):
        y = 4.42 + i * 0.58
        text(slide, value, 0.90, y, 1.48, 0.28, 15, "blue", True)
        text(slide, label, 2.42, y + 0.04, 1.92, 0.25, 9.3, "gray")
    text(slide, "Sample random 仅 3 个 eligible pools；cap 效应区间跨 0。", 0.90, 6.12, 3.38, 0.26, 8.6, "gray")
    rect(slide, 4.90, 1.47, 7.75, 5.05, "white", "line")
    card_title(slide, "split-v2 角色与使用权限", 5.20, 1.75, 6.9)
    rows = [
        ("ORIGINAL", "限定范围 PASS", "historical outer vault 58,424 行；E1 eligible 3,163 行。\n只允许既有 fixed-pool、eGFP unmod、designed single-assay 等声明上下文。", "green"),
        ("RETRAINED", "NO-GO", "2,695 条重训练序列清单及关联谱系未知；\nquarantine_retrain_lineage_unknown = 96,043 行，outer vault = 0。", "red"),
        ("REPLAY", "0 行授权", "清洗阶段 5,710 行 trajectory-replay eligible；\n经训练、scaler、control 与谱系排除后，两个模型实际 replay eligible 均为 0。", "amber"),
    ]
    for i, (tag, status, body, color) in enumerate(rows):
        y = 2.25 + i * 1.15
        pill(slide, tag, 5.20, y, 1.18, f"{color}_light" if color != "amber" else "amber_light", color, 7.5)
        text(slide, status, 6.57, y + 0.02, 1.45, 0.25, 10.5, color, True)
        text(slide, body, 8.02, y - 0.01, 4.20, 0.78, 9.2, "gray")
        if i < 2:
            line(slide, 5.20, y + 0.92, 7.02, "line")
    rect(slide, 5.20, 5.88, 7.02, 0.42, "red_light", None)
    text(slide, "不得把 historical measured-state outer 写成 never-evaluated / 新盲测。", 5.38, 5.99, 6.64, 0.20, 9.4, "red", True)


def add_pressure(prs):
    slide = base_slide(prs, "04  STRICT H1", "严格 H1｜T-cell 可重复获益，N50 / HepG2 划定边界", 4,
                       "T-cell cap 2–16 名义及全20对比 Bonferroni 区间均负；HepG2 cap8 校正后跨0，仅探索性。")
    picture_contain(slide, FIGURES["pressure"], 0.62, 1.33, 12.08, 5.65)


def add_mechanism(prs):
    slide = base_slide(prs, "05  MECHANISM", "机制｜分数匹配后，优化器特异增量不稳定", 5,
                       "四个外部情境、各 16 pools；BoN cap 16 与池内最近 proxy-score match 的描述性比较。")
    picture_contain(slide, FIGURES["mechanism"], 0.62, 1.42, 12.08, 4.45)
    rect(slide, 0.72, 6.03, 12.0, 0.72, "blue_light", None)
    text(slide, "4 / 4 名义 95% CI 均跨 0", 0.98, 6.20, 3.55, 0.30, 14, "blue", True)
    text(slide, "未识别稳定的 BoN 特异误差增量；匹配不能消除所有分布偏移。", 4.45, 6.22, 7.88, 0.28, 11, "navy", True)


def add_common(prs):
    summary = pd.read_csv(DATA / "strict_h1_figure4_cross_architecture_summary.csv")
    slide = base_slide(prs, "06  COMMON ERROR", "共同误差｜换架构，并不等于换掉误差", 6,
                       "CNN ensemble 与 4-mer MLP 共享训练数据与终点定义；高度相关指向共同 proxy 误差。")
    picture_contain(slide, FIGURES["common"], 0.50, 1.35, 6.30, 5.62)
    rect(slide, 7.00, 1.48, 5.62, 1.30, "navy", None)
    text(slide, "within-pool error correlation", 7.30, 1.72, 4.95, 0.24, 9, "CFCECE", True)
    text(slide, f"r = {summary.architecture_error_correlation.min():.3f}–{summary.architecture_error_correlation.max():.3f}", 7.30, 2.08, 4.95, 0.40, 23, "white", True)
    rect(slide, 7.00, 3.02, 5.62, 2.05, "white", "line")
    card_title(slide, "四个外部情境", 7.28, 3.26, 4.90)
    for i, row in enumerate(summary.itertuples()):
        y = 3.72 + i * 0.30
        name = row.context.replace("Defined ", "")
        text(slide, name, 7.30, y, 2.85, 0.22, 10, "gray")
        text(slide, f"{row.architecture_error_correlation:.3f}", 10.80, y, 1.20, 0.22, 11, "blue", True, PP_ALIGN.RIGHT)
    rect(slide, 7.00, 5.30, 5.62, 1.30, "teal_light", None)
    text(slide, "64 pool estimates  ·  16 pools / context", 7.28, 5.53, 4.95, 0.28, 12, "teal", True)
    text(slide, "CNN 的 MAE 在四情境均更低且区间不跨 0；\n但高相关仍排除“独立功能验证”的解释。", 7.28, 5.94, 4.95, 0.48, 9.5, "gray")


def add_noise(prs):
    slide = base_slide(prs, "07  SENSITIVITY", "敏感性｜proxy 噪声并未被 BoN 单调放大", 7,
                       "N50 HEK293T · 16 pools · 5 个噪声水平 · cap 1 / 4 / 16；池内配对后再重采样。")
    picture_contain(slide, FIGURES["noise"], 0.62, 1.42, 12.08, 4.65)
    rect(slide, 0.70, 6.15, 12.0, 0.64, "amber_light", None)
    text(slide, "解释边界", 0.94, 6.34, 1.15, 0.24, 10, "amber", True)
    text(slide, "部分负向配对效应仅作探索性结果；cap 1 在此协议中不变，不能据此建议注入噪声。", 2.10, 6.31, 10.15, 0.28, 10.8, "amber", True)


def add_vgsc(prs):
    data = pd.read_csv(DATA / "extended_data_vgsc.csv").set_index("context")
    designed = data.loc["Designed outer"]
    slide = base_slide(prs, "08  VGSC + Q", "VGSC / Q｜概率校准改善，未转化为选择收益", 8,
                       "原试点与 dev-only Q 重校准均保留；outer 仅探索性回顾敏感性，未用于调参。")
    picture_contain(slide, FIGURES["vgsc"], 0.62, 1.40, 12.08, 4.25)
    rect(slide, 0.68, 5.77, 5.88, 1.02, "red_light", None)
    text(slide, "ORIGINAL PILOT · NO-GO", 0.94, 5.96, 2.80, 0.24, 10, "red", True)
    text(slide, f"Δ = {designed.difference:+.3f}  [{designed.ci_low:+.3f}, {designed.ci_high:+.3f}]  ·  n=16", 0.94, 6.30, 4.95, 0.26, 12, "red", True)
    text(slide, "历史 Q 全拒选；random n=3 区间跨 0", 3.35, 5.98, 2.95, 0.22, 8.5, "red", False, PP_ALIGN.RIGHT)
    rect(slide, 6.76, 5.77, 5.88, 1.02, "white", "line")
    text(slide, "Q REPAIR · 仍无选择收益", 7.02, 5.96, 3.25, 0.24, 10, "gray", True)
    text(slide, "calibration：V+Q 3.4041 > V-only 3.2983", 7.02, 6.30, 5.00, 0.26, 11.5, "navy", True)
    text(slide, "outer 无拒选；V+Q=V-only，成本增加", 9.72, 5.98, 2.50, 0.22, 8.5, "gray", False, PP_ALIGN.RIGHT)


def add_leakage(prs):
    slide = base_slide(prs, "09  LEAKAGE AUDIT", "泄漏审计｜严格 H1 通过，但不升级为前瞻盲测", 9,
                       "距离定义：50 nt 替换 Hamming；exact=0、H1=1、H2=2。H2 仅作预声明诊断。")
    status_row(slide, 1.48, "PASS", "green_light", "green", "Sample 原始—清洗—终点",
               "行数、哈希、有限标签、45 对 exact overlap 与终点覆盖一致；3,193 个缺失标签保持缺失。")
    status_row(slide, 2.30, "PASS*", "green_light", "green", "split-v2 角色与权限",
               "5,507,770 条分配守恒；未见 raw row、unit、exact/H1-group 跨角色冲突。* 仅限声明范围。")
    status_row(slide, 3.12, "PASS†", "green_light", "green", "Castillo 严格 H1 重建",
               "四情境各两次确定性构建；池内/跨池及跨 Sample exact/H1 均通过。† 回顾性协议敏感性。")
    rect(slide, 0.65, 4.08, 12.0, 1.33, "white", "line")
    text(slide, "H2 诊断：跨 Sample H2 配对数（不作为门禁）", 0.90, 4.30, 5.25, 0.26, 11, "navy", True)
    contexts = [("HepG2 r1", "6"), ("T-cell r1", "48"), ("T-cell r2", "44"), ("N50", "0")]
    for i, (label, value) in enumerate(contexts):
        x = 0.92 + i * 2.88
        text(slide, label, x, 4.72, 1.70, 0.22, 9.5, "gray")
        text(slide, value, x + 1.80, 4.64, 0.72, 0.34, 18, "amber" if value != "0" else "green", True, PP_ALIGN.RIGHT)
    text(slide, "池内/跨池 H2 配对数为 0 / 2 / 0 / 0；诊断不回改冻结规则。", 0.92, 5.12, 10.85, 0.22, 8.7, "gray")
    rect(slide, 0.65, 5.67, 12.0, 0.97, "red_light", None)
    text(slide, "附加边界", 0.90, 5.88, 1.25, 0.25, 10, "red", True)
    text(slide, "严格 H1 不是新盲测；T-cell 两情境仍属同一研究。retrained outer 仍 NO-GO，Leppek / Schlusser 数据仍缺失。", 2.12, 5.85, 10.00, 0.48, 10.3, "red", True)


def add_next(prs):
    slide = base_slide(prs, "10  NEXT GATE", "下一门禁｜先解决剩余谱系，再启动前瞻盲测", 10,
                       "所有待重跑 / 待冻结事项均为未来工作；本报告不把它们写成已完成。")
    gates = [
        ("G0a", "retrained 谱系闭环", "定位 2,695 条序列及全部关联 allocation components；未完成前 outer 保持 NO-GO。", "NO-GO", "red"),
        ("G0b", "严格 H1 协议敏感性", "双构建与 exact/H1 检查已完成；H2 仅诊断，历史 outer 不升级为新盲测。", "已完成", "green"),
        ("G0c", "研究级数据齐备", "Leppek / Schlusser 补齐原始数据、来源哈希、终点语义与独立划分。", "待数据", "gray"),
        ("G1", "新的前瞻盲测", "冻结 pool、阈值、分析与日志；记录实际 CUDA、禁止 CPU 静默降级；只用 never-evaluated outer。", "未启动", "blue"),
    ]
    for i, (gate, title_value, body, status, color) in enumerate(gates):
        y = 1.46 + i * 1.22
        rect(slide, 0.65, y, 8.25, 1.00, "white", "line")
        pill(slide, gate, 0.86, y + 0.18, 0.74, "blue_light" if color == "blue" else (f"{color}_light" if color != "amber" else "amber_light"), color, 8)
        text(slide, title_value, 1.82, y + 0.15, 2.60, 0.26, 11.5, "navy", True)
        text(slide, status, 7.42, y + 0.16, 1.10, 0.24, 9.5, color, True, PP_ALIGN.RIGHT)
        text(slide, body, 1.82, y + 0.52, 6.64, 0.34, 9.1, "gray")
    rect(slide, 9.18, 1.46, 3.47, 2.15, "navy", None)
    text(slide, "当前决策", 9.46, 1.76, 2.90, 0.28, 11, "8BCF8B", True)
    text(slide, "不开启\nretrained outer", 9.46, 2.23, 2.90, 0.72, 20, "white", True)
    text(slide, "不把固定池结果升级为前瞻验证", 9.46, 3.14, 2.86, 0.25, 8.8, "CFCECE")
    rect(slide, 9.18, 3.85, 3.47, 2.49, "teal_light", None)
    text(slide, "Q / Paper3 已有报告进展", 9.46, 4.12, 2.90, 0.28, 11, "teal", True)
    text(slide, "• Q log loss 改善但 regret 未改善\n• 两类 outer 均无 Q 拒选，成本增加\n• Paper3：STOP_NO_REPAIR_HEADROOM\n• full-Q 与 zero-label regret 差 = 0\n• 未运行 acquisition 对比，不声称优势", 9.46, 4.57, 2.86, 1.42, 9.2, "navy")


def add_stats_backup(prs):
    slide = base_slide(prs, "BACKUP 01  METHODS", "备份｜统计与计算口径", 1,
                       "主报告只保留科学结论；重采样、预算与隐私口径集中在此。", True)
    cards = [
        ("区间与配对", "Pool-cluster bootstrap；10,000 draws\n图中为名义 95% CI；Bonferroni 仅保守敏感性\n20 对比 / 每情境 5 对比，不是新确证分析", "blue"),
        ("样本量", "多数情境 16 pools\nSample random / VGSC random：3 pools\n不是独立研究 meta-analysis", "teal"),
        ("BoN 候选预算", "Random 始终为单候选基线；BoN cap = 1–16\n压力曲线改变 BoN 候选数\n不是等候选预算比较", "violet"),
        ("隐私与产物", "严格 H1 仅匿名 pool index 与聚合误差\n无序列、sequence ID、measured-MRL label\n新三图均有 PNG / PDF / SVG", "green"),
    ]
    for i, (title_value, body, color) in enumerate(cards):
        x = 0.65 + (i % 2) * 6.10
        y = 1.50 + (i // 2) * 2.12
        rect(slide, x, y, 5.85, 1.82, "white", "line")
        rect(slide, x, y, 0.06, 1.82, color, None, False)
        text(slide, title_value, x + 0.28, y + 0.25, 5.20, 0.30, 13, color, True)
        text(slide, body, x + 0.28, y + 0.74, 5.15, 0.82, 10.2, "gray")
    rect(slide, 0.65, 5.88, 12.0, 0.72, "amber_light", None)
    text(slide, "VGSC 成本", 0.90, 6.09, 1.40, 0.26, 10.5, "amber", True)
    text(slide, "cap-16 CNN member budgets 匹配；V-model 成本额外。未进行 outer-pool threshold retuning。", 2.25, 6.07, 9.90, 0.28, 10.5, "amber", True)


def add_split_backup(prs):
    slide = base_slide(prs, "BACKUP 02  SPLIT-V2", "备份｜Sample split-v2 角色计数", 2,
                       "每模型 2,753,885 个唯一 raw row；每模型 776,514 个 allocation units。", True)
    rows = [
        ("train", "1,857,006", "1,581,588"), ("dev", "171,534", "32,417"),
        ("calibration", "127,962", "24,334"), ("query", "128,276", "23,872"),
        ("development", "26,906", "718,002"), ("historical_measured_state_outer", "58,424", "0"),
        ("official_seen_test", "123,880", "123,880"), ("official_boundary_ambiguous", "1,363", "1,363"),
        ("scaler_seen_quarantine", "124,113", "124,113"), ("control_nonexact", "28,273", "28,273"),
        ("quarantine_unsupported_context", "106,148", "0"), ("quarantine_retrain_lineage_unknown", "0", "96,043"),
        ("合计", "2,753,885", "2,753,885"),
    ]
    x0 = [0.70, 5.55, 8.55]
    headers = ["role", "original evolution", "retrained evolution"]
    rect(slide, 0.65, 1.46, 12.0, 0.42, "navy", None)
    for x, value in zip(x0, headers):
        text(slide, value, x, 1.56, 2.65 if x == x0[0] else 3.40, 0.22, 9.5, "white", True,
             PP_ALIGN.LEFT if x == x0[0] else PP_ALIGN.RIGHT)
    for i, row in enumerate(rows):
        y = 1.93 + i * 0.365
        fill = "blue_light" if i == len(rows) - 1 else ("white" if i % 2 == 0 else "bg")
        rect(slide, 0.65, y, 12.0, 0.34, fill, None, False)
        for j, (x, value) in enumerate(zip(x0, row)):
            text(slide, value, x, y + 0.065, 4.55 if j == 0 else 3.40, 0.20, 8.6,
                 "navy" if i == len(rows) - 1 else "gray", i == len(rows) - 1,
                 PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT)
    text(slide, "两模型合计 5,507,770 条分配；original E1 eligible = 3,163，retrained outer vault = 0。", 0.72, 6.78, 11.4, 0.22, 8.8, "gray")


def add_overlap_backup(prs):
    slide = base_slide(prs, "BACKUP 03  STRICT H1", "备份｜严格 H1 双构建与 H2 诊断明细", 3,
                       "每情境 16×64=1,024；exact/H1 为硬门禁，H2 仅为预声明诊断，不回改冻结规则。", True)
    headers = ["上下文", "独立确定性构建", "池内/跨池 exact/H1", "跨 Sample exact/H1", "池内/跨池 H2", "跨 Sample H2"]
    widths = [2.45, 1.80, 2.05, 2.05, 1.65, 1.80]
    xs = [0.65]
    for width in widths[:-1]:
        xs.append(xs[-1] + width)
    rect(slide, 0.65, 1.48, 11.80, 0.48, "navy", None)
    for x, width, value in zip(xs, widths, headers):
        text(slide, value, x + 0.06, 1.60, width - 0.12, 0.24, 8.2, "white", True, PP_ALIGN.CENTER)
    data_rows = [
        ("N50 HEK293T r1", "2 / hashes identical", "PASS", "PASS", "0", "0"),
        ("defined HepG2 r1", "2 / validated", "PASS", "PASS", "0", "6"),
        ("defined T-cell r1", "2 / validated", "PASS", "PASS", "2", "48"),
        ("defined T-cell r2", "2 / validated", "PASS", "PASS", "0", "44"),
    ]
    for i, row in enumerate(data_rows):
        y = 2.02 + i * 0.52
        rect(slide, 0.65, y, 11.80, 0.46, "white" if i % 2 == 0 else "bg", None, False)
        for j, (x, width, value) in enumerate(zip(xs, widths, row)):
            color = "amber" if j >= 4 and value != "0" else ("green" if value == "PASS" or value == "0" else "gray")
            text(slide, value, x + 0.08, y + 0.105, width - 0.16, 0.23, 8.7, color, j == 0 or color != "gray",
                 PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
    rect(slide, 0.65, 4.45, 12.0, 1.94, "white", "line")
    text(slide, "解释与历史记录边界", 0.90, 4.70, 3.40, 0.28, 11.5, "navy", True)
    text(slide, "• N50 learner hashes 与旧构建一致；HepG2 / T-cell 三情境已改变，旧数值仅作 pre-repair 历史记录。\n• H2 计数为配对数；未用于候选排除、模型选择、门禁或追求正结果的阈值调整。\n• 首轮 NPZ 缺少 utr50，在 CUDA 推理和科学评价前失败；日志保留，修正 retry 使用同一 Sample train NPZ。\n• 严格 H1 通过不消除细胞、文库、assay 或未知谱系偏移；所有 outer 仍是回顾性证据。", 0.90, 5.12, 11.15, 0.98, 8.9, "gray")


def add_evidence_backup(prs):
    slide = base_slide(prs, "BACKUP 04  PROVENANCE", "备份｜证据路径、审计范围与交付配置", 4,
                       "本地仅保存聚合证据；未复制远端原始大表、逐序列池、vault 或 outer 标签。", True)
    rect(slide, 0.65, 1.45, 7.50, 4.98, "white", "line")
    text(slide, "本地聚合证据", 0.92, 1.73, 6.90, 0.28, 12, "navy", True)
    paths = [
        "deliverables/reports/docs/PAPER1_STRICT_H1_RESULT.md",
        "deliverables/reports/docs/PAPER1_STRICT_H1_PROTOCOL.md",
        "deliverables/reports/source_data/paper1_strict_h1_v3/analysis/final_audit.json",
        "deliverables/reports/q_actual_cost_recalibration_20261124.md",
        "deliverables/reports/paper3_e0_repair_headroom_20261124.md",
        "deliverables/reports/docs/GATE0_READ_ONLY_AUDIT_20261124.md（历史审计）",
        "deliverables/figures/strict_h1_validation_report.json",
        "deliverables/figures/source_data/strict_h1_source_manifest.json",
        "paper1_strict_h1_v3/analysis/strict_h1_multiplicity_sensitivity.json",
        "paper1_strict_h1_v3/analysis/compute_strict_h1_multiplicity_sensitivity.py",
    ]
    for i, path in enumerate(paths):
        text(slide, path, 0.92, 2.20 + i * 0.39, 6.90, 0.24, 8.3, "gray")
    rect(slide, 8.42, 1.45, 4.23, 2.30, "navy", None)
    text(slide, "只读审计范围", 8.70, 1.73, 3.65, 0.28, 12, "8BCF8B", True)
    text(slide, "• 直接重读真实原始 / 派生表与 Learner 池\n• 未运行历史验证入口\n• 未把历史 PASS 当本轮结论\n• 未修改代码或启动训练\n• 审计收尾 HEAD：cc5e148", 8.70, 2.20, 3.55, 1.25, 9.1, "white")
    rect(slide, 8.42, 4.00, 4.23, 2.43, "teal_light", None)
    text(slide, "本次 deck 配置", 8.70, 4.28, 3.65, 0.28, 12, "teal", True)
    text(slide, "16:9 · 13.333×7.5 in\n10 页主报告 + 4 页备份\nArial Unicode MS（中英文统一无衬线）\n三张严格 H1 新图 + 两张历史扩展图\n本地 python-pptx / PowerPoint\n无 GPU 训练或推理；旧 deck 保留", 8.70, 4.74, 3.55, 1.35, 9.4, "navy")


def build(output: Path, config_path: Path = CONFIG):
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "Proxy–Function Gap：科学更新"
    prs.core_properties.subject = "严格 H1 选择压力、共同误差、Q/VGSC、Paper3 与证据边界"
    prs.core_properties.author = "Scientific update assembled locally"
    prs.core_properties.keywords = "proxy-function gap, regret, strict H1, Q, VGSC, Paper3, retrospective"
    prs.core_properties.comments = "10 main slides plus 4 backups; no new blinded evaluation claimed."
    add_cover(prs)
    add_contract(prs)
    add_sample(prs)
    add_pressure(prs)
    add_mechanism(prs)
    add_common(prs)
    add_noise(prs)
    add_vgsc(prs)
    add_leakage(prs)
    add_next(prs)
    add_stats_backup(prs)
    add_split_backup(prs)
    add_overlap_backup(prs)
    add_evidence_backup(prs)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output)
    old_decks = sorted(p for p in output.parent.glob("*.pptx") if p.name != output.name and not p.name.startswith("~$"))
    config = {
        "date": "2026-11-24",
        "output": str(output.relative_to(ROOT)),
        "slide_size_inches": [W, H],
        "main_slides": 10,
        "backup_slides": 4,
        "fonts": {"latin": FONT_LATIN, "cjk": FONT_CJK},
        "palette": C,
        "execution": {
            "local_assembly": True,
            "remote_connection": "read-only source inspection; no remote deliverable generation",
            "gpu_training_or_inference": False,
            "render_engine": "python-pptx; preview PDF exported by Microsoft PowerPoint / Quartz",
            "rejected_preview": "LibreOffice preview rejected after system OCR confirmed CJK glyph failure",
        },
        "scientific_boundary": {
            "outer_evidence": "retrospective strict-H1 fixed-pool / protocol-sensitivity; not a new blind test",
            "gate0": "not a full Gate 0 PASS",
            "strict_h1": "double-build exact/H1 checks passed; H2 diagnostic only",
            "multiplicity": "T-cell cap 2–16 remains negative under all-20 Bonferroni; HepG2 cap8 crosses zero under both corrections and is exploratory only",
            "comparison_budget": "Random is always a single-candidate baseline; BoN cap varies, so the pressure curve is not an equal-candidate-budget comparison",
            "intervals": "Figure intervals are nominal 95% CIs; Bonferroni is conservative sensitivity, not new confirmatory analysis",
            "retrained_outer": "NO-GO",
            "vgsc": "original fixed-pool pilot did not pass",
            "q_recalibration": "improved log loss without selection-regret improvement; no outer rejection and added cost",
            "paper3": "STOP_NO_REPAIR_HEADROOM; no acquisition comparison claimed",
            "pending_work_reported_as_complete": False,
        },
        "sources": {
            "audit": {"path": str(AUDIT.relative_to(ROOT)), "sha256": sha256(AUDIT)},
            "strict_report": {"path": str(STRICT_REPORT.relative_to(ROOT)), "sha256": sha256(STRICT_REPORT)},
            "multiplicity": {"path": str(MULTIPLICITY.relative_to(ROOT)), "sha256": sha256(MULTIPLICITY)},
            "q_report": {"path": str(Q_REPORT.relative_to(ROOT)), "sha256": sha256(Q_REPORT)},
            "paper3_report": {"path": str(PAPER3_REPORT.relative_to(ROOT)), "sha256": sha256(PAPER3_REPORT)},
            "figures": {key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for key, path in FIGURES.items()},
            "validation_report": str((FIG / "strict_h1_validation_report.json").relative_to(ROOT)),
            "source_manifest": str((DATA / "strict_h1_source_manifest.json").relative_to(ROOT)),
        },
        "old_deck_sha256_before": {p.name: sha256(p) for p in old_decks},
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "slides": len(prs.slides), "bytes": output.stat().st_size, "sha256": sha256(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    build(args.output.resolve(), args.config.resolve())
