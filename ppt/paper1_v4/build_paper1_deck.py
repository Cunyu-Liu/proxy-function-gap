from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from delivery_scripts.publication_paths import DELIVERABLES, EVIDENCE_ROOT, OUTPUT_ROOT

CANONICAL_PYTHON = Path(os.environ.get("PFG_CANONICAL_PYTHON", sys.executable)).expanduser().resolve()
if Path(sys.executable).resolve() != CANONICAL_PYTHON:
    if not CANONICAL_PYTHON.exists():
        raise RuntimeError(f"Canonical deck builder not found: {CANONICAL_PYTHON}")
    os.execv(str(CANONICAL_PYTHON), [str(CANONICAL_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import pptx
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = EVIDENCE_ROOT
FIG = DELIVERABLES / "figures" / "v4"
SRC = DELIVERABLES / "reports" / "source_data"
BUILD_OUT = OUTPUT_ROOT / "ppt" / "paper1_v4"
BUILD_OUT.mkdir(parents=True, exist_ok=True)
DIAG = SRC / "tcell_error_diagnosis_v1" / "analysis"
V1 = SRC / "tcell_target_adaptation_v1" / "target"
V2 = SRC / "tcell_listnet_adaptation_v2" / "target_v2"
ANCHOR = SRC / "tcell_cross_cell_anchor_v1" / "target"
RANK = SRC / "tcell_rank_bstrict_v2" / "artifacts" / "auxiliary_gate.json"
BACKBONE = SRC / "tcell_multicontext_backbone_v1" / "auxiliary_gate.json"
OUT = BUILD_OUT / "paper1_tcell_proxy_decision_20261126.pptx"
W, H = 13.333, 7.5
FONT = "PingFang SC"
C = {
    "ink": "17233B", "subink": "5D687A", "muted": "8B95A7", "line": "DDE3EB",
    "soft": "F4F7FA", "paper": "FFFFFF", "blue": "0F4D92", "blue_light": "EAF1F8",
    "teal": "238A7A", "teal_light": "E8F4F1", "coral": "C65D4B", "coral_light": "FAEEEB",
    "gold": "B8862D", "gold_light": "FBF4E7", "violet": "725A9C", "violet_light": "F0EDF6",
}


def rgb(key):
    return RGBColor.from_string(C[key])


def rect(slide, x, y, w, h, fill="paper", line=None, radius=False):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = rgb(fill)
    if line:
        shape.line.color.rgb = rgb(line); shape.line.width = Pt(0.8)
    else:
        shape.line.fill.background()
    shape.shadow.inherited = False
    return shape


def text(slide, x, y, w, h, value, size=14, color="ink", bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.08):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame; tf.clear(); tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(value.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = spacing
        r = p.add_run(); r.text = line
        r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = rgb(color)
    return shape


def pill(slide, x, y, w, value, kind="retrospective", size=9):
    schemes = {"diagnostic": ("gold", "gold_light"), "positive": ("teal", "teal_light"),
               "retrospective": ("blue", "blue_light"), "gap": ("coral", "coral_light"),
               "exploratory": ("violet", "violet_light")}
    color, light = schemes[kind]
    rect(slide, x, y, w, 0.32, light, radius=True)
    text(slide, x + 0.10, y + 0.055, w - 0.20, 0.22, value, size=size, color=color, bold=True, anchor=MSO_ANCHOR.MIDDLE)


def header(slide, section, title, subtitle, number):
    rect(slide, 0, 0, W, 0.10, "blue")
    text(slide, 0.55, 0.28, 3.4, 0.26, section, size=10, color="blue", bold=True)
    text(slide, 0.55, 0.64, 12.15, 0.52, title, size=25, color="ink", bold=True)
    text(slide, 0.55, 1.18, 12.15, 0.36, subtitle, size=11.5, color="subink")
    text(slide, 12.35, 7.15, 0.40, 0.20, f"{number:02d}", size=8.5, color="muted", align=PP_ALIGN.RIGHT)


def picture(slide, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih); pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), width=Inches(pw), height=Inches(ph))


def metric(slide, x, y, w, value, caption, color="blue", value_size=25):
    text(slide, x, y, w, 0.45, value, size=value_size, color=color, bold=True)
    text(slide, x, y + 0.54, w, 0.58, caption, size=11, color="subink", spacing=1.08)


def bullet_list(slide, x, y, w, items, size=13, gap=0.83):
    for i, (head, body, color) in enumerate(items):
        yy = y + i * gap
        rect(slide, x, yy + 0.06, 0.055, 0.48, color)
        text(slide, x + 0.18, yy, w - 0.18, 0.28, head, size=size, color=color, bold=True)
        text(slide, x + 0.18, yy + 0.31, w - 0.18, gap - 0.31, body, size=size - 1.3, color="subink", spacing=1.08)


def footer(slide, value):
    text(slide, 0.55, 7.05, 11.5, 0.25, value, size=8.5, color="muted")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


summary = load_json(DIAG / "summary.json")
v1p = load_json(V1 / "primary.json"); v1s = load_json(V1 / "secondary.json")
v2p = load_json(V2 / "primary.json"); v2r = load_json(V2 / "results.json")
anchorp = load_json(ANCHOR / "primary.json"); anchors = load_json(ANCHOR / "secondary.json")
rank = load_json(RANK); backbone = load_json(BACKBONE)
effects = summary["selection_effects"]
cap16 = {c: next(r for r in effects if r["context"] == c and r["model"] == "cnn" and r["contrast"] == "bon_minus_random_n" and r["cap"] == 16) for c in ("defined_tcell_r1", "defined_tcell_r2")}
assert all(cap16[c]["effect"] < 0 and cap16[c]["ci_high"] < 0 for c in cap16)
assert not summary["new_blind_test"] and not v1p["success"] and not v2p["success"] and not anchorp["success"]
assert not rank["target_evaluation_permitted"] and backbone["target_evaluation_count"] == 0

prs = Presentation(); prs.slide_width = Inches(W); prs.slide_height = Inches(H); blank = prs.slide_layouts[6]


def new():
    slide = prs.slides.add_slide(blank); rect(slide, 0, 0, W, H, "paper"); return slide


s = new(); rect(s, 0, 0, W, H, "ink"); rect(s, 0.78, 0.90, 0.75, 0.06, "teal")
text(s, 0.78, 1.22, 11.5, 0.35, "PAPER 1 · T-CELL PROXY–FUNCTION GAP", size=15, color="teal", bold=True)
text(s, 0.78, 1.85, 11.65, 1.40, "有效基线之后：\n真正的瓶颈位于高分尾部", size=36, color="paper", bold=True, spacing=1.10)
text(s, 0.78, 3.60, 11.5, 0.46, "有效基线  →  尾部瓶颈  →  回归–决策脱钩  →  设计指标", size=19, color="paper", bold=True)
rect(s, 0.78, 4.42, 5.55, 1.25, "blue", radius=True)
metric(s, 1.03, 4.63, 5.05, "−1.706 MRL", "T-cell r1 · cap16 等预算 Δregret\n95% CI [−2.169, −1.211]", "paper", 27)
rect(s, 6.58, 4.42, 5.55, 1.25, "teal", radius=True)
metric(s, 6.83, 4.63, 5.05, "−1.087 MRL", "T-cell r2 · cap16 等预算 Δregret\n95% CI [−1.812, −0.336]", "paper", 27)
text(s, 0.78, 6.20, 11.4, 0.55, "主阳性：同一 proposal prefix 下，BoN 的真实 MRL 选择优于 Random(N)。\n边界：历史 strict-H1 回顾性诊断；不宣称新盲测、普遍迁移或修复成功。", size=11.5, color="line")

s = new(); header(s, "01  SCIENTIFIC QUESTION", "科学问题不是“预测准不准”，而是“选择是否更好”", "同一候选预算、同一随机机会、同一实测 oracle，才能把排序价值从候选数效应中分离", 2)
steps = [
    ("01", "有效基线", "BoN(N) vs Random(N)\n共享 nested proposal prefix", "blue", "blue_light"),
    ("02", "尾部瓶颈", "regret = 机会损失\n+ 尾部内排序损失", "coral", "coral_light"),
    ("03", "回归–决策脱钩", "MAE、Spearman 与 cap16\nregret 分别报告", "violet", "violet_light"),
    ("04", "设计指标", "主终点锁定真实 MRL regret\n失败门禁先于目标评价", "teal", "teal_light"),
]
for i, (num, head, body, color, light) in enumerate(steps):
    x = 0.55 + i * 3.12
    rect(s, x, 1.85, 2.85, 3.55, light, radius=True)
    text(s, x + 0.22, 2.08, 0.65, 0.45, num, size=24, color=color, bold=True)
    text(s, x + 0.22, 2.78, 2.40, 0.42, head, size=18, color=color, bold=True)
    text(s, x + 0.22, 3.50, 2.40, 1.05, body, size=13, color="ink", bold=True, spacing=1.18)
    text(s, x + 0.22, 4.85, 2.40, 0.30, "→" if i < 3 else "✓", size=18, color=color, bold=True)
rect(s, 0.55, 5.78, 12.23, 0.92, "soft", radius=True)
text(s, 0.80, 6.00, 11.72, 0.48, "证据角色：16 pools × 5 proposal seeds / repeat；全部 cap 保留。目标适配的 test-v1、ListNet 的 test-v2、以及未开启 T-cell 的辅助门禁不可混写。", size=12, color="ink", bold=True)
footer(s, "主要效应：BoN(N) − Random(N) measured-MRL regret；负值有利。所有目标结果均为回顾性证据。")

s = new(); header(s, "02  VALID BASELINE → TAIL BOTTLENECK", "全局排序确实有用，但 top16 内顺序成为瓶颈", "Figure 16 · 高 MAE 与有效等预算排序可以并存；尾部损失需要单独定位", 3)
picture(s, FIG / "Figure16_v4_tcell_global_value_tail_bottleneck.png", 0.45, 1.52, 12.42, 5.35)
footer(s, "主阳性保留：r1/r2 cap16 等预算效应均为负且名义 CI 不跨 0；top16 尾部内损失均大于机会损失。")

s = new(); header(s, "03  MECHANISTIC LOCALIZATION", "尾部瓶颈不是抽象诊断，而是可分解的选择损失", "完整池 top16 第一名的 regret，可严格拆成 tail 未覆盖与 tail 内排序两部分", 4)
rect(s, 0.55, 1.72, 5.95, 2.05, "blue_light", radius=True)
text(s, 0.82, 1.96, 5.35, 0.35, "T-cell r1 · 总 regret 3.316", size=17, color="blue", bold=True)
metric(s, 0.82, 2.48, 2.40, "0.836", "机会损失\n25.2%", "gold", 24)
metric(s, 3.55, 2.48, 2.55, "2.480", "尾部内排序损失\n74.8%", "coral", 24)
rect(s, 6.80, 1.72, 5.98, 2.05, "teal_light", radius=True)
text(s, 7.07, 1.96, 5.35, 0.35, "T-cell r2 · 总 regret 4.071", size=17, color="teal", bold=True)
metric(s, 7.07, 2.48, 2.40, "1.953", "机会损失\n48.0%", "gold", 24)
metric(s, 9.80, 2.48, 2.55, "2.117", "尾部内排序损失\n52.0%", "coral", 24)
bullet_list(s, 0.72, 4.20, 5.85, [
    ("全局相关不等于尾部相关", "全局 Spearman 0.591/0.529；top16 内降至 0.206/0.144。", "violet"),
    ("第一名反复被翻转", "两个 repeat 都在 15/16 个池中发生尾部第一名翻转。", "coral"),
], size=13, gap=1.05)
rect(s, 6.80, 4.15, 5.98, 2.15, "soft", radius=True)
text(s, 7.07, 4.42, 5.42, 0.35, "对模型设计的直接含义", size=16, color="blue", bold=True)
text(s, 7.07, 4.98, 5.42, 0.92, "单调校准或全局 MAE 下降，最多改善分数解释；只有改变同预算下的实际选择，才能降低 regret。oracle 分解是上界诊断，不是已实现修复。", size=13, color="ink", bold=True, spacing=1.16)
footer(s, "来源：tcell_error_diagnosis_v1/top_tail_summary.tsv。机会损失与排序损失之和严格等于总 regret。")

s = new(); header(s, "04  REGRESSION–DECISION DECOUPLING", "改善回归指标，并没有自动改善 cap16 设计决策", "Figure 17 · test-v1 / test-v2 / paired cross-cell / 辅助门禁分层呈现", 5)
picture(s, FIG / "Figure17_v4_tcell_regression_decision_decoupling.png", 0.45, 1.50, 12.42, 5.38)
footer(s, "50k main、ListNet 与 HepG2-anchor 均未通过各自主门禁；辅助 pairwise/backbone 未评价 T-cell。")

s = new(); header(s, "05  TARGET ADAPTATION · TWO TESTS", "test-v1 与 test-v2 给出同一教训，但不能合并成一个测试", "两者均为 target-context retrospective grouped holdout；不同 component-disjoint 测试池、不同主要估计量", 6)
rect(s, 0.55, 1.72, 5.95, 4.85, "blue_light", radius=True)
pill(s, 0.82, 1.96, 2.30, "TEST-V1 · 50k MAIN", "retrospective")
text(s, 0.82, 2.50, 5.35, 0.42, "MAE 下降，cap16 regret 反向", size=18, color="blue", bold=True)
metric(s, 0.82, 3.18, 2.40, "−0.437653", "r1 MAE 变化\n1.352159 → 0.914506", "blue", 21)
metric(s, 3.55, 3.18, 2.40, "+0.038929", "r1 cap16 Δregret\n主终点变差", "coral", 21)
metric(s, 0.82, 4.45, 2.40, "−0.043556", "r2 MAE 变化\n1.062273 → 1.018716", "blue", 21)
metric(s, 3.55, 4.45, 2.40, "+0.012551", "r2 cap16 Δregret\n主终点变差", "coral", 21)
text(s, 0.82, 5.83, 5.30, 0.48, "合并 +0.025740 [−0.072019, +0.140994]；20k 次级点估计不替换预注册 50k 主结果。", size=10.5, color="subink", bold=True)
rect(s, 6.80, 1.72, 5.98, 4.85, "violet_light", radius=True)
pill(s, 7.07, 1.96, 2.50, "TEST-V2 · LISTNET", "exploratory")
text(s, 7.07, 2.50, 5.35, 0.42, "listwise 局部信号，联合主门禁失败", size=18, color="violet", bold=True)
metric(s, 7.07, 3.18, 5.25, "−0.037415", "vs 原 CNN · cap16 mean-replicate Δregret\n95% CI [−0.183204, +0.102494]", "violet", 23)
metric(s, 7.07, 4.45, 5.25, "−0.007585", "vs v1 main · cap16 mean-replicate Δregret\n95% CI [−0.028698, +0.012474]", "violet", 23)
text(s, 7.07, 5.83, 5.35, 0.48, "两项 CI 上界均 >0，改善均 <0.1 MRL；co-primary IUT FAIL。不能把 cap16 小幅点估计写成修复。", size=10.5, color="subink", bold=True)
footer(s, "test-v1 主分析分别约束两个 repeat；test-v2 主估计量为 per-sequence r1/r2 平均 MRL regret。")

s = new(); header(s, "06  NEGATIVE RESULTS RETAINED", "负结果限定“什么信息不够”，而不是证明所有方法无效", "跨细胞全局信息与辅助训练收益都不能替代目标细胞高分尾部的真实选择证据", 7)
rect(s, 0.55, 1.75, 3.85, 4.75, "coral_light", radius=True)
pill(s, 0.80, 2.00, 2.85, "PAIRED CROSS-CELL TEST", "gap", 8.5)
text(s, 0.80, 2.52, 3.35, 0.55, "HepG2-anchor\n一致反向", size=19, color="coral", bold=True)
metric(s, 0.80, 3.35, 3.25, "+0.428897", "合并 cap16 Δregret\n[+0.245460, +0.616474]", "coral", 23)
text(s, 0.80, 4.70, 3.25, 1.10, "HepG2 全局 ρ 略高，却使 T-cell r1/r2 分别变差 +0.463/+0.395。相同 proposal 机会下，损失来自选择本身。", size=11.5, color="ink", bold=True)
rect(s, 4.72, 1.75, 3.85, 4.75, "gold_light", radius=True)
pill(s, 4.97, 2.00, 2.55, "AUXILIARY GATE ONLY", "diagnostic", 8.5)
text(s, 4.97, 2.52, 3.35, 0.55, "Pairwise B-strict\n双门禁失败", size=19, color="gold", bold=True)
metric(s, 4.97, 3.35, 3.25, "0", "T-cell evaluation count\n未读取目标 MRL", "gold", 25)
text(s, 4.97, 4.70, 3.25, 1.10, "N50→HepG2：CI 跨 0；HepG2→N50：点估计反向。辅助目的地收益未迁移，按协议停止。", size=11.5, color="ink", bold=True)
rect(s, 8.89, 1.75, 3.89, 4.75, "soft", radius=True)
pill(s, 9.14, 2.00, 2.55, "AUXILIARY GATE ONLY", "diagnostic", 8.5)
text(s, 9.14, 2.52, 3.35, 0.55, "Multicontext backbone\n双门禁失败", size=18, color="blue", bold=True)
metric(s, 9.14, 3.35, 3.25, "0", "T-cell evaluation count\n未读取目标 MRL", "blue", 25)
text(s, 9.14, 4.70, 3.25, 1.10, "两个 destination auxiliary test 点估计均变差；N50→HepG2 区间完全位于有害方向。", size=11.5, color="ink", bold=True)
footer(s, "辅助门禁失败不是 T-cell 负效应估计；图中以 target evaluation=0 明确区分。")

s = new(); header(s, "07  DESIGN METRICS", "Paper 1 的方法学产出：把评价终点改成设计终点", "regret 不是附加指标，而是连接 proxy 排序与真实功能选择的主终点", 8)
rows = [
    ("主终点", "等预算 measured-MRL regret", "BoN(N) − Random(N)，共享 prefix 与完整池 oracle", "blue"),
    ("机制分解", "机会损失 + 尾部内排序损失", "区分 proposal 覆盖不足与 tail 内顺序错误", "coral"),
    ("次要指标", "MAE / RMSE / Spearman", "用于解释拟合与全局排序，不替代选择改善", "violet"),
    ("证据门槛", "方向 + CI + 最小实质改善", "两个 repeat / 两个 comparator 按预注册联合判定", "teal"),
]
for i, (role, name, rule, color) in enumerate(rows):
    y = 1.75 + i * 1.08
    rect(s, 0.55, y, 12.23, 0.90, "soft" if i % 2 == 0 else "paper", line="line", radius=True)
    text(s, 0.80, y + 0.20, 1.30, 0.30, role, size=12, color=color, bold=True)
    text(s, 2.20, y + 0.17, 4.15, 0.35, name, size=15, color="ink", bold=True)
    text(s, 6.55, y + 0.18, 5.85, 0.48, rule, size=11.5, color="subink")
rect(s, 0.55, 6.20, 12.23, 0.62, "ink", radius=True)
text(s, 0.80, 6.37, 11.70, 0.30, "决策规则：只降低 MAE、只提高全局相关、只改变选择或只在辅助集获益，均不能宣称 T-cell 功能设计改善。", size=11.5, color="paper", bold=True)
footer(s, "该指标框架由真实等预算阳性驱动；失败方法用于限定证据，而不是削弱已观察到的 BoN 基线价值。")

s = new(); header(s, "08  PAPER 1 TAKE-HOME", "可发表的主张：proxy 有价值，但其适用域必须写到高分尾部", "以真实等预算收益为主线，以尾部损失和回归–决策脱钩解释为何需要设计指标", 9)
rect(s, 0.55, 1.70, 7.35, 4.95, "blue_light", radius=True)
pill(s, 0.82, 1.95, 2.55, "SUPPORTED CLAIM", "positive")
text(s, 0.82, 2.47, 6.80, 0.95, "在既定 T-cell 池与预算内，\nproxy 排序带来可测量的真实 MRL 选择收益。", size=22, color="blue", bold=True, spacing=1.15)
bullet_list(s, 0.82, 3.72, 6.70, [
    ("收益不是候选数假象", "r1/r2 cap2–16 等预算点估计全部有利；cap16 名义 CI 均不跨 0。", "blue"),
    ("瓶颈可科学定位", "top16 尾部内排序损失大于机会损失；15/16 池第一名翻转。", "coral"),
    ("指标决定可证伪性", "回归改善不能替代 regret；新修复必须同预算、同 oracle、过门禁。", "teal"),
], size=12.5, gap=0.82)
rect(s, 8.20, 1.70, 4.58, 2.30, "coral_light", radius=True)
text(s, 8.47, 1.97, 4.05, 0.35, "明确不主张", size=17, color="coral", bold=True)
text(s, 8.47, 2.52, 4.00, 1.10, "不主张目标适配或 ListNet 已修复；不主张 HepG2 可直接锚定 T-cell；不把辅助门禁失败画成目标效果。", size=12.5, color="ink", bold=True)
rect(s, 8.20, 4.30, 4.58, 2.35, "soft", radius=True)
text(s, 8.47, 4.57, 4.05, 0.35, "下一项合法验证", size=17, color="teal", bold=True)
text(s, 8.47, 5.12, 4.00, 1.10, "在未触碰数据或新实验中，预注册 cap16 regret、等预算 Random(N)、机会/排序分解和最小改善门槛；目标标签不得回流选型。", size=12.5, color="ink", bold=True)
footer(s, "历史 outer 已查看：全部结果为回顾性证据。真正的独立生物验证仍需新数据或新实验。")

prs.save(OUT)
source_paths = [DIAG / "summary.json", V1 / "primary.json", V1 / "secondary.json", V2 / "primary.json", V2 / "results.json", ANCHOR / "primary.json", ANCHOR / "secondary.json", RANK, BACKBONE]
figure_paths = [FIG / "Figure16_v4_tcell_global_value_tail_bottleneck.png", FIG / "Figure17_v4_tcell_regression_decision_decoupling.png"]
config = {
    "deck": OUT.name, "slides": len(prs.slides), "size_inches": [W, H], "font": FONT,
    "runtime": {"python": platform.python_version(), "python_pptx": pptx.__version__},
    "story": "valid equal-budget baseline -> tail bottleneck -> regression-decision decoupling -> design metrics",
    "evidence_boundary": "retrospective; test-v1/test-v2 separated; auxiliary failures target evaluation=0; no failed method presented as improvement",
    "sources": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths],
    "figures": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in figure_paths],
}
(BUILD_OUT / "paper1_tcell_proxy_decision_20261126_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "PASS", "deck": OUT.name, "slides": len(prs.slides)}, ensure_ascii=False, indent=2))
