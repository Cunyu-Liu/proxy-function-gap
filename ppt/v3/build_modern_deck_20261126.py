from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from delivery_scripts.publication_paths import DELIVERABLES, EVIDENCE_ROOT, OUTPUT_ROOT

CANONICAL_PYTHON = Path(os.environ.get("PFG_CANONICAL_PYTHON", sys.executable)).expanduser()
if Path(sys.executable).resolve() != CANONICAL_PYTHON.resolve():
    if not CANONICAL_PYTHON.exists():
        raise RuntimeError(f"Configured deck builder not found: {CANONICAL_PYTHON}")
    os.execv(str(CANONICAL_PYTHON), [str(CANONICAL_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import pptx
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = EVIDENCE_ROOT
FIG = DELIVERABLES / "figures"
V3 = FIG / "v3"
OUT_DIR = OUTPUT_ROOT / "ppt" / "v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "proxy_function_gap_modern_20261126.pptx"
RISK_FROZEN = DELIVERABLES / "reports" / "source_data" / "paper2_risk_frozen_v1" / "frozen_protocol.json"
PERSIST_SUMMARY = DELIVERABLES / "reports" / "source_data" / "persist_mrl_external_v1" / "results" / "summary.json"
BUDGET_RESULTS = DELIVERABLES / "reports" / "source_data" / "paper1_budget_matched_v1" / "results.json"
SEQUENCE_RESULTS = DELIVERABLES / "reports" / "source_data" / "paper1_sequence_ablation_v1" / "analysis" / "results" / "summary.json"
W, H = 13.333, 7.5
FONT = "Arial Unicode MS"
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
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.08, margin=0):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear(); tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    for i, line in enumerate(value.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.line_spacing = spacing
        r = p.add_run(); r.text = line
        r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = rgb(color)
    return shape


def pill(slide, x, y, w, value, kind="diagnostic", size=9):
    schemes = {
        "diagnostic": ("gold", "gold_light"), "confirmatory": ("teal", "teal_light"),
        "retrospective": ("blue", "blue_light"), "gap": ("coral", "coral_light"),
    }
    color, light = schemes[kind]
    rect(slide, x, y, w, 0.32, light, radius=True)
    text(slide, x + 0.10, y + 0.055, w - 0.20, 0.22, value, size=size, color=color, bold=True,
         anchor=MSO_ANCHOR.MIDDLE)


def header(slide, section, title, subtitle=None, number=None):
    rect(slide, 0, 0, W, 0.10, "blue")
    text(slide, 0.55, 0.30, 2.2, 0.28, section, size=10, color="blue", bold=True)
    text(slide, 0.55, 0.65, 12.15, 0.55, title, size=25, color="ink", bold=True)
    if subtitle:
        text(slide, 0.55, 1.19, 12.15, 0.35, subtitle, size=11.5, color="subink")
    if number:
        text(slide, 12.35, 7.15, 0.40, 0.20, f"{number:02d}", size=8.5, color="muted", align=PP_ALIGN.RIGHT)


def picture(slide, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                    width=Inches(pw), height=Inches(ph))


def bullet_list(slide, x, y, w, items, size=13, gap=0.83):
    for i, (head, body, color) in enumerate(items):
        yy = y + i * gap
        rect(slide, x, yy + 0.06, 0.055, 0.48, color)
        text(slide, x + 0.18, yy, w - 0.18, 0.28, head, size=size, color=color, bold=True)
        text(slide, x + 0.18, yy + 0.31, w - 0.18, gap - 0.31, body, size=size - 1.3, color="subink", spacing=1.1)


def metric(slide, x, y, w, value, caption, color="blue"):
    text(slide, x, y, w, 0.48, value, size=25, color=color, bold=True)
    text(slide, x, y + 0.52, w, 0.60, caption, size=11.5, color="subink", spacing=1.1)


def footer(slide, value):
    text(slide, 0.55, 7.08, 11.5, 0.22, value, size=8.5, color="muted")


risk_frozen = json.loads(RISK_FROZEN.read_text())
calibration = risk_frozen["calibration"]
assert risk_frozen["threshold_locked"] and not risk_frozen["calibration_gate"]["pass"]
assert not risk_frozen["outer_scoring_authorized"] and not risk_frozen["outer_scored"]
persist = json.loads(PERSIST_SUMMARY.read_text())
assert persist["constructs"] == 59 and persist["utr_clusters"] == 49
assert persist["algorithm_seeds"] == 200 and persist["negative_results_retained"]
assert persist["primary"]["bon_minus_random_regret"] > 0
assert persist["primary"]["ci95"][0] < 0 < persist["primary"]["ci95"][1]
assert not persist["primary"]["significant_benefit"]
budget = json.loads(BUDGET_RESULTS.read_text())
sequence = json.loads(SEQUENCE_RESULTS.read_text())
assert not budget["new_blind_test"] and not sequence["new_blind_test"] and not sequence["causal_claim"]
equal = budget["families"]["bon_minus_random_n"]
random_budget = budget["families"]["random_n_minus_random1"]
r1_equal = [r for r in equal if r["context"] == "defined_tcell_r1" and r["cap"] > 1]
r2_equal = [r for r in equal if r["context"] == "defined_tcell_r2" and r["cap"] > 1]
assert all(r["holm_p"] < 0.05 for r in r1_equal)
assert [r["cap"] for r in r2_equal if r["holm_p"] < 0.05] == [4]
assert not any(r["holm_p"] < 0.05 for r in random_budget)
assert not any(r["adjusted_holm_p"] < 0.05 for r in sequence["primary"])
assert not any(r["holm_p"] < 0.05 for r in sequence["sequence_family_sensitivity"] if r["family"] == "no_uAUG")

prs = Presentation()
prs.slide_width = Inches(W); prs.slide_height = Inches(H)
blank = prs.slide_layouts[6]


def new():
    slide = prs.slides.add_slide(blank)
    rect(slide, 0, 0, W, H, "paper")
    return slide


s = new()
rect(s, 0, 0, W, H, "ink")
rect(s, 0.78, 1.03, 0.75, 0.06, "teal")
text(s, 0.78, 1.35, 11.5, 0.35, "PROXY–FUNCTION GAP", size=15, color="teal", bold=True)
text(s, 0.78, 2.00, 11.6, 1.55, "Proxy 不是准不准，\n而是一份有适用域的契约", size=36, color="paper", bold=True, spacing=1.12)
text(s, 0.78, 3.95, 11.6, 0.50, "等预算价值  →  序列组成边界  →  外部迁移边界", size=19, color="paper", bold=True)
text(s, 0.78, 4.68, 11.3, 0.65, "5′UTR 固定池历史 outer 回顾性敏感性 · 三论文统一框架 · 失败门禁保留", size=15, color="line")
pill(s, 0.78, 5.75, 2.85, "敏感性不作因果或新盲测", "confirmatory", size=10)
text(s, 0.78, 6.55, 11.3, 0.30, "科学汇报 · 2026-11-26 · 仅使用冻结 artifacts；未启动新训练或揭盲", size=10.5, color="muted")

s = new(); header(s, "01  EVIDENCE CHAIN", "从等预算价值到组成与迁移边界", "先排除候选数解释，再检查序列组成，最后用独立研究限定外部迁移", 2)
picture(s, V3 / "Figure11_v3_evidence_chain.png", 0.55, 1.55, 12.20, 5.30)
footer(s, "主线：等预算价值 → 序列组成边界 → 外部迁移边界；Paper2/3 失败门禁与 outer 角色不变。")

s = new(); header(s, "02  OBSERVE · PAPER 1", "原始压力阳性提示价值，等预算对照负责拆解", "BoN−Random(1) 的方向可重复，但仍需排除候选数与序列组成解释", 3)
picture(s, FIG / "Figure5_v2_pressure_heatmap.png", 0.45, 1.70, 7.85, 4.75)
pill(s, 8.55, 1.72, 3.85, "回顾性 strict-H1 · 非新盲测", "retrospective")
bullet_list(s, 8.55, 2.30, 4.20, [
    ("T-cell r1 · cap16", "配对 Δregret −1.243 [−1.710, −0.769]；cap2–16 经 all-20 Bonferroni 仍为负。", "blue"),
    ("T-cell r2 · cap8", "最佳 −1.510 [−2.132, −0.868]；cap16 −1.355，方向与幅度可重复。", "teal"),
    ("边界必须同页呈现", "N50 无持续收益；HepG2 cap8 多重校正后跨 0，仅探索性。", "coral"),
    ("科学含义", "收益方向是情境的函数，不能宣称压力普遍有益或有害。", "violet"),
], size=12.5, gap=1.02)
footer(s, "来源：strict_h1_figure2_paired_pressure.csv；strict_h1_multiplicity_sensitivity.json。")

s = new(); header(s, "03  EQUAL-BUDGET VALUE · PAPER 1", "等预算下，T-cell 仍显示 proxy 排序价值", "BoN(N) vs 同一 proposal prefix 的 Random(N)；负值有利于 BoN，候选数完全相同", 4)
picture(s, V3 / "Figure14_v3_equal_budget_value.png", 0.55, 1.52, 12.20, 5.34)
footer(s, "历史 outer 回顾性敏感性：r1 cap2–16 全过 all-20 Holm；r2 仅 cap4；Random(N)−Random(1) 0/20 通过。非新盲测。")

s = new(); header(s, "04  COMPOSITION BOUNDARY · PAPER 1", "组成调整后，T-cell 收益明显衰减", "GC / uAUG / longest-polyU / train-support 调整与完整配对排除；敏感性不等于因果中介", 5)
picture(s, V3 / "Figure15_v3_sequence_composition_sensitivity.png", 0.35, 1.52, 8.55, 5.35)
pill(s, 9.15, 1.72, 3.35, "回顾性 · 探索性 · 非因果", "retrospective", size=9)
bullet_list(s, 9.15, 2.30, 3.55, [
    ("调整后无 Holm 阳性", "0/20；T-cell r1 仅 cap4/8 保留名义负区间，r2 cap4 符号反转。", "coral"),
    ("uAUG 排除后衰减", "0/20 通过 Holm；T-cell cap4–16 区间均跨 0。", "violet"),
    ("polyU 仅局部保留", "r1 cap4 与 r2 cap8 通过；cap16 不通过，保留池减少。", "teal"),
    ("解释边界", "不重排、不替换候选；不能据此宣称 polyU/uAUG 因果介导。", "gold"),
], size=11.5, gap=1.00)
footer(s, "来源：paper1_sequence_ablation_v1；10,000 次 pool-cluster bootstrap，cap1 纳入 all-20 校正。所有跨零与符号反转均保留。")

s = new(); header(s, "05  ATTRIBUTE · PAPER 1", "更好的架构 ≠ 独立的功能验证", "误差高度共享；分数匹配不支持优化器特异误差增量", 6)
picture(s, FIG / "Figure6_v2_error_concordance.png", 0.45, 1.65, 7.25, 4.90)
pill(s, 8.00, 1.75, 2.25, "回顾性机制诊断", "diagnostic")
metric(s, 8.00, 2.38, 4.55, "0.878–0.962", "CNN 与 4-mer MLP 的逐池绝对误差相关\n四个情境区间均远离 0", "teal")
rect(s, 8.00, 3.85, 4.55, 1.05, "soft", radius=True)
text(s, 8.22, 4.03, 4.10, 0.28, "分数匹配：4/4 区间跨 0", size=14, color="blue", bold=True)
text(s, 8.22, 4.40, 4.10, 0.35, "没有一致的优化器特异 MAE 增量；零结果不等于证明高分外推机制。", size=10.5, color="subink")
text(s, 8.00, 5.35, 4.50, 0.95, "因此，ensemble 分歧或更换架构不能被包装成独立功能验证。瓶颈首先属于数据、表示与终点契约。", size=13, color="ink", bold=True, spacing=1.15)
footer(s, "来源：strict_h1_figure4_cross_architecture_summary.csv；strict_h1_figure3_score_matching.csv。")

s = new(); header(s, "06  ATTRIBUTE · CROSS-ENDPOINT OOD", "Proxy 契约必须写明终点", "稳定性、MRL 与内源翻译效率不是可互换的排序目标", 7)
picture(s, FIG / "Figure10_v2_leppek_endpoints.png", 0.35, 1.65, 8.35, 4.95)
rect(s, 8.85, 1.75, 3.95, 4.75, "teal_light", radius=True)
pill(s, 9.08, 2.00, 2.55, "CONFIRMATORY · 一次性揭盲", "confirmatory", size=8.5)
text(s, 9.08, 2.52, 3.48, 0.62, "Schlusser HepG2\n内源 TE", size=18, color="teal", bold=True, spacing=1.15)
metric(s, 9.08, 3.35, 3.45, "ρ = 0.072", "MRL proxy 与真实 TE 的 Spearman 相关", "teal")
text(s, 9.08, 4.55, 3.45, 0.75, "BoN-CNN − Random\nΔregret +0.627 [−0.828, +9.998]", size=12.5, color="ink", bold=True, spacing=1.2)
text(s, 9.08, 5.52, 3.45, 0.60, "区间跨 0：MRL proxy 未改善真实 TE。单研究、极端右偏，不作跨细胞泛化。", size=10.5, color="subink")
footer(s, "Leppek：探索性描述，无 p 值/CI；62% = 171/276，top-6 重叠 1/6。Schlusser：冻结配置后确认性评测。")

s = new(); header(s, "07  CONSTRAIN · PAPER 2", "冻结风险规则未通过 calibration 门禁", "选择器无普遍优势；候选级 ensemble-std 拒选也未显示真实 regret 改善", 8)
rect(s, 0.55, 1.70, 5.65, 4.95, "soft", radius=True)
pill(s, 0.82, 1.95, 2.85, "SELECTION · 回顾性诊断", "retrospective", size=9)
text(s, 0.82, 2.48, 5.05, 0.45, "选择器与池级风险均为阴性", size=18, color="blue", bold=True)
bullet_list(s, 0.82, 3.15, 5.05, [
    ("方法无通用优势", "VGSC / COM-style / LCB / diversity 均未显示情境通用的 regret 优势。", "blue"),
    ("Q 风险头不判别", "calibration ρ −0.200，AUC 0.297；pool-level 阴性结果保持不变。", "coral"),
], size=13, gap=1.25)
text(s, 0.82, 5.88, 5.00, 0.50, "历史 outer 经验曲线仅是被门禁否定的回顾性诊断，不作当前阳性证据。", size=10.5, color="subink", bold=True)
rect(s, 6.50, 1.70, 6.28, 4.95, "coral_light", radius=True)
pill(s, 6.78, 1.95, 3.85, "FROZEN CALIBRATION · GATE FAILED", "gap", size=8.5)
metric(s, 6.78, 2.52, 2.55, f"{calibration['coverage']:.3f}", f"coverage · {calibration['accepted_decisions']}/{calibration['decisions']} decisions\n{calibration['accepted_pools']}/{calibration['pools']} pools represented", "coral")
metric(s, 9.65, 2.52, 2.75, f"{calibration['regret_reduction']:.6f}".replace("-", "−"), "regret reduction\n负值不利于拒选", "coral")
rect(s, 6.78, 4.10, 5.72, 0.78, "paper", radius=True)
text(s, 7.00, 4.25, 5.28, 0.48, f"95% pool-cluster CI  [{calibration['regret_reduction_95ci'][0]:.6f}, {calibration['regret_reduction_95ci'][1]:+.6f}]  跨 0".replace("-", "−"), size=12.5, color="ink", bold=True)
text(s, 6.78, 5.10, 5.72, 0.35, "点估计 > 0：失败   ·   CI 下界 > 0：失败", size=13, color="coral", bold=True)
rect(s, 6.78, 5.67, 5.72, 0.65, "ink", radius=True)
text(s, 7.00, 5.85, 5.28, 0.30, "OUTER NOT SCORED · 未授权，硬锁拒绝调用", size=12, color="paper", bold=True)
footer(s, "来源：paper2_risk_frozen_v1/frozen_protocol.json；阈值 0.10402689717871444 已冻结。本版本不评分 outer、不调阈值。")

s = new(); header(s, "08  REPAIR · PAPER 3", "标签修复可行，但 headroom 依赖表示", "简单 EI 已能兑现部分 headroom；复杂获取函数不支配简单基线", 9)
picture(s, FIG / "Figure8_v2_paper3_headroom.png", 0.45, 1.65, 12.35, 4.55)
pill(s, 0.65, 6.28, 3.35, "DIAGNOSTIC · 永久 outer 未动", "diagnostic")
text(s, 4.25, 6.28, 8.25, 0.50, "kmer4：5/16 正门禁；EI@16 = 1.220，random = 3.497。summary4：0/16；regret-analytic 不支配。", size=12, color="ink", bold=True)
footer(s, "来源：paper3_headroom_v4/headroom_v4_round1/summary.json；固定 5 池 pooled 曲线。非新盲测，不声称普遍 acquisition 优势。")

s = new(); header(s, "09  GOVERNANCE & OOD", "覆盖哪些 OOD，也明确缺哪一块", "数据隔离支撑可审计性，但不自动推出生物学泛化", 10)
picture(s, V3 / "Figure12_v3_data_governance_ood.png", 0.65, 1.52, 12.00, 5.42)
footer(s, "PERSIST 已提供独立同终点回顾性外测：结果负向、区间跨 0；跨终点 OOD 与盲测角色仍不可混同。")

s = new(); header(s, "10  TRANSFER · PERSIST-SEQ", "内部收益 ≠ 跨研究迁移", "独立研究、同终点 MRL 外测未支持 BoN；frozen 4-mer MLP · 回顾性非盲测", 11)
picture(s, V3 / "Figure13_v3_persist_external.png", 0.55, 1.53, 12.20, 5.30)
footer(s, "59 构建体 / 49 UTR 簇；200 种子仅作重复提案。BoN−Random +0.104723 [−0.100964, +0.212706]，不宣称获益或普遍失效。")

s = new(); header(s, "11  CLAIMS & ROADMAP", "当前可主张到哪里，后续如何重建风险命题", "等预算价值受组成约束，且不能替代外部迁移；冻结失败完整保留", 12)
rect(s, 0.55, 1.65, 5.95, 3.95, "blue_light", radius=True)
text(s, 0.82, 1.90, 5.40, 0.35, "当前有证据支持", size=17, color="blue", bold=True)
bullet_list(s, 0.82, 2.48, 5.30, [
    ("等预算、情境依赖价值", "T-cell r1 cap2–16 全过 Holm；r2 仅 cap4；Random 对照 0/20。", "blue"),
    ("组成与终点边界", "调整后 0/20 Holm 阳性；跨架构相关 0.878–0.962，Schlusser 跨终点失效。", "teal"),
    ("表示依赖的修复空间", "kmer4 有 headroom，EI 可兑现；summary4 0/16，outer 未动。", "violet"),
], size=12.5, gap=0.88)
rect(s, 6.80, 1.65, 5.98, 3.95, "coral_light", radius=True)
text(s, 7.07, 1.90, 5.40, 0.35, "不能提前主张", size=17, color="coral", bold=True)
bullet_list(s, 7.07, 2.48, 5.30, [
    ("非普遍、非因果", "等预算价值有情境边界；组成调整与家族排除不证明因果中介。", "gold"),
    ("冻结拒选未获益", "Paper2 calibration 点估计为负、CI 跨 0；outer 未评分。", "coral"),
    ("跨研究迁移未获益", "PERSIST 回顾性非盲测：ρ −0.008447；Δregret +0.104723，CI 跨 0，不支持 BoN。", "coral"),
], size=12.5, gap=0.88)
rect(s, 0.55, 5.88, 12.23, 0.92, "ink", radius=True)
text(s, 0.82, 6.06, 1.75, 0.28, "后续路线图", size=12, color="teal", bold=True)
text(s, 2.50, 5.99, 9.90, 0.60, "本版本：outer 不评分、阈值不回调。PERSIST 负结果原样保留；下一项跨研究验证须预注册并采用真正前瞻盲测角色。", size=11.5, color="paper", bold=True)
footer(s, "历史 outer 经验曲线不得升级为阳性证据；内部收益不能替代独立迁移，失败结果不改写为成功。")

prs.save(OUT)
figures = [
    FIG / "Figure5_v2_pressure_heatmap.png", FIG / "Figure6_v2_error_concordance.png",
    FIG / "Figure10_v2_leppek_endpoints.png",
    FIG / "Figure8_v2_paper3_headroom.png", V3 / "Figure11_v3_evidence_chain.png",
    V3 / "Figure12_v3_data_governance_ood.png", V3 / "Figure13_v3_persist_external.png",
    V3 / "Figure14_v3_equal_budget_value.png", V3 / "Figure15_v3_sequence_composition_sensitivity.png",
]
config = {
    "deck": OUT.name, "slides": len(prs.slides), "size_inches": [W, H], "font": FONT,
    "runtime": {"python": platform.python_version(), "python_pptx": pptx.__version__},
    "evidence_boundary": "equal-budget value -> sequence-composition boundary -> external-transfer boundary; all new displays are retrospective historical-outer sensitivities, not new blind tests or causal analyses; PERSIST negative result and Paper2/3 failure gates retained",
    "paper1_budget_matched": {
        "tcell_r1_holm_caps": [2, 4, 8, 16], "tcell_r2_holm_caps": [4],
        "random_n_minus_random1_holm_significant": 0,
        "source": {"path": str(BUDGET_RESULTS.relative_to(ROOT)), "sha256": hashlib.sha256(BUDGET_RESULTS.read_bytes()).hexdigest()},
    },
    "paper1_sequence_ablation": {
        "adjusted_holm_significant": 0, "no_uaug_holm_significant": 0,
        "no_polyu_holm_significant": [["defined_tcell_r1", 4], ["defined_tcell_r2", 8]],
        "causal_claim": False,
        "source": {"path": str(SEQUENCE_RESULTS.relative_to(ROOT)), "sha256": hashlib.sha256(SEQUENCE_RESULTS.read_bytes()).hexdigest()},
    },
    "persist": {
        "constructs": persist["constructs"], "utr_clusters": persist["utr_clusters"],
        "algorithm_seeds": persist["algorithm_seeds"], "primary": persist["primary"],
        "spearman": persist["secondary"]["spearman"],
        "source": {"path": str(PERSIST_SUMMARY.relative_to(ROOT)), "sha256": hashlib.sha256(PERSIST_SUMMARY.read_bytes()).hexdigest()},
    },
    "paper2_risk_frozen": {
        "schema_version": risk_frozen["schema_version"],
        "coverage": calibration["coverage"],
        "regret_reduction": calibration["regret_reduction"],
        "regret_reduction_95ci": calibration["regret_reduction_95ci"],
        "calibration_gate_pass": risk_frozen["calibration_gate"]["pass"],
        "outer_scoring_authorized": risk_frozen["outer_scoring_authorized"],
        "outer_scored": risk_frozen["outer_scored"],
        "source": {"path": str(RISK_FROZEN.relative_to(ROOT)), "sha256": hashlib.sha256(RISK_FROZEN.read_bytes()).hexdigest()},
    },
    "roadmap": "do not score this outer or tune the threshold; new risk definitions require a new preregistered version and independent evaluation roles",
    "figures": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in figures],
}
(OUT_DIR / "proxy_function_gap_modern_20261126_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
print("saved", OUT, "slides", len(prs.slides))
