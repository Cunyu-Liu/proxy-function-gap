import json
from pathlib import Path

import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path('/teams/HoulabData_1788502323/liucunyu/proxy_function_gap')
DESIGNED = ROOT / 'evaluation/outer_designed_v1'
RANDOM = ROOT / 'evaluation/outer_random_v1_eligible'
OUT = ROOT / 'ppt'
OUT.mkdir(parents=True, exist_ok=True)
for root, count in [(DESIGNED, 16), (RANDOM, 3)]:
    for round_id in (1, 2, 3):
        report = json.loads((root / f'independent_validation_round{round_id}.json').read_text())
        assert report['pass'] and report['pools'] == count and not report['labels_returned_to_learner']
d = json.loads((DESIGNED / 'summary.json').read_text())
r = json.loads((RANDOM / 'summary.json').read_text())
d_fp = next(x for x in d['configurations'] if x['strategy'] == 'fp-rej' and x['alpha'] == 20 and x['proposal_cap'] == 16)
d_bon = next(x for x in d['configurations'] if x['strategy'] == 'bon' and x['alpha'] == 0 and x['proposal_cap'] == 16)
r_fp = next(x for x in r['configurations'] if x['strategy'] == 'fp-rej' and x['alpha'] == 20 and x['proposal_cap'] == 16)
r_bon = next(x for x in r['configurations'] if x['strategy'] == 'bon' and x['alpha'] == 0 and x['proposal_cap'] == 16)

NAVY = RGBColor(18, 35, 56)
TEAL = RGBColor(22, 154, 155)
BLUE = RGBColor(60, 110, 180)
GOLD = RGBColor(220, 158, 55)
INK = RGBColor(35, 47, 60)
MUTED = RGBColor(100, 113, 128)
LIGHT = RGBColor(242, 246, 249)
WHITE = RGBColor(255, 255, 255)
RED = RGBColor(184, 76, 70)
GREEN = RGBColor(45, 145, 95)
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def rect(slide, x, y, w, h, fill):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill; shape.line.fill.background()
    return shape


def text(slide, value, x, y, w, h, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = value
    run.font.name = 'Aptos'; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    return box


def header(slide, title, subtitle, number):
    rect(slide, 0, 0, 13.333, 0.12, TEAL)
    text(slide, title, 0.65, 0.42, 11.8, 0.55, 28, NAVY, True)
    text(slide, subtitle, 0.67, 1.0, 11.6, 0.38, 13, MUTED)
    text(slide, number, 12.25, 0.48, 0.45, 0.3, 12, MUTED, align=PP_ALIGN.RIGHT)


def footer(slide):
    text(slide, 'Proxy–Function Gap · verified fixed-pool pilot', 0.65, 7.12, 9.5, 0.2, 9, MUTED)
    text(slide, '2026-11-24', 11.4, 7.12, 1.25, 0.2, 9, MUTED, align=PP_ALIGN.RIGHT)


def bullets(slide, items, x, y, w, h, size=18):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = '•  ' + item; p.font.name = 'Aptos'; p.font.size = Pt(size); p.font.color.rgb = INK
        p.space_after = Pt(13)


def metric(slide, x, y, w, value, label, accent=TEAL):
    rect(slide, x, y, w, 1.22, LIGHT)
    rect(slide, x, y, 0.07, 1.22, accent)
    text(slide, value, x + 0.22, y + 0.18, w - 0.35, 0.45, 25, NAVY, True)
    text(slide, label, x + 0.22, y + 0.73, w - 0.35, 0.28, 11, MUTED)


def regret_chart(values, labels, path, title):
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    bars = ax.bar(labels, values, color=['#16a09b', '#3c6eb4'], width=.55)
    ax.bar_label(bars, fmt='%.3f', padding=5, fontsize=12)
    ax.set_ylabel('Mean outer regret (lower is better)'); ax.set_title(title, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.18)
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout(); fig.savefig(path, dpi=190, bbox_inches='tight', facecolor='white'); plt.close(fig)

# 1
s = prs.slides.add_slide(blank); rect(s, 0, 0, 13.333, 7.5, NAVY); rect(s, .75, .85, .12, 1.25, TEAL)
text(s, 'Proxy–Function Gap', 1.15, .82, 10.8, .7, 34, WHITE, True)
text(s, 'VGSC 固定池 pilot：门禁结果与负结论', 1.15, 1.62, 10.8, .55, 23, RGBColor(199, 220, 232))
rect(s, 1.15, 2.65, 3.55, .5, RED)
text(s, '不进入主线 · 保留基线与协议', 1.3, 2.76, 3.25, .24, 12, WHITE, True, align=PP_ALIGN.CENTER)
text(s, '真实 CUDA 选择 · 历史实验标签 · Learner / Evaluator 分离', 1.15, 3.65, 11.0, .4, 19, WHITE)
text(s, '结果是预注册 pilot 的判伪证据，不是预印本级普遍结论。', 1.15, 6.25, 10.8, .35, 13, RGBColor(173, 194, 210))

# 2
s = prs.slides.add_slide(blank); header(s, '可审计证据链', '所有结果使用 ensemble-v2 的 measured-MRL 单位', '02')
metric(s, .65, 1.55, 2.85, '32 / 16', 'dev / calibration 池')
metric(s, 3.75, 1.55, 2.85, '42,000', '开发+校准选择记录', BLUE)
metric(s, 6.85, 1.55, 2.85, '16', 'designed outer 池', GOLD)
metric(s, 9.95, 1.55, 2.75, '3 × 3', 'V / Q 三轮 CUDA 复核', GREEN)
bullets(s, ['32 dev 池冻结 FP-REJ：α=20，cap=16，5 个配对种子。', '三成员 CNN 与 V/Q 在每次提议上真实执行；无查表冒充 forward。', 'outer 选择 regret 先保持为空；Evaluator 冻结后才读取历史 MRL。', '完整分配单元、近重复、轨迹和序列跨池隔离。'], .85, 3.25, 11.6, 2.65)
footer(s)

# 3
chart = OUT / 'pilot_designed_regret.png'; regret_chart([d_fp['selector_mean_regret_before_q'], d_bon['mean_regret']], ['FP-REJ', 'BoN'], chart, 'Designed outer · cap 16')
s = prs.slides.add_slide(blank); header(s, '主结果：FP-REJ 未优于 BoN', '16 个 designed outer 池；未加 Q 的选择器比较', '03')
s.shapes.add_picture(str(chart), Inches(.65), Inches(1.35), width=Inches(7.5))
metric(s, 8.55, 1.65, 3.8, '+0.343', 'FP-REJ − BoN regret', RED)
text(s, '池簇 bootstrap 95% 区间\n[0.225, 0.467]', 8.75, 3.15, 3.35, .85, 20, NAVY, True)
text(s, '方向为正：FP-REJ 更差。\n匹配相同 CNN 成员预算；V 成本另计。', 8.75, 4.35, 3.35, 1.0, 15, MUTED)
footer(s)

# 4
s = prs.slides.add_slide(blank); header(s, 'Q 没有形成可用的拒选器', '阈值只由 dev 选择，calibration 仅做最终 Platt 校准', '04')
metric(s, .75, 1.55, 3.55, '0%', 'designed outer 覆盖率', RED)
metric(s, 4.9, 1.55, 3.55, '100%', 'Q 全部拒选', GOLD)
metric(s, 9.05, 1.55, 3.55, '不可定义', 'FP-REJ+Q 最终 regret', MUTED)
bullets(s, ['designed 上 Q 概率分布发生明显情境偏移，冻结阈值不再提供覆盖。', '不在同一 outer 上回调阈值、重选 α 或删除不利池。', '按合同：Q 不进入主线；风险预测优势不成立。'], .9, 3.35, 11.5, 2.2)
footer(s)

# 5
chart = OUT / 'pilot_random_regret.png'; regret_chart([r_fp['mean_regret'], r_bon['mean_regret']], ['FP-REJ+Q', 'BoN'], chart, 'Eligible random outer · cap 16')
s = prs.slides.add_slide(blank); header(s, '次要 random 对照不支持普遍优势', '仅 3 个完整池；003 因 2 条缺失 MRL 整池排除', '05')
s.shapes.add_picture(str(chart), Inches(.65), Inches(1.35), width=Inches(7.5))
metric(s, 8.55, 1.65, 3.8, '−0.055', 'FP-REJ+Q − BoN', BLUE)
text(s, '池簇 bootstrap 95% 区间\n[−1.400, 1.051]', 8.75, 3.15, 3.35, .85, 20, NAVY, True)
text(s, '区间跨 0；Q 在这些池未拒选。\n不作为独立研究复验。', 8.75, 4.35, 3.35, 1.0, 15, MUTED)
footer(s)

# 6
s = prs.slides.add_slide(blank); header(s, '门禁决定与下一步', '保留负结果，转向有证据支持的 Paper 1 主线', '06')
bullets(s, ['VGSC 固定池方法：保留为基线与压力协议，不进入核心方法主张。', 'Q 风险头：本轮判为不可用；后续若重做必须更换开发依据并寻找新独立测试。', '数据与代码：单位错误、部分种子丢失和缺失标签均已记录并修复。', '下一阶段：历史轨迹 / best-of-N 现象、负对照、噪声协议与独立研究复验。', '当前不声称预印本完成、湿实验功能或跨研究普适性。'], .85, 1.55, 11.7, 4.4, 18)
rect(s, .85, 6.15, 11.55, .52, RGBColor(250, 242, 226))
text(s, '结论：pilot 完成了有效判伪；课题主线不因方法偏好而偏离证据。', 1.05, 6.28, 11.1, .25, 14, NAVY, True)
footer(s)

path = OUT / 'proxy_function_gap_vgsc_pilot_20261124.pptx'
prs.save(path)
print(path)
