import json
from pathlib import Path

import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path('/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/mechanism')
OUT = ROOT.parent / 'ppt'
contexts = {'N50': 'n50_v1', 'HepG2': 'hepg2_v1', 'T-cell r1': 'tcell_r1_v1', 'T-cell r2': 'tcell_r2_v1'}
data = {name: json.loads((ROOT / folder / 'evaluation/summary.json').read_text()) for name, folder in contexts.items()}
for folder in contexts.values():
    for round_id in (1, 2, 3):
        assert json.loads((ROOT / folder / f'evaluation/independent_validation_round{round_id}.json').read_text())['pass']

NAVY = RGBColor(18, 35, 56); TEAL = RGBColor(22, 154, 155); BLUE = RGBColor(60, 110, 180)
INK = RGBColor(35, 47, 60); MUTED = RGBColor(100, 113, 128); LIGHT = RGBColor(242, 246, 249); WHITE = RGBColor(255, 255, 255)
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5); blank = prs.slide_layouts[6]


def rect(s, x, y, w, h, color):
    shape = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h)); shape.fill.solid(); shape.fill.fore_color.rgb = color; shape.line.fill.background()


def text(s, value, x, y, w, h, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame; tf.clear(); tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align; run = p.add_run(); run.text = value
    run.font.name = 'Aptos'; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color


def header(s, title, subtitle, number):
    rect(s, 0, 0, 13.333, .12, TEAL); text(s, title, .65, .42, 11.8, .55, 28, NAVY, True)
    text(s, subtitle, .67, 1.0, 11.6, .38, 13, MUTED); text(s, number, 12.25, .48, .45, .3, 12, MUTED, align=PP_ALIGN.RIGHT)


def footer(s):
    text(s, 'Proxy–Function Gap · mechanism audit', .65, 7.12, 9.5, .2, 9, MUTED); text(s, '2026-11-24', 11.4, 7.12, 1.25, .2, 9, MUTED, align=PP_ALIGN.RIGHT)


def bullets(s, items, x, y, w, h, size=18):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame; tf.clear(); tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = '•  ' + item
        p.font.name = 'Aptos'; p.font.size = Pt(size); p.font.color.rgb = INK; p.space_after = Pt(13)

# 1
s = prs.slides.add_slide(blank); rect(s, 0, 0, 13.333, 7.5, NAVY); rect(s, .75, .85, .12, 1.25, TEAL)
text(s, 'Paper 1 · 机制判别', 1.15, .82, 10.8, .7, 34, WHITE, True)
text(s, '分数匹配与共同 proxy 风险', 1.15, 1.62, 10.8, .55, 23, RGBColor(199, 220, 232))
text(s, '结论：更像高分外推风险，而非优化器特异失真', 1.15, 3.1, 10.8, .5, 22, WHITE, True)
text(s, '四个外部上下文 · 每池 192 次真实成员 forward · 三轮独立复核', 1.15, 4.0, 11.0, .4, 18, RGBColor(199, 220, 232))
text(s, '机制结论受当前数据与 proxy 谱系限制。', 1.15, 6.25, 10.8, .35, 13, RGBColor(173, 194, 210))

# 2 chart
names = list(data)
differences = [data[name]['score_matched_mean_abs_error_difference'] for name in names]
intervals = [data[name]['score_matched_cluster_bootstrap_95_percent_interval'] for name in names]
errors = [[value - interval[0], interval[1] - value] for value, interval in zip(differences, intervals)]
fig, ax = plt.subplots(figsize=(9.6, 3.8)); ax.axvline(0, color='#637180', lw=1)
ax.barh(names, differences, xerr=list(zip(*errors)), color=['#16a09b', '#3c6eb4', '#dc9e37', '#8b6db1'], capsize=5)
ax.set_xlabel('BoN − score-matched absolute error'); ax.set_title('Pool-cluster bootstrap intervals all include zero', fontweight='bold'); ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=.18)
fig.tight_layout(); chart = OUT / 'paper1_score_matched.png'; fig.savefig(chart, dpi=190, bbox_inches='tight', facecolor='white'); plt.close(fig)
s = prs.slides.add_slide(blank); header(s, '同分数下，没有稳定的优化器附加误差', '负值表示 BoN 更低，正值表示 BoN 更高；方向跨情境变化', '02')
s.shapes.add_picture(str(chart), Inches(.65), Inches(1.35), width=Inches(10.8)); footer(s)

# 3
s = prs.slides.add_slide(blank); header(s, '低分歧不等于安全', '共同 proxy 风险具有明显情境差异', '03')
rows = [('N50', '67 / 1024', '2.741 vs 2.826'), ('HepG2', '75 / 1024', '3.095 vs 2.845'), ('T-cell r1', '60 / 1024', '2.091 vs 2.071'), ('T-cell r2', '59 / 1024', '2.467 vs 2.601')]
text(s, 'Context', .85, 1.55, 2.0, .35, 14, MUTED, True); text(s, '低分歧 ∩ 高误差', 3.35, 1.55, 2.4, .35, 14, MUTED, True); text(s, '低分歧 vs 其他 |误差|', 6.45, 1.55, 3.2, .35, 14, MUTED, True)
for i, (name, count, values) in enumerate(rows):
    y = 2.15 + i * .78
    if i % 2 == 0: rect(s, .7, y - .12, 11.7, .65, LIGHT)
    text(s, name, .85, y, 2.0, .3, 17, NAVY, True); text(s, count, 3.35, y, 2.4, .3, 17); text(s, values, 6.45, y, 3.2, .3, 17)
text(s, '不能用 ensemble 投票认证真实功能；低分歧区域仍需历史实验或独立终点。', .85, 5.65, 11.5, .5, 16, NAVY, True)
footer(s)

# 4
s = prs.slides.add_slide(blank); header(s, '机制结论与剩余工作', '保留边界，不把负结果改写成方法优势', '04')
bullets(s, ['支持：高分尾部的外推风险和 proxy 适用域依赖。', '不支持：优化器特异失真、低分歧安全证书或跨情境统一机制。', 'uAUG、GC、poly-U、支持度统计作为探索性机制附录，不用于回调 outer。', '下一步：统一 Paper 1 证据表、敏感性分析和预印本前独立复核。'], .85, 1.55, 11.7, 3.8, 19)
rect(s, .85, 5.85, 11.55, .62, LIGHT); text(s, '当前阶段：机制对照完成；完整预印本仍需统一审查。', 1.05, 6.02, 11.1, .28, 15, NAVY, True)
footer(s)

path = OUT / 'proxy_function_gap_paper1_mechanism_20261124.pptx'
prs.save(path)
print(path)
