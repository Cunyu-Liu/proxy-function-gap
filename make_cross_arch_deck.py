import json
from pathlib import Path

import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path('/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/mechanism')
OUT = ROOT.parent / 'ppt'
folders = {'N50': 'n50_v1', 'HepG2': 'hepg2_v1', 'T-cell r1': 'tcell_r1_v1', 'T-cell r2': 'tcell_r2_v1'}
data = {name: json.loads((ROOT / folder / 'cross_arch/summary.json').read_text()) for name, folder in folders.items()}
for folder in folders.values():
    for round_id in (1, 2, 3): assert json.loads((ROOT / folder / f'cross_arch/independent_validation_round{round_id}.json').read_text())['pass']
NAVY = RGBColor(18, 35, 56); TEAL = RGBColor(22, 154, 155); INK = RGBColor(35, 47, 60); MUTED = RGBColor(100, 113, 128); LIGHT = RGBColor(242, 246, 249); WHITE = RGBColor(255, 255, 255)
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
    text(s, 'Proxy–Function Gap · cross-architecture audit', .65, 7.12, 9.5, .2, 9, MUTED); text(s, '2026-11-24', 11.4, 7.12, 1.25, .2, 9, MUTED, align=PP_ALIGN.RIGHT)


def bullets(s, items, x, y, w, h, size=18):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame; tf.clear(); tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = '•  ' + item
        p.font.name = 'Aptos'; p.font.size = Pt(size); p.font.color.rgb = INK; p.space_after = Pt(13)

# 1
s = prs.slides.add_slide(blank); rect(s, 0, 0, 13.333, 7.5, NAVY); rect(s, .75, .85, .12, 1.25, TEAL)
text(s, '共同 proxy 风险跨越模型架构', 1.15, .82, 11.0, .7, 34, WHITE, True)
text(s, 'CNN vs 4-mer MLP · 四个外部上下文', 1.15, 1.62, 10.8, .55, 23, RGBColor(199, 220, 232))
text(s, '绝对误差相关性：0.931–0.965', 1.15, 3.1, 10.8, .5, 25, WHITE, True)
text(s, '换架构不能替代独立实验终点。', 1.15, 4.0, 11.0, .4, 19, RGBColor(199, 220, 232))
text(s, '三种子收敛 · 三轮训练复核 · 三轮指标复核', 1.15, 6.25, 10.8, .35, 13, RGBColor(173, 194, 210))

# 2
names = list(data); correlations = [data[name]['architecture_error_correlation'] for name in names]
fig, ax = plt.subplots(figsize=(9.4, 3.7)); bars = ax.bar(names, correlations, color=['#16a09b', '#3c6eb4', '#dc9e37', '#8b6db1'])
ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=12); ax.set_ylim(0, 1.08); ax.set_ylabel('Absolute-error correlation'); ax.set_title('CNN and 4-mer MLP make strongly correlated errors', fontweight='bold'); ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.18)
fig.tight_layout(); chart = OUT / 'paper1_cross_arch_correlation.png'; fig.savefig(chart, dpi=190, bbox_inches='tight', facecolor='white'); plt.close(fig)
s = prs.slides.add_slide(blank); header(s, '误差不是模型家族独立产生的', '每个点代表一个外部上下文；越高表示共同误差越强', '02')
s.shapes.add_picture(str(chart), Inches(.65), Inches(1.35), width=Inches(10.8)); footer(s)

# 3
s = prs.slides.add_slide(blank); header(s, '机制结论：共同 proxy 风险是主线证据', '停止用“更多同数据模型”替代独立验证', '03')
bullets(s, ['四个上下文均出现高误差相关，说明共享训练分布与终点定义是主要限制。', '异构 MLP 仅用于机制分析，不进入主方法或性能排行榜。', '低架构分歧区域仍需要历史实验或独立 oracle，不能投票认证。', '下一步：完成统一敏感性审查和 Paper 1 证据包，而非继续增加模型。'], .85, 1.55, 11.7, 3.8, 19)
rect(s, .85, 5.85, 11.55, .62, LIGHT); text(s, '当前结论：共同 proxy 风险跨架构成立，但所有结论仍受当前数据集限制。', 1.05, 6.02, 11.1, .28, 15, NAVY, True)
footer(s)

path = OUT / 'proxy_function_gap_cross_arch_20261124.pptx'
prs.save(path)
print(path)
