import json
from pathlib import Path

import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path('/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/evaluation')
OUT = ROOT.parent / 'ppt'
original = json.loads((ROOT / 'castillo_n50_v2/bon_pressure.json').read_text())
alternative = json.loads((ROOT / 'castillo_n50_sensitivity_v1/bon_pressure.json').read_text())
noise = json.loads((ROOT / 'castillo_n50_noise_v1/summary.json').read_text())
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
    text(s, 'Proxy–Function Gap · sensitivity and noise', .65, 7.12, 9.5, .2, 9, MUTED); text(s, '2026-11-24', 11.4, 7.12, 1.25, .2, 9, MUTED, align=PP_ALIGN.RIGHT)


def bullets(s, items, x, y, w, h, size=18):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame; tf.clear(); tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = '•  ' + item
        p.font.name = 'Aptos'; p.font.size = Pt(size); p.font.color.rgb = INK; p.space_after = Pt(13)

# 1
fig, ax = plt.subplots(figsize=(9.5, 3.8))
for name, values in [('Original', original), ('Alternative', alternative)]:
    rows = values['rows']; ax.plot([r['proposal_cap'] for r in rows], [r['mean_regret'] for r in rows], marker='o', lw=2, label=name)
ax.set_xscale('log', base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(['1', '2', '4', '8', '16']); ax.set_xlabel('BoN cap'); ax.set_ylabel('Mean measured-MRL regret'); ax.set_title('Alternative pool series also reverses', fontweight='bold'); ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.18)
fig.tight_layout(); chart = OUT / 'paper1_alternative_pools.png'; fig.savefig(chart, dpi=190, bbox_inches='tight', facecolor='white'); plt.close(fig)
s = prs.slides.add_slide(blank); header(s, '非单调不是单组候选池的偶然', 'N50 原始与替代候选池系列', '01')
s.shapes.add_picture(str(chart), Inches(.65), Inches(1.35), width=Inches(10.8)); footer(s)

# 2
rows = [row for row in noise['rows'] if row['proposal_cap'] == 16]
fig, ax = plt.subplots(figsize=(9.2, 3.5)); ax.plot([row['noise_sigma'] for row in rows], [row['mean_regret'] for row in rows], marker='o', color='#16a09b', lw=2)
ax.axhline(rows[0]['mean_regret'], color='#637180', ls='--', lw=1); ax.set_xlabel('Sigma on normalized CNN output'); ax.set_ylabel('Mean regret at cap 16'); ax.set_title('No monotone noise amplification', fontweight='bold'); ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.18)
fig.tight_layout(); chart = OUT / 'paper1_proxy_noise.png'; fig.savefig(chart, dpi=190, bbox_inches='tight', facecolor='white'); plt.close(fig)
s = prs.slides.add_slide(blank); header(s, '噪声协议也是负结果', 'sigma=0 与原网格逐键一致；中等扰动可产生探索效应', '02')
s.shapes.add_picture(str(chart), Inches(.65), Inches(1.35), width=Inches(10.3)); footer(s)

# 3
s = prs.slides.add_slide(blank); header(s, '统一证据边界', 'Paper 1 的结论来自多种对照，而非单个方法胜负', '03')
bullets(s, ['固定池压力：效应随文库、细胞和研究变化，不呈普遍单调方向。', '分数匹配：没有稳定的优化器附加误差，更支持高分外推。', '共同 proxy：CNN 与 4-mer MLP 误差高度相关，换架构不能替代独立验证。', '敏感性：替代候选池仍非单调；高斯输出噪声不必然被 BoN 放大。', 'VGSC 与 Q 未通过 pilot 门禁，不进入主线。'], .85, 1.45, 11.7, 4.3, 19)
rect(s, .85, 6.0, 11.55, .55, LIGHT); text(s, '下一步：将这套证据整理为 Paper 1 初稿与可复现附录。', 1.05, 6.14, 11.1, .28, 15, NAVY, True)
footer(s)

path = OUT / 'proxy_function_gap_sensitivity_noise_20261124.pptx'
prs.save(path)
print(path)
