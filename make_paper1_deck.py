import json
from pathlib import Path

import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path('/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/evaluation')
OUT = ROOT.parent / 'ppt'
OUT.mkdir(exist_ok=True)
paths = {
    'Sample designed': ROOT / 'outer_designed_v1/bon_pressure.json',
    'Sample random': ROOT / 'outer_random_v1_eligible/bon_pressure.json',
    'N50 HEK293T': ROOT / 'castillo_n50_v2/bon_pressure.json',
    'Defined HepG2': ROOT / 'castillo_hepg2_v1/bon_pressure.json',
    'Defined T-cell r1': ROOT / 'castillo_tcell_r1_v1/bon_pressure.json',
    'Defined T-cell r2': ROOT / 'castillo_tcell_r2_v1/bon_pressure.json',
}
data = {name: json.loads(path.read_text()) for name, path in paths.items()}
for name in ['N50 HEK293T', 'Defined HepG2', 'Defined T-cell r1', 'Defined T-cell r2']:
    for round_id in (1, 2, 3):
        assert json.loads((paths[name].parent / f'independent_validation_round{round_id}.json').read_text())['pass']

NAVY = RGBColor(18, 35, 56); TEAL = RGBColor(22, 154, 155); BLUE = RGBColor(60, 110, 180)
GOLD = RGBColor(220, 158, 55); INK = RGBColor(35, 47, 60); MUTED = RGBColor(100, 113, 128)
LIGHT = RGBColor(242, 246, 249); WHITE = RGBColor(255, 255, 255); RED = RGBColor(184, 76, 70)
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
    text(s, 'Proxy–Function Gap · Paper 1 fixed-pool pressure', .65, 7.12, 9.5, .2, 9, MUTED)
    text(s, '2026-11-24', 11.4, 7.12, 1.25, .2, 9, MUTED, align=PP_ALIGN.RIGHT)


def bullets(s, items, x, y, w, h, size=18):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = box.text_frame; tf.clear(); tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = '•  ' + item
        p.font.name = 'Aptos'; p.font.size = Pt(size); p.font.color.rgb = INK; p.space_after = Pt(13)


def metric(s, x, y, w, value, label, accent=TEAL):
    rect(s, x, y, w, 1.22, LIGHT); rect(s, x, y, .07, 1.22, accent)
    text(s, value, x + .22, y + .18, w - .35, .45, 25, NAVY, True); text(s, label, x + .22, y + .73, w - .35, .28, 11, MUTED)


def chart(names, path, title):
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    for name in names:
        rows = data[name]['rows']; ax.plot([r['proposal_cap'] for r in rows], [r['mean_regret'] for r in rows], marker='o', lw=2, label=name)
    ax.set_xscale('log', base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(['1', '2', '4', '8', '16'])
    ax.set_xlabel('Best-of-N proposal cap'); ax.set_ylabel('Mean measured-MRL regret'); ax.set_title(title, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.18); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=190, bbox_inches='tight', facecolor='white'); plt.close(fig)

# 1
s = prs.slides.add_slide(blank); rect(s, 0, 0, 13.333, 7.5, NAVY); rect(s, .75, .85, .12, 1.25, TEAL)
text(s, 'Paper 1 · 固定池选择压力', 1.15, .82, 10.8, .7, 34, WHITE, True)
text(s, '跨研究外部上下文：收益并非普遍单调', 1.15, 1.62, 10.8, .55, 23, RGBColor(199, 220, 232))
rect(s, 1.15, 2.65, 3.65, .5, TEAL); text(s, '外部数据核验完成 · 机制分析待补', 1.3, 2.76, 3.35, .24, 12, WHITE, True, align=PP_ALIGN.CENTER)
text(s, 'GSE232927 · 16 池 / 情境 · 真实 CUDA · 三轮独立评分复核', 1.15, 3.65, 11.0, .4, 19, WHITE)
text(s, '当前不声称历史轨迹复现、VGSC 优势或完整预印本。', 1.15, 6.25, 10.8, .35, 13, RGBColor(173, 194, 210))

# 2
chart_path = OUT / 'paper1_sample_pressure.png'; chart(['Sample designed', 'Sample random'], chart_path, 'Sample 2019 · fixed-pool best-of-N')
s = prs.slides.add_slide(blank); header(s, 'Sample 内部：两种候选分布给出不同方向', 'designed 单调改善；random 非单调，不能合并叙述', '02')
s.shapes.add_picture(str(chart_path), Inches(.65), Inches(1.35), width=Inches(9.1))
metric(s, 9.95, 1.65, 2.55, '1.905 → 0.681', 'designed cap 1 → 16')
text(s, '这些是对 pilot outer 的二次分析，\n不是新的独立盲测。', 10.0, 3.35, 2.4, .85, 14, MUTED)
footer(s)

# 3
chart_path = OUT / 'paper1_external_pressure.png'; chart(['N50 HEK293T', 'Defined HepG2', 'Defined T-cell r1', 'Defined T-cell r2'], chart_path, 'Castillo-Hair · independent external contexts')
s = prs.slides.add_slide(blank); header(s, '外部数据：没有复现持续单调收益', 'N50、HepG2、T-cell 两个重复均为非单调或后期反转', '03')
s.shapes.add_picture(str(chart_path), Inches(.55), Inches(1.25), width=Inches(11.9))
text(s, '曲线方向不是预设结论；所有 cap 与所有负结果均保留。', .85, 6.25, 11.5, .35, 14, MUTED, align=PP_ALIGN.CENTER)
footer(s)

# 4
s = prs.slides.add_slide(blank); header(s, '可支持的结论：压力效应依赖情境', '外部结果收紧了 proxy 的适用域', '04')
metric(s, .65, 1.55, 2.85, '4 / 4', '外部情境未单调改善')
metric(s, 3.75, 1.55, 2.85, '16 × 64', '每情境独立候选', BLUE)
metric(s, 6.85, 1.55, 2.85, '3 ×', '每情境独立评分复核', GOLD)
bullets(s, ['支持：proxy 选择压力的真实功能效应必须按文库、细胞和研究限定。', '不支持：压力越大必然越好、必然越差，或 VGSC 自动解决 proxy 失效。', 'N25 全长不兼容而排除；未用 padding 或重训制造一致性。'], .85, 3.35, 11.6, 2.45)
footer(s)

# 5
s = prs.slides.add_slide(blank); header(s, 'Paper 1 下一道证据门', '从现象边界进入机制判别', '05')
bullets(s, ['分数匹配对照：区分高分尾部外推与优化器特异失真。', '机制消融：poly-U、GC、uAUG、精确/近重复和共同 proxy 风险。', '完整报告各 cap 的池簇区间与实际 forward 成本。', '补独立研究与负对照后，再判断 Paper 1 是否达到预印本证据包。'], .85, 1.55, 11.7, 3.8, 19)
rect(s, .85, 5.85, 11.55, .62, LIGHT); text(s, '当前阶段：外部固定池结果完成；完整 Paper 1 尚未完成。', 1.05, 6.02, 11.1, .28, 15, NAVY, True)
footer(s)

path = OUT / 'proxy_function_gap_paper1_external_20261124.pptx'
prs.save(path)
print(path)
