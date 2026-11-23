import json
from pathlib import Path

import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path('/teams/HoulabData_1788502323/liucunyu/proxy_function_gap')
TRAIN = ROOT / 'fresh_proxy_cnn_v2/training'
RUN = TRAIN / 'formal_run1_retry1'
OUT = ROOT / 'ppt'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((TRAIN / 'independent_summary.json').read_text())
assert summary['status'] == 'PASS' and summary['all_converged_dev_plateau']
assert not summary['outer_labels_read'] and not summary['outer_metrics_computed']

NAVY = RGBColor(18, 35, 56)
TEAL = RGBColor(22, 154, 155)
BLUE = RGBColor(60, 110, 180)
GOLD = RGBColor(220, 158, 55)
INK = RGBColor(35, 47, 60)
MUTED = RGBColor(100, 113, 128)
LIGHT = RGBColor(242, 246, 249)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(45, 145, 95)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def rect(slide, x, y, w, h, fill, radius=False):
    shape = slide.shapes.add_shape(5 if radius else 1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    return shape


def text(slide, value, x, y, w, h, size=20, color=INK, bold=False,
         align=PP_ALIGN.LEFT, font='Aptos', valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True; tf.vertical_anchor = valign
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = value
    r.font.name = font; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return box


def bullets(slide, items, x, y, w, h, size=19, color=INK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item; p.font.name = 'Aptos'; p.font.size = Pt(size); p.font.color.rgb = color
        p.space_after = Pt(13); p.level = 0
        p._p.get_or_add_pPr().insert(0, p._p._new_buChar()) if False else None
        p.text = '•  ' + item
    return box


def header(slide, title, subtitle=None, number=None):
    rect(slide, 0, 0, 13.333, 0.12, TEAL)
    text(slide, title, 0.65, 0.42, 11.8, 0.55, 28, NAVY, True)
    if subtitle:
        text(slide, subtitle, 0.67, 1.0, 11.6, 0.38, 13, MUTED)
    if number:
        text(slide, number, 12.25, 0.48, 0.45, 0.3, 12, MUTED, align=PP_ALIGN.RIGHT)


def footer(slide, label='Proxy–Function Gap · verified technical progress'):
    text(slide, label, 0.65, 7.12, 9.5, 0.2, 9, MUTED)
    text(slide, '2026-11-24', 11.4, 7.12, 1.25, 0.2, 9, MUTED, align=PP_ALIGN.RIGHT)


def metric(slide, x, y, w, value, label, accent=TEAL):
    rect(slide, x, y, w, 1.22, LIGHT, True)
    rect(slide, x, y, 0.07, 1.22, accent)
    text(slide, value, x + 0.22, y + 0.18, w - 0.35, 0.45, 25, NAVY, True)
    text(slide, label, x + 0.22, y + 0.73, w - 0.35, 0.28, 11, MUTED)

# 1
s = prs.slides.add_slide(blank)
rect(s, 0, 0, 13.333, 7.5, NAVY)
rect(s, 0.75, 0.85, 0.12, 1.25, TEAL)
text(s, 'Proxy–Function Gap', 1.15, 0.82, 10.8, 0.7, 34, WHITE, True)
text(s, '固定池选择压力：数据与训练阶段进展', 1.15, 1.62, 10.8, 0.55, 23, RGBColor(199, 220, 232))
rect(s, 1.15, 2.65, 3.25, 0.5, TEAL, True)
text(s, '训练门禁通过 · Outer 未揭盲', 1.3, 2.76, 2.95, 0.24, 12, WHITE, True, align=PP_ALIGN.CENTER)
text(s, '三种子 fresh CNN 并行 CUDA 训练', 1.15, 3.65, 10.5, 0.4, 20, WHITE)
text(s, '无 epoch / step 上限  ·  dual-plateau 收敛  ·  全行复算', 1.15, 4.2, 10.8, 0.35, 16, RGBColor(199, 220, 232))
text(s, '本稿仅记录技术进展，不声称方法优势或生物功能结论。', 1.15, 6.25, 10.8, 0.35, 13, RGBColor(173, 194, 210))
text(s, '2026-11-24', 11.1, 6.9, 1.25, 0.25, 10, RGBColor(173, 194, 210), align=PP_ALIGN.RIGHT)

# 2
s = prs.slides.add_slide(blank); header(s, '数据契约与标签隔离', 'Sample 2019 · fresh proxy 专用谱系 · random / designed 分层', '02')
metric(s, 0.65, 1.55, 2.85, '2,753,885', '原始序列—来源行')
metric(s, 3.75, 1.55, 2.85, '293,091', 'fresh train / scaler 行', BLUE)
metric(s, 6.85, 1.55, 2.85, '4,236', 'random main outer 行', GOLD)
metric(s, 9.95, 1.55, 2.75, '2,879', 'designed outer 行', GREEN)
bullets(s, [
    '固定 50 nt 标准化；原始序列、来源行与哈希完整保留。',
    '精确重复、Hamming-1、allocation unit 与完整轨迹按组隔离。',
    'train / dev / calibration / query / outer 权限分离；query 无标签。',
    'outer 属历史实测状态，不声称“从未被任何人评估”。'
], 0.85, 3.25, 11.6, 2.65, 18)
rect(s, 0.85, 6.15, 11.55, 0.52, RGBColor(232, 242, 242), True)
text(s, '门禁结论：具备 fresh 训练资格；E1 科学评分仍需校准与配置冻结。', 1.05, 6.28, 11.1, 0.25, 14, NAVY, True)
footer(s)

# 3 chart
histories = {}
for seed in (11, 23, 37):
    rows = [json.loads(line) for line in (RUN / f'seed_{seed}/history.jsonl').read_text().splitlines() if line.strip()]
    histories[seed] = rows
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.65), sharey=True)
colors = {'train': '#16a09b', 'dev': '#3c6eb4'}
for ax, seed in zip(axes, (11, 23, 37)):
    rows = histories[seed]
    epochs = [r['epoch'] for r in rows]
    ax.plot(epochs, [r['train_eval_loss'] for r in rows], color=colors['train'], lw=1.8, label='Full train eval')
    ax.plot(epochs, [r['dev_eval_loss'] for r in rows], color=colors['dev'], lw=1.8, label='Full dev eval')
    best = next(x for x in summary['seeds'] if x['seed'] == seed)
    ax.axvline(best['best_epoch'], color='#dc9e37', lw=1.2, ls='--')
    ax.scatter([best['best_epoch']], [best['best_dev_eval_loss']], s=32, color='#dc9e37', zorder=5)
    ax.set_title(f'Seed {seed} · best epoch {best["best_epoch"]}', fontsize=11)
    ax.set_xlabel('Epoch'); ax.grid(alpha=.18); ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('MSE loss (train-scaled label)')
axes[1].legend(frameon=False, fontsize=9, loc='upper right')
fig.tight_layout()
chart = OUT / 'fresh_proxy_convergence.png'
fig.savefig(chart, dpi=190, bbox_inches='tight', facecolor='white'); plt.close(fig)
s = prs.slides.add_slide(blank); header(s, '三种子均达到 dual-plateau 收敛', '完整 train / dev eval 用于停止；金色虚线为最佳 dev checkpoint', '03')
s.shapes.add_picture(str(chart), Inches(0.65), Inches(1.45), width=Inches(12.0))
text(s, '无最大 epoch / step；patience = 5，min_delta = 0.0001。训练曲线仅作开发诊断。', 0.8, 6.25, 11.8, 0.35, 14, MUTED, align=PP_ALIGN.CENTER)
footer(s)

# 4 table
s = prs.slides.add_slide(blank); header(s, '收敛与最佳 checkpoint', '最佳模型按完整 dev loss 选择；最终 loss 不等于最佳 checkpoint loss', '04')
cols = [0.8, 2.0, 3.45, 5.2, 7.15, 9.1, 11.0]
headers = ['Seed', 'Epochs', 'Best epoch', 'Best train', 'Best dev', 'Final dev', 'Peak GPU']
for x, h in zip(cols, headers): text(s, h, x, 1.55, 1.55, 0.35, 13, MUTED, True)
rect(s, 0.7, 2.0, 11.95, 0.02, RGBColor(210, 220, 228))
for i, item in enumerate(summary['seeds']):
    y = 2.25 + i * 0.88
    if i % 2 == 0: rect(s, 0.7, y - 0.12, 11.95, 0.7, LIGHT)
    vals = [str(item['seed']), str(item['epochs']), str(item['best_epoch']),
            f"{item['best_train_eval_loss']:.4f}", f"{item['best_dev_eval_loss']:.4f}",
            f"{item['final_dev_eval_loss']:.4f}", f"{item['max_memory_allocated_bytes']/1e9:.2f} GB"]
    for x, val in zip(cols, vals): text(s, val, x, y, 1.55, 0.32, 16, NAVY, val in (str(item['seed']),))
rect(s, 0.8, 5.35, 11.7, 0.85, RGBColor(232, 242, 242), True)
text(s, '3 / 3 converged', 1.05, 5.55, 2.6, 0.35, 20, GREEN, True)
text(s, '无 OOM · 无 CPU fallback · 无后台残留', 3.75, 5.59, 7.9, 0.3, 16, NAVY)
footer(s)

# 5
s = prs.slides.add_slide(blank); header(s, '验证证据链', '训练制品、代码与硬件身份均可追溯', '05')
metric(s, 0.75, 1.55, 3.55, '201 passed', '训练后完整测试', GREEN)
metric(s, 4.9, 1.55, 3.55, '3 × CUDA', '实际模型与输入设备', TEAL)
metric(s, 9.05, 1.55, 3.55, '2 × 全行', 'train / dev 最佳权重复算', BLUE)
bullets(s, [
    '逐 epoch 保存 loss、数据顺序、配置与模型状态哈希。',
    'best / latest checkpoint 原子写入；恢复训练校验完整身份。',
    'fresh 初始化；未使用 official 权重或蒸馏。',
    'outer_labels_read = false；outer_metrics_computed = false。'
], 0.9, 3.25, 11.5, 2.55, 18)
text(s, '训练代码：fd5e673  |  冻结配置：9c333998…  |  pack：1df96fab…', 0.9, 6.15, 11.4, 0.3, 12, MUTED)
footer(s)

# 6
s = prs.slides.add_slide(blank); header(s, '下一道门禁', '先完成开发集校准与冻结，再进行一次性 outer 评测', '06')
steps = [
    ('01', '固定池 v2', '非重叠完整簇；random / designed 分层'),
    ('02', 'V / Q 校准', 'dev grouped OOF；calibration 仅最终校准'),
    ('03', '内层冻结', 'α、实际预算、基线、覆盖与统计规则'),
    ('04', 'Outer 评测', '选择清单先冻结；随后由 Evaluator 评分')
]
for i, (num, title, desc) in enumerate(steps):
    y = 1.55 + i * 1.13
    rect(s, 0.85, y, 0.62, 0.62, TEAL if i < 3 else NAVY, True)
    text(s, num, 0.85, y + 0.14, 0.62, 0.25, 13, WHITE, True, align=PP_ALIGN.CENTER)
    text(s, title, 1.75, y - 0.02, 2.5, 0.35, 19, NAVY, True)
    text(s, desc, 4.25, y + 0.02, 7.75, 0.35, 16, INK)
rect(s, 0.85, 6.2, 11.55, 0.48, RGBColor(250, 242, 226), True)
text(s, '当前不声称：相对基线优势、噪声鲁棒性、Q 增量或预印本级科学结论。', 1.05, 6.31, 11.1, 0.25, 14, RGBColor(133, 91, 25), True)
footer(s)

path = OUT / 'proxy_function_gap_training_progress_20261124.pptx'
prs.save(path)
print(path)
