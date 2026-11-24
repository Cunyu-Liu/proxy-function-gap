import hashlib
import json
import subprocess
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path('/teams/HoulabData_1788502323/liucunyu/proxy_function_gap')
EVALUATION = ROOT / 'evaluation'
OUT = ROOT / 'ppt'
OUT.mkdir(exist_ok=True)

pressure_paths = {
    'Sample designed': EVALUATION / 'outer_designed_v1/bon_pressure.json',
    'Sample random': EVALUATION / 'outer_random_v1_eligible/bon_pressure.json',
    'N50 HEK293T': EVALUATION / 'castillo_n50_v2/bon_pressure.json',
    'Defined HepG2': EVALUATION / 'castillo_hepg2_v1/bon_pressure.json',
    'Defined T-cell r1': EVALUATION / 'castillo_tcell_r1_v1/bon_pressure.json',
    'Defined T-cell r2': EVALUATION / 'castillo_tcell_r2_v1/bon_pressure.json',
}
pressure = {name: json.loads(path.read_text()) for name, path in pressure_paths.items()}
for name in ('N50 HEK293T', 'Defined HepG2', 'Defined T-cell r1', 'Defined T-cell r2'):
    for round_id in (1, 2, 3):
        report = json.loads((pressure_paths[name].parent / f'independent_validation_round{round_id}.json').read_text())
        assert report['pass']

cross = {}
for name, folder in {'N50': 'n50_v1', 'HepG2': 'hepg2_v1', 'T-cell r1': 'tcell_r1_v1', 'T-cell r2': 'tcell_r2_v1'}.items():
    root = ROOT / 'mechanism' / folder / 'cross_arch'
    cross[name] = json.loads((root / 'summary.json').read_text())
    for round_id in (1, 2, 3):
        assert json.loads((root / f'independent_validation_round{round_id}.json').read_text())['pass']

designed = [row['mean_regret'] for row in pressure['Sample designed']['rows']]
assert all(abs(a - b) < 5e-4 for a, b in zip(designed, [1.905, 1.528, 1.161, 0.770, 0.681]))
correlations = [cross[name]['architecture_error_correlation'] for name in cross]
assert min(correlations) > 0.93 and max(correlations) < 0.966

NAVY = RGBColor(18, 35, 56)
TEAL = RGBColor(22, 154, 155)
BLUE = RGBColor(60, 110, 180)
GOLD = RGBColor(220, 158, 55)
INK = RGBColor(35, 47, 60)
MUTED = RGBColor(100, 113, 128)
LIGHT = RGBColor(242, 246, 249)
WHITE = RGBColor(255, 255, 255)
PALE = RGBColor(199, 220, 232)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def rect(slide, x, y, w, h, color):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def text(slide, value, x, y, w, h, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = 'Aptos'
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def header(slide, title, subtitle, number):
    rect(slide, 0, 0, 13.333, 0.12, TEAL)
    text(slide, title, 0.65, 0.42, 11.6, 0.55, 28, NAVY, True)
    text(slide, subtitle, 0.67, 1.0, 11.5, 0.38, 13, MUTED)
    text(slide, number, 12.25, 0.48, 0.45, 0.3, 12, MUTED, align=PP_ALIGN.RIGHT)


def footer(slide):
    text(slide, 'Proxy–Function Gap · Paper 1 preprint evidence', 0.65, 7.12, 9.5, 0.2, 9, MUTED)
    text(slide, '2026-11-24', 11.4, 7.12, 1.25, 0.2, 9, MUTED, align=PP_ALIGN.RIGHT)


def bullets(slide, items, x, y, w, h, size=17, color=INK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = '•  ' + item
        paragraph.font.name = 'Aptos'
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = color
        paragraph.space_after = Pt(12)
    return box


def metric(slide, x, y, w, value, label, accent=TEAL, dark=False):
    rect(slide, x, y, w, 1.25, NAVY if dark else LIGHT)
    rect(slide, x, y, 0.07, 1.25, accent)
    text(slide, value, x + 0.22, y + 0.18, w - 0.35, 0.45, 24, WHITE if dark else NAVY, True)
    text(slide, label, x + 0.22, y + 0.76, w - 0.35, 0.28, 11, PALE if dark else MUTED)


slide = prs.slides.add_slide(blank)
rect(slide, 0, 0, 13.333, 7.5, NAVY)
rect(slide, 0.78, 0.72, 0.12, 1.55, TEAL)
text(slide, 'PAPER 1  /  PREPRINT EVIDENCE', 1.15, 0.65, 10.8, 0.3, 12, PALE, True)
text(slide, 'Proxy 选择压力的情境边界\n与跨架构共同失效', 1.15, 1.08, 11.1, 1.25, 31, WHITE, True)
text(slide, '5′ UTR 固定池历史实验回放', 1.15, 2.48, 10.8, 0.42, 20, PALE)
metric(slide, 1.15, 3.35, 4.85, '1.905 → 0.681', 'Sample designed · cap 1 → 16 实测 regret', TEAL, True)
metric(slide, 6.25, 3.35, 4.85, '0.931–0.965', 'CNN vs 4-mer MLP · 四情境误差相关', GOLD, True)
text(slide, '真实 CUDA 推理  ·  历史实测 MRL  ·  池簇区间  ·  三轮独立复核', 1.15, 5.35, 10.9, 0.35, 16, WHITE)
text(slide, '结论限定于当前数据与回顾性证据，不构成前瞻湿实验验证。', 1.15, 6.45, 10.9, 0.35, 12, PALE)

slide = prs.slides.add_slide(blank)
header(slide, '固定池中，proxy 高分选择带来真实收益', '候选生成过程不变，仅增加 best-of-N 选择压力', '02')
slide.shapes.add_picture(str(OUT / 'paper1_sample_pressure.png'), Inches(0.55), Inches(1.45), width=Inches(8.85))
metric(slide, 9.7, 1.65, 2.8, '64.3%', 'designed regret 相对下降', TEAL)
metric(slide, 9.7, 3.15, 2.8, '5 档', 'cap {1, 2, 4, 8, 16}', BLUE)
text(slide, '主终点是历史实测 MRL regret，\n不是 proxy 分数或训练集相关性。', 9.78, 4.85, 2.55, 0.9, 14, MUTED)
footer(slide)

slide = prs.slides.add_slide(blank)
header(slide, '外部上下文界定 proxy 的适用边界', '四个 Castillo-Hair 情境呈现不同的压力响应曲线', '03')
slide.shapes.add_picture(str(OUT / 'paper1_external_pressure.png'), Inches(0.55), Inches(1.32), width=Inches(12.1))
rect(slide, 0.8, 6.18, 11.75, 0.55, LIGHT)
text(slide, '结论：proxy 的有效性必须按文库、细胞与实验情境声明，不能预设统一方向。', 1.0, 6.33, 11.35, 0.26, 14, NAVY, True, align=PP_ALIGN.CENTER)
footer(slide)

slide = prs.slides.add_slide(blank)
header(slide, '共同 proxy 误差跨越模型架构', '独立训练的 CNN 与 4-mer MLP 在四个外部情境中呈现强误差相关', '04')
slide.shapes.add_picture(str(OUT / 'paper1_cross_arch_correlation.png'), Inches(0.45), Inches(1.55), width=Inches(8.25))
metric(slide, 9.0, 1.65, 3.35, '0.931–0.965', '绝对误差相关范围', GOLD)
bullets(slide, ['两种架构、同一训练分布', '四个外部情境一致复现', '模型投票不能替代独立终点'], 9.05, 3.35, 3.15, 2.0, 15)
footer(slide)

slide = prs.slides.add_slide(blank)
header(slide, '预印本证据链与复现边界', '正向主结果、外部情境与机制证据已统一', '05')
for x, title, body, accent in [
    (0.65, '选择收益', 'Sample designed 中，增加选择压力持续降低历史实测 regret。', TEAL),
    (4.55, '共同风险', '两种架构在四个外部情境中呈现高度一致的误差结构。', GOLD),
    (8.45, '适用边界', '结论限定于回顾性固定池证据，不外推为普遍生物功能保证。', BLUE),
]:
    rect(slide, x, 1.55, 3.55, 2.05, LIGHT)
    rect(slide, x, 1.55, 3.55, 0.07, accent)
    text(slide, title, x + 0.25, 1.85, 3.0, 0.35, 19, NAVY, True)
    text(slide, body, x + 0.25, 2.38, 3.0, 0.9, 14, INK)
metric(slide, 0.65, 4.15, 2.75, '3 seeds', '模型训练至开发损失平台', TEAL)
metric(slide, 3.65, 4.15, 2.75, '3 rounds', '关键指标独立复核', BLUE)
metric(slide, 6.65, 4.15, 2.75, '224 tests', '完整测试通过', GOLD)
metric(slide, 9.65, 4.15, 2.75, '6 pages', 'A4 预印本 PDF', TEAL)
text(slide, '代码：github.com/Cunyu-Liu/proxy-function-gap', 0.75, 6.12, 11.8, 0.35, 15, NAVY, True, align=PP_ALIGN.CENTER)
text(slide, '大型数据、权重与评分产物保存在 /teams；代码与提交保存在 /home。', 0.75, 6.55, 11.8, 0.3, 12, MUTED, align=PP_ALIGN.CENTER)
footer(slide)

path = OUT / 'proxy_function_gap_preprint_results_20261124.pptx'
prs.save(path)

check = Presentation(path)
assert len(check.slides) == 5
slide_width = check.slide_width
slide_height = check.slide_height
texts = []
for slide in check.slides:
    for shape in slide.shapes:
        assert shape.left >= 0 and shape.top >= 0
        assert shape.left + shape.width <= slide_width + 1
        assert shape.top + shape.height <= slide_height + 1
        if hasattr(shape, 'text'):
            texts.append(shape.text)
joined = '\n'.join(texts)
assert 'VGSC' not in joined and '尚未完成' not in joined and '机制分析待补' not in joined
assert '1.905 → 0.681' in joined and '0.931–0.965' in joined and '224 tests' in joined
report = {
    'pass': True,
    'slides': len(check.slides),
    'shape_bounds': 'pass',
    'content_boundary': 'positive_main_results_no_vgsc_pilot',
    'source_commit': subprocess.check_output(['git', '-C', '/home/proxy-function-gap', 'rev-parse', 'HEAD'], text=True).strip(),
    'pptx': str(path),
    'bytes': path.stat().st_size,
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
}
(OUT / 'preprint_deck_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
