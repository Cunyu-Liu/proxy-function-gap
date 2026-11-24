import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
LEFT = 0.09
RIGHT = 0.91
BOTTOM = 0.055
BODY_SIZE = 9.5


def clean_inline(value):
    value = value.replace('\\(', '$').replace('\\)', '$')
    value = value.replace('**', '').replace('`', '')
    return value.strip()


def wrap_text(value, width, subsequent_indent=''):
    return textwrap.wrap(
        clean_inline(value),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=False,
        subsequent_indent=subsequent_indent,
    ) or ['']


def parse_table(lines):
    rows = []
    for line in lines:
        cells = [clean_inline(cell) for cell in line.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{3,}:?', cell.replace(' ', '')) for cell in cells):
            continue
        rows.append(cells)
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('invalid Markdown table')
    return rows


def parse_blocks(source):
    blocks = []
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            index += 1
            continue
        if line.startswith('|'):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                table_lines.append(lines[index].strip())
                index += 1
            blocks.append(('table', parse_table(table_lines)))
            continue
        if line.startswith('### '):
            blocks.append(('subsection', clean_inline(line[4:])))
        elif line.startswith('## '):
            blocks.append(('section', clean_inline(line[3:])))
        elif line.startswith('# '):
            blocks.append(('title', clean_inline(line[2:])))
        elif line.startswith('**') and line.endswith('**'):
            blocks.append(('caption', clean_inline(line)))
        else:
            blocks.append(('paragraph', raw))
        index += 1
    return blocks


class PaperRenderer:
    def __init__(self, pdf):
        self.pdf = pdf
        self.fig = None
        self.y = 0.0
        self.page = 0
        self.in_references = False

    def new_page(self):
        if self.fig is not None:
            self.fig.text(0.5, 0.025, str(self.page), ha='center', fontsize=8, family='DejaVu Serif')
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.page += 1
        self.fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        if self.page > 1:
            self.fig.text(LEFT, 0.963, 'CONTEXTUAL BOUNDARIES OF PROXY SELECTION PRESSURE', fontsize=7.2, family='DejaVu Serif', color='#4b5563')
            self.fig.text(RIGHT, 0.963, 'PREPRINT', ha='right', fontsize=7.2, family='DejaVu Serif', color='#4b5563')
            self.fig.add_artist(Line2D([LEFT, RIGHT], [0.952, 0.952], transform=self.fig.transFigure, color='#cbd5e1', linewidth=0.6))
        self.y = 0.935 if self.page == 1 else 0.925

    def ensure(self, height):
        if self.fig is None or self.y - height < BOTTOM:
            self.new_page()

    def paragraph(self, value, size=BODY_SIZE, weight='normal', width=98, indent=LEFT, color='#111827', gap=0.007, subsequent_indent=''):
        lines = wrap_text(value, width, subsequent_indent=subsequent_indent)
        line_height = size / 72 / PAGE_HEIGHT * 1.42
        self.ensure(len(lines) * line_height + gap)
        for line in lines:
            if self.y < BOTTOM:
                self.new_page()
            self.fig.text(indent, self.y, line, fontsize=size, fontweight=weight, family='DejaVu Serif', color=color, va='top')
            self.y -= line_height
        self.y -= gap

    def heading(self, value, level):
        if value == 'References':
            self.in_references = True
        size = 13 if level == 1 else 10.5
        height = 0.052 if level == 1 else 0.038
        self.ensure(height)
        self.y -= 0.008 if level == 1 else 0.003
        self.fig.text(LEFT, self.y, value, fontsize=size, fontweight='bold', family='DejaVu Serif', color='#0f172a', va='top')
        self.y -= 0.027 if level == 1 else 0.024

    def caption(self, value):
        self.ensure(0.038)
        self.fig.text(LEFT, self.y, value, fontsize=8.5, fontweight='bold', family='DejaVu Serif', color='#1e293b', va='top')
        self.y -= 0.025

    def table(self, rows):
        columns = len(rows[0])
        first = {6: 0.30, 5: 0.29, 4: 0.34}.get(columns, 0.28)
        widths = [first] + [(RIGHT - LEFT - first) / (columns - 1)] * (columns - 1)
        font_size = 7.4
        line_height = 0.013
        wrapped = []
        for row in rows:
            cells = []
            for value, width in zip(row, widths):
                chars = max(8, int(width * PAGE_WIDTH * 72 / (font_size * 0.52)) - 2)
                cells.append(wrap_text(value, chars))
            wrapped.append(cells)
        heights = [max(len(cell) for cell in row) * line_height + 0.010 for row in wrapped]
        total = sum(heights) + 0.018
        self.ensure(total)
        for row_index, (row, row_height) in enumerate(zip(wrapped, heights)):
            top = self.y
            if row_index == 0:
                self.fig.add_artist(Rectangle((LEFT, top - row_height), RIGHT - LEFT, row_height, transform=self.fig.transFigure, facecolor='#eaf0f6', edgecolor='none', zorder=-1))
            self.fig.add_artist(Line2D([LEFT, RIGHT], [top, top], transform=self.fig.transFigure, color='#64748b', linewidth=0.65))
            x = LEFT
            for value, width in zip(row, widths):
                for line_index, line in enumerate(value):
                    self.fig.text(x + 0.004, top - 0.007 - line_index * line_height, line, fontsize=font_size, fontweight='bold' if row_index == 0 else 'normal', family='DejaVu Serif', va='top')
                x += width
            self.y -= row_height
            if row_index == 0:
                self.fig.add_artist(Line2D([LEFT, RIGHT], [self.y, self.y], transform=self.fig.transFigure, color='#94a3b8', linewidth=0.5))
        self.fig.add_artist(Line2D([LEFT, RIGHT], [self.y, self.y], transform=self.fig.transFigure, color='#64748b', linewidth=0.65))
        self.y -= 0.018

    def render(self, blocks):
        for kind, value in blocks:
            if kind == 'title':
                self.paragraph(value, size=18, weight='bold', width=54, gap=0.012, color='#0f172a')
            elif kind == 'section' and value == 'Fixed-pool evidence from 5′ UTR optimization':
                self.paragraph(value, size=12, width=70, gap=0.012, color='#334155')
            elif kind == 'section':
                self.heading(value, 1)
            elif kind == 'subsection':
                if value == 'Abstract':
                    self.heading(value, 2)
                else:
                    self.heading(value, 2)
            elif kind == 'caption':
                self.caption(value)
            elif kind == 'table':
                self.table(value)
            elif kind == 'paragraph':
                stripped = value.strip()
                if stripped == 'Anonymous authors':
                    self.paragraph(stripped, size=10, width=100, gap=0.016, color='#475569')
                elif stripped.startswith('Keywords:'):
                    self.paragraph(stripped, size=8.5, width=108, gap=0.012, color='#475569')
                elif re.match(r'^\[\d+\]', stripped):
                    self.paragraph(stripped, size=8.2, width=110, gap=0.006, subsequent_indent='    ')
                elif re.match(r'^\d+\. ', stripped):
                    self.paragraph(stripped, size=BODY_SIZE, width=98, subsequent_indent='   ')
                else:
                    self.paragraph(stripped, size=8.2 if self.in_references else BODY_SIZE, width=110 if self.in_references else 98)
        if self.fig is not None:
            self.fig.text(0.5, 0.025, str(self.page), ha='center', fontsize=8, family='DejaVu Serif')
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None


def main():
    source = Path('/home/proxy-function-gap/paper/PAPER1_DRAFT.md')
    target_dir = Path('/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper')
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / 'PAPER1_DRAFT.pdf'
    blocks = parse_blocks(source.read_text())
    metadata = {
        'Title': 'Contextual Boundaries of Proxy Selection Pressure in Biological Sequence Design',
        'Author': 'Anonymous authors',
        'Subject': 'Fixed-pool evidence from 5′ UTR optimization',
        'Keywords': 'proxy-function gap, biological sequence design, 5′ UTR, model-based optimization',
    }
    with PdfPages(target, metadata=metadata) as pdf:
        PaperRenderer(pdf).render(blocks)
    print(target)


if __name__ == '__main__':
    main()
