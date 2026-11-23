import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    source = Path('/home/proxy-function-gap/paper/PAPER1_DRAFT.md')
    target = source.with_suffix('.pdf')
    lines = source.read_text().splitlines()
    pdf = PdfPages(target)
    fig = None
    y = 0.0

    def new_page():
        nonlocal fig, y
        if fig is not None:
            pdf.savefig(fig)
            plt.close(fig)
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.94

    def write(value, size=9, weight='normal', family='DejaVu Sans', indent=0.08, gap=0.017):
        nonlocal y
        width = 96 if size <= 8 else 88
        for line in textwrap.wrap(value, width=width, break_long_words=False, replace_whitespace=False) or ['']:
            if y < 0.055:
                new_page()
            fig.text(indent, y, line, fontsize=size, fontweight=weight, family=family, va='top')
            y -= gap
        y -= gap * 0.45

    new_page()
    for raw in lines:
        line = raw.strip()
        if not line:
            y -= 0.008
            continue
        if line.startswith('# '):
            write(line[2:], size=17, weight='bold', gap=0.029)
        elif line.startswith('## '):
            write(line[3:], size=13, weight='bold', gap=0.024)
        elif line.startswith('### '):
            write(line[4:], size=11, weight='bold', gap=0.021)
        elif line.startswith('**') and line.endswith('**'):
            write(line.strip('*'), size=10, weight='bold', gap=0.019)
        elif line.startswith('|'):
            write(line, size=6.5, family='DejaVu Sans Mono', gap=0.014)
        elif line.startswith('```'):
            continue
        else:
            write(line, size=8.6, gap=0.017)
    if fig is not None:
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
    print(target)


if __name__ == '__main__':
    main()
