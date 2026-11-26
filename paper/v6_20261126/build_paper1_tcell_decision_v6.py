#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from delivery_scripts.publication_paths import DELIVERABLES, OUTPUT_ROOT

FIGURES = DELIVERABLES / "figures"
SOURCE = HERE / "PAPER1_TCELL_DECISION_V6_20261126.md"
OUT = OUTPUT_ROOT / "paper" / "v6_20261126"
OUT.mkdir(parents=True, exist_ok=True)
OUTPUT = SOURCE.with_suffix(".pdf")
SOURCE_DATE_EPOCH = int(datetime(2026, 11, 26, tzinfo=timezone.utc).timestamp())
FIGURE_MAP = [
    ("Figure 14 |", "v3/Figure14_v3_equal_budget_value.pdf"),
    ("Figure 16 |", "v4/Figure16_v4_tcell_global_value_tail_bottleneck.pdf"),
    ("Figure 17 |", "v4/Figure17_v4_tcell_regression_decision_decoupling.pdf"),
]
HEADER = r"""
\special{pdf:minorversion 7}
\usepackage{unicode-math}
\setmathfont{STIX Two Math}
\usepackage{graphicx}
\usepackage{caption}
\captionsetup{font=footnotesize,labelfont=bf,justification=justified,singlelinecheck=false}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Anonymous authors}
\fancyhead[R]{\small T-cell decision-focused Paper 1, V6}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}
\setlength{\emergencystretch}{6em}
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.12}
\usepackage{etoolbox}
\usepackage{seqsplit}
\let\preprinttexttt\texttt
\DeclareRobustCommand{\preprintseqtexttt}[1]{\preprinttexttt{\seqsplit{#1}}}
\let\texttt\preprintseqtexttt
\AtBeginEnvironment{longtable}{\footnotesize}
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_source(work: Path) -> tuple[Path, dict]:
    original = SOURCE.read_text(encoding="utf-8")
    replacements = {}

    def replace_display_math(match: re.Match) -> str:
        source_block = match.group(0)
        rendered_block = f"$$\n{match.group(1)}$$"
        replacements[rendered_block] = source_block
        return rendered_block

    rendered_text = re.sub(r"(?ms)^\\\[\n(.*?)^\\\]$", replace_display_math, original)

    def replace_inline_math(match: re.Match) -> str:
        source_math = match.group(0)
        rendered_math = f"${match.group(1)}$"
        replacements[rendered_math] = source_math
        return rendered_math

    rendered_text = re.sub(r"\\\(([^\n]*?)\\\)", replace_inline_math, rendered_text)
    for prefix, name in FIGURE_MAP:
        matches = [line for line in rendered_text.splitlines() if line.startswith(f"**{prefix}")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one caption beginning {prefix}, found {len(matches)}")
        caption = matches[0]
        if name not in caption:
            raise RuntimeError(f"{prefix} does not cite {name}")
        source_figure = FIGURES / name
        if not source_figure.is_file():
            raise RuntimeError(f"Missing figure: {source_figure}")
        shutil.copyfile(source_figure, work / Path(name).name)
        rendered_name = Path(name).name
        rendered_line = f"![{caption}]({rendered_name}){{width=\\linewidth}}"
        rendered_text = rendered_text.replace(caption, rendered_line, 1)
        replacements[rendered_line] = caption
    rendered = work / SOURCE.name
    rendered.write_text(rendered_text, encoding="utf-8")
    restored = rendered.read_text(encoding="utf-8")
    for rendered_line, source_line in replacements.items():
        restored = restored.replace(rendered_line, source_line)
    if restored != original:
        raise RuntimeError("Scientific content invariant failed")
    return rendered, replacements


def build_pdf(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / OUTPUT.name
    with tempfile.TemporaryDirectory(prefix=".paper1_v6_build_", dir=output_dir) as temporary:
        work = Path(temporary)
        rendered, _ = prepare_source(work)
        header = work / "paper1_v6_header.tex"
        header.write_text(HEADER, encoding="utf-8")
        tex_output = work / "paper1_v6.tex"
        environment = os.environ.copy()
        environment.update({"SOURCE_DATE_EPOCH": str(SOURCE_DATE_EPOCH), "TZ": "UTC"})
        command = [
            "pandoc",
            str(rendered),
            "--from",
            "markdown+tex_math_dollars+tex_math_single_backslash+raw_attribute+implicit_figures",
            "--resource-path",
            str(work),
            "--standalone",
            "--to",
            "latex",
            "--include-in-header",
            str(header),
            "-V",
            "geometry:margin=0.70in",
            "-V",
            "fontsize=10pt",
            "-V",
            "mainfont=Times New Roman",
            "-V",
            "sansfont=Helvetica",
            "-V",
            "monofont=Menlo",
            "-V",
            "colorlinks=true",
            "-V",
            "linkcolor=blue",
            "-V",
            "urlcolor=blue",
            "-o",
            str(tex_output),
        ]
        pandoc = subprocess.run(command, cwd=work, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if pandoc.returncode:
            raise RuntimeError(f"Pandoc conversion failed:\n{pandoc.stdout + pandoc.stderr}")
        latex = tex_output.read_text(encoding="utf-8")
        latex = re.sub(r"\(\\columnwidth - (\d+)\\tabcolsep\)", lambda match: f"(\\columnwidth - {int(match.group(1)) + 3}\\tabcolsep)", latex)
        tex_output.write_text(latex, encoding="utf-8")
        completed = subprocess.run(["tectonic", "--outdir", str(work), str(tex_output)], cwd=work, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log = pandoc.stdout + pandoc.stderr + completed.stdout + completed.stderr
        if completed.returncode:
            raise RuntimeError(f"Tectonic build failed:\n{log}")
        warnings = [line for line in log.splitlines() if re.search(r"warning|overfull|undefined|missing character", line, re.I)]
        serious = [line for line in warnings if re.search(r"overfull|undefined|missing character", line, re.I)]
        if serious:
            raise RuntimeError("Serious layout warnings:\n" + "\n".join(serious))
        shutil.copyfile(work / "paper1_v6.pdf", output)
    return {
        "source": SOURCE.name,
        "output": output.name,
        "source_sha256": sha256(SOURCE),
        "output_sha256": sha256(output),
        "content_invariant": True,
        "warnings": warnings,
        "serious_warnings": serious,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "command": "pandoc <rendered-source> --standalone --to latex; tectonic <rendered-tex>",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    record = build_pdf(args.output_dir)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
