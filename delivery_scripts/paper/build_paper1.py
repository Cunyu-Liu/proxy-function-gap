#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "pandoc", str(HERE / "PAPER1_DRAFT.md"),
        "--from", "markdown+tex_math_dollars+raw_attribute",
        "--resource-path", str(HERE),
        "--pdf-engine", "tectonic",
        "--toc",
        "-V", "geometry:margin=0.85in",
        "-V", "fontsize=11pt",
        "-V", "mainfont=STIX Two Text",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue",
        "-o", str(output),
    ], check=True, cwd=ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "PAPER1_DRAFT.pdf")
    args = parser.parse_args()
    build(args.output.resolve())
