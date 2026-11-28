#!/usr/bin/env python3
"""Build and validate the single-column preprint review PDF."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
SOURCE = MANUSCRIPT / "preprint_draft.md"
BIBLIOGRAPHY = MANUSCRIPT / "references.bib"
DEFAULT_OUTPUT = MANUSCRIPT / "preprint_draft.pdf"
DEFAULT_CUTOFF = "2026-11-27"
FIGURES = {
    1: "Fig1_E0_data_and_leakage_audit_300dpi.png",
    2: "Fig2_eclip_tail_forest_v2_20261128.png",
    3: "Fig3_E5_PARCLIP_cross_protocol_300dpi.png",
    4: "Fig4_E6_GRIMM_five_split_frequency_and_OOD_v2_300dpi.png",
    5: "Fig5_ood_metric_conflicts_v2_20261128.png",
    6: "Fig6_rnacompete_dual_track_v2_20261128.png",
}

CSL = """<?xml version="1.0" encoding="utf-8"?>
<style xmlns="http://purl.org/net/xbiblio/csl" class="in-text" version="1.0" default-locale="en-US">
  <info>
    <title>ProxyFunctionGap numeric bioinformatics</title>
    <id>https://github.com/Cunyu-Liu/proxy-function-gap/numeric-bioinformatics</id>
    <category citation-format="numeric"/>
    <category field="biology"/>
    <updated>2026-11-27T00:00:00+00:00</updated>
  </info>
  <macro name="author">
    <names variable="author">
      <name initialize-with=". " delimiter=", " and="text" name-as-sort-order="all"/>
      <substitute><text variable="title"/></substitute>
    </names>
  </macro>
  <macro name="year">
    <date variable="issued"><date-part name="year"/></date>
  </macro>
  <citation collapse="citation">
    <sort><key variable="citation-number"/></sort>
    <layout prefix="[" suffix="]" delimiter=","><text variable="citation-number"/></layout>
  </citation>
  <bibliography et-al-min="7" et-al-use-first="6" hanging-indent="true">
    <layout suffix=".">
      <text variable="citation-number" suffix=". "/>
      <text macro="author" suffix=". "/>
      <text variable="title" suffix=". "/>
      <group delimiter=" ">
        <text variable="container-title" font-style="italic"/>
        <text macro="year"/>
        <group delimiter=":"><text variable="volume"/><text variable="page"/></group>
        <text variable="URL"/>
      </group>
    </layout>
  </bibliography>
</style>
"""


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result


def latex_caption(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    rendered = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            rendered.append(r"\path{" + part[1:-1] + "}")
            continue
        escaped = part
        for old, new in [
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("#", r"\#"),
            ("_", r"\_"),
        ]:
            escaped = escaped.replace(old, new)
        rendered.append(escaped)
    return "".join(rendered)


def preamble(cutoff: str) -> str:
    display_date = datetime.strptime(cutoff, "%Y-%m-%d").strftime("%d %B %Y").lstrip("0")
    return r"""
\usepackage{graphicx}
\usepackage{xeCJK}
\setCJKmainfont{Arial Unicode MS}
\setCJKmonofont{Arial Unicode MS}
\usepackage{microtype}
\usepackage{caption}
\usepackage{etoolbox}
\usepackage{needspace}
\appto\UrlBreaks{\do\_\do\-}
\captionsetup{font=small,skip=6pt}
\AtBeginEnvironment{longtable}{\scriptsize\setlength{\tabcolsep}{3pt}}
\setlength{\LTpre}{8pt}
\setlength{\LTpost}{10pt}
\setlength{\emergencystretch}{2em}
\renewcommand{\arraystretch}{1.18}
\setlength{\parskip}{0.2em}
\makeatletter
\renewcommand{\maketitle}{%
\begin{titlepage}
\centering
\vspace*{8mm}
{\Huge\bfseries DRAFT\par}
\vspace{3mm}
{\Large\bfseries INTERNAL REVIEW COPY\par}
\vspace{8mm}
\rule{\textwidth}{0.8pt}
\vspace{9mm}
{\LARGE\bfseries\@title\par}
\vspace{8mm}
\rule{\textwidth}{0.8pt}
\vspace{10mm}
{\large\bfseries Evidence cutoff: """ + display_date + r"""\par}
\vspace{5mm}
{\normalsize Repository evidence snapshot: """ + cutoff + r"""\par}
\vfill
\begin{minipage}{0.90\textwidth}
\centering
\small
RNAcompete five-fold evaluation at the cutoff: \textbf{60/60 neural and 10/10 deterministic kNN evaluations accepted}. RiNALMo, LoRA/full, and CAFA3 matrices remain ongoing and contribute no accepted results.
\end{minipage}
\vfill
\begin{minipage}{0.90\textwidth}
\centering
\small
Author names are intentionally not listed in this review copy. No submission, acceptance, or endorsement status is asserted.
\end{minipage}
\vspace{8mm}
\end{titlepage}}
\makeatother
"""


def prepare_source(text: str) -> str:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("The manuscript must begin with one level-1 title")
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:])
    body = body.replace(
        "**Preprint draft — single-column bioinformatics format**", "", 1
    )
    for number, filename in FIGURES.items():
        pattern = re.compile(
            rf"!\[Figure {number}\]\(\.\./figures/{re.escape(filename)}\)\n\n"
            rf"\*\*(Figure {number} \| [^*]+)\*\*([^\n]+)"
        )
        match = pattern.search(body)
        if not match:
            raise ValueError(f"Figure {number} image/caption pair is missing or changed")
        caption = latex_caption(match.group(1)) + latex_caption(match.group(2))
        replacement = (
            "```{=latex}\n"
            "\\clearpage\n"
            "\\begin{figure}[!ht]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\linewidth,height=0.72\\textheight,keepaspectratio]{{figure{number}.png}}\n"
            f"\\caption*{{\\textbf{{{caption}}}}}\n"
            "\\end{figure}\n"
            "```"
        )
        body = pattern.sub(lambda _: replacement, body, count=1)
    verification_start = body.index("The primary verification files are ")
    verification_end = body.index("The PAR-CLIP small-table", verification_start)
    verification_paths = re.findall(
        r"`([^`]+)`", body[verification_start:verification_end]
    )
    if len(verification_paths) != 4:
        raise ValueError("Expected four primary verification paths")
    verification_list = "The primary verification files are:\n\n" + "\n".join(
        f"- `{path}`" for path in verification_paths
    ) + "\n\n"
    body = body[:verification_start] + verification_list + body[verification_end:]
    placeholder = "References are supplied in `references.bib` and use the citation keys embedded in this Markdown draft."
    if placeholder not in body:
        raise ValueError("The references placement sentinel is missing")
    body = body.replace(placeholder, "::: {#refs}\n:::", 1)
    def breakable_code(match: re.Match[str]) -> str:
        value = match.group(1)
        if len(value) > 24 or "/" in value or re.search(r"[\u4e00-\u9fff]", value):
            escaped = value.replace("_", r"\_\allowbreak{}").replace("%", r"\%")
            escaped = escaped.replace("/", r"/\allowbreak{}").replace("-", r"-\allowbreak{}")
            return f"`\\texttt{{{escaped}}}`{{=latex}}"
        return match.group(0)

    body = re.sub(r"(?<!`)`([^`\n]+)`(?!`)", breakable_code, body)
    body = body.replace("′", r"$^{\prime}$")
    body = body.replace("∈", "in")
    body = body.replace("τ", "tau")
    body = body.replace("≥", ">=")
    body = body.replace("Δ=", "Δ = ")
    body = re.sub(r"(?<=\d\.\d{6})/(?=\d\.\d{6})", " / ", body)
    body = re.sub(
        r"\n### ",
        lambda _: "\n\n```{=latex}\n\\Needspace{10\\baselineskip}\n```\n\n### ",
        body,
    )
    body = body.replace("## Abstract\n", "## Abstract {.unnumbered}\n", 1)
    body = body.replace("## References\n", "## References {.unnumbered}\n", 1)
    body = re.sub(
        r"^(#{2,6}) ",
        lambda match: match.group(1)[1:] + " ",
        body,
        flags=re.MULTILINE,
    )
    return f"---\ntitle: {json.dumps(title, ensure_ascii=False)}\nlang: en-US\n---\n\n{body}\n"


def prepare_latex(text: str) -> str:
    ledger_start = text.rfind(r"\begin{longtable}")
    ledger_end = text.index(r"\toprule", ledger_start)
    ledger_spec = (
        "\\begin{longtable}[]{@{}\n"
        "  >{\\raggedright\\arraybackslash}p{(\\columnwidth - 2\\tabcolsep) * \\real{0.24}}\n"
        "  >{\\raggedright\\arraybackslash}p{(\\columnwidth - 2\\tabcolsep) * \\real{0.76}}@{}}\n"
    )
    text = text[:ledger_start] + ledger_spec + text[ledger_end:]

    def normalize(match: re.Match[str]) -> str:
        header = match.group(0)
        fractions = [float(value) for value in re.findall(r"\\real\{([0-9.]+)\}", header)]
        if not fractions or sum(fractions) <= 1:
            return header
        total = sum(fractions)
        index = 0

        def scaled(value: re.Match[str]) -> str:
            nonlocal index
            normalized = fractions[index] / total
            index += 1
            return f"\\real{{{normalized:.6f}}}"

        return re.sub(r"\\real\{([0-9.]+)\}", scaled, header)

    return re.sub(r"\\begin\{longtable\}.*?\\toprule", normalize, text, flags=re.DOTALL)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def spot_checks(source_text: str, pdf_text: str) -> list[str]:
    fig2 = read_tsv(ROOT / "figures/verification/data/fig2/paired_bootstrap.tsv")
    fig3_rows = (ROOT / "figures/data/parclip_p20/three_seed_macro_summary.tsv").read_text().splitlines()
    fig3 = {
        row.split("\t")[0]: row.split("\t")[1:]
        for row in fig3_rows[3:]
        if row.strip()
    }
    fig5 = read_tsv(ROOT / "figures/verification/data/fig5/macro_three_seed.tsv")
    fig4_function = read_tsv(ROOT / "figures/verification/data/fig4/function_split_summary.tsv")
    fig4_ood = read_tsv(ROOT / "figures/verification/data/fig4/ood_v2_split_summary.tsv")
    rnacompete_report = (ROOT / "E1_强基线/RNAcompete_Ray2013_RNAFM五折最终报告_2026-11-27.md").read_text(encoding="utf-8")
    rnacompete_values = ["0.475605", "0.430202", "0.352559", "0.185360", "0.111699", "0.098188"]
    for value in rnacompete_values:
        if value not in rnacompete_report:
            raise ValueError(f"RNAcompete final report is missing {value}")

    def rounded(value: str, digits: int = 6) -> str:
        return f"{float(value):.{digits}f}"

    e1_tail = next(row for row in fig2 if row["track"] == "neg1c" and row["slice"] == "tail" and row["metric"] == "auroc")
    e1_tail_pr = next(row for row in fig2 if row["track"] == "neg2" and row["slice"] == "tail" and row["metric"] == "auprc")
    deep_rna = next(row for row in fig5 if row["track"] == "neg2" and row["method"] == "deep_knn")
    protein_bucket = next(row for row in fig4_function if row["bucket"] == "tail_10_29")
    protein_msp = next(row for row in fig4_ood if row["method"] == "msp" and row["metric"] == "fpr95")
    passes = [
        (
            "pass 1: strict3train RNA",
            [
                rounded(e1_tail["observed_delta"]),
                rounded(e1_tail_pr["observed_delta"]),
                rounded(e1_tail["p_two_sided"], 8),
            ],
        ),
        (
            "pass 2: PAR-CLIP and RNA OOD",
            [
                rounded(fig3["eCLIP-neg1c-fold0"][0]),
                rounded(fig3["PARCLIP-otherRBP-geneMatched-excl300"][0]),
                rounded(deep_rna["auroc_ood_mean"]),
                rounded(deep_rna["fpr95_mean"]),
            ],
        ),
        (
            "pass 3: protein classification and pseudo-OOD",
            [
                rounded(protein_bucket["split_mean"]),
                rounded(protein_bucket["split_sd"]),
                rounded(protein_msp["split_mean"]),
                rounded(protein_msp["split_min"]),
            ],
        ),
        ("pass 4: RNAcompete sequence and RBP-OOD", rnacompete_values),
    ]
    normalized_pdf = " ".join(pdf_text.split())
    messages = []
    for label, values in passes:
        for value in values:
            if value not in source_text:
                raise ValueError(f"{label}: {value} is absent from the reviewed Markdown")
            if value not in normalized_pdf:
                raise ValueError(f"{label}: {value} is absent from the rendered PDF")
        messages.append(f"{label}: {', '.join(values)}")
    return messages


def validate_pdf(
    pdf: Path,
    source_text: str,
    latex_text: str,
    engine_log: str,
    cited_keys: set[str],
) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for render validation") from exc

    for tool in ("pdfinfo", "pdffonts", "pdftotext"):
        if not shutil.which(tool):
            raise RuntimeError(f"Required PDF validation tool is missing: {tool}")
    info = run(["pdfinfo", str(pdf)], ROOT).stdout
    page_match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    if not page_match:
        raise ValueError("pdfinfo did not report a page count")
    page_count = int(page_match.group(1))
    if page_count < 10:
        raise ValueError(f"Unexpectedly short preprint: {page_count} pages")

    font_lines = run(["pdffonts", str(pdf)], ROOT).stdout.splitlines()[2:]
    if not font_lines:
        raise ValueError("No fonts were reported")
    nonembedded = [line for line in font_lines if line.split()[-5] != "yes"]
    if nonembedded:
        raise ValueError("Non-embedded fonts:\n" + "\n".join(nonembedded))

    extracted = run(["pdftotext", "-layout", str(pdf), "-"], ROOT).stdout
    for marker in ("[@", "\ufffd", "(cid:", "???"):
        if marker in extracted:
            position = extracted.index(marker)
            context = extracted[max(0, position - 100) : position + 100]
            raise ValueError(
                f"Unresolved citation or missing-glyph marker in PDF: {marker!r}; context={context!r}"
            )
    for key in cited_keys:
        if f"@{key}" in extracted:
            raise ValueError(f"Unresolved citation key in PDF: {key}")

    document = fitz.open(pdf)
    if len(document) != page_count:
        raise ValueError("Poppler and PyMuPDF disagree on page count")
    full_text = "\n".join(page.get_text() for page in document)
    figure_pages = {}
    image_xrefs = set()
    for page_index, page in enumerate(document):
        page_text = page.get_text()
        for number in FIGURES:
            if f"Figure {number} |" in page_text:
                figure_pages[number] = page_index + 1
                if not page.get_images(full=True):
                    raise ValueError(f"Figure {number} caption has no embedded raster image on its page")
        image_xrefs.update(image[0] for image in page.get_images(full=True))
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            if x0 < -0.5 or y0 < -0.5 or x1 > page.rect.width + 0.5 or y1 > page.rect.height + 0.5:
                raise ValueError(
                    f"Text block exceeds page {page_index + 1} bounds: {block[:4]}; "
                    f"text={block[4][:300]!r}"
                )
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), colorspace=fitz.csGRAY, alpha=False)
        samples = pixmap.samples
        dark = sum(value < 250 for value in samples)
        if dark / len(samples) < 0.0005:
            raise ValueError(f"Blank or nearly blank rendered page: {page_index + 1}")
    if set(figure_pages) != set(FIGURES):
        raise ValueError(f"Missing figure captions: {sorted(set(FIGURES) - set(figure_pages))}")
    if len(image_xrefs) < len(FIGURES):
        raise ValueError(f"Only {len(image_xrefs)} unique embedded images were found")

    for number in range(1, 8):
        if f"Table {number}." not in full_text:
            raise ValueError(f"Table {number} caption is missing")
    table_count = latex_text.count(r"\begin{longtable}")
    if table_count < 8:
        raise ValueError(f"Expected at least eight longtables, found {table_count}")
    missing_glyphs = re.findall(r"Missing character:[^\n]*", engine_log)
    if missing_glyphs:
        raise ValueError("LaTeX missing glyphs detected:\n" + "\n".join(missing_glyphs))
    overflow = re.findall(r"Overfull \\[hv]box[^\n]*", engine_log)
    if overflow:
        first = engine_log.index("Overfull")
        detail = engine_log[first : first + 800]
        raise ValueError("LaTeX overflow detected:\n" + "\n".join(overflow) + "\n" + detail)

    references_text = extracted[extracted.rfind("References") :]
    reference_numbers = {int(value) for value in re.findall(r"^\s*(\d+)\.\s+", references_text, re.MULTILINE)}
    missing_references = set(range(1, len(cited_keys) + 1)) - reference_numbers
    if missing_references:
        raise ValueError(f"Missing numbered bibliography entries: {sorted(missing_references)}")

    numeric_messages = spot_checks(source_text, extracted)
    return "\n".join(
        [
            f"PDF validation passed: {page_count} pages; {len(font_lines)} embedded fonts; {len(image_xrefs)} unique images",
            f"Figures on pages: {figure_pages}; tables use {table_count} longtables; {len(cited_keys)} numbered references",
            "Render checks passed: no blank pages, out-of-page text, LaTeX overfull boxes, unresolved citations, or replacement glyphs",
            *numeric_messages,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence-cutoff", default=DEFAULT_CUTOFF)
    args = parser.parse_args()
    datetime.strptime(args.evidence_cutoff, "%Y-%m-%d")
    for tool in ("pandoc", "tectonic"):
        if not shutil.which(tool):
            raise RuntimeError(f"Required build tool is missing: {tool}")
    for path in (SOURCE, BIBLIOGRAPHY, *(ROOT / "figures" / name for name in FIGURES.values())):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_text = SOURCE.read_text(encoding="utf-8")
    cited_keys = set(re.findall(r"@([A-Za-z0-9_:-]+)", source_text))
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", BIBLIOGRAPHY.read_text(encoding="utf-8"), re.MULTILINE))
    if cited_keys != bib_keys:
        raise ValueError(f"Citation/BibTeX mismatch: uncited={bib_keys - cited_keys}, undefined={cited_keys - bib_keys}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preprint_build_", dir=MANUSCRIPT) as temporary:
        build = Path(temporary)
        for number, filename in FIGURES.items():
            shutil.copy2(ROOT / "figures" / filename, build / f"figure{number}.png")
        shutil.copy2(BIBLIOGRAPHY, build / "references.bib")
        (build / "numeric.csl").write_text(CSL, encoding="utf-8")
        (build / "preamble.tex").write_text(preamble(args.evidence_cutoff), encoding="utf-8")
        prepared = prepare_source(source_text)
        (build / "preprint.md").write_text(prepared, encoding="utf-8")
        command = [
            "pandoc",
            "preprint.md",
            "--from=markdown+raw_attribute",
            "--to=latex",
            "--standalone",
            "--citeproc",
            "--bibliography=references.bib",
            "--csl=numeric.csl",
            "--include-in-header=preamble.tex",
            "--variable=documentclass=article",
            "--variable=classoption=onecolumn",
            "--variable=fontsize=10pt",
            "--variable=geometry:a4paper",
            "--variable=geometry:top=20mm",
            "--variable=geometry:bottom=20mm",
            "--variable=geometry:left=18mm",
            "--variable=geometry:right=18mm",
            "--variable=colorlinks=true",
            "--variable=linkcolor=black",
            "--variable=citecolor=black",
            "--variable=urlcolor=blue",
            "--variable=pagestyle=plain",
            "--metadata=link-citations:true",
            "--number-sections",
            "--output=preprint.tex",
        ]
        run(command, build)
        latex_text = prepare_latex((build / "preprint.tex").read_text(encoding="utf-8"))
        (build / "preprint.tex").write_text(latex_text, encoding="utf-8")
        engine_log = run(["tectonic", "preprint.tex"], build).stdout
        candidate = build / "preprint.pdf"
        report = validate_pdf(
            candidate,
            source_text,
            latex_text,
            engine_log,
            cited_keys,
        )
        shutil.copy2(candidate, args.output)

    print(f"Built: {args.output}")
    print(report)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
