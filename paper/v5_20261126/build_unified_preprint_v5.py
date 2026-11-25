#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from delivery_scripts.publication_paths import DELIVERABLES, EVIDENCE_ROOT, OUTPUT_ROOT, REPO_ROOT, display_path

ROOT = EVIDENCE_ROOT
FIGURES = DELIVERABLES / "figures"
OUT = OUTPUT_ROOT / "paper" / "v5_20261126"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "UNIFIED_PREPRINT_V5_RELEASE_MANIFEST.json"
SOURCE_DATE_EPOCH = int(datetime(2026, 11, 26, tzinfo=timezone.utc).timestamp())

DOCUMENTS = [
    {
        "key": "main",
        "source": "UNIFIED_PREPRINT_MAIN_V5_20261126.md",
        "output": "UNIFIED_PREPRINT_MAIN_V5_20261126.pdf",
        "figures": [
            ("Graphical abstract |", ["v3/Figure11_v3_evidence_chain.pdf"]),
            ("Figure 1 |", ["Figure2_v2_selection_pressure.pdf"]),
            ("Figure 2 |", ["Figure3_v2_score_matching.pdf"]),
            ("Figure 3 |", ["Figure4_v2_common_proxy_error.pdf"]),
            ("Figure 4 |", ["Figure9_v2_schlusser_te.pdf", "Figure10_v2_leppek_endpoints.pdf"]),
            ("Figure 5 |", ["Figure7_v2_paper2_baselines.pdf"]),
            ("Figure 6 |", ["Figure8_v2_paper3_headroom.pdf"]),
            ("Figure 13 |", ["v3/Figure13_v3_persist_external.pdf"]),
            ("Figure 14 |", ["v3/Figure14_v3_equal_budget_value.pdf"]),
            ("Figure 15 |", ["v3/Figure15_v3_sequence_composition_sensitivity.pdf"]),
            ("Extended Data Figure 1 |", ["ExtendedData_proxy_noise.pdf"]),
            ("Extended Data Figure 2 |", ["v3/Figure12_v3_data_governance_ood.pdf"]),
        ],
        "required_source": [
            "Candidate-level uncertainty abstention fails its frozen calibration gate",
            "outer scoring was hard-locked and not performed",
            "Outer was not scored",
            "no new outer result was generated",
            "permanent outer labels were neither read nor scored",
            "not a new blind test",
            "numerical reproducibility, not biological or outer effectiveness",
            "59 eligible constructs spanning 49 unique 5′ UTRs",
            "Exact and equal-length Hamming-1 audits",
            "+0.104723",
            "[−0.100964, 0.212706]",
            "p=0.245175",
            "−0.008447",
            "not a blind test",
            "not proof of universal proxy failure",
            "training lineage, architecture, and construct",
            "−0.799949",
            "−1.448756",
            "−1.579340",
            "−1.705760",
            "only cap 4 survived Holm",
            "Holm p=0.055994",
            "no adjusted effect surviving Holm",
            "not evidence that uAUG or poly-U causally mediates",
            "SeedSequence(20261125)",
            "prediction descending, then ascending `Sequence ID`, then row position",
            "SHA256 alternative",
            "requires its own gate",
        ],
        "required_pdf": [
            "Proxy ranking helps at equal budget",
            "Abstract",
            "0.104026897",
            "2.866883",
            "Outer was not scored",
            "Figure 6 |",
            "Figure 13 |",
            "Figure 14 |",
            "Figure 15 |",
            "−0.799949",
            "−1.705760",
            "0.055994",
            "SeedSequence(20261125)",
            "SHA256 alternative",
            "+0.104723",
            "0.245175",
            "−0.008447",
            "Extended Data Figure 1 |",
            "Extended Data Figure 2 |",
            "Graphical abstract |",
            "Data and code availability",
            "References",
            "[1] J. Skalse",
            "[15] J. Chen",
            "10.1038/s41587-019-0164-5",
        ],
    },
    {
        "key": "supplement_s2",
        "source": "UNIFIED_PREPRINT_SUPPLEMENT_S2_V5_20261126.md",
        "output": "UNIFIED_PREPRINT_SUPPLEMENT_S2_V5_20261126.pdf",
        "figures": [
            ("Supplementary Figure S2.1 |", ["ExtendedData_VGSC_paired_difference.pdf"]),
            ("Supplementary Figure S2.2 |", ["Figure7_v2_paper2_baselines.pdf"]),
            ("Supplementary Figure S2.3 |", ["ExtendedData_proxy_noise.pdf"]),
        ],
        "required_source": [
            "retrospective sensitivity evidence, not blind evidence",
            "outer_scoring_authorized=false",
            "outer_scored=false",
            "PermissionError",
            "retrospective signal did not pass the frozen calibration gate",
        ],
        "required_pdf": [
            "Supplementary Study S2",
            "Scope and evidence status",
            "0.104026897",
            "PermissionError",
            "outer_scoring_authorized=false",
            "outer_scored=false",
            "Supplementary Figure S2.3 |",
            "S2.9 Verification",
        ],
    },
    {
        "key": "supplement_s3",
        "source": "UNIFIED_PREPRINT_SUPPLEMENT_S3_V5_20261126.md",
        "output": "UNIFIED_PREPRINT_SUPPLEMENT_S3_V5_20261126.pdf",
        "figures": [
            ("Supplementary Figure S3.1 |", ["Figure8_v2_paper3_headroom.pdf"]),
        ],
        "required_source": [
            "Permanent outer labels were not read and outer metrics were not computed",
            "not a new blind test",
            "do not establish label efficiency",
            "Numerical convergence does not remove representation/model misspecification",
        ],
        "required_pdf": [
            "Supplementary Study S3",
            "Scope and evidence status",
            "384/384",
            "Permanent outer labels were not read",
            "Supplementary Figure S3.1 |",
            "S3.11 Verification",
        ],
    },
]

FORBIDDEN_WORDING = [
    "remaining positive risk-control signal",
    "positive risk-control signal",
    "risk–coverage signal",
    "risk-coverage signal",
    "signal awaiting confirmation",
    "awaits a frozen-threshold test",
    "待outer确认",
    "剩余阳性风险信号",
    "There is no independent-study, same-endpoint MRL replication",
    "独立同终点复验仍缺失",
    "独立同终点复验尚缺失",
    "PERSIST supports BoN benefit",
    "PERSIST proves universal proxy failure",
]

HEADER = r"""
\special{pdf:minorversion 7}
\usepackage{unicode-math}
\setmathfont{STIX Two Math}
\usepackage{graphicx}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Anonymous authors}
\fancyhead[R]{\small Unified preprint v5}
\fancyfoot[C]{\thepage}
\setlength{\headheight}{14pt}
\setlength{\emergencystretch}{6em}
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.15}
\usepackage{etoolbox}
\usepackage{seqsplit}
\let\preprinttexttt\texttt
\DeclareRobustCommand{\preprintseqtexttt}[1]{\preprinttexttt{\seqsplit{#1}}}
\let\texttt\preprintseqtexttt
\AtBeginEnvironment{longtable}{\footnotesize}
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict:
    return {
        "path": display_path(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00ad", "")).strip()


def combine_figures(sources: list[Path], destination: Path) -> None:
    documents = [fitz.open(path) for path in sources]
    try:
        widths = []
        heights = []
        for document in documents:
            if len(document) != 1:
                raise RuntimeError(f"Expected one-page figure: {document.name}")
            rect = document[0].rect
            widths.append(rect.width)
            heights.append(rect.height)
        width = max(widths)
        scale = width / max(widths)
        scaled_heights = [height * width / source_width for height, source_width in zip(heights, widths)]
        gap = 12
        combined = fitz.open()
        page = combined.new_page(width=width, height=sum(scaled_heights) + gap * (len(sources) - 1))
        y = 0.0
        for document, height in zip(documents, scaled_heights):
            source_rect = document[0].rect
            placed_width = source_rect.width * scale
            x = (width - placed_width) / 2
            target = fitz.Rect(x, y, x + placed_width, y + height)
            page.show_pdf_page(target, document, 0)
            y += height + gap
        combined.set_metadata({})
        combined.save(destination, garbage=4, deflate=True, no_new_id=1)
        combined.close()
    finally:
        for document in documents:
            document.close()


def prepare_document(spec: dict, work: Path) -> tuple[Path, dict]:
    source = HERE / spec["source"]
    original = source.read_text(encoding="utf-8")
    replacements = {}
    copied = set()

    def replace_display_math(match: re.Match) -> str:
        source_block = match.group(0)
        rendered_block = f"$$\n{match.group(1)}$$"
        replacements[rendered_block] = source_block
        return rendered_block

    original = re.sub(r"(?ms)^\\\[\n(.*?)^\\\]$", replace_display_math, original)

    def replace_inline_math(match: re.Match) -> str:
        source_math = match.group(0)
        rendered_math = f"${match.group(1)}$"
        replacements[rendered_math] = source_math
        return rendered_math

    original = re.sub(r"\\\(([^\n]*?)\\\)", replace_inline_math, original)
    if "∈" in original:
        original = original.replace("∈", "$\\in$")
        replacements["$\\in$"] = "∈"
    for prefix, figure_names in spec["figures"]:
        matches = [line for line in original.splitlines() if line.startswith(f"**{prefix}")]
        if len(matches) != 1:
            raise RuntimeError(f"{source.name}: expected one caption beginning {prefix}, found {len(matches)}")
        caption = matches[0]
        for name in figure_names:
            if name not in caption:
                raise RuntimeError(f"{source.name}: {prefix} does not cite {name}")
            if name not in copied:
                source_figure = FIGURES / name
                if not source_figure.is_file():
                    raise RuntimeError(f"Missing figure: {source_figure}")
                shutil.copyfile(source_figure, work / Path(name).name)
                copied.add(name)
        if len(figure_names) == 1:
            rendered_name = Path(figure_names[0]).name
        else:
            rendered_name = "Figure4_endpoint_mismatch_composite.pdf"
            combine_figures([work / name for name in figure_names], work / rendered_name)
        rendered_line = f"![{caption}]({rendered_name}){{width=\\linewidth}}"
        if caption not in original:
            raise RuntimeError(f"Caption replacement failed for {prefix}")
        original = original.replace(caption, rendered_line, 1)
        replacements[rendered_line] = caption
    rendered = work / spec["source"]
    rendered.write_text(original, encoding="utf-8")
    return rendered, replacements


def build_documents(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    if output_dir != OUT and OUT not in output_dir.parents:
        raise ValueError("Build outputs must remain inside the configured V5 paper output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in DOCUMENTS:
        output = output_dir / spec["output"]
        if output.exists():
            output.unlink()
    build_records = {}
    with tempfile.TemporaryDirectory(prefix=".unified_preprint_v5_build_", dir=output_dir) as temporary:
        work = Path(temporary)
        header = work / "preprint_header.tex"
        header.write_text(HEADER, encoding="utf-8")
        for spec in DOCUMENTS:
            rendered, replacements = prepare_document(spec, work)
            restored = rendered.read_text(encoding="utf-8")
            for rendered_line, caption in replacements.items():
                restored = restored.replace(rendered_line, caption, 1)
            if restored != (HERE / spec["source"]).read_text(encoding="utf-8"):
                raise RuntimeError(f"Scientific content invariant failed for {spec['source']}")
            output = output_dir / spec["output"]
            tex_output = work / f"{spec['key']}.tex"
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
                "--toc",
                "--toc-depth",
                "3",
                "--include-in-header",
                str(header),
                "-V",
                f"geometry:margin={0.65 if spec['key'] == 'supplement_s3' else 0.73}in",
                "-V",
                "fontsize=10pt",
                "-V",
                "mainfont=Arial",
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
            environment = os.environ.copy()
            environment.update({"SOURCE_DATE_EPOCH": str(SOURCE_DATE_EPOCH), "TZ": "UTC"})
            pandoc = subprocess.run(
                command,
                cwd=work,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if pandoc.returncode:
                raise RuntimeError(f"Pandoc conversion failed for {spec['source']}:\n{pandoc.stdout + pandoc.stderr}")
            latex = tex_output.read_text(encoding="utf-8")
            latex = re.sub(
                r"\(\\columnwidth - (\d+)\\tabcolsep\)",
                lambda match: f"(\\columnwidth - {int(match.group(1)) + 3}\\tabcolsep)",
                latex,
            )
            tex_output.write_text(latex, encoding="utf-8")
            completed = subprocess.run(
                ["tectonic", "--outdir", str(work), str(tex_output)],
                cwd=work,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            log = pandoc.stdout + pandoc.stderr + completed.stdout + completed.stderr
            if completed.returncode:
                raise RuntimeError(f"Tectonic build failed for {spec['source']}:\n{log}")
            shutil.copyfile(work / f"{spec['key']}.pdf", output)
            warnings = [line for line in log.splitlines() if re.search(r"warning|overfull|undefined|missing character", line, re.I)]
            serious = [line for line in warnings if re.search(r"overfull|undefined|missing character", line, re.I)]
            if serious:
                raise RuntimeError(f"Serious layout warnings for {spec['output']}:\n" + "\n".join(serious))
            build_records[spec["key"]] = {
                "command": "pandoc <rendered-source> --standalone --to latex; tectonic <rendered-tex>",
                "pandoc_longtable_edge_padding_corrected": True,
                "warnings": warnings,
                "content_invariant": True,
            }
    return build_records


def round_one(source_hashes_before: dict, build_records: dict) -> dict:
    documents = {}
    all_pass = True
    combined = "\n".join((HERE / spec["source"]).read_text(encoding="utf-8") for spec in DOCUMENTS)
    normalized_combined = re.sub(r"\s+", "", combined).lower()
    forbidden = [phrase for phrase in FORBIDDEN_WORDING if re.sub(r"\s+", "", phrase).lower() in normalized_combined]
    for spec in DOCUMENTS:
        path = HERE / spec["source"]
        text = path.read_text(encoding="utf-8")
        missing = [token for token in spec["required_source"] if token not in text]
        figure_records = []
        for prefix, names in spec["figures"]:
            for name in names:
                figure = FIGURES / name
                figure_records.append({
                    "caption": prefix,
                    **artifact(figure),
                    "pdf_valid": fitz.open(figure).page_count == 1,
                })
        source_after = sha256(path)
        passed = not missing and source_after == source_hashes_before[spec["source"]] and build_records[spec["key"]]["content_invariant"]
        all_pass &= passed and all(record["pdf_valid"] for record in figure_records)
        documents[spec["key"]] = {
            "source": artifact(path),
            "missing_required_boundaries": missing,
            "source_hash_unchanged": source_after == source_hashes_before[spec["source"]],
            "rendered_content_restores_source_exactly": build_records[spec["key"]]["content_invariant"],
            "build_warnings": build_records[spec["key"]]["warnings"],
            "figures": figure_records,
            "pass": passed,
        }
    all_pass &= not forbidden
    return {
        "name": "Source Markdown hashes and claim boundaries",
        "status": "PASS" if all_pass else "FAIL",
        "source_scientific_content_modified": False,
        "forbidden_stale_positive_wording_found": forbidden,
        "dry_run_promoted_to_scientific_validity": False,
        "documents": documents,
    }


def font_audit(path: Path) -> dict:
    completed = subprocess.run(["pdffonts", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    rows = []
    for line in completed.stdout.splitlines()[2:]:
        match = re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line)
        if match:
            rows.append({"font": line.split()[0], "embedded": match.group(1) == "yes", "unicode": match.group(3) == "yes"})
    return {
        "fonts": rows,
        "all_embedded": bool(rows) and all(row["embedded"] for row in rows),
        "all_unicode": bool(rows) and all(row["unicode"] for row in rows),
    }


def validate_pdf(path: Path, spec: dict, full: bool = True) -> dict:
    with path.open("rb") as handle:
        header = handle.read(5)
        handle.seek(max(0, path.stat().st_size - 2048))
        trailer = handle.read()
    document = fitz.open(path)
    pages = []
    all_text = []
    outside_words = []
    outside_objects = []
    blank_pages = []
    xobject_count = 0
    for index, page in enumerate(document):
        text = page.get_text("text")
        all_text.append(text)
        rect = page.rect
        words = page.get_text("words")
        for word in words:
            box = fitz.Rect(word[:4])
            if box.x0 < rect.x0 - 0.5 or box.y0 < rect.y0 - 0.5 or box.x1 > rect.x1 + 0.5 or box.y1 > rect.y1 + 0.5:
                outside_words.append({"page": index + 1, "text": word[4], "bbox": list(box)})
        xobjects = page.get_xobjects()
        xobject_count += len(xobjects)
        object_boxes = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
        object_boxes.extend(drawing["rect"] for drawing in page.get_drawings())
        for box in object_boxes:
            if all(math.isfinite(value) for value in box) and not box.is_empty and (box.x0 < rect.x0 - 1 or box.y0 < rect.y0 - 1 or box.x1 > rect.x1 + 1 or box.y1 > rect.y1 + 1):
                outside_objects.append({"page": index + 1, "bbox": list(box)})
        page_record = {"page": index + 1, "characters": len(text.strip()), "words": len(words), "xobjects": len(xobjects)}
        if full:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY, alpha=False)
            pixels = np.frombuffer(pixmap.samples, dtype=np.uint8)
            standard_deviation = float(pixels.std())
            page_record["render_pixels"] = [pixmap.width, pixmap.height]
            page_record["render_std"] = round(standard_deviation, 3)
            if standard_deviation < 1 or not text.strip():
                blank_pages.append(index + 1)
        pages.append(page_record)
    text = normalize("\n".join(all_text))
    compact_text = re.sub(r"\s+", "", text)
    required_tokens = spec["required_pdf"] + [prefix for prefix, _ in spec["figures"]]
    missing = [token for token in required_tokens if normalize(token) not in text and re.sub(r"\s+", "", token) not in compact_text]
    replacement_glyphs = text.count("\ufffd")
    fonts = font_audit(path)
    pdfinfo = subprocess.run(["pdfinfo", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    expected_xobjects = len(spec["figures"])
    passed = (
        header == b"%PDF-"
        and b"%%EOF" in trailer
        and len(document) > 0
        and not missing
        and not outside_words
        and not outside_objects
        and not blank_pages
        and replacement_glyphs == 0
        and fonts["all_embedded"]
        and fonts["all_unicode"]
        and xobject_count >= expected_xobjects
        and pdfinfo.returncode == 0
    )
    result = {
        "path": display_path(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "pages": len(document),
        "pdfinfo_valid": pdfinfo.returncode == 0,
        "header_and_eof_valid": header == b"%PDF-" and b"%%EOF" in trailer,
        "missing_required_text": missing,
        "replacement_glyphs": replacement_glyphs,
        "words_outside_page": outside_words,
        "page_objects_outside_page": outside_objects,
        "blank_or_textless_pages": blank_pages,
        "xobject_count": xobject_count,
        "expected_figure_count": expected_xobjects,
        "font_audit": fonts,
        "page_records": pages,
        "pass": passed,
    }
    document.close()
    return result


def round_two() -> dict:
    documents = {spec["key"]: validate_pdf(OUT / spec["output"], spec) for spec in DOCUMENTS}
    return {
        "name": "PDF page text, overflow, rendering, and font checks",
        "status": "PASS" if all(record["pass"] for record in documents.values()) else "FAIL",
        "documents": documents,
    }


def round_three(initial_hashes: dict, source_hashes_before: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix=".unified_preprint_v5_rebuild_", dir=OUT) as temporary:
        rebuild_dir = Path(temporary)
        build_documents(rebuild_dir)
        outputs = {}
        all_match = True
        all_valid = True
        for spec in DOCUMENTS:
            rebuilt = rebuild_dir / spec["output"]
            rebuilt_hash = sha256(rebuilt)
            validation = validate_pdf(rebuilt, spec, full=False)
            matches = rebuilt_hash == initial_hashes[spec["key"]]
            all_match &= matches
            all_valid &= validation["pass"]
            outputs[spec["key"]] = {
                "initial_sha256": initial_hashes[spec["key"]],
                "clean_rebuild_sha256": rebuilt_hash,
                "byte_identical": matches,
                "rebuilt_pdf_valid": validation["pass"],
                "pages": validation["pages"],
            }
    sources_unchanged = all(sha256(HERE / name) == digest for name, digest in source_hashes_before.items())
    passed = all_match and all_valid and sources_unchanged
    return {
        "name": "Clean rebuild determinism and file validity",
        "status": "PASS" if passed else "FAIL",
        "clean_empty_output_directory": True,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "source_hashes_unchanged_after_rebuild": sources_unchanged,
        "outputs": outputs,
    }


def write_manifest(rounds: list[dict]) -> None:
    source_hashes = {spec["source"]: sha256(HERE / spec["source"]) for spec in DOCUMENTS}
    figure_paths = sorted({FIGURES / name for spec in DOCUMENTS for _, names in spec["figures"] for name in names})
    status = "PASS" if all(item["status"] == "PASS" for item in rounds) else "FAIL"
    manifest = {
        "release": "UNIFIED_PREPRINT_V5_20261126",
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Code-only V5 archive of equal-candidate-budget Random(N), sequence fixed-effect/family sensitivities, PERSIST source-confirmed tie audit, Figures 14/15, narrative boundaries, and deterministic PDF packaging. Scientific source text is unchanged; historical binaries and logs remain in the external evidence package.",
        "history_archive": "external evidence package; not duplicated in this code-only archive",
        "history_preserved": [],
        "source_markdown_sha256": source_hashes,
        "scientific_content_changed": False,
        "evidence_boundaries": {
            "paper1_equal_budget": "Retrospective historical-outer sensitivity, not a new blind test. T-cell r1 cap2-16 and r2 cap4 survive complete all-20 Holm including cap1; HepG2 cap2 also survives, cap4 Holm p=0.055994; N50 has no nominal support.",
            "paper1_sequence": "Retrospective/exploratory composition sensitivity only. GC/uAUG/polyU/support adjustment attenuates T-cell effects with no adjusted all-20 Holm survivor. uAUG exclusions have none; polyU exclusions retain exactly r1 cap4 and r2 cap8. No sequence causation or mediation claim.",
            "persist_mrl_external_v1": "Completed small independent-study, same-endpoint retrospective full-MLP-contract evaluation: 59 constructs/49 UTRs, exact/H1=0; +0.104723 [-0.100964,0.212706], p=0.245175, Spearman -0.008447. No detectable transfer; not blind, not direct T-cell refutation, not universal failure or BoN benefit.",
            "persist_tie_audit_v1_1": "Source confirms prediction descending/Sequence ID/row order. SHA256 alternative leaves original selections, primary effect, CI and p unchanged; local bootstrap changes and pre-existing manifest mismatch remain disclosed.",
            "paper2": "Frozen candidate-risk calibration failed; outer scoring was hard-locked and not performed. Historical outer curves are retrospective sensitivity evidence only; risk control requires an independent gate.",
            "paper3": "Development/calibration diagnostics only; permanent outer labels and metrics remain untouched. Label repair requires an independent gate; numerical convergence is not scientific or outer validation.",
        },
        "verification_rounds": rounds,
        "artifacts": [artifact(HERE / spec["source"]) for spec in DOCUMENTS]
        + [artifact(OUT / spec["output"]) for spec in DOCUMENTS]
        + [artifact(Path(__file__).resolve()), artifact(HERE / "verify_unified_preprint_v5_20261126.py")]
        + [artifact(REPO_ROOT / "docs" / name) for name in (
            "SCIENTIFIC_STORY_V5_20261126.md",
            "UNIFIED_PREPRINT_CLAIM_EVIDENCE_LEDGER_V5_20261126.md",
            "PERSIST_MRL_EXTERNAL_V1.md",
            "PAPER1_BUDGET_MATCHED_V1.md",
            "PAPER1_SEQUENCE_ABLATION_V1.md",
        )]
        + [artifact(ROOT / "deliverables/reports/docs" / f"UNIFIED_PREPRINT_SOURCE_VERIFICATION_V5_20261126_ROUND{index}.json") for index in (1, 2, 3)]
        + [artifact(ROOT / "deliverables/reports/source_data/paper1_budget_matched_v1/results.json")]
        + [artifact(ROOT / "deliverables/reports/source_data/paper1_sequence_ablation_v1/analysis/results/summary.json")]
        + [artifact(ROOT / "deliverables/reports/source_data/persist_mrl_external_v1/results/summary.json")]
        + [artifact(ROOT / "PERSIST_TIE_AUDIT_V1_1.md")]
        + [artifact(ROOT / "persist_mrl_external_v1/tie_audit_v1_1/results" / name) for name in ("summary_v1_1.json", "comparison_v1_vs_v1_1.json")]
        + [artifact(path) for path in figure_paths],
        "tools": {
            "python": sys.version.split()[0],
            "pandoc": subprocess.run(["pandoc", "--version"], text=True, stdout=subprocess.PIPE, check=True).stdout.splitlines()[0],
            "tectonic": subprocess.run(["tectonic", "--version"], text=True, stdout=subprocess.PIPE, check=True).stdout.strip(),
            "pymupdf": fitz.VersionBind,
            "numpy": np.__version__,
        },
        "old_release_manifest_overwritten": False,
        "remote_access_or_unblinding_performed": False,
    }
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    arguments = parser.parse_args()
    if arguments.build_only:
        build_documents(arguments.output_dir)
        return
    if arguments.output_dir.resolve() != OUT:
        raise ValueError("Full release verification requires the configured V5 paper output directory")
    source_hashes_before = {spec["source"]: sha256(HERE / spec["source"]) for spec in DOCUMENTS}
    build_records = build_documents(OUT)
    first = round_one(source_hashes_before, build_records)
    if first["status"] != "PASS":
        write_manifest([first])
        raise RuntimeError("Round 1 failed; see release manifest")
    second = round_two()
    if second["status"] != "PASS":
        write_manifest([first, second])
        raise RuntimeError("Round 2 failed; see release manifest")
    initial_hashes = {spec["key"]: sha256(OUT / spec["output"]) for spec in DOCUMENTS}
    third = round_three(initial_hashes, source_hashes_before)
    write_manifest([first, second, third])
    if third["status"] != "PASS":
        raise RuntimeError("Round 3 failed; see release manifest")
    print(json.dumps({
        "status": "PASS",
        "manifest": str(MANIFEST),
        "pdfs": {spec["key"]: str(OUT / spec["output"]) for spec in DOCUMENTS},
    }, indent=2))


if __name__ == "__main__":
    main()
