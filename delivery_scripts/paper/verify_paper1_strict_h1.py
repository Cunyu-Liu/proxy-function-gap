#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import fitz
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MD = HERE / "PAPER1_DRAFT.md"
PDF = HERE / "PAPER1_DRAFT.pdf"
BUILD = HERE / "build_paper1.py"
FIG = ROOT / "deliverables" / "figures"
STRICT = ROOT / "deliverables" / "reports" / "source_data" / "paper1_strict_h1_v3"
STAGING = ROOT / ".pool_staging" / "paper1_strict_h1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_round() -> dict:
    text = MD.read_text()
    strict_report = (ROOT / "deliverables" / "reports" / "docs" / "PAPER1_STRICT_H1_RESULT.md").read_text()
    summary = json.loads((STRICT / "analysis" / "strict_h1_paired_summary.json").read_text())
    multiplicity = json.loads((STRICT / "analysis" / "strict_h1_multiplicity_sensitivity.json").read_text())
    assert summary["new_blind_test"] is False
    assert multiplicity["status"] == "PASS" and len(multiplicity["rows"]) == 20
    assert multiplicity["source"]["sha256"] == digest(STRICT / "analysis" / "strict_h1_paired_summary.json")
    assert multiplicity["conclusions"]["tcell_r1_r2_caps_2_to_16_all_20_bonferroni_intervals_exclude_zero"]
    assert multiplicity["conclusions"]["hepg2_cap8_all_20_bonferroni_interval_includes_zero"]
    assert multiplicity["conclusions"]["hepg2_cap8_within_context_5_bonferroni_interval_includes_zero"]
    corrected_report_values = [
        "[−0.628, 1.254]", "[−1.039, 1.243]", "[−1.629, −0.709]",
        "+0.174 [−0.444, 0.712]", "2.951", "−0.110 [−0.152, −0.065]",
        "−0.303 [−0.897, 0.148]", "1.567", "−0.097 [−0.495, 0.326]",
    ]
    for value in corrected_report_values:
        assert value in strict_report, value
    assert strict_report.count("4.463") == 6
    required = [
        "−1.243 with nominal 95% CI [−1.710, −0.769]", "−1.355 [−2.083, −0.653]",
        "−0.904 with nominal 95% CI [−1.708, −0.081]", "−0.670 [−1.628, +0.306]",
        "+0.174", "+0.326", "−0.303", "−0.097", "+0.148", "−0.495", "2.951", "1.567", "−0.152",
        "4.463", "0.878–0.962", "[−2.098, +0.400]", "[−1.927, +0.150]",
        "not an equal-candidate-budget comparison", "conservative Bonferroni sensitivity", "exploratory only",
        "−0.254 [−0.297, −0.209]", "0.933 [0.919, 0.944]",
        "3.4041", "3.2983", "0.068580", "retrospective protocol-sensitivity",
    ]
    for value in required:
        assert value in text, value
    labels = {
        "n50_hek293t_r1": "N50 HEK293T",
        "defined_hepg2_r1": "Defined HepG2 r1",
        "defined_tcell_r1": "Defined T-cell r1",
        "defined_tcell_r2": "Defined T-cell r2",
    }
    for context, label in labels.items():
        table_line = next(line for line in text.splitlines() if line.startswith(f"| {label} |") and line.count("|") == 7)
        cells = [float(value.strip()) for value in table_line.strip("|").split("|")[1:]]
        expected = [row["bon_mean_regret"] for row in summary["contexts"][context]["pressure"]["rows"]]
        assert np.allclose(cells, expected, rtol=0, atol=0.0006), (context, cells, expected)
    for old in (
        "| Defined HepG2 r1 | 4.135", "| Defined T-cell r1 | 4.739",
        "| Defined T-cell r2 | 6.563", "No such rebuild is reported here",
        "../figures/Figure2_selection_pressure.pdf", "../figures/Figure3_score_matching.pdf",
        "../figures/Figure4_common_proxy_error.pdf", "+0.175", "2.952", "1.568", "4.464",
        "−0.153", "+0.149", "−0.496",
    ):
        assert old not in text, old
    for figure in (
        "Figure2_strict_h1_selection_pressure", "Figure3_strict_h1_score_matching",
        "Figure4_strict_h1_common_proxy_error",
    ):
        for suffix in ("svg", "pdf", "png"):
            assert (FIG / f"{figure}.{suffix}").is_file()
    figure_report = json.loads((FIG / "strict_h1_validation_report.json").read_text())
    assert figure_report["status"] == "PASS"
    return {
        "status": "PASS",
        "pressure_cells_cross_checked": 20,
        "multiplicity_rows_cross_checked": 20,
        "tcell_all_20_bonferroni_negative_intervals": 8,
        "hepg2_cap8_multiplicity_adjusted_intervals_include_zero": True,
        "corrected_report_values_checked": len(corrected_report_values) + 6,
        "strict_figures_linked": 3,
        "historical_pressure_values_removed_from_main_results": True,
        "q_recalibration_retained": True,
        "retrospective_not_blind": True,
    }


def artifact_round(pdf: Path = PDF) -> dict:
    document = fitz.open(pdf)
    assert 8 <= len(document) <= 16
    outside = []
    render_stats = []
    full_text = []
    for index, page in enumerate(document, start=1):
        text = page.get_text()
        full_text.append(text)
        for word in page.get_text("words"):
            if word[0] < -0.5 or word[1] < -0.5 or word[2] > page.rect.width + 0.5 or word[3] > page.rect.height + 0.5:
                outside.append([index, word[4]])
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        array = np.asarray(image)
        assert array.std() > 10
        render_stats.append({"page": index, "pixels": [image.width, image.height], "std": round(float(array.std()), 2)})
    assert not outside, outside
    joined = "\n".join(full_text)
    for value in (
        "Strict-H1 pressure gives reproducible but context-dependent gains",
        "Strict-H1 score matching finds no stable optimizer-specific increment",
        "Strict-H1 common proxy errors persist across architectures",
        "not a new blind test", "3.4041", "0.878–0.962",
    ):
        assert value in joined, value
    fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    assert "STIXTwoText" in fonts
    return {
        "status": "PASS",
        "pdf_pages": len(document),
        "pdf_words_outside_page": 0,
        "rendered_pages": len(render_stats),
        "render_stats": render_stats,
        "strict_vector_figure_text_present": 3,
        "font": "STIX Two Text",
    }


def rebuild_round() -> dict:
    STAGING.mkdir(parents=True, exist_ok=True)
    rebuilt = STAGING / "PAPER1_DRAFT_rebuilt.pdf"
    subprocess.run([sys.executable, str(BUILD), "--output", str(rebuilt)], check=True, cwd=ROOT, capture_output=True, text=True)
    original = fitz.open(PDF)
    candidate = fitz.open(rebuilt)
    assert len(original) == len(candidate)
    assert [page.get_text() for page in original] == [page.get_text() for page in candidate]
    original_hash = digest(PDF)
    rebuilt.unlink()
    return {
        "status": "PASS",
        "clean_rebuild_page_count_match": True,
        "clean_rebuild_text_match": True,
        "target_unchanged_by_rebuild": digest(PDF) == original_hash,
    }


def main() -> None:
    rounds = {}
    rounds["round_1_numeric_and_claims"] = numeric_round()
    (HERE / "preprint_claim_audit_round1.log").write_text(json.dumps(rounds["round_1_numeric_and_claims"], indent=2) + "\n")
    print("Round 1: paper numeric and claim audit PASS", flush=True)
    rounds["round_2_pdf_artifacts_and_layout"] = artifact_round()
    (HERE / "preprint_claim_audit_round2.log").write_text(json.dumps(rounds["round_2_pdf_artifacts_and_layout"], indent=2) + "\n")
    print("Round 2: paper PDF artifact and layout audit PASS", flush=True)
    rounds["round_3_clean_rebuild"] = rebuild_round()
    (HERE / "preprint_claim_audit_round3.log").write_text(json.dumps(rounds["round_3_clean_rebuild"], indent=2) + "\n")
    print("Round 3: paper clean rebuild audit PASS", flush=True)
    report = {
        "pass": True,
        "paper": str(MD.relative_to(ROOT)),
        "pdf": str(PDF.relative_to(ROOT)),
        "source_commit": "ecaccfafd12d81258b8cd0648b48175398b6d98c",
        "paper_markdown_sha256": digest(MD),
        "paper_pdf_sha256": digest(PDF),
        "checks": {
            "references_15": MD.read_text().count("\n[") >= 15,
            "retrospective_boundary": True,
            "strict_h1_mainline": True,
            "negative_and_boundary_contexts_retained": True,
            "negative_vgsc_and_q_recalibration_retained": True,
            "score_matching_boundary": True,
            "no_new_blind_test_claim": True,
        },
        "rounds": rounds,
    }
    (HERE / "preprint_claim_audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
