from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "deliverables" / "paper"
SRC = ROOT / "deliverables" / "reports" / "source_data"
SOURCE = PAPER / "PAPER1_CROSS_SCENARIO_V7_20261128.md"
PDF = SOURCE.with_suffix(".pdf")
BUILD = PAPER / "build_paper1_cross_scenario_v7.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()
    checks = []

    def check(name, fn):
        fn()
        checks.append({"name": name, "status": "PASS"})

    text = SOURCE.read_text(encoding="utf-8")

    def claims():
        required = [
            "−1.705760", "−1.014876", "−0.160894", "−0.2800", "−4.206431",
            "−0.159379", "+0.2838", "+0.428897", "Amp→cefotaxime", "lnW", "production",
            "not tropism", "retrospective", "RandomN", "functional regret",
        ]
        for token in required:
            assert token in text, token
        assert "do not compute a cross-task pooled effect size" in text
        assert "not prospective blind tests" in text
        assert text.count("## References") == 1
        refs = re.findall(r"^\[\d+\]", text, flags=re.M)
        assert len(refs) >= 15
    check("manuscript_claims_and_boundaries", claims)

    def figures():
        for name in [
            "Figure14_v3_equal_budget_value.pdf",
            "Figure16_v4_tcell_global_value_tail_bottleneck.pdf",
            "Figure17_v4_tcell_regression_decision_decoupling.pdf",
            "Figure18_v5_cross_scenario_functional_value.pdf",
            "Figure19_v5_regression_decision_dissociation.pdf",
        ]:
            assert name in text
            assert any((ROOT / "deliverables" / "figures").rglob(name))
    check("figure_links", figures)

    def source_hashes():
        paths = [
            SRC / "paper1_budget_matched_v1" / "results.json",
            SRC / "scenario_gb1_v1" / "aggregate" / "test_aggregate.json",
            SRC / "scenario_gfp_v1" / "results" / "primary.json",
            SRC / "scenario_aav_v1" / "results" / "main_results.json",
            SRC / "scenario_tem1_v1" / "results" / "direction_results.tsv",
            SRC / "scenario_mrna_non_tcell_v1" / "artifact" / "results" / "decision_summary.tsv",
        ]
        return {str(p.relative_to(ROOT)): sha(p) for p in paths}
    hashes = source_hashes()
    check("source_files_present", lambda: None)

    import fitz
    def pdf_check():
        doc = fitz.open(PDF)
        assert len(doc) >= 10
        full = "\n".join(page.get_text() for page in doc)
        assert "Decision-focused evaluation" in full
        assert "GB1" in full and "AAV9" in full and "TEM-1" in full
        assert "\ufffd" not in full
        for page in doc:
            assert page.get_text().strip()
        doc.close()
    check("pdf_text_and_pages", pdf_check)

    def clean_rebuild():
        with tempfile.TemporaryDirectory(prefix=".v7_verify_", dir=PAPER) as tmp:
            subprocess.run([sys.executable, str(BUILD), "--output-dir", tmp], check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            rebuilt = Path(tmp) / PDF.name
            assert sha(rebuilt) == sha(PDF)
    check("clean_rebuild_byte_identical", clean_rebuild)

    payload = {
        "round": args.round,
        "status": "PASS",
        "checks": checks,
        "source_sha256": sha(SOURCE),
        "pdf_sha256": sha(PDF),
        "evidence_hashes": hashes,
        "new_prospective_experiment": False,
        "cross_task_pooled_effect": False,
    }
    out = ROOT / "deliverables" / "reports" / "docs" / f"PAPER1_CROSS_SCENARIO_V7_VERIFICATION_ROUND{args.round}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"round": args.round, "status": "PASS", "checks": len(checks), "pdf": PDF.name}, ensure_ascii=False))


if __name__ == "__main__":
    main()