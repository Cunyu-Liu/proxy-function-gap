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
SOURCE = PAPER / "PAPER1_TAIL_RELIABILITY_V8_20261128.md"
PDF = SOURCE.with_suffix(".pdf")
BUILD = PAPER / "build_paper1_tail_reliability_v8.py"


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
        for token in ["tail reliability", "−1.705760", "−0.160894", "−0.2800", "−4.206431", "+0.2838", "Amp→cefotaxime", "not tropism", "retrospective", "RandomN"]:
            assert token in text, token
        assert "does not pool raw effect sizes" in text
        assert "not a causal intervention proof" in text
        refs = re.findall(r"^\[\d+\]", text, flags=re.M)
        assert len(refs) >= 15
    check("v8_claims", claims)

    def figures():
        for name in ["Figure14_v3_equal_budget_value.pdf", "Figure16_v4_tcell_global_value_tail_bottleneck.pdf", "Figure17_v4_tcell_regression_decision_decoupling.pdf", "Figure18_v5_cross_scenario_functional_value.pdf", "Figure19_v5_regression_decision_dissociation.pdf", "Figure20_v6_decision_value_map.pdf", "Figure21_v6_unified_protocol_governance.pdf"]:
            assert name in text
            assert any((ROOT / "deliverables" / "figures").rglob(name))
    check("v8_figures", figures)

    import fitz
    def pdf_check():
        doc = fitz.open(PDF)
        assert len(doc) >= 10
        full = "\n".join(page.get_text() for page in doc)
        assert "tail reliability" in full and "GB1" in full and "AAV9" in full
        assert "\ufffd" not in full
        for page in doc:
            assert page.get_text().strip()
        doc.close()
    check("v8_pdf", pdf_check)

    def clean():
        with tempfile.TemporaryDirectory(prefix=".v8_verify_", dir=PAPER) as tmp:
            subprocess.run([sys.executable, str(BUILD), "--output-dir", tmp], check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert sha(Path(tmp) / PDF.name) == sha(PDF)
    check("v8_clean_rebuild", clean)

    payload = {"round": args.round, "status": "PASS", "checks": checks, "source_sha256": sha(SOURCE), "pdf_sha256": sha(PDF)}
    out = ROOT / "deliverables" / "reports" / "docs" / f"PAPER1_TAIL_RELIABILITY_V8_VERIFICATION_ROUND{args.round}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"round": args.round, "status": "PASS", "checks": len(checks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()