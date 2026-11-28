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
SOURCE = PAPER / "PAPER1_TAIL_RELIABILITY_V10_20261128.md"
PDF = SOURCE.with_suffix(".pdf")
BUILD = PAPER / "build_paper1_tail_reliability_v10.py"


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
        for token in ["tail reliability", "−1.705760", "−0.160894", "−0.2800", "−4.206431",
                      "+0.2838", "Amp→cefotaxime", "not tropism", "retrospective", "RandomN",
                      "UPTS", "0.029558", "auxiliary gate", "DRAFT_NOT_SUBMITTED",
                      "0.8788", "0.00076", "0.8333", "0.7697"]:
            assert token in text, token
        assert "no cross-task pooled raw effect size" in text
        assert "not a causal intervention proof" in text
        assert "not repairable by reweighting" in text
        assert "touching zero" in text or "touched zero" in text
        for q in ["Does proxy ranking improve", "Where does the remaining loss",
                  "Does repairing regression", "Does repairing the selection rule",
                  "when does proxy ranking have value", "What single diagnostic",
                  "Does the pattern extend"]:
            assert q in text, q
        refs = re.findall(r"^\[\d+\]", text, flags=re.M)
        assert len(refs) >= 18, len(refs)
    check("v10_claims", claims)

    def figures():
        for name in ["Figure1_v8_equal_budget_value.pdf", "Figure2_v8_tail_bottleneck.pdf",
                     "Figure3_v8_regression_decision_dissociation.pdf", "Figure4_v8_upts_falsification.pdf",
                     "Figure5_v8_cross_scenario_forest.pdf", "Figure6_v8_unified_tail_diagnostic.pdf",
                     "ED_Figure1_v8_label_distributions.pdf"]:
            assert name in text, name
            assert any((ROOT / "deliverables" / "figures").rglob(name)), name
    check("v10_figures", figures)

    import fitz

    def pdf_check():
        doc = fitz.open(PDF)
        assert len(doc) >= 9
        full = "\n".join(page.get_text() for page in doc)
        for token in ["tail reliability", "GB1", "AAV9", "UPTS", "0.8788"]:
            assert token in full, token
        assert "\ufffd" not in full
        for page in doc:
            assert page.get_text().strip()
        doc.close()
    check("v10_pdf", pdf_check)

    def clean():
        with tempfile.TemporaryDirectory(prefix=".v10_verify_", dir=PAPER) as tmp:
            subprocess.run([sys.executable, str(BUILD), "--output-dir", tmp], check=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert sha(Path(tmp) / PDF.name) == sha(PDF)
    check("v10_clean_rebuild", clean)

    payload = {"round": args.round, "status": "PASS", "checks": checks, "source_sha256": sha(SOURCE), "pdf_sha256": sha(PDF)}
    out = ROOT / "deliverables" / "reports" / "docs" / f"PAPER1_TAIL_RELIABILITY_V10_VERIFICATION_ROUND{args.round}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"round": args.round, "status": "PASS", "checks": len(checks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()