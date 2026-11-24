#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DELIVERABLES = ROOT / "deliverables"
REMOTE_ROOT = "/teams/HoulabData_1788502323/liucunyu/proxy_function_gap"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-commit", default=None)
    parser.add_argument("--github-upload", default="not_claimed_by_this_update")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        manifest = json.loads((HERE / "release_manifest.json").read_text())
        for record in manifest["artifacts"]:
            path = ROOT / record["local_path"]
            assert path.stat().st_size == record["bytes"], record["local_path"]
            assert digest(path) == record["sha256"], record["local_path"]
        assert manifest["paper_markdown_sha256"] == digest(HERE / "PAPER1_DRAFT.md")
        print(json.dumps({"status": "PASS", "artifacts": len(manifest["artifacts"]), "hashes": "SHA-256 and byte sizes match"}))
        return
    figure_report = json.loads((DELIVERABLES / "figures" / "strict_h1_validation_report.json").read_text())
    paper_report = json.loads((HERE / "preprint_claim_audit.json").read_text())
    ppt_report = json.loads((DELIVERABLES / "ppt" / "proxy_function_gap_scientific_update_20261124_verification.json").read_text())
    assert figure_report["status"] == "PASS" and paper_report["pass"] is True and ppt_report["status"] == "PASS"
    artifacts = [
        "deliverables/paper/PAPER1_DRAFT.md",
        "deliverables/paper/PAPER1_DRAFT.pdf",
        "deliverables/paper/preprint_claim_audit.json",
        "deliverables/reports/source_data/paper1_strict_h1_v3/analysis/strict_h1_multiplicity_sensitivity.json",
        "deliverables/reports/source_data/paper1_strict_h1_v3/analysis/compute_strict_h1_multiplicity_sensitivity.py",
        "deliverables/ppt/proxy_function_gap_scientific_update_20261124.pptx",
        "deliverables/ppt/proxy_function_gap_scientific_update_20261124.pdf",
        "deliverables/ppt/proxy_function_gap_scientific_update_20261124_verification.json",
        "deliverables/reports/docs/PAPER1_STRICT_H1_RESULT.md",
        "deliverables/reports/q_actual_cost_recalibration_20261124.md",
        "deliverables/reports/paper3_e0_repair_headroom_20261124.md",
        "deliverables/figures/strict_h1_validation_report.json",
        "deliverables/figures/source_data/strict_h1_source_manifest.json",
        "deliverables/ppt/proxy_function_gap_vgsc_pilot_20261124.pptx",
        "deliverables/ppt/proxy_function_gap_paper1_external_20261124.pptx",
        "deliverables/ppt/proxy_function_gap_paper1_mechanism_20261124.pptx",
        "deliverables/ppt/proxy_function_gap_cross_arch_20261124.pptx",
        "deliverables/ppt/proxy_function_gap_sensitivity_noise_20261124.pptx",
    ]
    for base in (
        "Figure2_strict_h1_selection_pressure",
        "Figure3_strict_h1_score_matching",
        "Figure4_strict_h1_common_proxy_error",
    ):
        artifacts.extend(f"deliverables/figures/{base}.{suffix}" for suffix in ("svg", "pdf", "png"))
    records = []
    for relative in artifacts:
        path = ROOT / relative
        assert path.is_file(), relative
        remote_path = relative.replace("deliverables/", f"{REMOTE_ROOT}/", 1)
        remote_path = remote_path.replace("/reports/docs/", "/reports/")
        remote_path = remote_path.replace("/reports/source_data/paper1_strict_h1_v3/", "/paper1_strict_h1_v3/")
        records.append({
            "path": remote_path,
            "local_path": relative,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        })
    manifest = {
        "code_commit": "ecaccfafd12d81258b8cd0648b48175398b6d98c",
        "release_commit": args.release_commit,
        "related_source_commits": {
            "paper1_strict_h1": "ecaccfafd12d81258b8cd0648b48175398b6d98c",
            "q_actual_cost": [
                "52ce8ef151eb4b9420fd54bea5a33b3713dc0607",
                "096d4daade3c494f57ba4fffe22f2c668b6fedc5",
                "cdc95523486430059bead212663a3146b3c6d103",
                "e2d18e1b8d7bdc9a7d0c3e957a902e91b87c4ba7",
            ],
            "paper3_e0_repair_headroom": "dde960959d8197c5b1ccf7a143289afeb834f68b",
        },
        "paper_markdown_sha256": digest(HERE / "PAPER1_DRAFT.md"),
        "evidence_boundary": "Retrospective strict-H1 protocol sensitivity; not a new blind test.",
        "statistical_boundary": {
            "intervals": "Nominal 95% pool-cluster bootstrap CIs; Bonferroni intervals are conservative sensitivity, not new confirmatory analysis.",
            "tcell": "All eight T-cell r1/r2 cap 2–16 intervals exclude zero under all-20 Bonferroni sensitivity.",
            "hepg2_cap8": "Both all-20 and within-context-5 Bonferroni intervals include zero; exploratory only.",
            "candidate_budget": "Random is always a single-candidate baseline; BoN cap varies, so the pressure curve is not an equal-candidate-budget comparison.",
        },
        "validation": {
            "strict_figures": "PASS: numeric and multiplicity, artifacts/layout, deterministic rerun",
            "paper": "PASS: numeric/claims, PDF layout, clean rebuild",
            "ppt": "PASS: numeric/boundary, embedded images/PDF layout, clean rebuild and old-deck preservation",
            "rounds": 3,
        },
        "artifacts": records,
        "review": "passed",
        "github_upload": args.github_upload,
    }
    (HERE / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"artifacts": len(records), "paper_markdown_sha256": manifest["paper_markdown_sha256"], "review": manifest["review"]}))


if __name__ == "__main__":
    main()
