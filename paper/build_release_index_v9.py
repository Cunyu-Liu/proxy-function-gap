#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DELIV = ROOT / "deliverables"
INDEX = HERE / "release_index_v9.json"
PAPER_MD = DELIV / "paper" / "PAPER1_TAIL_RELIABILITY_V9_20261128.md"

FIGURES = [
    "v3/Figure14_v3_equal_budget_value",
    "v4/Figure16_v4_tcell_global_value_tail_bottleneck",
    "v4/Figure17_v4_tcell_regression_decision_decoupling",
    "v5/Figure18_v5_cross_scenario_functional_value",
    "v5/Figure19_v5_regression_decision_dissociation",
    "v6/Figure20_v6_decision_value_map",
    "v6/Figure21_v6_unified_protocol_governance",
    "v7/Figure22_v7_upts_auxiliary_gate_falsification",
]
REPORTS = [
    "P0_STATUS_REPORT_20261128.md",
    "P0_REMOTE_ARTIFACT_AUDIT_20261128.md",
    "P1_MECHANISM_PROTOCOL_DRAFT_20261128.md",
    "TCELL_UPTS_V1_20261128.md",
    "P3_INDEPENDENT_DATA_SEARCH_20261128.md",
    "P4_BASELINE_OOD_AUDIT_20261128.md",
    "CROSS_SCENARIO_EVIDENCE_SYNTHESIS_20261128.md",
    "PAPER1_SELF_REFLECTION_AND_PUBLICATION_PLAN_20261128.md",
    "verify_paper1_tail_reliability_v9.py",
]
MANUAL_ITEMS = [
    "作者署名与顺序",
    "单位/通讯作者",
    "利益冲突声明",
    "数据与代码许可（含各公开数据集原始许可）",
    "目标期刊/仓库与投稿格式",
    "是否需要公开数据/代码 DOI（Zenodo）",
    "参考文献 [17] T3PO-mRNA 作者/年份补核（审计时 OpenReview 403）",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_paths():
    paths = [
        "deliverables/paper/PAPER1_TAIL_RELIABILITY_V9_20261128.md",
        "deliverables/paper/PAPER1_TAIL_RELIABILITY_V9_20261128.pdf",
        "deliverables/paper/build_paper1_tail_reliability_v9.py",
    ]
    for r in (1, 2, 3):
        paths.append(f"deliverables/reports/docs/PAPER1_TAIL_RELIABILITY_V9_VERIFICATION_ROUND{r}.json")
        paths.append(f"deliverables/reports/docs/CROSS_SCENARIO_SYNTHESIS_VERIFICATION_20261128_ROUND{r}.json")
    for fig in FIGURES:
        for ext in ("pdf", "png", "svg"):
            paths.append(f"deliverables/figures/{fig}.{ext}")
    for rep in REPORTS:
        paths.append(f"deliverables/reports/docs/{rep}")
    paths += [
        "deliverables/reports/source_data/cross_scenario_tail_audit_v1/tail_audit.json",
        "deliverables/reports/source_data/cross_scenario_tail_audit_v1/tail_audit.tsv",
        "deliverables/reports/source_data/tcell_upts_v1/artifact/input_hashes.json",
        "deliverables/reports/source_data/tcell_upts_v1/artifact/frozen_thresholds.json",
        "deliverables/reports/source_data/tcell_upts_v1/artifact/auxiliary_gate.json",
        "deliverables/reports/source_data/tcell_upts_v1/artifact/verification/round2_independent.json",
        "deliverables/reports/source_data/tcell_upts_v1/artifact/verification/round3_delivery.json",
    ]
    for d in ("n50_hek293t_r1_to_defined_hepg2_r1", "defined_hepg2_r1_to_n50_hek293t_r1"):
        paths.append(f"deliverables/reports/source_data/tcell_upts_v1/artifact/auxiliary/{d}/selection.json")
    for code in ("configs/upts_frozen.json", "src/upts.py", "scripts/freeze_inputs.py", "scripts/run_auxiliary.py", "scripts/verify_round2.py", "tests/test_upts.py"):
        paths.append(f"deliverables/reports/source_data/tcell_upts_v1/code/{code}")
    paths += [
        "deliverables/ppt/proxy_function_gap_paper1_v9_20261128.pptx",
        "deliverables/ppt/build_paper1_v9_deck.py",
    ]
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-commit", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--round", type=int, default=None)
    args = parser.parse_args()

    if args.check:
        manifest = json.loads(INDEX.read_text())
        for record in manifest["artifacts"]:
            path = ROOT / record["local_path"]
            assert path.stat().st_size == record["bytes"], record["local_path"]
            assert digest(path) == record["sha256"], record["local_path"]
        assert manifest["paper_markdown_sha256"] == digest(PAPER_MD)
        print(json.dumps({"status": "PASS", "round": args.round, "artifacts": len(manifest["artifacts"]), "hashes": "SHA-256 and byte sizes match"}))
        return

    artifacts = []
    for rel in artifact_paths():
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing release artifact: {rel}")
        artifacts.append({"local_path": rel, "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest = {
        "release": "Paper 1 tail-reliability V9",
        "release_commit": args.release_commit,
        "github_upload": "pushed to origin/paper23-gates-v1" if args.release_commit else "not_claimed",
        "paper_markdown_sha256": digest(PAPER_MD),
        "evidence_boundary": "All retrospective historical-data studies; grouped holdouts are not prospective blind tests. UPTS v1 failed its auxiliary gate; T-cell target not evaluated. DRAFT_NOT_SUBMITTED.",
        "statistical_boundary": "No cross-task pooled raw effect; endpoint-specific regret; cluster bootstrap; algorithm seeds are not biological replicates.",
        "ppt_status": "updated_to_V9_20261128",
        "manual_items_before_submission": MANUAL_ITEMS,
        "artifacts": artifacts,
    }
    INDEX.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "BUILT", "artifacts": len(artifacts), "release_commit": args.release_commit, "index": str(INDEX.relative_to(ROOT))}))


if __name__ == "__main__":
    main()