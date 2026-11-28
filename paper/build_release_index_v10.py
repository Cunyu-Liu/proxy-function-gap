#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DELIV = ROOT / "deliverables"
INDEX = HERE / "release_index_v10.json"
PAPER_MD = DELIV / "paper" / "PAPER1_TAIL_RELIABILITY_V10_20261128.md"

FIGURES = [
    "v8/Figure1_v8_equal_budget_value",
    "v8/Figure2_v8_tail_bottleneck",
    "v8/Figure3_v8_regression_decision_dissociation",
    "v8/Figure4_v8_upts_falsification",
    "v8/Figure5_v8_cross_scenario_forest",
    "v8/Figure6_v8_unified_tail_diagnostic",
    "v8/ED_Figure1_v8_label_distributions",
]
REPORTS = [
    "V10_P0_RECHECK_20261128.md",
    "CROSS_SCENARIO_TAIL_ASSOCIATION_20261128.md",
    "DATA_BASELINE_OOD_AUDIT_V10_20261128.md",
    "SCENARIO_CAO2021_V1_20261128.md",
    "CROSS_SCENARIO_EVIDENCE_SYNTHESIS_20261128.md",
    "P4_BASELINE_OOD_AUDIT_20261128.md",
    "TCELL_UPTS_V1_20261128.md",
    "verify_paper1_tail_reliability_v10.py",
]
MANUAL_ITEMS = [
    "作者署名与顺序",
    "单位/通讯作者",
    "利益冲突声明",
    "数据与代码许可（含各公开数据集原始许可）",
    "目标期刊/仓库与投稿格式",
    "是否需要公开数据/代码 DOI（Zenodo）",
    "参考文献 [18] T3PO-mRNA 作者/年份补核（审计时 OpenReview 403）",
    "独立数据评价（Cao 内源 TE / Ogden AAV2 tropism）需另立冻结协议与可行数据通道",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_paths():
    paths = [
        "deliverables/paper/PAPER1_TAIL_RELIABILITY_V10_20261128.md",
        "deliverables/paper/PAPER1_TAIL_RELIABILITY_V10_20261128.pdf",
        "deliverables/paper/build_paper1_tail_reliability_v10.py",
    ]
    for r in (1, 2, 3):
        paths.append(f"deliverables/reports/docs/PAPER1_TAIL_RELIABILITY_V10_VERIFICATION_ROUND{r}.json")
    for fig in FIGURES:
        for ext in ("pdf", "png", "svg"):
            paths.append(f"deliverables/figures/{fig}.{ext}")
    paths += [
        "deliverables/figures/v8/figure_style_v8.py",
        "deliverables/figures/v8/prep_v8_data.py",
        "deliverables/figures/v8/build_v8_figures.py",
        "deliverables/figures/v8/build_v8_extended.py",
        "deliverables/figures/v8/verify_v8_figures.py",
        "deliverables/figures/v8/figures_v8_data.json",
        "deliverables/figures/v8/v8_build_config.json",
    ]
    for r in (1, 2, 3):
        paths.append(f"deliverables/figures/v8/v8_verification_round{r}.json")
    for rep in REPORTS:
        paths.append(f"deliverables/reports/docs/{rep}")
    paths += [
        "deliverables/reports/source_data/cross_scenario_tail_audit_v2/tail_audit_v2.json",
        "deliverables/reports/source_data/cross_scenario_tail_audit_v2/tail_audit_v2.tsv",
        "deliverables/reports/source_data/cross_scenario_tail_audit_v2/input_hashes.json",
        "deliverables/reports/source_data/cross_scenario_tail_audit_v2/association_protocol_frozen.json",
        "deliverables/reports/source_data/cross_scenario_tail_audit_v2/association_results.json",
        "deliverables/reports/source_data/cross_scenario_tail_audit_v2/run_association.py",
        "deliverables/reports/source_data/cross_scenario_tail_audit_v2/verify_association.py",
        "deliverables/reports/source_data/cross_scenario_tail_audit_v2/label_distributions.json",
    ]
    for r in (1, 2, 3):
        paths.append(f"deliverables/reports/source_data/cross_scenario_tail_audit_v2/association_verification_round{r}.json")
    paths += [
        "deliverables/ppt/proxy_function_gap_paper1_v10_20261128.pptx",
        "deliverables/ppt/build_paper1_v10_deck.py",
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
        "release": "Paper 1 tail-reliability V10",
        "release_commit": args.release_commit,
        "github_upload": "pushed to origin/paper23-gates-v1" if args.release_commit else "not_claimed",
        "paper_markdown_sha256": digest(PAPER_MD),
        "evidence_boundary": "All retrospective historical-data studies; grouped holdouts are not prospective blind tests. UPTS v1 failed its auxiliary gate; T-cell target not evaluated. Independent candidates (Cao endogenous TE, Ogden AAV2 tropism) qualified but access-barred and not evaluated. DRAFT_NOT_SUBMITTED.",
        "statistical_boundary": "No cross-task pooled raw effect; endpoint-specific regret; cluster bootstrap; cross-scenario tail association is associational (n=10), difference CI touches zero; algorithm seeds are not biological replicates.",
        "new_in_v10": "Seven-question narrative chain; unified top-decile tail-recall metric + pre-registered association (rho=0.879, p=0.00076); v8 journal-style figure set; data/baseline/OOD consolidated audit; Cao endpoint correction.",
        "ppt_status": "updated_to_V10_20261128",
        "manual_items_before_submission": MANUAL_ITEMS,
        "artifacts": artifacts,
    }
    INDEX.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "BUILT", "artifacts": len(artifacts), "release_commit": args.release_commit, "index": str(INDEX.relative_to(ROOT))}))


if __name__ == "__main__":
    main()