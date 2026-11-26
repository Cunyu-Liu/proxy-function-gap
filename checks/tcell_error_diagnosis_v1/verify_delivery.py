#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap")
OUT = BASE / "tcell_error_diagnosis_v1"
CODE = Path("/home/proxy-function-gap-tcell-diagnosis")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    checks = []
    def require(condition, name):
        assert condition, name
        checks.append(name)

    protocol = json.loads((CODE / "frozen_protocol.json").read_text())
    require(protocol["analysis_id"] == "tcell_error_diagnosis_v1", "protocol_identity")
    require("not a new blind test" in protocol["evidence_boundary"], "protocol_retrospective_boundary")
    raw = json.loads((OUT / "raw_audit/raw_audit.json").read_text())
    require(raw["status"] == "PASS" and raw["original_cnn_field_checks"] == 6400, "raw_status")
    for record in raw["input_files"]:
        path = Path(record["path"])
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], f"input:{path.name}")
    independent = json.loads((OUT / "verification/round2_independent.json").read_text())
    require(independent["status"] == "PASS" and independent["numeric_comparisons"] == 530, "independent_status")
    require(independent["maximum_absolute_difference"] < 2e-10, "independent_precision")

    tables = {}
    for path in sorted((OUT / "analysis").glob("*.tsv")):
        original = pd.read_csv(path, sep="\t")
        exported = pd.read_csv(OUT / "export" / path.name, sep="\t")
        pd.testing.assert_frame_equal(original.drop(columns=["pool_id"], errors="ignore"), exported,
                                      check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
        tables[path.name] = original
        require(not {"sequence_id", "utr50", "selected_id", "selected_index", "y", "pool_id"}.intersection(exported.columns),
                f"deidentified:{path.name}")
    require(len(tables) == 9, "aggregate_table_count")

    point = tables["pointwise_metrics.tsv"]
    require(len(point) == 4 and np.all(point.n == 1024), "pointwise_counts")
    require(np.max(np.abs(point.mse - point.bias_squared - point.centered_mse)) < 1e-10, "mse_decomposition")
    require(np.all(point.bias > 0), "systematic_bias_retained")
    tail = tables["top_tail_summary.tsv"]
    require(len(tail) == 20, "tail_counts")
    require(np.max(np.abs(tail.regret - tail.opportunity_loss - tail.within_tail_ranking_loss)) < 1e-10,
            "tail_decomposition")
    selection = tables["selection_summary.tsv"]
    require(len(selection) == 60, "selection_counts")
    effects = tables["selection_effects.tsv"]
    require(len(effects) == 70, "effect_counts")
    for _, row in effects.iterrows():
        if row.contrast == "mlp_minus_cnn_bon":
            left = selection[(selection.context == row.context) & (selection.model == "mlp")
                             & (selection.cap == row.cap) & (selection.strategy == "bon")].regret.iloc[0]
            right = selection[(selection.context == row.context) & (selection.model == "cnn")
                              & (selection.cap == row.cap) & (selection.strategy == "bon")].regret.iloc[0]
        else:
            left_name, right_name = {"bon_minus_random_n": ("bon", "random_n"),
                                     "bon_minus_random_1": ("bon", "random_1"),
                                     "random_n_minus_random_1": ("random_n", "random_1")}[row.contrast]
            left = selection[(selection.context == row.context) & (selection.model == row.model)
                             & (selection.cap == row.cap) & (selection.strategy == left_name)].regret.iloc[0]
            right = selection[(selection.context == row.context) & (selection.model == row.model)
                              & (selection.cap == row.cap) & (selection.strategy == right_name)].regret.iloc[0]
        require(abs(row.effect - (left - right)) < 1e-10,
                f"effect_arithmetic:{row.context}:{row.model}:{row.cap}:{row.contrast}")
    random_n = effects[(effects.contrast == "random_n_minus_random_1") & (effects.cap > 1)]
    require(np.all(random_n.ci_low <= 0) and np.all(random_n.ci_high >= 0), "random_n_negative_result")
    arch_diff = effects[(effects.contrast == "mlp_minus_cnn_bon") & (effects.cap > 1)]
    require(np.all(arch_diff.effect > 0), "mlp_unfavorable_results_retained")
    require(np.all((tail.winner_flipped_within_tail >= 0) & (tail.winner_flipped_within_tail <= 1)), "flip_range")

    report = (OUT / "REPORT.md").read_text()
    for text in ["回顾性诊断与敏感性分析", "不是新盲测", "2.480", "2.117", "1.835/2.955",
                 "0.936/0.963", "314,563", "Random(N)−Random(1) 没有任何区间排除零",
                 "没有用 T-cell 标签训练、校准、构造选择特征或选择修复方法"]:
        require(text in report, f"report:{text}")
    require("utr50" not in (OUT / "export/summary.json").read_text(), "summary_no_sequences")
    require((OUT / "logs/round1_tests.log").exists() and (OUT / "logs/round2_independent.log").exists(),
            "failure_logs_retained")

    result = {"status": "PASS", "checks": len(checks),
              "aggregate_tables": len(tables), "raw_input_hashes": len(raw["input_files"]),
              "negative_results_retained": True, "local_candidate_labels_or_sequences_exported": False,
              "rounds": {"round1": "PASS", "round2": "PASS", "round3_remote": "PASS"}}
    (OUT / "verification/round3_delivery.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = {"analysis": "tcell_error_diagnosis_v1", "status": "PASS", "files": {}}
    paths = [CODE / "run_analysis.py", CODE / "independent_verify.py", CODE / "verify_delivery.py",
             CODE / "verify_local_delivery.py", CODE / "frozen_protocol.json", CODE / "tests/test_analysis.py",
             OUT / "PROTOCOL.md", OUT / "REPORT.md", OUT / "raw_audit/raw_audit.json",
             OUT / "verification/environment.json", OUT / "verification/round2_independent.json",
             OUT / "verification/round3_delivery.json"]
    paths += sorted((OUT / "analysis").glob("*.tsv")) + sorted((OUT / "analysis").glob("*.json"))
    paths += sorted((OUT / "export").glob("*"))
    paths += [p for p in sorted((OUT / "logs").glob("*")) if not p.name.startswith("round3_delivery")]
    for path in paths:
        if path.is_file():
            manifest["files"][str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    result["manifest_files"] = len(manifest["files"])
    (OUT / "verification/round3_delivery.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest["files"][str(OUT / "verification/round3_delivery.json")] = {
        "sha256": sha(OUT / "verification/round3_delivery.json"),
        "bytes": (OUT / "verification/round3_delivery.json").stat().st_size}
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
