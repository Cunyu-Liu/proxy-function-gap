#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from delivery_scripts.publication_paths import DELIVERABLES, EVIDENCE_ROOT, OUTPUT_ROOT, REPO_ROOT, display_path

ROOT = EVIDENCE_ROOT
PAPER = DELIVERABLES / "paper"
FIGURES = DELIVERABLES / "figures"
SOURCE = DELIVERABLES / "reports" / "source_data"
OUT = OUTPUT_ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)
PDF_OUT = OUTPUT_ROOT / "paper" / "v6_1_20261126"
PDF_OUT.mkdir(parents=True, exist_ok=True)
MANUSCRIPT = HERE / "PAPER1_TCELL_DECISION_V6_1_20261126.md"
PDF = PAPER / MANUSCRIPT.with_suffix(".pdf").name
LEDGER = REPO_ROOT / "docs" / "PAPER1_TCELL_DECISION_V6_1_CLAIM_EVIDENCE_LEDGER_20261126.md"
BUILDER = HERE / "build_paper1_tcell_decision_v6_1.py"
FIGURE_PATHS = [
    FIGURES / "v3/Figure14_v3_equal_budget_value.pdf",
    FIGURES / "v4/Figure16_v4_tcell_global_value_tail_bottleneck.pdf",
    FIGURES / "v4/Figure17_v4_tcell_regression_decision_decoupling.pdf",
]
INPUT_PATHS = [
    SOURCE / "paper1_budget_matched_v1/results.json",
    SOURCE / "paper1_budget_matched_v1/pool_effects.tsv",
    SOURCE / "tcell_error_diagnosis_v1/analysis/summary.json",
    SOURCE / "tcell_error_diagnosis_v1/analysis/pointwise_metrics.tsv",
    SOURCE / "tcell_error_diagnosis_v1/analysis/top_tail_summary.tsv",
    SOURCE / "tcell_error_diagnosis_v1/analysis/selection_summary.tsv",
    SOURCE / "tcell_target_adaptation_v1/target/results.json",
    SOURCE / "tcell_listnet_adaptation_v2/target_v2/results.json",
    SOURCE / "tcell_listnet_adaptation_v2/verification/round1_independent_split.json",
    SOURCE / "tcell_cross_cell_anchor_v1/target/results.json",
    SOURCE / "tcell_residual_bstrict_v1/artifacts/auxiliary_gate.json",
    SOURCE / "tcell_rank_bstrict_v2/artifacts/auxiliary_gate.json",
    SOURCE / "tcell_multicontext_backbone_v1/auxiliary_gate.json",
    FIGURES / "v4/paper1_figures_16_17_config.json",
] + FIGURE_PATHS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict:
    return {"path": display_path(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00ad", "")).strip()


class Checks:
    def __init__(self, manuscript: str):
        self.manuscript = manuscript
        self.rows = []

    def require(self, condition: bool, name: str, claim: str, detail=""):
        self.rows.append({"name": name, "claim": claim, "pass": bool(condition), "detail": detail})

    def close(self, observed: float, expected: float, name: str, claim: str, token: str | None = None, tol=1e-9):
        passed = math.isclose(float(observed), float(expected), rel_tol=0, abs_tol=tol)
        if token is not None:
            passed = passed and token in self.manuscript
        self.require(passed, name, claim, {"observed": observed, "expected": expected, "required_manuscript_token": token})

    def finish(self, round_name: str):
        failed = [row for row in self.rows if not row["pass"]]
        return {
            "release": "PAPER1_TCELL_DECISION_V6_1_20261126",
            "round": round_name,
            "status": "PASS" if not failed else "FAIL",
            "checks_total": len(self.rows),
            "checks_passed": len(self.rows) - len(failed),
            "checks_failed": len(failed),
            "checks": self.rows,
        }


def indexed(rows: list[dict], **keys) -> dict:
    return {tuple(row[key] for key in keys): row for row in rows}


def round_one() -> dict:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    ledger = LEDGER.read_text(encoding="utf-8")
    c = Checks(text)
    for path in INPUT_PATHS:
        c.require(path.is_file(), f"input_exists:{path.name}", "C25", str(path))
    if any(not row["pass"] for row in c.rows):
        return c.finish("round1_numeric_claims")

    budget = load_json(INPUT_PATHS[0])
    pool_rows = load_tsv(INPUT_PATHS[1])
    c.require(len(pool_rows) == 320, "budget_pool_rows_320", "C01/C04", len(pool_rows))
    c.require(budget["status"] == "PASS" and budget["new_blind_test"] is False, "budget_retrospective_status", "C01")
    groups = {}
    for row in pool_rows:
        groups.setdefault((row["context"], int(row["cap"])), []).append(row)
    family = {(r["context"], r["cap"]): r for r in budget["families"]["bon_minus_random_n"]}
    context_summaries = {(r["context"], r["cap"]): r for r in budget["context_summaries"]}
    expected_budget = {
        ("defined_tcell_r1", 2): (4.1218072491858955, 4.921756194003953, -0.7999489448180579, -1.201008, -0.449730, 0.003999600039996),
        ("defined_tcell_r1", 4): (3.6424426444305946, 5.091198791659766, -1.4487561472291708, -1.809149, -1.094514, 0.003999600039996),
        ("defined_tcell_r1", 8): (3.5993160094451992, 5.178655663931107, -1.5793396544859084, -1.975375, -1.183081, 0.003999600039996),
        ("defined_tcell_r1", 16): (3.5551806849011736, 5.260940340371057, -1.705759655469883, -2.163445, -1.211914, 0.003999600039996),
        ("defined_tcell_r2", 2): (5.421317194225926, 6.102141480546617, -0.6808242863206915, -1.248022, -0.135771, 0.177582),
        ("defined_tcell_r2", 4): (5.254169702339798, 6.269045374564331, -1.014875672224533, -1.544490, -0.465832, 0.006399360063993601),
        ("defined_tcell_r2", 8): (4.888535226777166, 5.829408638615157, -0.940873411837991, -1.738304, -0.151800, 0.222178),
        ("defined_tcell_r2", 16): (5.043376614172306, 6.130317295921176, -1.08694068174887, -1.833685, -0.318589, 0.064994),
    }
    for key, expected in expected_budget.items():
        rows = groups[key]
        c.require(len(rows) == 16, f"budget_16_pools:{key}", "C02/C03")
        observed = [float(np.mean([float(r[field]) for r in rows])) for field in ("bon_regret", "random_n_regret", "bon_minus_random_n")]
        claim = "C02" if key[0].endswith("r1") else "C03"
        for j, field in enumerate(("bon", "random_n", "effect")):
            token = f"{expected[j]:.6f}" if field == "effect" else None
            if field == "effect" and expected[j] < 0:
                token = f"−{abs(expected[j]):.6f}"
            expected_value = expected[j] if field == "effect" else context_summaries[key][f"{field}_regret"]
            c.close(observed[j], expected_value, f"budget_recompute:{key}:{field}", claim, token, tol=2e-8)
        inference = family[key]
        c.close(inference["effect"], expected[2], f"budget_json_effect:{key}", claim)
        c.close(inference["nominal_95_ci"][0], expected[3], f"budget_ci_low:{key}", claim, tol=5e-7)
        c.close(inference["nominal_95_ci"][1], expected[4], f"budget_ci_high:{key}", claim, tol=5e-7)
        c.close(inference["holm_p"], expected[5], f"budget_holm:{key}", claim, tol=5e-7)
        c.require(f"{expected[3]:.6f}".replace("-", "−") in text and f"{expected[4]:.6f}".replace("-", "−") in text, f"budget_ci_tokens:{key}", claim)
        if key[0].endswith("r1") or key == ("defined_tcell_r2", 4):
            c.require(inference["holm_p"] < 0.05, f"budget_holm_survives:{key}", claim)
        else:
            c.require(inference["holm_p"] >= 0.05, f"budget_holm_not_survives:{key}", claim)
    for key, rows in groups.items():
        if key[1] == 1:
            c.require(all(float(r["bon_minus_random_n"]) == 0 for r in rows), f"cap1_identity:{key}", "C04")
    random_family = budget["families"]["random_n_minus_random1"]
    c.require(len(random_family) == 20 and all(r["holm_p"] >= 0.05 for r in random_family), "random_n_random1_no_holm_survivor", "C04")

    diagnosis = load_json(INPUT_PATHS[2])
    pointwise = {(r["context"], r["model"]): r for r in load_tsv(INPUT_PATHS[3])}
    expected_pointwise = {
        "defined_tcell_r1": (1.0643927420067529, 1.5674544802689514, 1.9710659268090611, 0.5910524812945454, 0.5838694625265951, 1.1083578156690632),
        "defined_tcell_r2": (0.4074804505165193, 1.543969499774867, 2.3087319106531221, 0.5292216226399566, 0.5150549602572645, 1.4757177144636591),
    }
    for context, expected in expected_pointwise.items():
        row = pointwise[(context, "cnn")]
        for j, field in enumerate(("bias", "mae", "rmse", "spearman", "mean_within_pool_spearman", "oracle_global_median_shift_mae")):
            c.close(float(row[field]), expected[j], f"pointwise:{context}:{field}", "C05/C06", f"{expected[j]:.6f}")
        c.close(float(row["mse"]), float(row["bias_squared"]) + float(row["centered_mse"]), f"mse_identity:{context}", "C05")
        summary_row = next(r for r in diagnosis["pointwise"] if r["context"] == context and r["model"] == "cnn")
        c.close(summary_row["mae"], expected[1], f"pointwise_json_tsv_agree:{context}", "C05")

    tail = {(r["context"], r["model"], int(r["cap"])): r for r in load_tsv(INPUT_PATHS[4])}
    expected_tail = {
        "defined_tcell_r1": (3.315826448847611, 2.4799977092008927, 0.8358287396467186, 0.2060575502396873, 0.4309392507002801),
        "defined_tcell_r2": (4.0705708474667865, 2.11722079513684, 1.9533500523299472, 0.143984242067949, 0.4498026753074111),
    }
    for context, expected in expected_tail.items():
        row = tail[(context, "cnn", 16)]
        for j, field in enumerate(("regret", "within_tail_ranking_loss", "opportunity_loss", "spearman", "pair_inversion_rate")):
            token = f"{expected[j]:.6f}" if j < 4 else None
            c.close(float(row[field]), expected[j], f"tail:{context}:{field}", "C07/C08", token)
        c.close(float(row["regret"]), float(row["within_tail_ranking_loss"]) + float(row["opportunity_loss"]), f"tail_identity:{context}", "C08")
        c.close(float(row["winner_flipped_within_tail"]) * 16, 15, f"tail_winner_overturned:{context}", "C07")
        share = float(row["within_tail_ranking_loss"]) / float(row["regret"])
        expected_share = 0.748 if context.endswith("r1") else 0.520
        c.close(share, expected_share, f"tail_ranking_share:{context}", "C08", tol=0.0005)

    selection = {(r["context"], r["model"], int(r["cap"]), r["strategy"]): r for r in load_tsv(INPUT_PATHS[5])}
    expected_prefix = {
        "defined_tcell_r1": (3.555180684901174, 1.719882499855096, 1.8352981850460774),
        "defined_tcell_r2": (5.043376614172306, 2.088226640997843, 2.9551499731744633),
    }
    for context, expected in expected_prefix.items():
        row = selection[(context, "cnn", 16, "bon")]
        for j, field in enumerate(("regret", "opportunity_loss", "within_prefix_ranking_loss")):
            c.close(float(row[field]), expected[j], f"prefix:{context}:{field}", "C09")
        c.close(float(row["regret"]), float(row["opportunity_loss"]) + float(row["within_prefix_ranking_loss"]), f"prefix_identity:{context}", "C09")
        random_row = selection[(context, "cnn", 16, "random_n")]
        c.close(float(row["opportunity_loss"]), float(random_row["opportunity_loss"]), f"prefix_shared_opportunity:{context}", "C09")

    for context, n, mse_share, events in (("defined_tcell_r1", 62, 0.4145044734133946, 5), ("defined_tcell_r2", 102, 0.665006627455886, 11)):
        row = next(r for r in diagnosis["outliers"] if r["context"] == context and r["model"] == "cnn")
        c.require(row["outlier_n"] == n and row["cap16_bon_outlier_events"] == events, f"outlier_counts:{context}", "C10")
        c.close(row["mse_contribution"], mse_share, f"outlier_mse_share:{context}", "C10")

    v1 = load_json(INPUT_PATHS[6])
    c.require(v1["evaluation_count"] == 1 and v1["new_blind_test"] is False, "v1_one_retrospective_evaluation", "C11")
    expected_v1 = {
        "defined_tcell_r1": (1.3521591065469438, 0.914506247476038, -0.4376528590709058, 2.389656748113372, 2.4285856769522525, 0.03892892883888),
        "defined_tcell_r2": (1.0622726490779133, 1.0187163057577855, -0.0435563433201278, 2.8730018816132494, 2.8855531993597245, 0.0125513177464756),
    }
    for context, expected in expected_v1.items():
        errors = v1["pointwise_errors_mrl_units"][context]
        cap = v1["by_repeat_and_cap"][context]["16"]
        observed = (errors["original_cnn"]["mae"], errors["main_50k"]["mae"], errors["main_50k"]["mae"] - errors["original_cnn"]["mae"], cap["mean_regret"]["original_cnn"], cap["mean_regret"]["main_50k"], cap["mean_regret"]["main_50k"] - cap["mean_regret"]["original_cnn"])
        for j, value in enumerate(expected):
            c.close(observed[j], value, f"v1_adaptation:{context}:{j}", "C11", tol=2e-8)
        c.close(cap["main_minus_original"]["effect"], expected[5], f"v1_effect_json:{context}", "C11")
        for cap_n in (1, 2, 4, 8, 16):
            means = v1["by_repeat_and_cap"][context][str(cap_n)]["mean_regret"]
            c.close(means["main_50k"], means["mse_only_50k"], f"v1_main_mse_same:{context}:{cap_n}", "C13")
            c.close(v1["by_repeat_and_cap"][context][str(cap_n)]["main_minus_original"]["p_holm_10_repeat_cap_tests"], 1, f"v1_holm_one:{context}:{cap_n}", "C12")
    primary = v1["primary"]
    combined = primary["combined_main_minus_original"]
    repeat_mean = float(np.mean(list(primary["repeat_main_minus_original_effects"].values())))
    c.close(repeat_mean, combined["effect"], "v1_combined_repeat_mean", "C12")
    c.close(combined["effect"], 0.0257401232926778, "v1_combined_effect", "C12", "+0.025740")
    c.close(combined["ci_low"], -0.07201905000880485, "v1_combined_ci_low", "C12")
    c.close(combined["ci_high"], 0.14099380381511187, "v1_combined_ci_high", "C12")
    c.close(primary["minimum_required_absolute_reduction"], max(0.1, 0.05 * primary["combined_original_cap16_regret"]), "v1_required_reduction", "C12")
    c.require(primary["success"] is False and primary["both_repeats_improve"] is False, "v1_primary_fail", "C12")

    v2 = load_json(INPUT_PATHS[7])
    split = load_json(INPUT_PATHS[8])
    c.require(v2["evaluation_count"] == 1 and v2["historical_retrospective_boundary_retained"], "v2_one_retrospective_evaluation", "C14")
    c.require(split["status"] == "PASS" and split["test_components"] == 1004 and split["eligible_components"] == 39222, "v2_component_split", "C14")
    c.require(split["test_v2_to_v1_exact_h1"] == split["test_v2_to_sample_exact_h1"] == split["cross_test_pool_exact_h1"] == 0, "v2_zero_conflicts", "C14")
    mean16 = v2["by_context_and_cap"]["mean_replicate"]["16"]["mean_regret"]
    for comparator, expected_effect, ci_low, ci_high in (("original_cnn", -0.0374154783699216, -0.1832037112435067, 0.1024938883642236), ("v1_main", -0.0075845675072431, -0.02869787621135, 0.0124741182041318)):
        effect = mean16["listnet"] - mean16[comparator]
        c.close(effect, expected_effect, f"v2_effect_recompute:{comparator}", "C15")
        record = v2["primary"]["comparisons"][comparator]
        c.close(record["effect"]["effect"], expected_effect, f"v2_effect_json:{comparator}", "C15")
        c.close(record["effect"]["ci_low"], ci_low, f"v2_ci_low:{comparator}", "C15")
        c.close(record["effect"]["ci_high"], ci_high, f"v2_ci_high:{comparator}", "C15")
        c.close(record["absolute_reduction"], -expected_effect, f"v2_reduction:{comparator}", "C15")
        c.require(record["absolute_reduction"] < record["minimum_required_absolute_reduction"] and record["effect"]["p_holm_10_comparison_cap_tests"] == 1 and not record["effect"]["ci_high"] < 0, f"v2_gate_fails:{comparator}", "C15")
        c.require(v2["primary"]["comparison_pass"][comparator] is False, f"v2_comparison_pass_false:{comparator}", "C15")
    c.require(v2["primary"]["success"] is False and v2["primary"]["post_evaluation_protocol_changes"] == 0, "v2_primary_fail_no_change", "C15")
    for comparator, r1, r2 in (("v1_main", 0.0209456258193571, -0.0361147608338432),):
        for context, expected in (("defined_tcell_r1", r1), ("defined_tcell_r2", r2)):
            row = v2["by_context_and_cap"][context]["16"]
            effect = row["mean_regret"]["listnet"] - row["mean_regret"][comparator]
            c.close(effect, expected, f"v2_repeat_effect:{context}", "C16")
    mean_errors = v2["pointwise_errors_mrl_units"]["mean_replicate"]
    c.close(mean_errors["listnet"]["mae"] - mean_errors["v1_main"]["mae"], 0.0021040176173872, "v2_mae_increase_vs_v1", "C16")
    c.close(mean_errors["original_cnn"]["mae"] - mean_errors["listnet"]["mae"], 0.3069153580945532, "v2_mae_reduction_vs_original", "C16")
    for comparator, expected in (("original_cnn", 0.01875), ("v1_main", 0.0114583333333333)):
        c.close(v2["top_tail"]["mean_replicate"][comparator]["tail_pair_accuracy_change"], expected, f"v2_pair_accuracy:{comparator}", "C16")

    anchor = load_json(INPUT_PATHS[9])
    primary = anchor["primary"]
    for context, expected in (("defined_tcell_r1", 0.463089560864378), ("defined_tcell_r2", 0.3947045943160953)):
        cap = anchor["by_repeat_and_cap"][context]["16"]
        effect = cap["mean_regret"]["hepg2_anchor"] - cap["mean_regret"]["cnn"]
        c.close(effect, expected, f"anchor_effect_recompute:{context}", "C17")
        c.close(cap["anchor_minus_cnn"]["effect"], expected, f"anchor_effect_json:{context}", "C17")
        oracle = cap["oracle"]
        c.close(oracle["cnn"]["proposal_headroom_mean"], oracle["hepg2_anchor"]["proposal_headroom_mean"], f"anchor_same_headroom:{context}", "C20")
    c.close(float(np.mean(list(primary["repeat_anchor_minus_cnn_effects"].values()))), primary["combined_anchor_minus_cnn"]["effect"], "anchor_combined_mean", "C17")
    c.close(primary["combined_anchor_minus_cnn"]["effect"], 0.4288970775902367, "anchor_combined_effect", "C17", "+0.428897")
    c.close(primary["combined_anchor_minus_cnn"]["ci_low"], 0.2454604622370106, "anchor_ci_low", "C17")
    c.close(primary["combined_anchor_minus_cnn"]["ci_high"], 0.6164744196656033, "anchor_ci_high", "C17")
    c.require(primary["combined_anchor_minus_cnn"]["ci_low"] > 0 and primary["success"] is False, "anchor_significant_reverse", "C17")
    c.close(anchor["by_repeat_and_cap"]["defined_tcell_r1"]["16"]["anchor_minus_cnn"]["p_holm_10_repeat_cap_tests"], 0.053994600539946, "anchor_r1_holm", "C18")
    c.close(anchor["by_repeat_and_cap"]["defined_tcell_r2"]["16"]["anchor_minus_cnn"]["p_holm_10_repeat_cap_tests"], 0.017998200179982, "anchor_r2_holm", "C18")
    c.close(anchor["secondary"]["fusion_cap16_combined_minus_cnn"]["effect"], 0.1545268005100963, "anchor_fusion_effect", "C18")
    for context, cnn_rho, hepg2_rho in (("defined_tcell_r1", 0.7220913244262841, 0.7300011924122529), ("defined_tcell_r2", 0.724563853193763, 0.7501573082554026)):
        stats = anchor["secondary"]["spearman"][context]
        c.close(stats["cnn"]["overall_spearman"], cnn_rho, f"anchor_cnn_spearman:{context}", "C19")
        c.close(stats["hepg2_anchor"]["overall_spearman"], hepg2_rho, f"anchor_hepg2_spearman:{context}", "C19")
        tail_stats = anchor["secondary"]["top_tail"][context]
        c.require(tail_stats["top1_flip_pools"] == 16, f"anchor_top1_flips:{context}", "C19")
        c.close(tail_stats["top16_membership_jaccard_mean"], 0.4704316126284089, f"anchor_tail_jaccard:{context}", "C19")
        turnover = anchor["by_repeat_and_cap"][context]["16"]["turnover_anchor_vs_cnn"]
        c.require(turnover["changed"] == 61 and turnover["worsened_among_changed"] == 39, f"anchor_turnover:{context}", "C19")

    residual = load_json(INPUT_PATHS[10])
    c.require(residual["all_directions_pass"] is False and residual["target_mrl_read"] is False and residual["target_evaluation_permitted"] is False, "residual_fail_closed", "C21")
    for direction, expected in zip(residual["directions"], (0.0003429119388008, 0.0046384833559658)):
        c.close(-direction["test"]["cap16_regret_improvement"], expected, f"residual_effect:{direction['direction']}", "C21")
        c.require(direction["gate_pass"] is False, f"residual_gate:{direction['direction']}", "C21")
    c.require(not (SOURCE / "tcell_residual_bstrict_v1/target").exists(), "residual_no_target_directory", "C21")

    rank = load_json(INPUT_PATHS[11])
    c.require(rank["all_directions_pass"] is False and rank["target_mrl_read"] is False and rank["target_evaluation_permitted"] is False, "rank_fail_closed", "C22")
    for direction in rank["directions"]:
        effect = direction["test"]["corrected_minus_frozen"]
        recomputed = direction["test"]["corrected_cap16_mean_regret"] - direction["test"]["frozen_cap16_mean_regret"]
        c.close(recomputed, effect["effect"], f"rank_effect:{direction['direction']}", "C22")
        c.require(direction["gate_pass"] is False and direction["test"]["target_mrl_read"] is False, f"rank_gate:{direction['direction']}", "C22")
    c.require(not (SOURCE / "tcell_rank_bstrict_v2/target").exists(), "rank_no_target_directory", "C22")

    backbone = load_json(INPUT_PATHS[12])
    c.require(backbone["both_directions_pass"] is False and backbone["target_evaluation_count"] == 0 and backbone["target_mrl_read"] is False, "backbone_fail_closed", "C23")
    for name, direction in backbone["directions"].items():
        c.close(direction["main_cap16_regret"] - direction["baseline_cap16_regret"], direction["effect"]["effect"], f"backbone_effect:{name}", "C23")
        c.require(direction["point_pass"] is False and direction["ci_pass"] is False, f"backbone_gate:{name}", "C23")
    c.require(not (SOURCE / "tcell_multicontext_backbone_v1/target").exists(), "backbone_no_target_directory", "C23")

    figure_config = load_json(INPUT_PATHS[13])
    for record in figure_config["sources"]:
        path = ROOT / record["path"]
        c.require(path.is_file() and sha256(path) == record["sha256"], f"figure_source_hash:{path.name}", "C25")
    for path in FIGURE_PATHS:
        with fitz.open(path) as document:
            c.require(document.page_count == 1, f"figure_one_page:{path.name}", "C25")
        c.require(path.name in text, f"figure_cited:{path.name}", "C25")
    for index in range(1, 26):
        c.require(f"C{index:02d}" in ledger, f"ledger_claim_C{index:02d}", "C25")
    c.require("Figure16_v4_tcell_error_decomposition" not in text and "Figure17_v4_tcell_adaptation_decision" not in text, "no_placeholder_figure_names", "C25")

    result = c.finish("round1_numeric_claims")
    result["scientific_input_artifacts"] = [artifact(path) for path in INPUT_PATHS]
    result["recomputation_scope"] = "Read-only aggregate TSV/JSON arithmetic, source-hash, isolation, gate, and manuscript-token checks; no model run or target evaluation."
    return result


def round_two() -> dict:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    ledger = LEDGER.read_text(encoding="utf-8")
    c = Checks(text)
    required = {
        "specific_positive_title": "Equal-budget CNN ranking reduces functional regret in T-cell 5′ UTR selection",
        "abstract_equal_budget": "reduced regret relative to Random(N) at every cap 2–16 in both repeats",
        "abstract_decision_endpoint": "measured functional regret—not MAE—as the primary endpoint",
        "abstract_50k_repeat_specific": "reduced MAE in both repeats (substantially in r1 and modestly in r2)",
        "abstract_listnet_point_estimates": "small favorable point estimates that failed both co-primary gates",
        "random_n_conditional_intervals": "pool-bootstrap intervals condition on those realized draws and do not integrate selector-randomization Monte Carlo uncertainty",
        "retrospective_status": "All experimental analyses are retrospective analyses of historical data with strict sequence/grouped holdouts",
        "no_blind": "None is a new prospective blind test",
        "no_causal": "or evidence of causation",
        "r2_multiplicity_boundary": "only cap 4 survived the all-20 Holm family",
        "tail_not_global_failure": "substantial absolute error and useful global ordering are simultaneously observed",
        "tail_diagnostic_not_repair": "oracle components describe headroom rather than realized repair",
        "v1_grouped": "target-context retrospective grouped holdout evidence",
        "v2_grouped": "another 16 component-disjoint pools of 64",
        "different_tests_not_compared": "absolute regret values are not compared across their different test pools",
        "listnet_gate_failure": "The joint co-primary intersection–union gate therefore failed",
        "listnet_not_improvement": "These are not reported as significant or substantive improvements",
        "anchor_reverse": "This combined interval is wholly in the harmful direction",
        "anchor_repeat_correction": "whereas r1 cap16 was nominal but did not survive",
        "auxiliary_not_tcell_failure": "they are not mislabeled as T-cell failures or as validated improvements",
        "auxiliary_target_zero": "target evaluation count remained zero for these programs",
        "mae_not_irrelevant": "This recommendation does not imply that MAE is irrelevant",
        "primary_regret_recommendation": "recommend measured functional regret as the primary acceptance endpoint",
        "no_new_model": "V6.1 performed no model training or inference",
        "availability": "## Data and code availability",
        "limitations": "## 4. Limitations",
        "methods": "## 5. Methods",
        "figure_legends": "## Figure legends",
        "references": "## References",
    }
    for name, token in required.items():
        c.require(token in text, name, "narrative", token)
    c.require("proxy contract" not in text.split("---", 2)[1].lower(), "title_not_proxy_contract", "narrative")
    c.require("all methods fail" in text, "not_all_negative_frame", "narrative")
    forbidden = [
        "This is a new blind test",
        "These are new blind tests",
        "prospective wet-lab validation was performed",
        "causal mediation is demonstrated",
        "ListNet significantly improves",
        "ListNet substantially improves",
        "50k adaptation improves cap16",
        "Target-context adaptation with 50,000 labels markedly reduced MAE",
        "Fifty-thousand target labels produced markedly lower MAE",
        "small, gate-failing point improvements",
        "HepG2 anchoring improves T-cell",
        "both repeats survived Holm at cap16",
        "residual, pairwise, and multi-context models failed in T-cell",
        "Figure16_v4_tcell_error_decomposition",
        "Figure17_v4_tcell_adaptation_decision",
        "TBD",
        "PLACEHOLDER",
    ]
    for token in forbidden:
        c.require(token not in text, f"forbidden_wording:{token}", "narrative")
    for path in FIGURE_PATHS:
        c.require(path.name in text, f"actual_figure_reference:{path.name}", "C25")
        c.require(f"**Figure {14 if 'Figure14' in path.name else 16 if 'Figure16' in path.name else 17} |" in text, f"figure_legend_present:{path.name}", "C25")
    for section in ("### 5.1", "### 5.2", "### 5.3", "### 5.4", "### 5.5", "### 5.6", "### 5.7", "### 5.8", "### 5.9", "### 5.10", "### 5.11"):
        c.require(section in text, f"methods_subsection:{section}", "narrative")
    body, references = text.split("## References", 1)
    reference_ids = set(int(x) for x in re.findall(r"^\[(\d+)\]", references, flags=re.M))
    c.require(reference_ids == set(range(1, 16)), "references_1_to_15", "narrative", sorted(reference_ids))
    cited = set()
    for match in re.finditer(r"\[(\d+)(?:[–-](\d+))?\]", body):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if 1 <= start <= end <= 15:
            cited.update(range(start, end + 1))
    c.require(cited == reference_ids, "all_references_cited", "narrative", {"cited": sorted(cited), "references": sorted(reference_ids)})
    for index in range(1, 26):
        c.require(f"C{index:02d}" in ledger, f"ledger_complete_C{index:02d}", "C25")
    c.require("Round 1—主张数值" in ledger and "Round 2—叙事边界" in ledger and "Round 3—PDF clean rebuild" in ledger, "ledger_three_round_scopes", "C25")
    c.require("V4 and all earlier" not in text, "no_stale_release_narrative", "narrative")
    result = c.finish("round2_narrative_and_scope")
    result["audit_scope"] = "Title/abstract contribution, retrospective/grouped-holdout language, correction families, test separation, gate semantics, figure references, complete sections, citations, and prohibited wording. No numeric or PDF check is inferred from this round."
    return result


def font_audit(path: Path) -> dict:
    completed = subprocess.run(["pdffonts", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    rows = []
    for line in completed.stdout.splitlines()[2:]:
        match = re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line)
        if match:
            rows.append({"font": line.split()[0], "embedded": match.group(1) == "yes", "unicode": match.group(3) == "yes"})
    return {"fonts": rows, "all_embedded": bool(rows) and all(row["embedded"] for row in rows), "all_unicode": bool(rows) and all(row["unicode"] for row in rows)}


def validate_pdf(path: Path, full: bool = True) -> dict:
    required_tokens = [
        "Equal-budget CNN ranking reduces functional regret",
        "Abstract",
        "−0.799949",
        "−1.705760",
        "0.006399",
        "2.479998",
        "2.117221",
        "+0.025740",
        "−0.037415",
        "+0.428897",
        "target evaluation count remained zero",
        "Figure 14 |",
        "Figure 16 |",
        "Figure 17 |",
        "Data and code availability",
        "References",
        "[15] N. Schlusser",
    ]
    with path.open("rb") as handle:
        header = handle.read(5)
        handle.seek(max(0, path.stat().st_size - 2048))
        trailer = handle.read()
    document = fitz.open(path)
    texts = []
    outside_words = []
    outside_objects = []
    blank_pages = []
    page_records = []
    xobject_count = 0
    for index, page in enumerate(document):
        text = page.get_text("text")
        texts.append(text)
        rect = page.rect
        words = page.get_text("words")
        for word in words:
            box = fitz.Rect(word[:4])
            if box.x0 < rect.x0 - 0.5 or box.y0 < rect.y0 - 0.5 or box.x1 > rect.x1 + 0.5 or box.y1 > rect.y1 + 0.5:
                outside_words.append({"page": index + 1, "text": word[4], "bbox": list(box)})
        xobjects = page.get_xobjects()
        xobject_count += len(xobjects)
        boxes = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
        boxes.extend(drawing["rect"] for drawing in page.get_drawings())
        for box in boxes:
            if all(math.isfinite(v) for v in box) and not box.is_empty and (box.x0 < rect.x0 - 1 or box.y0 < rect.y0 - 1 or box.x1 > rect.x1 + 1 or box.y1 > rect.y1 + 1):
                outside_objects.append({"page": index + 1, "bbox": list(box)})
        record = {"page": index + 1, "characters": len(text.strip()), "words": len(words), "xobjects": len(xobjects)}
        if full:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), colorspace=fitz.csGRAY, alpha=False)
            pixels = np.frombuffer(pixmap.samples, dtype=np.uint8)
            record["render_pixels"] = [pixmap.width, pixmap.height]
            record["render_std"] = round(float(pixels.std()), 3)
            if pixels.std() < 1 or not text.strip():
                blank_pages.append(index + 1)
        page_records.append(record)
    text = normalize("\n".join(texts))
    compact = re.sub(r"\s+", "", text)
    missing = [token for token in required_tokens if normalize(token) not in text and re.sub(r"\s+", "", token) not in compact]
    fonts = font_audit(path)
    pdfinfo = subprocess.run(["pdfinfo", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    passed = header == b"%PDF-" and b"%%EOF" in trailer and len(document) > 0 and not missing and not outside_words and not outside_objects and not blank_pages and "\ufffd" not in text and fonts["all_embedded"] and fonts["all_unicode"] and xobject_count >= 3 and pdfinfo.returncode == 0
    result = {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "pages": len(document),
        "header_and_eof_valid": header == b"%PDF-" and b"%%EOF" in trailer,
        "pdfinfo_valid": pdfinfo.returncode == 0,
        "missing_required_text": missing,
        "replacement_glyphs": text.count("\ufffd"),
        "words_outside_page": outside_words,
        "objects_outside_page": outside_objects,
        "blank_or_textless_pages": blank_pages,
        "xobject_count": xobject_count,
        "expected_min_figure_xobjects": 3,
        "font_audit": fonts,
        "page_records": page_records,
        "pass": passed,
    }
    document.close()
    return result


def round_three() -> dict:
    c = Checks(MANUSCRIPT.read_text(encoding="utf-8"))
    for name in ("ROUND1_NUMERIC", "ROUND2_NARRATIVE"):
        path = OUT / f"PAPER1_TCELL_DECISION_V6_1_{name}_20261126.json"
        c.require(path.is_file(), f"prior_round_exists:{name}", "round3")
        if path.is_file():
            c.require(load_json(path)["status"] == "PASS", f"prior_round_pass:{name}", "round3")
    c.require(PDF.is_file(), "delivered_pdf_exists", "round3")
    if any(not row["pass"] for row in c.rows):
        return c.finish("round3_pdf_clean_rebuild")
    initial_hash = sha256(PDF)
    initial_validation = validate_pdf(PDF, full=True)
    c.require(initial_validation["pass"], "delivered_pdf_full_validation", "round3", initial_validation)
    source_hash_before = sha256(MANUSCRIPT)
    spec = importlib.util.spec_from_file_location("paper1_v6_1_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix=".paper1_v6_1_clean_rebuild_", dir=PDF_OUT) as temporary:
        rebuild_dir = Path(temporary)
        c.require(not any(rebuild_dir.iterdir()), "clean_output_directory_empty", "round3")
        build_record = module.build_pdf(rebuild_dir)
        rebuilt = rebuild_dir / PDF.name
        c.require(rebuilt.is_file(), "clean_rebuilt_pdf_exists", "round3")
        if rebuilt.is_file():
            rebuilt_hash = sha256(rebuilt)
            c.require(rebuilt_hash == initial_hash, "clean_rebuild_byte_identical", "round3", {"initial": initial_hash, "rebuilt": rebuilt_hash})
            rebuilt_validation = validate_pdf(rebuilt, full=True)
            c.require(rebuilt_validation["pass"], "clean_rebuilt_pdf_full_validation", "round3", rebuilt_validation)
            c.require(build_record.get("content_invariant") is True, "scientific_content_invariant", "round3")
            c.require(not build_record.get("serious_warnings"), "no_serious_layout_warnings", "round3", build_record.get("serious_warnings"))
    c.require(sha256(MANUSCRIPT) == source_hash_before, "manuscript_unchanged_by_build", "round3")
    round1 = load_json(OUT / "PAPER1_TCELL_DECISION_V6_1_ROUND1_NUMERIC_20261126.json")
    for record in round1["scientific_input_artifacts"]:
        path = ROOT / record["path"]
        if "figures" in Path(record["path"]).parts:
            continue
        c.require(sha256(path) == record["sha256"], f"scientific_input_unchanged:{path.name}", "scope")
    figure_config = load_json(FIGURES / "v4/paper1_figures_16_17_config.json")
    for record in figure_config["sources"] + figure_config["figures"]:
        path = ROOT / record["path"]
        c.require(path.is_file() and sha256(path) == record["sha256"], f"current_figure_delivery_hash:{path.name}", "scope")
    result = c.finish("round3_pdf_clean_rebuild")
    result["delivered_pdf_validation"] = initial_validation
    result["clean_rebuild"] = {"byte_identical_required": True, "source_date_epoch": module.SOURCE_DATE_EPOCH}
    result["audit_scope"] = "Delivered PDF text, rendering, embedded Unicode fonts, overflow, required figures, clean empty-directory rebuild, byte identity, and scientific-input hash preservation."
    return result


def write_result(result: dict, stem: str) -> None:
    json_path = OUT / f"{stem}.json"
    log_path = OUT / f"{stem}.log"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"release={result['release']}",
        f"round={result['round']}",
        f"status={result['status']}",
        f"checks_total={result['checks_total']}",
        f"checks_passed={result['checks_passed']}",
        f"checks_failed={result['checks_failed']}",
    ]
    for row in result["checks"]:
        if not row["pass"]:
            lines.append(f"FAIL {row['name']} claim={row['claim']} detail={json.dumps(row['detail'], ensure_ascii=False)}")
    if result["status"] == "PASS":
        lines.append("All scoped checks passed; this round does not upgrade retrospective evidence to prospective or causal evidence.")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def write_manifest(round3: dict) -> None:
    paths = [MANUSCRIPT, PDF, BUILDER, Path(__file__).resolve(), LEDGER] + FIGURE_PATHS
    for stem in ("PAPER1_TCELL_DECISION_V6_1_ROUND1_NUMERIC_20261126", "PAPER1_TCELL_DECISION_V6_1_ROUND2_NARRATIVE_20261126", "PAPER1_TCELL_DECISION_V6_1_ROUND3_PDF_20261126"):
        paths.extend([OUT / f"{stem}.json", OUT / f"{stem}.log"])
    manifest = {
        "release": "PAPER1_TCELL_DECISION_V6_1_20261126",
        "status": "PASS" if round3["status"] == "PASS" else "FAIL",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent T-cell decision-focused Paper 1 V6.1 main text, claim-evidence ledger, three scoped read-only verification rounds, and deterministic PDF packaging. No source_data, figures, PPT, model, or target evaluation was modified or run.",
        "evidence_boundary": "All experiments are retrospective/grouped holdouts as individually specified. No new prospective blind test, independent wet-lab replication, or causal claim. Auxiliary residual/pairwise/multi-context programs have target evaluation count zero.",
        "primary_positive_claim": "Frozen Sample CNN BoN lowers measured T-cell MRL regret relative to same-prefix Random(N); r1 caps 2-16 and r2 cap4 survive the complete all-20 Holm family.",
        "verification_rounds": {
            "round1_numeric": "PASS",
            "round2_narrative": "PASS",
            "round3_pdf_clean_rebuild": round3["status"],
        },
        "artifacts": [artifact(path) for path in paths],
        "source_data_modified": False,
        "figures_modified": False,
        "ppt_modified": False,
        "new_model_training_or_inference": False,
        "new_target_unblinding_or_evaluation": False,
    }
    (PDF_OUT / "PAPER1_TCELL_DECISION_V6_1_RELEASE_MANIFEST_20261126.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.round == 1:
        result = round_one()
        stem = "PAPER1_TCELL_DECISION_V6_1_ROUND1_NUMERIC_20261126"
    elif args.round == 2:
        result = round_two()
        stem = "PAPER1_TCELL_DECISION_V6_1_ROUND2_NARRATIVE_20261126"
    else:
        result = round_three()
        stem = "PAPER1_TCELL_DECISION_V6_1_ROUND3_PDF_20261126"
    write_result(result, stem)
    if args.round == 3:
        write_manifest(result)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
