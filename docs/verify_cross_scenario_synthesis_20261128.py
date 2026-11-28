from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "deliverables" / "reports" / "source_data"
DOCS = ROOT / "deliverables" / "reports" / "docs"
REPORT = DOCS / "CROSS_SCENARIO_EVIDENCE_SYNTHESIS_20261128.md"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def close(a, b, tol=1e-9):
    assert math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol), (a, b)


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()
    checks = []

    def check(name, fn):
        fn()
        checks.append({"name": name, "status": "PASS"})

    budget_path = SRC / "paper1_budget_matched_v1" / "results.json"
    budget = load_json(budget_path)
    family = { (r["context"], r["cap"]): r for r in budget["families"]["bon_minus_random_n"] }

    def tcell():
        r1 = family[("defined_tcell_r1", 16)]
        r2 = family[("defined_tcell_r2", 16)]
        close(r1["effect"], -1.7057596554698837)
        close(r2["effect"], -1.086940681749869)
        assert r1["nominal_95_ci"][1] < 0 and r2["nominal_95_ci"][1] < 0
        close(r1["holm_p"], 0.003999600039996)
        close(r2["holm_p"], 0.064993500649935)
        assert all(family[("defined_tcell_r1", c)]["holm_p"] < 0.05 for c in [2,4,8,16])
        assert family[("defined_tcell_r2", 4)]["holm_p"] < 0.05
        assert not budget["new_blind_test"]
    check("tcell_equal_budget_primary", tcell)

    gb1_path = SRC / "scenario_gb1_v1" / "aggregate" / "test_aggregate.json"
    gb1 = load_json(gb1_path)
    def gb1_check():
        b = gb1["budgets"]["16"]
        close(b["delta_regret_bon_minus_random"], -0.16089381057826665)
        close(gb1["bootstrap"]["16"]["ci95"][0], -0.28285336364321556)
        close(gb1["bootstrap"]["16"]["ci95"][1], -0.05051518912930951)
        assert gb1["label"] == "lnW" and gb1["n_pools"] == 55
        assert gb1["bootstrap"]["16"]["cluster_unit"] == "position-pair pool"
    check("gb1_lnw_cap16", gb1_check)

    gfp_path = SRC / "scenario_gfp_v1" / "results" / "primary.json"
    gfp = load_json(gfp_path)
    def gfp_check():
        methods = {m["method"]: m for m in gfp["methods"]}
        cnn = next(r for r in methods["cnn_ensemble"]["primary"] if r["budget"] == 16)
        add = next(r for r in methods["additive"]["primary"] if r["budget"] == 16)
        close(cnn["mean_bon_minus_random_regret"], -0.26083015723296876)
        close(add["mean_bon_minus_random_regret"], -0.2800, tol=1e-4)
        assert cnn["ci95_bon_minus_random_regret"]["high"] < 0
        assert add["ci95_bon_minus_random_regret"]["high"] < 0
        assert add["mean_bon_regret"] < cnn["mean_bon_regret"]
        close(methods["cnn_ensemble"]["point_metrics"]["spearman"], 0.43402505842428907)
        order9 = next(r for r in methods["cnn_ensemble"]["point_metrics_by_order"] if r["order"] == 9)
        assert order9["spearman"] < 0
    check("gfp_high_order_selection", gfp_check)

    aav_path = SRC / "scenario_aav_v1" / "results" / "main_results.json"
    aav = load_json(aav_path)
    def aav_check():
        b = aav["caps"]["16"]
        close(b["paired_delta_bon_minus_random"], -4.206431097252901)
        close(b["ci95_bootstrap_percentile"][0], -4.564111062543161)
        close(b["ci95_bootstrap_percentile"][1], -3.931098295982311)
        assert b["pools"] == 3996 and not aav["meeting6_used"]
        assert "not tropism" in aav["endpoint_semantics"]
        assert not aav["new_blinded_validation"]
    check("aav9_production_cap16", aav_check)

    tem_overall_path = SRC / "scenario_tem1_v1" / "results" / "overall_results.tsv"
    tem_dir_path = SRC / "scenario_tem1_v1" / "results" / "direction_results.tsv"
    tem_overall = load_tsv(tem_overall_path)
    tem_dir = load_tsv(tem_dir_path)
    def tem_check():
        row = next(r for r in tem_overall if r["split"] == "test" and r["N"] == "16")
        close(row["delta_bon_minus_random"], -0.15937934064620415)
        close(row["delta_bon_minus_random_ci_low"], -0.20787367629407305)
        close(row["delta_bon_minus_random_ci_high"], -0.11393345569025302)
        cef = [r for r in tem_dir if r["split"] == "test" and r["target"].startswith("cefotaxime")]
        assert len(cef) == 15
        assert all(float(r["delta_bon_minus_random_ci_low"]) <= 0 <= float(r["delta_bon_minus_random_ci_high"]) for r in cef)
        amp = [r for r in tem_dir if r["split"] == "test" and not r["target"].startswith("cefotaxime")]
        assert len(amp) == 60
    check("tem1_condition_dependence", tem_check)

    mrna_dir = SRC / "scenario_mrna_non_tcell_v1" / "artifact" / "results"
    decision_path = mrna_dir / "decision_summary.tsv"
    decision = load_tsv(decision_path)
    def mrna_check():
        stab = next(r for r in decision if r["pool"] == "stability_kmer_27" and r["context"] == "primary|integrated_abundance" and r["cap"] == "16")
        close(stab["bon_minus_random_n"], 0.283767186261695)
        assert float(stab["bon_minus_random_n"]) > 0
        indel = [r for r in decision if r["pool"] == "indel_kmer_48" and r["context"].startswith("primary|") and r["cap"] == "16"]
        assert len(indel) == 4
        assert any(float(r["bon_minus_random_n"]) > 0 for r in indel)
        assert any(float(r["bon_minus_random_n"]) < 0 for r in indel)
        metrics = load_json(mrna_dir / "unified_scenario_metrics.json")
        n50 = next(r for r in metrics if r["scenario"] == "n50_hek293t_r1" and r["cap"] == 16)
        hepg2 = next(r for r in metrics if r["scenario"] == "defined_hepg2_r1" and r["cap"] == 16)
        close(n50["bon_minus_random_n"], 0.1359392963)
        close(hepg2["bon_minus_random_n"], -1.0868857332)
        assert hepg2["holm_p"] > 0.05
    check("mrna_non_tcell_boundaries", mrna_check)

    def report_check():
        text = REPORT.read_text(encoding="utf-8")
        for token in ["−1.705760", "−0.160894", "−0.260830", "−4.206431", "−0.159379", "+0.2838", "Amp→cefotaxime", "production 不是 tropism", "lnW 不是 Kd"]:
            assert token in text, token
        assert "不得把原始效应量跨场景合并" in text
        assert "没有新前瞻湿实验" in text
    check("synthesis_report_boundaries", report_check)

    sources = [budget_path, gb1_path, gfp_path, aav_path, tem_overall_path, tem_dir_path, decision_path, mrna_dir / "unified_scenario_metrics.json", REPORT]
    payload = {
        "round": args.round,
        "status": "PASS",
        "checks": checks,
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "cross_scenario_pooled_effect_computed": False,
        "new_prospective_experiment": False,
    }
    out = DOCS / f"CROSS_SCENARIO_SYNTHESIS_VERIFICATION_20261128_ROUND{args.round}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"round": args.round, "status": "PASS", "checks": len(checks), "output": str(out.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()