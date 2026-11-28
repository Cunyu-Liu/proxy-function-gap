from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "deliverables" / "reports" / "source_data"
FIGURES = ["Figure18_v5_cross_scenario_functional_value", "Figure19_v5_regression_decision_dissociation"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def numeric_checks():
    checks = []
    budget = load_json(SRC / "paper1_budget_matched_v1" / "results.json")
    fam = {(r["context"], r["cap"]): r for r in budget["families"]["bon_minus_random_n"]}
    assert abs(fam[("defined_tcell_r1", 16)]["effect"] + 1.7057596554698837) < 1e-12
    assert abs(fam[("defined_tcell_r2", 16)]["effect"] + 1.086940681749869) < 1e-12
    checks.append("tcell_effects")
    gb1 = load_json(SRC / "scenario_gb1_v1" / "aggregate" / "test_aggregate.json")
    assert abs(gb1["budgets"]["16"]["delta_regret_bon_minus_random"] + 0.16089381057826665) < 1e-12
    assert gb1["label"] == "lnW"
    checks.append("gb1_lnw")
    gfp = load_json(SRC / "scenario_gfp_v1" / "results" / "primary.json")
    methods = {m["method"]: m for m in gfp["methods"]}
    g = next(r for r in methods["additive"]["primary"] if r["budget"] == 16)
    assert g["mean_bon_minus_random_regret"] < 0 and g["ci95_bon_minus_random_regret"]["high"] < 0
    assert next(r for r in methods["cnn_ensemble"]["point_metrics_by_order"] if r["order"] == 9)["spearman"] < 0
    checks.append("gfp_order_decay_and_selection")
    aav = load_json(SRC / "scenario_aav_v1" / "results" / "main_results.json")
    assert abs(aav["caps"]["16"]["paired_delta_bon_minus_random"] + 4.206431097252901) < 1e-12
    assert not aav["meeting6_used"] and "not tropism" in aav["endpoint_semantics"]
    checks.append("aav_production_semantics")
    tem = load_tsv(SRC / "scenario_tem1_v1" / "results" / "direction_results.tsv")
    cef = [r for r in tem if r["split"] == "test" and r["N"] == "16" and r["target"].startswith("cefotaxime")]
    assert len(cef) == 5 and all(float(r["delta_bon_minus_random_ci_low"]) <= 0 <= float(r["delta_bon_minus_random_ci_high"]) for r in cef)
    checks.append("tem1_cross_drug_null")
    mrna = load_tsv(SRC / "scenario_mrna_non_tcell_v1" / "artifact" / "results" / "decision_summary.tsv")
    stab = next(r for r in mrna if r["pool"] == "stability_kmer_27" and r["context"] == "primary|integrated_abundance" and r["cap"] == "16")
    assert float(stab["bon_minus_random_n"]) > 0
    checks.append("stability_harm")
    return checks


def rebuild():
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "987654321"
    subprocess.run([sys.executable, str(HERE / "build_cross_scenario_figures.py")], check=True, env=env, cwd=ROOT)


def main():
    before = {f"{name}.{ext}": sha(HERE / f"{name}.{ext}") for name in FIGURES for ext in ["png", "pdf", "svg"]}
    checks = numeric_checks()
    rebuild()
    after = {f"{name}.{ext}": sha(HERE / f"{name}.{ext}") for name in FIGURES for ext in ["png", "pdf", "svg"]}
    assert before == after, [k for k in before if before[k] != after[k]]
    import fitz
    formats = {}
    for name in FIGURES:
        png = HERE / f"{name}.png"
        pdf = HERE / f"{name}.pdf"
        svg = HERE / f"{name}.svg"
        doc = fitz.open(pdf)
        assert len(doc) == 1
        text = svg.read_text(encoding="utf-8")
        assert "<text" in text or "font-family" in text
        formats[name] = {"png_bytes": png.stat().st_size, "pdf_pages": len(doc), "svg_bytes": svg.stat().st_size}
        doc.close()
    report = {
        "overall": "PASS",
        "numeric_checks": checks,
        "deterministic_rebuild": True,
        "formats": formats,
        "source_hashes": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                SRC / "paper1_budget_matched_v1" / "results.json",
                SRC / "scenario_gb1_v1" / "aggregate" / "test_aggregate.json",
                SRC / "scenario_gfp_v1" / "results" / "primary.json",
                SRC / "scenario_aav_v1" / "results" / "main_results.json",
                SRC / "scenario_tem1_v1" / "results" / "direction_results.tsv",
                SRC / "scenario_mrna_non_tcell_v1" / "artifact" / "results" / "decision_summary.tsv",
            ]
        },
    }
    (HERE / "cross_scenario_v5_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overall": "PASS", "figures": FIGURES}, ensure_ascii=False))


if __name__ == "__main__":
    main()