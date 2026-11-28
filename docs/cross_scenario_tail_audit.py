from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "deliverables" / "reports" / "source_data"
OUT = ROOT / "deliverables" / "reports" / "source_data" / "cross_scenario_tail_audit_v1"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def rel(delta, random):
    return -float(delta) / float(random) * 100.0


rows = []

budget = load_json(SRC / "paper1_budget_matched_v1" / "results.json")
summaries = {(r["context"], r["cap"]): r for r in budget["context_summaries"]}
fam = {(r["context"], r["cap"]): r for r in budget["families"]["bon_minus_random_n"]}
tail = load_tsv(SRC / "tcell_error_diagnosis_v1" / "analysis" / "top_tail_summary.tsv")
point = load_tsv(SRC / "tcell_error_diagnosis_v1" / "analysis" / "pointwise_metrics.tsv")
for ctx, label in [("defined_tcell_r1", "T-cell MRL r1"), ("defined_tcell_r2", "T-cell MRL r2")]:
    r = fam[(ctx, 16)]
    s = summaries[(ctx, 16)]
    t = next(x for x in tail if x["context"] == ctx and x["model"] == "cnn" and x["cap"] == "16")
    p = next(x for x in point if x["context"] == ctx and x["model"] == "cnn")
    rows.append({
        "scenario": label, "family": "mRNA", "endpoint": "measured MRL",
        "global_spearman": float(p["spearman"]), "tail_metric": "top16 Spearman",
        "tail_value": float(t["spearman"]), "within_tail_loss_fraction": float(t["within_tail_ranking_loss"]) / float(t["regret"]),
        "relative_regret_reduction_pct": rel(r["effect"], s["random_n_regret"]),
        "cap16_delta": r["effect"], "ci95": r["nominal_95_ci"], "status": "positive" if r["holm_p"] < 0.05 else "nominal",
    })

gb1 = load_json(SRC / "scenario_gb1_v1" / "aggregate" / "test_aggregate.json")
b = gb1["budgets"]["16"]
rows.append({
    "scenario": "GB1 binding", "family": "protein", "endpoint": "IgG-Fc lnW",
    "global_spearman": 0.3478596529400488, "tail_metric": "true top-decile recall",
    "tail_value": b["top_decile_recall"], "within_tail_loss_fraction": None,
    "relative_regret_reduction_pct": rel(b["delta_regret_bon_minus_random"], b["random_regret"]),
    "cap16_delta": b["delta_regret_bon_minus_random"], "ci95": gb1["bootstrap"]["16"]["ci95"], "status": "positive",
})

gfp = load_json(SRC / "scenario_gfp_v1" / "results" / "primary.json")
methods = {m["method"]: m for m in gfp["methods"]}
g = next(r for r in methods["additive"]["primary"] if r["budget"] == 16)
rows.append({
    "scenario": "GFP brightness", "family": "protein", "endpoint": "log brightness",
    "global_spearman": methods["additive"]["point_metrics"]["spearman"], "tail_metric": "true top5% lift at B16",
    "tail_value": 10.47, "within_tail_loss_fraction": None,
    "relative_regret_reduction_pct": rel(g["mean_bon_minus_random_regret"], g["mean_random_regret"]),
    "cap16_delta": g["mean_bon_minus_random_regret"], "ci95": [g["ci95_bon_minus_random_regret"]["low"], g["ci95_bon_minus_random_regret"]["high"]], "status": "positive",
})

aav = load_json(SRC / "scenario_aav_v1" / "results" / "main_results.json")
a = aav["caps"]["16"]
rows.append({
    "scenario": "AAV9 production", "family": "gene therapy", "endpoint": "Production enrichment",
    "global_spearman": 0.820908, "tail_metric": "true top1% recall",
    "tail_value": 0.1875, "within_tail_loss_fraction": None,
    "relative_regret_reduction_pct": rel(a["paired_delta_bon_minus_random"], a["random_mean_regret"]),
    "cap16_delta": a["paired_delta_bon_minus_random"], "ci95": a["ci95_bootstrap_percentile"], "status": "positive",
})

tem = load_tsv(SRC / "scenario_tem1_v1" / "results" / "direction_results.tsv")
corr = load_tsv(SRC / "scenario_tem1_v1" / "results" / "correlations.tsv")
amp16 = [r for r in tem if r["split"] == "test" and r["N"] == "16" and not r["target"].startswith("cefotaxime")]
cef16 = [r for r in tem if r["split"] == "test" and r["N"] == "16" and r["target"].startswith("cefotaxime")]
amp_corr = [r for r in corr if r["split"] == "test" and not r["target"].startswith("cefotaxime")]
cef_corr = [r for r in corr if r["split"] == "test" and r["target"].startswith("cefotaxime")]
rows.append({
    "scenario": "TEM-1 Amp→Amp", "family": "enzyme", "endpoint": "relative resistance",
    "global_spearman": sum(float(r["spearman"]) for r in amp_corr) / len(amp_corr),
    "tail_metric": "source top10% Spearman", "tail_value": sum(float(r["source_top10_spearman"]) for r in amp_corr) / len(amp_corr),
    "within_tail_loss_fraction": None,
    "relative_regret_reduction_pct": rel(sum(float(r["delta_bon_minus_random"]) for r in amp16) / len(amp16), sum(float(r["random_n_regret"]) for r in amp16) / len(amp16)),
    "cap16_delta": sum(float(r["delta_bon_minus_random"]) for r in amp16) / len(amp16), "ci95": None, "status": "condition-dependent",
})
rows.append({
    "scenario": "TEM-1 Amp→cefotaxime", "family": "enzyme", "endpoint": "relative resistance",
    "global_spearman": sum(float(r["spearman"]) for r in cef_corr) / len(cef_corr),
    "tail_metric": "source top10% Spearman", "tail_value": sum(float(r["source_top10_spearman"]) for r in cef_corr) / len(cef_corr),
    "within_tail_loss_fraction": None,
    "relative_regret_reduction_pct": rel(sum(float(r["delta_bon_minus_random"]) for r in cef16) / len(cef16), sum(float(r["random_n_regret"]) for r in cef16) / len(cef16)),
    "cap16_delta": sum(float(r["delta_bon_minus_random"]) for r in cef16) / len(cef16), "ci95": None, "status": "null",
})

mrna = load_tsv(SRC / "scenario_mrna_non_tcell_v1" / "artifact" / "results" / "decision_summary.tsv")
diag = load_json(SRC / "scenario_mrna_non_tcell_v1" / "artifact" / "results" / "decision_diagnostics.json")
metrics = load_json(SRC / "scenario_mrna_non_tcell_v1" / "artifact" / "results" / "unified_scenario_metrics.json")
for scenario, label in [("n50_hek293t_r1", "HEK293T N50 MRL"), ("defined_hepg2_r1", "HepG2 defined-end MRL")]:
    m = next(r for r in metrics if r["scenario"] == scenario and r["cap"] == 16)
    rows.append({
        "scenario": label, "family": "mRNA", "endpoint": "measured MRL",
        "global_spearman": m["spearman"], "tail_metric": "true top-decile recall",
        "tail_value": m["top_decile_recall"], "within_tail_loss_fraction": None,
        "relative_regret_reduction_pct": rel(m["bon_minus_random_n"], m["random_n_regret"]),
        "cap16_delta": m["bon_minus_random_n"], "ci95": m["ci95"], "status": "null" if scenario.startswith("n50") else "nominal",
    })
stab = next(r for r in mrna if r["pool"] == "stability_kmer_27" and r["context"] == "primary|integrated_abundance" and r["cap"] == "16")
stab_diag = next(r for r in diag if r["pool"] == "stability_kmer_27" and r["context"] == "primary|integrated_abundance")
rows.append({
    "scenario": "mRNA stability", "family": "mRNA", "endpoint": "integrated abundance",
    "global_spearman": stab_diag["spearman"], "tail_metric": "true top-decile recall",
    "tail_value": stab_diag["top_decile_recall"], "within_tail_loss_fraction": None,
    "relative_regret_reduction_pct": rel(stab["bon_minus_random_n"], stab["random_n_regret"]),
    "cap16_delta": float(stab["bon_minus_random_n"]), "ci95": [0.0824, 0.4435], "status": "harmful",
})
indel = [r for r in mrna if r["pool"] == "indel_kmer_48" and r["context"].startswith("primary|") and r["cap"] == "16"]
indel_diag = [r for r in diag if r["pool"] == "indel_kmer_48" and r["context"].startswith("primary|")]
rows.append({
    "scenario": "Castillo indel", "family": "mRNA", "endpoint": "normalized indel",
    "global_spearman": sum(r["spearman"] for r in indel_diag) / len(indel_diag),
    "tail_metric": "true top-decile recall", "tail_value": sum(r["top_decile_recall"] for r in indel_diag) / len(indel_diag),
    "within_tail_loss_fraction": None,
    "relative_regret_reduction_pct": None,
    "cap16_delta": None, "ci95": None, "status": "mixed/null",
})

payload = {
    "analysis": "cross_scenario_tail_audit_v1",
    "purpose": "Unify scenarios by tail reliability and decision benefit, not by pooled raw effect size.",
    "rows": rows,
    "interpretation": "Positive decision benefit coexists with heterogeneous global correlation; failures show low or negative tail reliability. Metrics are task-specific and not pooled inferentially.",
    "retrospective": True,
}
(OUT / "tail_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with (OUT / "tail_audit.tsv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"rows": len(rows), "output": str(OUT.relative_to(ROOT))}, ensure_ascii=False))