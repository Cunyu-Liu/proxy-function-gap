#!/usr/bin/env python3
"""Extract frozen-artifact numbers for the V10 (v8) figure set."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "deliverables" / "reports" / "source_data"
OUT = Path(__file__).resolve().parent


def jload(p):
    return json.loads((SRC / p).read_text())


def tsv(p):
    with (SRC / p).open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


data = {}

# Fig 1: T-cell equal-budget value
budget = jload("paper1_budget_matched_v1/results.json")
fam = {(r["context"], r["cap"]): r for r in budget["families"]["bon_minus_random_n"]}
summ = {(r["context"], r["cap"]): r for r in budget["context_summaries"]}
tcell = {}
for ctx in ["defined_tcell_r1", "defined_tcell_r2"]:
    caps = []
    for cap in [2, 4, 8, 16]:
        f = fam[(ctx, cap)]; s = summ[(ctx, cap)]
        caps.append({"cap": cap, "effect": f["effect"], "ci": f["nominal_95_ci"],
                     "holm_p": f["holm_p"], "bon_regret": s["bon_regret"],
                     "random_n_regret": s["random_n_regret"]})
    tcell[ctx] = caps
data["tcell_equal_budget"] = tcell

# Fig 2: tail bottleneck
tail = tsv("tcell_error_diagnosis_v1/analysis/top_tail_summary.tsv")
point = tsv("tcell_error_diagnosis_v1/analysis/pointwise_metrics.tsv")
bottleneck = {}
for ctx in ["defined_tcell_r1", "defined_tcell_r2"]:
    t = next(x for x in tail if x["context"] == ctx and x["model"] == "cnn" and x["cap"] == "16")
    p = next(x for x in point if x["context"] == ctx and x["model"] == "cnn")
    bottleneck[ctx] = {
        "global_spearman": float(p["spearman"]), "global_mae": float(p["mae"]),
        "top16_spearman": float(t["spearman"]), "regret": float(t["regret"]),
        "opportunity_loss": float(t["opportunity_loss"]),
        "within_tail_ranking_loss": float(t["within_tail_ranking_loss"]),
        "winner_flipped_within_tail": float(t["winner_flipped_within_tail"]),
        "full_oracle_mrl": float(t["full_oracle_mrl"]), "tail_oracle_mrl": float(t["tail_oracle_mrl"]),
    }
data["tcell_bottleneck"] = bottleneck

# Fig 3: regression-decision dissociation
adapt = jload("tcell_target_adaptation_v1/target/primary.json")
adapt_sec = jload("tcell_target_adaptation_v1/target/secondary.json")
mae = {}
for ctx in ["defined_tcell_r1", "defined_tcell_r2"]:
    pe = adapt_sec["pointwise_errors_mrl_units"][ctx]
    mae[ctx] = {"original_mae": pe["original_cnn"]["mae"], "adapted_mae": pe["main_50k"]["mae"],
                "original_spearman": pe["original_cnn"]["spearman"], "adapted_spearman": pe["main_50k"]["spearman"]}
data["adaptation"] = {
    "mae": mae,
    "repeat_regret_delta": adapt["repeat_main_minus_original_effects"],
    "combined": adapt["combined_main_minus_original"],
    "success": adapt["success"],
}
listnet = jload("tcell_listnet_adaptation_v2/target_v2/primary.json")
data["listnet"] = {k: {"effect": v["effect"], "baseline_regret": v["baseline_cap16_regret"],
                       "min_reduction": v["minimum_required_absolute_reduction"]}
                   for k, v in listnet["comparisons"].items()}
anchor = jload("tcell_cross_cell_anchor_v1/target/primary.json")
data["anchor"] = {"combined": anchor["combined_anchor_minus_cnn"],
                  "repeat": anchor["repeat_anchor_minus_cnn_effects"],
                  "cnn_baseline_regret": anchor["combined_cnn_baseline_regret"]}

# Fig 4: UPTS gate
data["upts"] = jload("tcell_upts_v1/artifact/auxiliary_gate.json")
upts_sel = {}
for d in ["n50_hek293t_r1_to_defined_hepg2_r1", "defined_hepg2_r1_to_n50_hek293t_r1"]:
    sel_path = SRC / "tcell_upts_v1/artifact/auxiliary" / d / "selection.json"
    if sel_path.exists():
        upts_sel[d] = json.loads(sel_path.read_text())
data["upts_selection"] = upts_sel

# Fig 5/6: cross-scenario audit + association
data["tail_audit_v2"] = jload("cross_scenario_tail_audit_v2/tail_audit_v2.json")["rows"]
data["association"] = jload("cross_scenario_tail_audit_v2/association_results.json")

(OUT / "figures_v8_data.json").write_text(json.dumps(data, indent=2) + "\n")
print("wrote figures_v8_data.json")
print("tcell r1 cap16:", tcell["defined_tcell_r1"][-1])
print("bottleneck r1:", bottleneck["defined_tcell_r1"])
print("upts_selection keys:", list(upts_sel))