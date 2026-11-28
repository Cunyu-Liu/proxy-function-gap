#!/usr/bin/env python3
"""Label distribution summaries for Extended Data figure (CPU, frozen artifacts)."""
import gzip, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew

BASE = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap")
OUT = BASE / "cross_scenario_tail_audit_v2"


def summ(y, name, unit):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    q = np.percentile(y, [0, 5, 25, 50, 75, 95, 100])
    return {"scenario": name, "unit": unit, "n": int(len(y)), "mean": float(y.mean()),
            "sd": float(y.std()), "skew": float(skew(y)),
            "min": q[0], "q5": q[1], "q25": q[2], "median": q[3], "q75": q[4], "q95": q[5], "max": q[6]}


rows = []
# T-cell
for ctx, lab in [("defined_tcell_r1", "T-cell MRL r1"), ("defined_tcell_r2", "T-cell MRL r2")]:
    pr = BASE / "data/derived/castillo_fixed_pools_v3_strict_h1_retry1" / ctx
    labels = pd.read_csv(pr / "evaluator/labels.tsv.gz", sep="\t")
    rows.append(summ(labels["measured_rl"], lab, "measured MRL"))
# GB1
d = np.load(BASE / "scenario_gb1_v1/predictions/test_predictions.npz", allow_pickle=True)
lab = pd.read_csv(BASE / "scenario_gb1_v1/data/evaluator/test_labels.csv")
j = pd.DataFrame({"mutant": d["mutants"]}).merge(lab, on="mutant")
rows.append(summ(j["DMS_score"], "GB1 binding", "IgG-Fc lnW"))
# GFP
d = np.load(BASE / "scenario_gfp_v1/predictions/additive.npz", allow_pickle=True)
lab = pd.read_csv(BASE / "scenario_gfp_v1/evaluator/high_order_labels.csv")
j = pd.DataFrame({"variant_id": d["variant_id"]}).merge(lab, on="variant_id")
rows.append(summ(j["DMS_score"], "GFP brightness", "log brightness"))
# AAV
d = np.load(BASE / "scenario_aav_v1/evaluation/test_predictions.npz", allow_pickle=True)
rows.append(summ(d["measured_production"], "AAV9 production", "Production enrichment"))
# mRNA stability / indel from candidate_master
cm = pd.read_csv(BASE / "scenario_mrna_non_tcell_v1/results/candidate_master.tsv", sep="\t")
rows.append(summ(cm["integrated_abundance"].dropna(), "mRNA stability", "integrated abundance"))
# TEM-1 cefotaxime target
f = pd.read_csv(BASE / "scenario_tem1_v1/raw/stiffler2015_f.csv", usecols=["hgvs_pro", "score"])
rows.append(summ(pd.to_numeric(f["score"], errors="coerce"), "TEM-1 cefotaxime", "relative resistance"))

(OUT / "label_distributions.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps([{k: r[k] for k in ["scenario", "n", "mean", "sd", "skew", "median"]} for r in rows], indent=1))