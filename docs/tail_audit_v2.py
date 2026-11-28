#!/usr/bin/env python3
"""Cross-scenario tail audit v2: unified tail-reliability metrics.

Computes a SINGLE metric definition at the global evaluation-population level
for every scenario, replacing the heterogeneous task-specific tail metrics of
v1. Pure CPU statistics on frozen artifacts; no training, no inference, no
modification of any frozen test. Retrospective.

Unified definition (per scenario, over all evaluation candidates):
  k = max(3, ceil(0.10 * N))                      (top decile)
  global_spearman        = Spearman(score, label) over all N
  tail_recall_decile     = |pred_top_k ∩ true_top_k| / k
  tail_spearman_pred_dec = Spearman(score, label) within predicted top-k
  tail_spearman_true_dec = Spearman(score, label) within true top-k
  top1pct_recall         = |pred_top_k1 ∩ true_top_k1| / k1, k1=max(1,ceil(0.01N))
Ties broken by ascending candidate index (deterministic).
"""
from __future__ import annotations
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap")
OUT = BASE / "cross_scenario_tail_audit_v2"
OUT.mkdir(parents=True, exist_ok=True)
V1 = BASE / "cross_scenario_tail_audit_v1" / "tail_audit.json"
INPUTS = []


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def register(path: Path):
    INPUTS.append({"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size})


def unified(score, label) -> dict:
    s = np.asarray(score, float)
    y = np.asarray(label, float)
    mask = np.isfinite(s) & np.isfinite(y)
    s, y = s[mask], y[mask]
    n = len(s)
    idx = np.arange(n)

    def topk(values, frac, min_k):
        k = max(min_k, math.ceil(frac * n))
        order = np.lexsort((idx, -values))[:k]
        return order, k

    pred_idx, k = topk(s, 0.10, 3)
    true_idx, _ = topk(y, 0.10, 3)
    recall = len(set(pred_idx.tolist()) & set(true_idx.tolist())) / k
    pred1, k1 = topk(s, 0.01, 1)
    true1, _ = topk(y, 0.01, 1)
    recall1 = len(set(pred1.tolist()) & set(true1.tolist())) / k1

    def sp(ix):
        if len(ix) >= 3 and np.std(s[ix]) > 0 and np.std(y[ix]) > 0:
            return float(spearmanr(s[ix], y[ix]).statistic)
        return float("nan")

    return {
        "n": int(n), "k_decile": int(k),
        "global_spearman": float(spearmanr(s, y).statistic),
        "tail_recall_decile": float(recall),
        "tail_spearman_pred_decile": sp(pred_idx),
        "tail_spearman_true_decile": sp(true_idx),
        "top1pct_recall": float(recall1),
    }


def load_tcell(context: str) -> dict:
    pool_root = BASE / "data/derived/castillo_fixed_pools_v3_strict_h1_retry1" / context
    result_root = BASE / "paper1_v3_strict_h1_retry1" / context
    labels_path = pool_root / "evaluator/labels.tsv.gz"
    register(labels_path)
    labels = pd.read_csv(labels_path, sep="\t")
    label_map = dict(zip(labels["sequence_id"], labels["measured_rl"].astype(float)))
    scores, ys = [], []
    for path in sorted((pool_root / "learner").glob("*.jsonl.gz")):
        register(path)
        with gzip.open(path, "rt") as f:
            learner = [json.loads(line) for line in f]
        pool_id = learner[0]["pool_id"]
        cnn_path = result_root / "cnn_scores" / f"{pool_id}.json"
        register(cnn_path)
        cnn = {r["sequence_id"]: r for r in json.loads(cnn_path.read_text())["records"]}
        for base in learner:
            sid = base["sequence_id"]
            scores.append(float(cnn[sid]["proxy_mean"]))
            ys.append(float(label_map[sid]))
    out = unified(scores, ys)
    out["proxy"] = "cnn_ensemble_proxy_mean"
    out["source"] = "paper1_v3_strict_h1_retry1 + castillo_fixed_pools_v3_strict_h1_retry1"
    return out


def load_gfp() -> dict:
    pred_path = BASE / "scenario_gfp_v1/predictions/additive.npz"
    lab_path = BASE / "scenario_gfp_v1/evaluator/high_order_labels.csv"
    register(pred_path); register(lab_path)
    d = np.load(pred_path, allow_pickle=True)
    pred = pd.DataFrame({"variant_id": d["variant_id"], "prediction": d["prediction"]})
    lab = pd.read_csv(lab_path)
    j = pred.merge(lab, on="variant_id", how="inner")
    assert len(j) == len(pred), "GFP label join lost rows"
    out = unified(j["prediction"].to_numpy(), j["DMS_score"].to_numpy())
    out["proxy"] = "additive_ridge"
    out["source"] = "scenario_gfp_v1/predictions/additive.npz + evaluator/high_order_labels.csv"
    return out


def load_gb1() -> dict:
    pred_path = BASE / "scenario_gb1_v1/predictions/test_predictions.npz"
    lab_path = BASE / "scenario_gb1_v1/data/evaluator/test_labels.csv"
    register(pred_path); register(lab_path)
    d = np.load(pred_path, allow_pickle=True)
    mean_score = d["scores"].mean(axis=0)
    pred = pd.DataFrame({"mutant": d["mutants"], "prediction": mean_score})
    lab = pd.read_csv(lab_path)
    j = pred.merge(lab, on="mutant", how="inner")
    assert len(j) == len(pred), "GB1 label join lost rows"
    out = unified(j["prediction"].to_numpy(), j["DMS_score"].to_numpy())
    out["proxy"] = "additive_ridge_seed_mean"
    out["source"] = "scenario_gb1_v1/predictions/test_predictions.npz + data/evaluator/test_labels.csv"
    return out


def load_aav() -> dict:
    pred_path = BASE / "scenario_aav_v1/evaluation/test_predictions.npz"
    register(pred_path)
    d = np.load(pred_path, allow_pickle=True)
    out = unified(d["scores"], d["measured_production"])
    out["proxy"] = "cnn_ensemble_scores"
    out["source"] = "scenario_aav_v1/evaluation/test_predictions.npz"
    return out


def load_mrna() -> dict:
    path = BASE / "scenario_mrna_non_tcell_v1/results/unified_scenario_metrics.json"
    register(path)
    metrics = json.loads(path.read_text())
    rows = {}
    for scenario in ["n50_hek293t_r1", "defined_hepg2_r1", "stability_kmer_27", "indel_kmer_48"]:
        sel = [r for r in metrics if r["scenario"] == scenario and r["cap"] == 16]
        assert sel, f"missing mRNA scenario {scenario}"
        recall = float(np.mean([r["top_decile_recall"] for r in sel]))
        spear = float(np.mean([r["spearman"] for r in sel]))
        n = int(sel[0]["n_candidates"])
        rows[scenario] = {
            "n": n, "k_decile": max(3, math.ceil(0.10 * n)),
            "global_spearman": spear, "tail_recall_decile": recall,
            "tail_spearman_pred_decile": float("nan"),
            "tail_spearman_true_decile": float("nan"),
            "top1pct_recall": float("nan"),
            "proxy": "frozen_CNN/kmer_mlp", "n_contexts": len(sel),
            "source": "scenario_mrna_non_tcell_v1/results/unified_scenario_metrics.json (stored frozen global top-decile recall)",
        }
    return rows


def load_tem1() -> dict:
    proto_path = BASE / "scenario_tem1_v1/source_snapshot/frozen_protocol.json"
    if not proto_path.exists():
        proto_path = BASE / "scenario_tem1_v1/frozen_protocol.json"
    register(proto_path)
    proto = json.loads(proto_path.read_text())
    cond_files = proto["data"]["conditions"]
    raw_dir = BASE / "scenario_tem1_v1/raw"
    scores = {}
    for name, fn in cond_files.items():
        p = raw_dir / fn
        register(p)
        f = pd.read_csv(p, usecols=["hgvs_pro", "score"], dtype={"hgvs_pro": str})
        f["score"] = pd.to_numeric(f["score"], errors="coerce")
        scores[name] = f.set_index("hgvs_pro")["score"]
    frame = pd.DataFrame(scores)
    frame["hgvs_pro"] = frame.index
    frame["position"] = frame["hgvs_pro"].str.extract(r"p\.[A-Za-z*]{3}(\d+)", expand=False).astype(int)
    positions = np.array(sorted(frame["position"].unique()))
    seed = proto["split"]["seed"]
    ordered = sorted(positions, key=lambda p: hashlib.sha256(f"scenario_tem1_v1|split|{seed}|{p}".encode()).hexdigest())
    n_train = int(len(ordered) * proto["split"]["train_fraction"])
    n_dev = int(len(ordered) * proto["split"]["dev_fraction"])
    split_map = {p: "train" for p in ordered[:n_train]}
    split_map.update({p: "dev" for p in ordered[n_train:n_train + n_dev]})
    split_map.update({p: "test" for p in ordered[n_train + n_dev:]})
    frame["split"] = frame["position"].map(split_map)
    test = frame[frame["split"] == "test"].copy()
    conditions = list(cond_files)
    per_dir = []
    for source in proto["directions"]["sources"]:
        for target in proto["directions"]["targets"]:
            if source == target:
                continue
            paired = test[[source, target]].dropna()
            m = unified(paired[source].to_numpy(), paired[target].to_numpy())
            m["source"] = source; m["target"] = target
            per_dir.append(m)
    groups = {}
    for gname, pred in [("amp", lambda t: not t.startswith("cefotaxime")),
                        ("cefotaxime", lambda t: t.startswith("cefotaxime"))]:
        sel = [m for m in per_dir if pred(m["target"])]
        agg = {"n_directions": len(sel)}
        for key in ["global_spearman", "tail_recall_decile", "tail_spearman_pred_decile", "tail_spearman_true_decile", "top1pct_recall"]:
            agg[key] = float(np.nanmean([m[key] for m in sel]))
        agg["n"] = int(np.mean([m["n"] for m in sel]))
        agg["k_decile"] = int(np.mean([m["k_decile"] for m in sel]))
        agg["proxy"] = "source_condition_fitness"
        agg["source"] = "scenario_tem1_v1/raw/stiffler2015_*.csv + frozen_protocol.json (test split, recomputed)"
        groups[gname] = agg
    return groups


def main():
    detail = {}
    detail["T-cell MRL r1"] = load_tcell("defined_tcell_r1")
    detail["T-cell MRL r2"] = load_tcell("defined_tcell_r2")
    detail["GB1 binding"] = load_gb1()
    detail["GFP brightness"] = load_gfp()
    detail["AAV9 production"] = load_aav()
    mrna = load_mrna()
    detail["HEK293T N50 MRL"] = mrna["n50_hek293t_r1"]
    detail["HepG2 defined-end MRL"] = mrna["defined_hepg2_r1"]
    detail["mRNA stability"] = mrna["stability_kmer_27"]
    detail["Castillo indel"] = mrna["indel_kmer_48"]
    tem = load_tem1()
    detail["TEM-1 Amp→Amp"] = tem["amp"]
    detail["TEM-1 Amp→cefotaxime"] = tem["cefotaxime"]

    v1 = {r["scenario"]: r for r in json.loads(V1.read_text())["rows"]}
    register(V1)
    rows = []
    for scenario, m in detail.items():
        base = v1[scenario]
        row = {
            "scenario": scenario, "family": base["family"], "endpoint": base["endpoint"],
            "status": base["status"],
            "relative_regret_reduction_pct": base["relative_regret_reduction_pct"],
            "cap16_delta": base["cap16_delta"], "ci95": base["ci95"],
            "v1_global_spearman": base["global_spearman"],
            "v1_tail_metric": base["tail_metric"], "v1_tail_value": base["tail_value"],
            "n_eval": m["n"], "k_decile": m["k_decile"],
            "global_spearman": m["global_spearman"],
            "tail_recall_decile": m["tail_recall_decile"],
            "tail_spearman_pred_decile": m["tail_spearman_pred_decile"],
            "tail_spearman_true_decile": m["tail_spearman_true_decile"],
            "top1pct_recall": m["top1pct_recall"],
            "proxy": m["proxy"], "metric_source": m["source"],
        }
        rows.append(row)

    payload = {
        "analysis": "cross_scenario_tail_audit_v2",
        "purpose": "Unified tail-reliability metrics (single definition, global evaluation population) to support a pre-registered cross-scenario association test.",
        "unified_definition": "k=max(3,ceil(0.1N)); tail_recall_decile=|pred_top_k ∩ true_top_k|/k; tail_spearman_pred/true=Spearman within predicted/true top-k; global top-decile.",
        "retrospective": True,
        "no_frozen_test_modified": True,
        "rows": rows,
    }
    (OUT / "tail_audit_v2.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "tail_audit_v2.tsv", sep="\t", index=False)
    (OUT / "input_hashes.json").write_text(json.dumps(INPUTS, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "inputs": len(INPUTS), "out": str(OUT)}, ensure_ascii=False))
    for r in rows:
        print(f"{r['scenario']:28s} N={r['n_eval']:6d} recall={r['tail_recall_decile']:.4f} "
              f"sp_pred={r['tail_spearman_pred_decile']:.4f} sp_true={r['tail_spearman_true_decile']:.4f} "
              f"global={r['global_spearman']:.4f} reduction={r['relative_regret_reduction_pct']}")


if __name__ == "__main__":
    main()