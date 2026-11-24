#!/usr/bin/env python3
"""E0 汇总报告：聚合全部 E0 产出 → E0_report.md（数据卡 v1.0 与 G0 判决的数字骨架）。"""
import json, os
import numpy as np
import pandas as pd

E = "/root/rareseq/data/e0"
OUT = "/root/rareseq/data/e0/E0_report.md"
L = []
def p(s=""):
    L.append(s)

def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None

def tsv(path):
    return pd.read_csv(path, sep="\t") if os.path.exists(path) else None

p("# E0 数据审计汇总报告（自动生成）")
p()
p("## 1. 窗口提取")
wt = jload(f"{E}/windows/window_totals.json")
if wt:
    p(f"- 输入位点 {wt['n_input']:,}；唯一 {wt['n_unique']:,}；去重 {wt['n_dup_removed']:,}")
    p(f"- 染色体边界截断 {wt['n_clipped']:,}；非 ACGU 跳过 {wt['n_nonACGU_skipped']:,}；最终窗口 {wt['n_final']:,}")
ws = tsv(f"{E}/windows/window_stats.tsv")
if ws is not None:
    p(f"- 实验数 {len(ws)}；截断窗口占比 {ws.n_clipped.sum()/max(ws.n_input.sum(),1):.4%}")
p()
p("## 2. 划分冻结")
sm = jload(f"{E}/splits/split_manifest.id90.json")
if sm:
    p(f"- 聚类窗口 {sm['n_windows_clustered']:,} → 簇 {sm['n_clusters']:,}（压缩比 {sm['n_clusters']/max(sm['n_windows_clustered'],1):.3f}）")
    p(f"- 区域 {sm['n_regions']:,} → 连通分量 {sm['n_components']:,}；跨区链接 {sm['n_span_links']:,}")
    p(f"- 折区域数 {sm['fold_region_counts']}；折窗口数 {sm.get('fold_window_counts')}")
    p(f"- CD-HIT 用时 {sm['cluster_seconds']}s；冻结时间 {sm['frozen_utc']}")
cs = tsv(f"{E}/leakage/cluster_stats.tsv")
if cs is not None:
    p(f"- n_eff/n_pos 均值 {cs.n_eff_ratio.mean():.3f}；最小 {cs.n_eff_ratio.min():.3f}（{cs.loc[cs.n_eff_ratio.idxmin(),'experiment']}）")
p()
p("## 3. 负例三轨")
ns = tsv(f"{E}/neg/negatives_stats.tsv")
if ns is not None:
    agg = ns.groupby("track")[["n_pos", "n_sampled", "n_nonACGU", "n_final"]].sum()
    p(agg.to_markdown())
na = tsv(f"{E}/neg/neg_audit.tsv")
if na is not None:
    p(f"- **neg1f 落入正例±300nt 邻域比例：均值 {na.neg1f_in_ext300_frac.mean():.4f}**（no-op 泄漏量化）")
    p(f"- neg1f 到最近正例距离中位数（中位）{na.neg1f_dist_median.median():.0f}nt；≤100nt 比例均值 {na.neg1f_dist_frac_le100.mean():.4f}")
p()
p("## 4. OverfitNN vs 冻结探针（G0 参照）")
ov = tsv(f"{E}/g0/overfitnn_results.tsv")
pr_pilot = tsv(f"{E}/g0/probe_results.tsv")
pr_main = tsv(f"{E}/g0_mainfold0/probe_results.tsv")
if ov is not None:
    ovs = ov.groupby(["track", "feature", "k"])[["auroc", "auprc"]].mean().round(4)
    p("### OverfitNN 全主集（5 折均值）")
    p(ovs.to_markdown())
if pr_pilot is not None:
    p("### 冻结探针 pilot（20 RBP，5 折均值）")
    p(pr_pilot.groupby("track")[["auroc", "auprc"]].mean().round(4).to_markdown())
if pr_main is not None:
    p("### 冻结探针主集 fold0")
    p(pr_main.groupby("track")[["auroc", "auprc"]].mean().round(4).to_markdown())
# G0 对照表: 每轨 best-OverfitNN vs probe
if ov is not None and pr_pilot is not None:
    p("### G0 对照（pilot，每轨 OverfitNN 最优配置 vs 探针）")
    rows = []
    for tr in sorted(set(ov.track) & set(pr_pilot.track)):
        o = ov[ov.track == tr].groupby(["feature", "k"])[["auroc", "auprc"]].mean()
        ob = o.auroc.idxmax()
        oa = o.loc[ob, "auroc"]; op = o.loc[ob, "auprc"]
        pp = pr_pilot[pr_pilot.track == tr][["auroc", "auprc"]].mean()
        rows.append({"track": tr, "ov_feat_k": f"{ob[0]}/k{ob[1]}",
                     "ov_auroc": round(oa, 4), "probe_auroc": round(pp.auroc, 4),
                     "d_auroc": round(pp.auroc - oa, 4),
                     "ov_auprc": round(op, 4), "probe_auprc": round(pp.auprc, 4),
                     "d_auprc": round(pp.auprc - op, 4)})
    p(pd.DataFrame(rows).to_markdown(index=False))
    p("判读：d>0.02（探针显著高于 OverfitNN）→ 信号非纯记忆；d<0.01–0.02 且处处如此 → G0 触发。")
# 尾/头分层
pa = tsv(f"{E}/positives_audit.tsv")
if ov is not None and pa is not None:
    per_rbp = pa.groupby("rbp")["n_sites"].sum()
    q1, q2 = per_rbp.quantile([1/3, 2/3])
    tier = {r: ("tail" if v <= q1 else ("head" if v > q2 else "mid")) for r, v in per_rbp.items()}
    ov2 = ov.copy()
    ov2["rbp"] = ov2.experiment.str.rsplit("_", n=1).str[0]
    ov2["tier"] = ov2.rbp.map(tier)
    p("### OverfitNN 按 RBP 峰数三分层（emb 特征，k=10，AUROC）")
    sub = ov2[(ov2.feature == "emb") & (ov2.k == 10)]
    p(sub.pivot_table(index="tier", columns="track", values="auroc", aggfunc="mean").round(4).to_markdown())
    if pr_pilot is not None:
        pr2 = pr_pilot.copy()
        pr2["rbp"] = pr2.experiment.str.rsplit("_", n=1).str[0]
        pr2["tier"] = pr2.rbp.map(tier)
        p("### 冻结探针 pilot 按三分层（AUROC）")
        p(pr2.pivot_table(index="tier", columns="track", values="auroc", aggfunc="mean").round(4).to_markdown())
p()
p("## 5. 四通道泄漏量化")
ls = jload(f"{E}/leakage/leakage_summary.json")
if ls:
    p("```json")
    p(json.dumps(ls, ensure_ascii=False, indent=1))
    p("```")
nb = tsv(f"{E}/leakage/nn_identity_buckets.tsv")
if nb is not None:
    fr = [c for c in nb.columns if c.startswith("frac_")]
    p("### NN identity 桶（全主集均值）")
    p(nb.groupby("track")[fr + ["nn_identity_mean"]].mean().round(4).to_markdown())
p()
p("## 6. 全量窗口 d_hom 分布")
ids = jload(f"{E}/identity/identity_summary.json")
if ids:
    p("```json")
    p(json.dumps(ids, indent=1))
    p("```")
p()
p("## 7. RBP 家族冻结")
fs = jload(f"{E}/families/family_summary.json")
if fs:
    p(f"- 阈值 {fs['threshold']}（silhouette {fs['silhouette']}）；{fs['n_rbps']} RBP → {fs['n_families']} 家族")
    p(f"- 家族大小 {fs['family_sizes']}")
p()
p("## 8. 吞吐（实测）")
bf = jload(f"{E}/bench_fm.json")
if bf:
    p("```json")
    p(json.dumps(bf, indent=1)[:1500])
    p("```")
p()
open(OUT, "w").write("\n".join(L))
print(f"report written: {OUT} ({len(L)} lines)")