#!/usr/bin/env python3
"""E6 GRIMM 五划分汇总：逐划分频率桶 Δmacro AUPRC(BCE−kNN)+CI+p 与 OOD 工作点。

只读取各划分 results_split{s}_convergence/summary.json（新协议：冻结 ESMC-300M、
960 维均值嵌入、BCE 线性头、validation<100 标签宏 AUPRC 早停、10000 次 bootstrap）。
五划分未全部完成时明确标注 incomplete，不写统一蛋白结论。
"""
import argparse
import json
import os

import pandas as pd

BUCKETS = ["tail_lt10", "tail_10_29", "mid_30_99", "head_ge100", "all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e6-root", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e6")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="1,2,3,4,5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    splits = [int(x) for x in args.splits.split(",")]
    bucket_rows, ood_rows, meta = [], [], {}
    missing = []
    for s in splits:
        path = os.path.join(args.e6_root, f"results_split{s}_convergence", "summary.json")
        if not os.path.exists(path):
            missing.append(s)
            continue
        d = json.load(open(path))
        flb = {b["bucket"]: b for b in d.get("function_label_bootstrap", [])}
        for b in BUCKETS:
            if b not in flb:
                continue
            x = flb[b]
            bucket_rows.append({"split": s, "bucket": b, "n_labels": x["n_labels"],
                                "delta_macro_auprc": x["delta_macro_auprc_neural_minus_knn"],
                                "ci_low": x["ci_low"], "ci_high": x["ci_high"],
                                "p_two_sided": x["p_two_sided"], "n_bootstrap": x["n_bootstrap"]})
        for m in d.get("ood_metrics", []):
            ood_rows.append({"split": s, "method": m["method"], "auroc_ood": m["auroc_ood"],
                             "fpr95": m["fpr95"], "aurc_ood_error": m["aurc_ood_error"]})
        fm = d.get("function_metrics", [])
        meta[s] = {"seeds": d.get("seeds"),
                   "best_epochs": sorted({m.get("best_epoch") for m in fm if m.get("model") in ("bce", "neural", "mlp")}),
                   "models": sorted({m.get("model") for m in fm}),
                   "selected_tau": d.get("selected_tau")}
    bdf = pd.DataFrame(bucket_rows)
    odf = pd.DataFrame(ood_rows)
    bdf.to_csv(os.path.join(args.out, "five_split_buckets.tsv"), sep="\t", index=False)
    odf.to_csv(os.path.join(args.out, "five_split_ood.tsv"), sep="\t", index=False)
    complete = not missing
    summary = {"complete": complete, "missing_splits": missing, "per_split_meta": meta}
    json.dump(summary, open(os.path.join(args.out, "five_split_summary.json"), "w"), indent=2)
    if not bdf.empty:
        pivot = bdf.pivot_table(index="bucket", columns="split", values="delta_macro_auprc")
        print("Δmacro AUPRC (BCE−kNN) by bucket × split:\n", pivot.round(5).to_string())
        print("\nOOD AUROC-OOD by method × split:\n",
              odf.pivot_table(index="method", columns="split", values="auroc_ood").round(5).to_string())
    print("\nCOMPLETE" if complete else f"\nINCOMPLETE missing={missing}")


if __name__ == "__main__":
    main()