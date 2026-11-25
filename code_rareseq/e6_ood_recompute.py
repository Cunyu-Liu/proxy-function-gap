#!/usr/bin/env python3
"""从五划分保存的逐样本 OOD 分数重算语义修正后的指标，不执行训练。"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from e6_grimm_baselines import OOD_METRIC_DEFINITION, ood_statistics
from log_run import RunLogger, sha256_file

METHODS = ["msp", "energy", "deep_knn"]


def check_synthetic_counterexamples():
    known = np.r_[np.zeros(99), 1.0]
    unknown = np.linspace(0.1, 1.0, 10)
    y = np.r_[np.zeros(len(known), dtype=int), np.ones(len(unknown), dtype=int)]
    result = ood_statistics(y, np.r_[known, unknown])
    assert abs(result["fpr95"] - 0.01) < 1e-12
    assert abs(result["ood_tpr_at_known_fpr5"] - 1.0) < 1e-12
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(ood_statistics(y, score)["aurc_ood_error"] - 0.14583333333333333) < 1e-12
    assert abs(ood_statistics(y, score[::-1])["aurc_ood_error"] - 0.6041666666666666) < 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e6-root", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e6")
    ap.add_argument("--splits", default="1,2,3,4,5")
    ap.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    args = ap.parse_args()
    splits = [int(value) for value in args.splits.split(",")]
    lg = RunLogger(args.log, stage="E6", kind="ood-metric-recompute-v2")
    lg.start(config=vars(args), data_version="GRIMM-EC_v2/five-splits/saved-ood-scores",
             inputs=[args.e6_root], code_version=sha256_file(os.path.abspath(__file__)))
    try:
        check_synthetic_counterexamples()
        outputs = []
        for split in splits:
            result_dir = os.path.join(args.e6_root, f"results_split{split}_convergence")
            summary_path = os.path.join(result_dir, "summary.json")
            with open(summary_path) as handle:
                summary = json.load(handle)
            n_test2 = int(summary["test2_coverage"]["test2_rows"])
            ood_rows = []
            score_hashes = {}
            for method in METHODS:
                score_path = os.path.join(result_dir, f"ood_scores_{method}.npy")
                score_hash = sha256_file(score_path)
                score = np.load(score_path, allow_pickle=False)
                if score.ndim != 1 or len(score) <= n_test2 or n_test2 <= 0 or not np.isfinite(score).all():
                    raise ValueError(f"Invalid saved OOD scores: split {split}/{method}")
                y = np.r_[np.zeros(len(score) - n_test2, dtype=np.uint8),
                          np.ones(n_test2, dtype=np.uint8)]
                row = {"method": method, **ood_statistics(y, score), "score_sha256": score_hash,
                       "score_dtype": str(score.dtype), "n_samples": len(score)}
                ood_rows.append(row)
                score_hashes[method] = score_hash
                order = np.argsort(score, kind="quicksort")
                risk = np.cumsum(y[order]) / np.arange(1, len(score) + 1)
                points = np.linspace(0.05, 1, 20)
                pd.DataFrame({"coverage": points,
                              "ood_error": risk[np.ceil(points * len(score)).astype(int) - 1]}).to_csv(
                    os.path.join(result_dir, f"risk_coverage_{method}.tsv"), sep="\t", index=False)
            pd.DataFrame(ood_rows).to_csv(os.path.join(result_dir, "ood_metrics.tsv"), sep="\t", index=False)
            summary["ood_metrics"] = ood_rows
            summary["ood_metric_definition"] = OOD_METRIC_DEFINITION
            summary["ood_recompute"] = {
                "run_id": lg.rec["run_id"], "source": "saved per-sample seed-ensemble scores; no retraining",
                "score_sha256": score_hashes, "synthetic_counterexamples_passed": True,
                "numpy_version": np.__version__, "argsort_kind": "quicksort; tied-order effects recorded",
                "quantile_method": "linear",
                "fpr95": "fraction of ID scores >= quantile(OOD scores, 0.05)",
                "aurc": "trapezoid of cumulative OOD fraction with samples sorted by ascending OOD score",
                "ood_tpr_at_known_fpr5": "legacy mislabeled fpr95 retained only as a separately named audit field",
                "bootstrap_note": "existing paired bootstrap intervals cover AUROC-OOD only; no new FPR95/AURC CI",
            }
            with open(summary_path, "w") as handle:
                json.dump(summary, handle, indent=2)
            outputs.append(result_dir)
            print("OOD_V2", split, json.dumps(ood_rows), flush=True)
        lg.finish(metrics={"metric_definition": OOD_METRIC_DEFINITION,
                           "synthetic_counterexamples_passed": True, "n_splits": len(splits),
                           "retraining": False}, outputs=outputs)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=outputs if "outputs" in locals() else [])
        raise


if __name__ == "__main__":
    main()
