#!/usr/bin/env python3
"""独立回算 E6 五划分报告数字，并复核运行配置与原始产物。"""
import argparse
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from log_run import RunLogger

BUCKETS = ["all", "tail_lt10", "tail_10_29", "mid_30_99", "head_ge100"]
METHODS = ["msp", "energy", "deep_knn"]
OOD_METRIC_DEFINITION = "e6-ood-v2:fpr95=id_at_ood_tpr95;aurc=ascending_ood_score"
TRAINING_CODE_HASH = "ba170e4067f8370e9e5a6aadfc4d8b45330ddbd730b81b056c91f68f00852177"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_json(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{path}[{index}]")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value at {path}")


def runs_by_output(path):
    latest = {}
    with open(path) as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id"):
                latest[row["run_id"]] = row
    result = {}
    for row in latest.values():
        if row.get("status") == "completed" and row.get("kind") == "grimm-frozen-baselines":
            for output in row.get("outputs", []):
                result[output] = row
    return result


def ood_statistics(y, score):
    order = np.argsort(score, kind="quicksort")
    risk = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1) / len(y)
    threshold = np.quantile(score[y == 1], 0.05)
    return {"auroc_ood": float(roc_auc_score(y, score)),
            "fpr95": float((score[y == 0] >= threshold).mean()),
            "aurc_ood_error": float(np.trapezoid(risk, coverage))}


def check_synthetic_counterexamples():
    known = np.r_[np.zeros(99), 1.0]
    unknown = np.linspace(0.1, 1.0, 10)
    y = np.r_[np.zeros(len(known), dtype=int), np.ones(len(unknown), dtype=int)]
    assert abs(ood_statistics(y, np.r_[known, unknown])["fpr95"] - 0.01) < 1e-12
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(ood_statistics(y, score)["aurc_ood_error"] - 0.14583333333333333) < 1e-12
    assert abs(ood_statistics(y, score[::-1])["aurc_ood_error"] - 0.6041666666666666) < 1e-12


def completed_recompute_runs(path):
    result = set()
    with open(path) as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "ood-metric-recompute-v2" and row.get("status") == "completed":
                result.add(row.get("run_id"))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e6-root", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e6")
    ap.add_argument("--agg-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    ap.add_argument("--baseline-code", default="/home/ProxyFunctionGap/code_rareseq/e6_grimm_baselines.py")
    ap.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    args = ap.parse_args()
    lg = RunLogger(args.log, stage="E6", kind="five-split-independent-verification")
    lg.start(config=vars(args), data_version="GRIMM-EC_v2/five-splits+ESMC-300M-frozen",
             inputs=[args.e6_root, args.agg_dir, args.run_log], code_version=sha256(os.path.abspath(__file__)))
    try:
        aggregate = json.load(open(os.path.join(args.agg_dir, "five_split_summary.json")))
        if not aggregate.get("accepted") or not aggregate.get("ood_semantics_verified"):
            raise ValueError("Aggregate is not scientifically accepted")
        if aggregate.get("ood_metric_definition") != OOD_METRIC_DEFINITION:
            raise ValueError("Aggregate OOD definition mismatch")
        finite_json(aggregate)
        check_synthetic_counterexamples()
        bucket_file = pd.read_csv(os.path.join(args.agg_dir, "five_split_buckets.tsv"), sep="\t")
        bucket_summary_file = pd.read_csv(os.path.join(args.agg_dir, "five_split_bucket_summary.tsv"), sep="\t")
        ood_file = pd.read_csv(os.path.join(args.agg_dir, "five_split_ood.tsv"), sep="\t")
        ood_summary_file = pd.read_csv(os.path.join(args.agg_dir, "five_split_ood_summary.tsv"), sep="\t")
        convergence_file = pd.read_csv(os.path.join(args.agg_dir, "five_split_convergence.tsv"), sep="\t")
        runs = runs_by_output(args.run_log)
        recompute_runs = completed_recompute_runs(args.run_log)
        corrected_baseline_hash = sha256(args.baseline_code)
        per_split = {}
        for split in range(1, 6):
            result_dir = os.path.join(args.e6_root, f"results_split{split}_convergence")
            run = runs[result_dir]
            config = run["config"]
            if run["code_version"] != TRAINING_CODE_HASH:
                raise ValueError(f"Training code hash mismatch in split {split}")
            expected = {"seeds": "0,1,2", "epochs": 1000, "patience": 20, "min_delta": 0.0001,
                        "batch": 512, "lr": 0.001, "weight_decay": 0.0001,
                        "taus": "0,0.5,1", "k": 20, "bootstrap": 10000}
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Frozen config mismatch in split {split}")
            if config["checkpoint_selection"] != "validation macro AUPRC, train labels with support <100":
                raise ValueError(f"Checkpoint selection mismatch in split {split}")
            if "validation" not in config["tau_selection"] or "validation" not in config["fmax_threshold"]:
                raise ValueError(f"Validation selection mismatch in split {split}")
            source = json.load(open(os.path.join(result_dir, "summary.json")))
            finite_json(source, f"split{split}.summary")
            recompute = source.get("ood_recompute", {})
            if source.get("ood_metric_definition") != OOD_METRIC_DEFINITION:
                raise ValueError(f"Source OOD definition mismatch in split {split}")
            if recompute.get("run_id") not in recompute_runs or not recompute.get("synthetic_counterexamples_passed"):
                raise ValueError(f"OOD semantic provenance mismatch in split {split}")
            source_ood = {row["method"]: row for row in source["ood_metrics"]}
            for method in ("msp", "energy"):
                deltas = np.load(os.path.join(result_dir, f"bootstrap_{method}.npz"))["deltas"]
                if deltas.shape != (10000,) or not np.isfinite(deltas).all():
                    raise ValueError(f"Invalid OOD bootstrap array in split {split}/{method}")

            labels = pd.read_csv(os.path.join(result_dir, "per_label_metrics.tsv"), sep="\t")
            labels["seed"] = labels.seed.astype(str)
            neural = labels[labels.seed != "deterministic"].groupby(
                ["ec", "n_train", "tier"], as_index=False).auprc.mean()
            knn = labels[labels.seed == "deterministic"][["ec", "auprc"]]
            paired = neural.merge(knn, on="ec", suffixes=("_neural", "_knn"), validate="one_to_one").dropna()
            paired["delta_auprc"] = paired.auprc_neural - paired.auprc_knn
            rng = np.random.default_rng(20261124)
            groups = [group for _, group in paired.groupby("tier")]
            recalculated = []
            for bucket in BUCKETS:
                selected = paired if bucket == "all" else paired[paired.tier == bucket]
                deltas = np.empty(10000)
                for index in range(10000):
                    if bucket == "all":
                        estimate = sum(len(group) * group.delta_auprc.to_numpy()[
                            rng.integers(0, len(group), len(group))].mean() for group in groups) / len(paired)
                    else:
                        estimate = selected.delta_auprc.to_numpy()[
                            rng.integers(0, len(selected), len(selected))].mean()
                    deltas[index] = estimate
                p = min(1.0, 2 * (1 + min((deltas <= 0).sum(), (deltas >= 0).sum())) / 10001)
                recalculated.append({"split": split, "bucket": bucket, "n_labels": len(selected),
                                     "delta_macro_auprc": selected.delta_auprc.mean(),
                                     "ci_low": np.quantile(deltas, 0.025), "ci_high": np.quantile(deltas, 0.975),
                                     "p_two_sided": p, "n_bootstrap": 10000})
            recalculated = pd.DataFrame(recalculated)
            expected_buckets = bucket_file[bucket_file.split == split].set_index("bucket").loc[
                recalculated.bucket].reset_index()[recalculated.columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(recalculated.reset_index(drop=True), expected_buckets,
                                          check_exact=False, rtol=0, atol=1e-12)

            for name in ("train", "validation", "test1", "test2"):
                frame = pd.read_csv(os.path.join(config["data_dir"], name + ".tsv"), sep="\t", usecols=["Entry"])
                ids = pd.read_csv(os.path.join(config["emb_dir"], name + "_ids.tsv"), sep="\t")
                array = np.load(os.path.join(config["emb_dir"], name + ".npy"), mmap_mode="r")
                if not ids.Entry.equals(frame.Entry) or array.shape != (len(frame), 960):
                    raise ValueError(f"Embedding alignment mismatch in split {split}/{name}")
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite embedding in split {split}/{name}")

            y_ood = np.concatenate([np.zeros(array_shape[0], dtype=np.uint8) for array_shape in [
                np.load(os.path.join(config["emb_dir"], "test1.npy"), mmap_mode="r").shape,
                np.load(os.path.join(config["emb_dir"], "test2.npy"), mmap_mode="r").shape]])
            y_ood[np.load(os.path.join(config["emb_dir"], "test1.npy"), mmap_mode="r").shape[0]:] = 1
            for method in METHODS:
                score_path = os.path.join(result_dir, f"ood_scores_{method}.npy")
                score_hash = sha256(score_path)
                source_row = source_ood[method]
                if source_row.get("metric_definition") != OOD_METRIC_DEFINITION:
                    raise ValueError(f"OOD row definition mismatch in split {split}/{method}")
                if source_row.get("score_sha256") != score_hash or recompute["score_sha256"].get(method) != score_hash:
                    raise ValueError(f"OOD score hash mismatch in split {split}/{method}")
                score = np.load(score_path)
                observed = ood_statistics(y_ood, score)
                expected_row = ood_file[(ood_file.split == split) & (ood_file.method == method)].iloc[0]
                if expected_row["metric_definition"] != OOD_METRIC_DEFINITION or expected_row["score_sha256"] != score_hash:
                    raise ValueError(f"Aggregate OOD semantic provenance mismatch in split {split}/{method}")
                for metric, value in observed.items():
                    if not np.isclose(value, expected_row[metric], rtol=0, atol=1e-12):
                        raise ValueError(f"OOD recalculation mismatch in split {split}/{method}/{metric}")
                    if not np.isclose(value, source_row[metric], rtol=0, atol=1e-12):
                        raise ValueError(f"Source OOD recalculation mismatch in split {split}/{method}/{metric}")
                order = np.argsort(score, kind="quicksort")
                risk = np.cumsum(y_ood[order]) / np.arange(1, len(score) + 1)
                points = np.linspace(0.05, 1, 20)
                expected_risk = risk[np.ceil(points * len(score)).astype(int) - 1]
                saved_risk = pd.read_csv(os.path.join(result_dir, f"risk_coverage_{method}.tsv"), sep="\t")
                if not np.allclose(saved_risk.ood_error, expected_risk, rtol=0, atol=1e-12):
                    raise ValueError(f"Risk-coverage mismatch in split {split}/{method}")
            per_split[str(split)] = {"run_id": run["run_id"], "config_verified": True,
                                     "embedding_960_finite_id_aligned": True,
                                     "bootstrap_recalculated": True, "ood_recalculated": True,
                                     "ood_metric_definition": OOD_METRIC_DEFINITION,
                                     "ood_score_hashes_verified": True, "risk_coverage_recalculated": True}

        recomputed_summary = bucket_file.groupby("bucket").delta_macro_auprc.agg(
            ["count", "mean", "std", "min", "max"]).reset_index()
        recomputed_summary.columns = ["bucket", "n_splits", "split_mean", "split_sd", "split_min", "split_max"]
        expected_summary = bucket_summary_file.set_index("bucket").loc[recomputed_summary.bucket].reset_index()
        pd.testing.assert_frame_equal(recomputed_summary,
                                      expected_summary[["bucket", "n_splits", "split_mean", "split_sd",
                                                        "split_min", "split_max"]].reset_index(drop=True),
                                      check_exact=False, rtol=0, atol=1e-12)
        for row in ood_summary_file.itertuples():
            values = ood_file[ood_file.method == row.method].sort_values("split")[row.metric].to_numpy()
            if not np.allclose([len(values), values.mean(), values.std(ddof=1), values.min(), values.max()],
                               [row.n_splits, row.split_mean, row.split_sd, row.split_min, row.split_max],
                               rtol=0, atol=1e-12):
                raise ValueError(f"Cross-split OOD summary mismatch: {row.method}/{row.metric}")
        if len(convergence_file) != 15 or set(convergence_file.seed) != {0, 1, 2}:
            raise ValueError("Convergence table is incomplete")
        result = {"accepted": True, "verification_run_id": lg.rec["run_id"],
                  "schema_and_finite_values": True, "config_seed_and_alignment": True,
                  "report_numbers_recalculated": True,
                  "ood_metric_definition": OOD_METRIC_DEFINITION, "ood_semantics_verified": True,
                  "synthetic_counterexamples_passed": True,
                  "training_code_hash": TRAINING_CODE_HASH, "corrected_metric_code_hash": corrected_baseline_hash,
                  "per_split": per_split,
                  "cross_split_statistics": "equal-weight split mean/sample SD/range; no pooled inference"}
        finite_json(result)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
        lg.finish(metrics={"accepted": True, "n_splits": 5, "n_seeds_per_split": 3,
                           "ood_metric_definition": OOD_METRIC_DEFINITION,
                           "ood_semantics_verified": True}, outputs=[args.out])
        print("VERIFIED", json.dumps(result), flush=True)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
