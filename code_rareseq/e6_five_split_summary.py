#!/usr/bin/env python3
"""验收并汇总 E6 GRIMM 五划分冻结基线；跨 split 仅作等权描述统计。"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from e6_grimm_baselines import OOD_METRIC_DEFINITION, ood_statistics
from log_run import RunLogger, sha256_file

BUCKETS = ["tail_lt10", "tail_10_29", "mid_30_99", "head_ge100", "all"]
METHODS = ["msp", "energy", "deep_knn"]
METRICS = ["auroc_ood", "fpr95", "aurc_ood_error"]
SEEDS = [0, 1, 2]
TAUS = [0.0, 0.5, 1.0]


def require_finite(value, path):
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite JSON value: {path}")


def read_json(path):
    def invalid(value):
        raise ValueError(f"Invalid JSON constant {value}: {path}")

    with open(path) as handle:
        result = json.load(handle, parse_constant=invalid)
    require_finite(result, path)
    return result


def completed_runs(path):
    latest = {}
    with open(path) as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id"):
                latest[row["run_id"]] = row
    return [row for row in latest.values() if row.get("status") == "completed"]


def matching_run(runs, result_dir):
    matches = [row for row in runs if row.get("kind") == "grimm-frozen-baselines"
               and result_dir in (row.get("outputs") or [])]
    if len(matches) != 1:
        raise ValueError(f"Expected one completed run for {result_dir}, found {len(matches)}")
    return matches[0]


def validate_history(frame, metrics, epochs, patience, min_delta):
    required = {"seed", "epoch", "train_loss", "val_tail100_macro_auprc"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing history columns: {required - set(frame.columns)}")
    numeric = frame[list(required)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite validation history")
    rows = []
    for item in metrics:
        seed = int(item["seed"])
        history = frame[frame.seed == seed].sort_values("epoch")
        if history.empty or history.epoch.duplicated().any():
            raise ValueError(f"Missing or duplicate history for seed {seed}")
        best_score = -np.inf
        best_epoch = None
        for row in history.itertuples():
            score = float(row.val_tail100_macro_auprc)
            if score > best_score + min_delta:
                best_score = score
                best_epoch = int(row.epoch)
        recorded_epoch = int(item["best_epoch"])
        last_epoch = int(history.epoch.max())
        if best_epoch != recorded_epoch:
            raise ValueError(f"Best epoch mismatch for seed {seed}: {best_epoch} != {recorded_epoch}")
        if last_epoch != min(epochs, best_epoch + patience):
            raise ValueError(f"Early stopping mismatch for seed {seed}: best={best_epoch}, last={last_epoch}")
        if best_epoch >= epochs:
            raise ValueError(f"Epoch cap is not convergence evidence for seed {seed}")
        rows.append({"seed": seed, "best_epoch": best_epoch, "last_epoch": last_epoch,
                     "epochs_after_best": last_epoch - best_epoch,
                     "best_val_tail100_macro_auprc": best_score,
                     "val_fmax_threshold": float(item["val_fmax_threshold"])})
    return rows


def validate_embeddings_and_labels(config, paired):
    shapes = {}
    expected_labels = None
    for name in ("train", "validation", "test1", "test2"):
        frame = pd.read_csv(os.path.join(config["data_dir"], name + ".tsv"), sep="\t")
        ids = pd.read_csv(os.path.join(config["emb_dir"], name + "_ids.tsv"), sep="\t")
        if not ids.Entry.equals(frame.Entry):
            raise ValueError(f"Embedding ID order mismatch: {name}")
        array = np.load(os.path.join(config["emb_dir"], name + ".npy"), mmap_mode="r")
        if array.ndim != 2 or array.shape != (len(frame), 960):
            raise ValueError(f"Embedding shape mismatch: {name} {array.shape}")
        for start in range(0, len(array), 4096):
            if not np.isfinite(array[start:start + 4096]).all():
                raise ValueError(f"Non-finite embedding: {name} rows {start}:")
        shapes[name] = list(array.shape)
        if name == "train":
            counts = frame["EC number"].str.split(";").explode().value_counts()
            expected_labels = set(counts.index)
            observed = paired.drop_duplicates("ec").set_index("ec").n_train
            if not set(observed.index).issubset(expected_labels) or not observed.equals(counts.loc[observed.index]):
                raise ValueError("Evaluated EC labels/frequencies do not align with training data")
    return shapes


def paired_label_deltas(frame):
    required = {"ec", "n_train", "tier", "auprc", "seed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing per-label columns: {required - set(frame.columns)}")
    frame = frame.copy()
    frame["seed"] = frame.seed.astype(str)
    neural = frame[frame.seed != "deterministic"]
    knn = frame[frame.seed == "deterministic"]
    if set(neural.seed) != {str(seed) for seed in SEEDS} or len(knn) == 0:
        raise ValueError("Unexpected per-label model/seed tags")
    label_sets = [set(group.ec) for _, group in frame.groupby("seed")]
    if any(labels != label_sets[0] for labels in label_sets[1:]):
        raise ValueError("EC labels are not aligned across seeds/models")
    if frame.groupby(["seed", "ec"]).size().max() != 1:
        raise ValueError("Duplicate EC label within a seed/model")
    if frame.groupby(["seed", "ec"])[["n_train", "tier"]].nunique().max().max() != 1:
        raise ValueError("Frequency/tier mismatch within a label")
    neural = neural.groupby(["ec", "n_train", "tier"], as_index=False).auprc.mean()
    paired = neural.merge(knn[["ec", "auprc"]], on="ec", suffixes=("_neural", "_knn"),
                          validate="one_to_one").dropna()
    paired["delta_auprc"] = paired.auprc_neural - paired.auprc_knn
    return paired


def describe(rows, value_key, group_keys):
    output = []
    for key, group in rows.groupby(group_keys, sort=False):
        values = group[value_key].to_numpy(dtype=float)
        record = dict(zip(group_keys if isinstance(group_keys, list) else [group_keys],
                          key if isinstance(key, tuple) else (key,)))
        record.update({"n_splits": len(values), "split_mean": values.mean(),
                       "split_sd": values.std(ddof=1), "split_min": values.min(),
                       "split_max": values.max()})
        output.append(record)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e6-root", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e6")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="1,2,3,4,5")
    ap.add_argument("--run-log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    ap.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    splits = [int(x) for x in args.splits.split(",")]
    lg = RunLogger(args.log, stage="E6", kind="five-split-summary")
    lg.start(config=vars(args), data_version="GRIMM-EC_v2/five-splits+ESMC-300M-frozen",
             inputs=[args.e6_root, args.run_log], code_version=sha256_file(os.path.abspath(__file__)))
    try:
        runs = completed_runs(args.run_log)
        bucket_rows, ood_rows, convergence_rows, meta = [], [], [], {}
        missing = []
        for split in splits:
            result_dir = os.path.join(args.e6_root, f"results_split{split}_convergence")
            path = os.path.join(result_dir, "summary.json")
            if not os.path.exists(path):
                missing.append(split)
                continue
            data = read_json(path)
            run = matching_run(runs, result_dir)
            config = run["config"]
            if config["seeds"] != "0,1,2" or config["bootstrap"] != 10000:
                raise ValueError(f"Seed/bootstrap contract mismatch in split {split}")
            if config["taus"] != "0,0.5,1" or config["k"] != 20:
                raise ValueError(f"Tau/kNN contract mismatch in split {split}")
            if "validation" not in config["checkpoint_selection"] or "validation" not in config["tau_selection"]:
                raise ValueError(f"Non-validation selection in split {split}")
            if data.get("seeds") != SEEDS:
                raise ValueError(f"Summary seeds mismatch in split {split}")

            metrics = data.get("function_metrics", [])
            neural = [row for row in metrics if row.get("model") == "frozen_bce_la"]
            knn = [row for row in metrics if row.get("model") == "deep_knn"]
            if sorted(int(row["seed"]) for row in neural) != SEEDS or len(knn) != 1:
                raise ValueError(f"Function model tags mismatch in split {split}")
            history = pd.read_csv(os.path.join(result_dir, "validation_history.tsv"), sep="\t")
            convergence = validate_history(history, neural, int(config["epochs"]), int(config["patience"]),
                                           float(config["min_delta"]))
            tau_frame = pd.read_csv(os.path.join(result_dir, "tau_validation.tsv"), sep="\t")
            if sorted(tau_frame.tau.tolist()) != TAUS or not np.isfinite(tau_frame.select_dtypes("number")).all().all():
                raise ValueError(f"Invalid tau validation grid in split {split}")
            selected_tau = float(tau_frame.sort_values("tau").iloc[tau_frame.val_tail100_macro_auprc.idxmax()].tau)
            if not np.isclose(selected_tau, data["selected_tau"]):
                raise ValueError(f"Selected tau mismatch in split {split}")
            if any(not np.isclose(float(row["tau"]), selected_tau) for row in neural):
                raise ValueError(f"Model tau mismatch in split {split}")

            paired = paired_label_deltas(pd.read_csv(os.path.join(result_dir, "per_label_metrics.tsv"), sep="\t"))
            embedding_shapes = validate_embeddings_and_labels(config, paired)
            bootstrap = {row["bucket"]: row for row in data.get("function_label_bootstrap", [])}
            if set(bootstrap) != set(BUCKETS):
                raise ValueError(f"Frequency buckets mismatch in split {split}")
            for bucket in BUCKETS:
                row = bootstrap[bucket]
                selected = paired if bucket == "all" else paired[paired.tier == bucket]
                observed = float(selected.delta_auprc.mean())
                if len(selected) != row["n_labels"] or not np.isclose(observed, row["delta_macro_auprc_neural_minus_knn"]):
                    raise ValueError(f"Label bootstrap alignment mismatch in split {split}/{bucket}")
                if row["n_bootstrap"] != 10000 or not 0 <= row["p_two_sided"] <= 1 or row["ci_low"] > row["ci_high"]:
                    raise ValueError(f"Invalid label bootstrap in split {split}/{bucket}")
                bucket_rows.append({"split": split, "bucket": bucket, "n_labels": row["n_labels"],
                                    "delta_macro_auprc": row["delta_macro_auprc_neural_minus_knn"],
                                    "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                                    "p_two_sided": row["p_two_sided"], "n_bootstrap": row["n_bootstrap"],
                                    "run_id": run["run_id"]})

            ood = {row["method"]: row for row in data.get("ood_metrics", [])}
            recompute = data.get("ood_recompute", {})
            if set(ood) != set(METHODS) or data.get("ood_metric_definition") != OOD_METRIC_DEFINITION:
                raise ValueError(f"OOD methods/definition mismatch in split {split}")
            if not recompute.get("synthetic_counterexamples_passed"):
                raise ValueError(f"OOD synthetic counterexamples not accepted in split {split}")
            recompute_matches = [item for item in runs if item.get("run_id") == recompute.get("run_id")
                                 and item.get("kind") == "ood-metric-recompute-v2"]
            if len(recompute_matches) != 1:
                raise ValueError(f"OOD recompute run is not completed in split {split}")
            n_test2 = int(data["test2_coverage"]["test2_rows"])
            for method in METHODS:
                row = ood[method]
                if row.get("metric_definition") != OOD_METRIC_DEFINITION:
                    raise ValueError(f"OOD row definition mismatch in split {split}/{method}")
                score_path = os.path.join(result_dir, f"ood_scores_{method}.npy")
                score_hash = sha256_file(score_path)
                if row.get("score_sha256") != score_hash or recompute.get("score_sha256", {}).get(method) != score_hash:
                    raise ValueError(f"OOD score hash mismatch in split {split}/{method}")
                score = np.load(score_path, allow_pickle=False)
                y = np.r_[np.zeros(len(score) - n_test2, dtype=np.uint8), np.ones(n_test2, dtype=np.uint8)]
                observed = ood_statistics(y, score)
                if any(not np.isclose(observed[metric], row[metric], rtol=0, atol=1e-12) for metric in METRICS):
                    raise ValueError(f"OOD semantic recalculation mismatch in split {split}/{method}")
                order = np.argsort(score, kind="quicksort")
                risk = np.cumsum(y[order]) / np.arange(1, len(score) + 1)
                points = np.linspace(0.05, 1, 20)
                expected_risk = risk[np.ceil(points * len(score)).astype(int) - 1]
                saved_risk = pd.read_csv(os.path.join(result_dir, f"risk_coverage_{method}.tsv"), sep="\t")
                if not np.allclose(saved_risk.coverage, points, rtol=0, atol=1e-12) or not np.allclose(
                        saved_risk.ood_error, expected_risk, rtol=0, atol=1e-12):
                    raise ValueError(f"Risk-coverage direction mismatch in split {split}/{method}")
                ood_rows.append({"split": split, "method": method, **{metric: row[metric] for metric in METRICS},
                                 "ood_tpr_at_known_fpr5": row["ood_tpr_at_known_fpr5"],
                                 "metric_definition": OOD_METRIC_DEFINITION, "score_sha256": score_hash,
                                 "ood_recompute_run_id": recompute["run_id"], "run_id": run["run_id"]})
            for row in data.get("ood_bootstrap", []):
                if row["n_bootstrap"] != 10000:
                    raise ValueError(f"OOD bootstrap count mismatch in split {split}")
            for row in convergence:
                convergence_rows.append({"split": split, **row, "selected_tau": selected_tau,
                                         "run_id": run["run_id"]})
            meta[str(split)] = {"run_id": run["run_id"], "code_version": run["code_version"],
                                "seeds": SEEDS, "best_epochs": {str(row["seed"]): row["best_epoch"] for row in convergence},
                                "last_epochs": {str(row["seed"]): row["last_epoch"] for row in convergence},
                                "early_stopping_verified": True, "selected_tau": selected_tau,
                                "label_alignment_verified": True, "bootstrap": 10000,
                                "embedding_dim": 960, "embedding_shapes": embedding_shapes,
                                "embeddings_finite_and_id_aligned": True,
                                "ood_metric_definition": OOD_METRIC_DEFINITION,
                                "ood_semantics_verified": True, "ood_recompute_run_id": recompute["run_id"],
                                "model": "frozen_bce_la", "knn_model": "deep_knn"}

        bdf = pd.DataFrame(bucket_rows)
        odf = pd.DataFrame(ood_rows)
        cdf = pd.DataFrame(convergence_rows)
        complete = not missing and len(meta) == len(splits)
        scientific_accepted = complete and all(meta[str(split)].get("ood_semantics_verified") is True
                                               for split in splits)
        bucket_summary = []
        if not bdf.empty:
            bucket_summary = describe(bdf, "delta_macro_auprc", ["bucket"])
            for row in bucket_summary:
                group = bdf[bdf.bucket == row["bucket"]]
                row.update({"n_positive": int((group.delta_macro_auprc > 0).sum()),
                            "n_negative": int((group.delta_macro_auprc < 0).sum()),
                            "direction_consistent": bool((group.delta_macro_auprc > 0).all() or
                                                         (group.delta_macro_auprc < 0).all()),
                            "ci_excludes_zero_splits": int(((group.ci_low > 0) | (group.ci_high < 0)).sum()),
                            "n_labels_min": int(group.n_labels.min()), "n_labels_max": int(group.n_labels.max())})
        ood_summary = []
        if not odf.empty:
            for method in METHODS:
                group = odf[odf.method == method]
                if group.empty:
                    continue
                for metric in METRICS:
                    values = group[metric].to_numpy(dtype=float)
                    baseline = odf[odf.method == "deep_knn"].set_index("split").loc[group.split, metric].to_numpy()
                    better = values > baseline if metric == "auroc_ood" else values < baseline
                    ood_summary.append({"method": method, "metric": metric, "n_splits": len(values),
                                        "split_mean": values.mean(), "split_sd": values.std(ddof=1),
                                        "split_min": values.min(), "split_max": values.max(),
                                        "n_better_than_deep_knn": int(better.sum())})

        bdf.to_csv(os.path.join(args.out, "five_split_buckets.tsv"), sep="\t", index=False)
        pd.DataFrame(bucket_summary).to_csv(os.path.join(args.out, "five_split_bucket_summary.tsv"), sep="\t", index=False)
        odf.to_csv(os.path.join(args.out, "five_split_ood.tsv"), sep="\t", index=False)
        pd.DataFrame(ood_summary).to_csv(os.path.join(args.out, "five_split_ood_summary.tsv"), sep="\t", index=False)
        cdf.to_csv(os.path.join(args.out, "five_split_convergence.tsv"), sep="\t", index=False)
        summary = {"complete": complete, "accepted": scientific_accepted, "missing_splits": missing,
                   "evidence_level": "five-split test results; cross-split descriptive only; no pooled inference",
                   "ood_metric_definition": OOD_METRIC_DEFINITION,
                   "ood_semantics_verified": scientific_accepted,
                   "acceptance_scope": "file completeness, training/validation contract, OOD definitions, score hashes, semantic recalculation",
                   "aggregation_run_id": lg.rec["run_id"], "per_split_meta": meta,
                   "bucket_descriptive": bucket_summary, "ood_descriptive": ood_summary}
        require_finite(summary, "five_split_summary")
        with open(os.path.join(args.out, "five_split_summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
        lg.finish(metrics={"complete": complete, "accepted": scientific_accepted,
                           "ood_semantics_verified": scientific_accepted,
                           "missing_splits": missing, "n_splits": len(meta)}, outputs=[args.out])
        if not bdf.empty:
            print("Δmacro AUPRC (BCE−kNN) by bucket × split:\n",
                  bdf.pivot(index="bucket", columns="split", values="delta_macro_auprc").round(6).to_string())
            print("\nCross-split descriptive summary:\n", pd.DataFrame(bucket_summary).round(6).to_string(index=False))
            print("\nOOD metrics by split:\n", odf.round(6).to_string(index=False))
        print("\nACCEPTED: completeness and OOD semantics verified" if scientific_accepted else f"\nINCOMPLETE missing={missing}")
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
