#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from e1_locked_logger import E1RunLogger as RunLogger
from log_run import sha256_file


KEYS = ["experiment", "rbp", "tier", "fold", "track", "window_id", "label"]


def metrics(y, score):
    return np.array([roc_auc_score(y, score), average_precision_score(y, score)])


def close_frame(actual, expected, columns):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    for column in columns:
        assert np.allclose(actual[column].to_numpy(), expected[column].to_numpy(), rtol=0, atol=1e-12), column


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--identity", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logger = RunLogger(args.log, stage="E1", kind="main-independent-recalculation")
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    logger.start(config=config, seed=20261123, data_version="e0-v2-id90-strict3train-main",
                 code_version=config["code_sha256"])
    try:
        audit = json.load(open(os.path.join(args.root, "audit", "main_audit.json")))
        assert audit["checks"]["n_modern_runs"] == 15 and audit["checks"]["n_knn_runs"] == 5
        assert audit["checks"]["all_prediction_keys_aligned"]
        assert audit["checks"]["all_raw_splits_aligned"]
        assert audit["checks"]["all_early_stopped_below_1000"]
        assert audit["checks"]["all_cuda0_numeric_probes_passed"]
        assert audit["checks"]["all_train_strict_three_folds"]
        assert audit["checks"]["no_fold_minus1_in_train"]
        modern_frames, knn_frames = [], []
        for fold in range(5):
            for seed in range(3):
                frame = pd.read_csv(os.path.join(args.root, f"f{fold}_modern_bce_s{seed}", "modern_results.tsv"), sep="\t")
                frame["seed"] = seed
                modern_frames.append(frame)
            knn_frames.append(pd.read_csv(os.path.join(args.root, f"f{fold}_knn", "knn_results.tsv"), sep="\t"))
        modern = pd.concat(modern_frames, ignore_index=True)
        knn = pd.concat(knn_frames, ignore_index=True)
        join_keys = ["experiment", "rbp", "tier", "fold", "track", "split"]
        paired = modern.merge(knn, on=join_keys, suffixes=("_modern", "_knn"), validate="many_to_one")
        paired["d_auroc"] = paired.auroc_modern - paired.auroc_knn
        paired["d_auprc"] = paired.auprc_modern - paired.auprc_knn
        value_columns = ["auroc_modern", "auprc_modern", "auroc_knn", "auprc_knn", "d_auroc", "d_auprc"]
        recalculated = paired.groupby(["track", "split", "tier"])[value_columns].mean().reset_index()
        published = pd.read_csv(os.path.join(args.root, "cv_summary", "cv_metrics.tsv"), sep="\t")
        published = published.set_index(["track", "split", "tier"]).sort_index()
        recalculated = recalculated.set_index(["track", "split", "tier"]).sort_index()
        for column in value_columns:
            assert np.allclose(recalculated[column], published[column + "_mean"], rtol=0, atol=1e-12), column
        recalculated.to_csv(os.path.join(args.out, "independent_cv_metrics.tsv"), sep="\t")
        reference = pd.read_csv(os.path.join(args.root, "cv_combined", "knn.tsv.gz"), sep="\t")
        reference["occurrence"] = reference.groupby(KEYS, sort=False).cumcount()
        reference = reference[KEYS + ["occurrence", "score"]].rename(columns={"score": "reference"})
        combined = None
        for seed in range(3):
            frame = pd.read_csv(os.path.join(args.root, "cv_combined", f"modern_bce_s{seed}.tsv.gz"), sep="\t",
                                usecols=KEYS + ["score"])
            frame["occurrence"] = frame.groupby(KEYS, sort=False).cumcount()
            frame = frame.rename(columns={"score": f"seed_{seed}"})
            combined = frame if combined is None else combined.merge(frame, on=KEYS + ["occurrence"], validate="one_to_one")
        data = reference.merge(combined, on=KEYS + ["occurrence"], validate="one_to_one")
        bootstrap = pd.read_csv(os.path.join(args.root, "cv_bootstrap", "paired_bootstrap.tsv"), sep="\t")
        output_rows = []
        for row in bootstrap.itertuples(index=False):
            subset = data[(data.track == row.track) & (data.tier == row.slice)]
            deltas = []
            for _, group in subset.groupby("experiment", sort=True):
                y = group.label.to_numpy()
                ref = metrics(y, group.reference.to_numpy())
                mod = np.mean([metrics(y, group[f"seed_{seed}"].to_numpy()) for seed in range(3)], axis=0)
                deltas.append(mod - ref)
            observed = np.nanmean(deltas, axis=0)
            metric_index = 0 if row.metric == "auroc" else 1
            values = np.load(os.path.join(args.root, "cv_bootstrap", f"bootstrap_{row.track}_{row.slice}.npz"))["deltas"][:, metric_index]
            values = values[np.isfinite(values)]
            p = min(1.0, 2 * (1 + min(np.sum(values <= 0), np.sum(values >= 0))) / (len(values) + 1))
            assert len(values) == 10000 and row.n_boot == 10000
            assert np.isclose(observed[metric_index], row.observed_delta, rtol=0, atol=1e-12)
            assert np.isclose(np.quantile(values, 0.025), row.ci_low, rtol=0, atol=1e-12)
            assert np.isclose(np.quantile(values, 0.975), row.ci_high, rtol=0, atol=1e-12)
            assert np.isclose(p, row.p_two_sided, rtol=0, atol=1e-15)
            output_rows.append({"track": row.track, "slice": row.slice, "metric": row.metric,
                                "n_experiments": len(deltas), "n_boot": len(values),
                                "observed_delta": observed[metric_index], "ci_low": np.quantile(values, 0.025),
                                "ci_high": np.quantile(values, 0.975), "p_two_sided": p})
        output = pd.DataFrame(output_rows)
        output.to_csv(os.path.join(args.out, "independent_bootstrap_recalculation.tsv"), sep="\t", index=False)
        identity = pd.read_csv(args.identity, sep="\t", usecols=["window_id", "nn_identity"]).set_index("window_id").nn_identity
        data["d_hom"] = data.window_id.map(identity)
        assert data.loc[data.label == 1, "d_hom"].notna().all()
        published_stratified = pd.read_csv(os.path.join(args.root, "cv_stratified", "stratified_summary.tsv"), sep="\t")
        low_rows = []
        for track, track_data in data.groupby("track"):
            model_rows = {}
            for model, column in (("knn", "reference"), ("modern_s0", "seed_0"),
                                  ("modern_s1", "seed_1"), ("modern_s2", "seed_2")):
                per_experiment = []
                for experiment, group in track_data.groupby("experiment"):
                    selected = group[(group.label == 0) | ((group.label == 1) & (group.d_hom < 0.60))]
                    value = metrics(selected.label.to_numpy(), selected[column].to_numpy())
                    per_experiment.append(value)
                mean = np.mean(per_experiment, axis=0)
                model_rows[model] = mean
                expected = published_stratified[(published_stratified.model == model) &
                                                (published_stratified.track == track) &
                                                (published_stratified.slice == "low_hom<0.60")]
                assert len(expected) == 1 and expected.n_experiments.iloc[0] == 146
                assert np.allclose(mean, expected[["auroc", "auprc"]].iloc[0], rtol=0, atol=1e-12)
                low_rows.append({"track": track, "model": model, "n_experiments": 146,
                                 "auroc": mean[0], "auprc": mean[1]})
            mean = np.mean(list(model_rows.values())[1:], axis=0)
            low_rows.append({"track": track, "model": "modern_three_seed_mean", "n_experiments": 146,
                             "auroc": mean[0], "auprc": mean[1]})
        low_output = pd.DataFrame(low_rows)
        low_output.to_csv(os.path.join(args.out, "independent_low_hom_recalculation.tsv"), sep="\t", index=False)
        checks = {"run_id": logger.rec["run_id"], "audit_run_id": audit["run_id"],
                  "n_pairs": len(paired), "n_bootstrap_rows": len(output),
                  "n_boot_per_slice": 10000, "cv_metrics_recalculated": True,
                  "bootstrap_observed_ci_p_recalculated": True, "all_npz_finite": True,
                  "low_hom_three_seeds_recalculated": True, "strict_train_and_cuda_audit_passed": True}
        with open(os.path.join(args.out, "verification.json"), "w") as handle:
            json.dump(checks, handle, indent=2)
        logger.finish(metrics=checks, outputs=[args.out])
        print(output.round(6).to_string(index=False))
    except Exception as exc:
        logger.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
