#!/usr/bin/env python3
"""M1 prediction validation, validation-only selection and paired acceptance gates."""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REQUIRED = {"experiment", "rbp", "tier", "fold", "track", "window_id", "label", "logit", "score", "split"}


def read_predictions(directory, split):
    path = os.path.join(directory, f"{split}_predictions.tsv.gz")
    frame = pd.read_csv(path, sep="\t")
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty or frame[list(REQUIRED)].isna().any().any():
        raise ValueError(f"{path}: empty predictions or missing required values")
    if not np.isfinite(frame[["label", "logit", "score"]].to_numpy()).all():
        raise ValueError(f"{path}: non-finite predictions")
    if not set(frame.label.unique()).issubset({0, 1}):
        raise ValueError(f"{path}: labels must be binary")
    if not frame.score.between(0, 1).all():
        raise ValueError(f"{path}: scores outside [0,1]")
    if np.max(np.abs(frame.score - 1 / (1 + np.exp(-frame.logit)))) > 1e-5:
        raise ValueError(f"{path}: score/logit mismatch")
    occurrence = frame.groupby(["experiment", "fold", "track", "window_id"]).cumcount()
    frame["occurrence"] = occurrence
    keys = ["experiment", "fold", "track", "window_id", "occurrence"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{path}: duplicate occurrence keys")
    return frame


def group_metrics(frame):
    rows = []
    for (track, tier), group in frame.groupby(["track", "tier"]):
        per_experiment = []
        for experiment, part in group.groupby("experiment"):
            y, score = part.label.to_numpy(), part.score.to_numpy()
            if len(np.unique(y)) == 2:
                per_experiment.append((roc_auc_score(y, score), average_precision_score(y, score)))
        if per_experiment:
            values = np.asarray(per_experiment)
            rows.append({"track": track, "tier": tier, "n_experiments": len(values),
                         "auroc": values[:, 0].mean(), "auprc": values[:, 1].mean()})
    return pd.DataFrame(rows)


def tail_score(frame):
    tail = frame[frame.tier == "tail"]
    if tail.empty:
        raise ValueError("No tail-RBP validation predictions")
    values = []
    for _, part in tail.groupby("experiment"):
        if len(np.unique(part.label)) == 2:
            values.append(average_precision_score(part.label, part.score))
    if not values:
        raise ValueError("No evaluable tail-RBP validation predictions")
    return float(np.mean(values))


def select_configurations(configs):
    validation = []
    for name, directory in configs.items():
        val = read_predictions(directory, "val")
        for group_key, part in val.groupby(["track", "seed"] if "seed" in val else ["track"]):
            track = group_key[0] if isinstance(group_key, tuple) else group_key
            seed_value = group_key[1] if isinstance(group_key, tuple) and len(group_key) == 2 else "single"
            validation.append({"config": name, "track": track, "seed": seed_value,
                               "tail_auprc": tail_score(part)})
    val = pd.DataFrame(validation)
    selected = {}
    for track, part in val.groupby("track"):
        scores = part.groupby("config").tail_auprc.agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
        selected[track] = scores.index[0]
    return val, selected


def paired_deltas(candidate, reference):
    keys = ["experiment", "fold", "track", "window_id", "occurrence"]
    a = candidate[keys + ["label", "score", "tier"]]
    b = reference[keys + ["label", "score"]]
    joined = a.merge(b, on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one")
    if len(joined) != len(a) or len(joined) != len(b):
        raise ValueError("Candidate/reference predictions are not occurrence-aligned")
    if not np.array_equal(joined.label_candidate, joined.label_reference):
        raise ValueError("Candidate/reference labels differ")
    rows = []
    for (track, tier), part in joined.groupby(["track", "tier"]):
        for experiment, group in part.groupby("experiment"):
            y = group.label_candidate.to_numpy()
            if len(np.unique(y)) < 2:
                continue
            for metric in ("auroc", "auprc"):
                function = roc_auc_score if metric == "auroc" else average_precision_score
                c = function(y, group.score_candidate)
                r = function(y, group.score_reference)
                rows.append({"track": track, "tier": tier, "experiment": experiment,
                             "metric": metric, "candidate": c, "reference": r, "delta": c - r})
    return pd.DataFrame(rows)


def bootstrap(joined, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    rows, draws = [], {}
    for (track, tier), part in joined.groupby(["track", "tier"]):
        experiments = []
        for _, group in part.groupby("experiment"):
            y = group.label_candidate.to_numpy()
            if len(np.unique(y)) == 2:
                experiments.append((y, group.score_candidate.to_numpy(), group.score_reference.to_numpy()))
        for metric in ("auroc", "auprc"):
            function = roc_auc_score if metric == "auroc" else average_precision_score
            observed_values = np.array([function(y, c) - function(y, r) for y, c, r in experiments])
            samples = np.empty(n_bootstrap)
            for iteration in range(n_bootstrap):
                values = []
                for y, candidate, reference in experiments:
                    index = rng.integers(0, len(y), size=len(y))
                    ys = y[index]
                    if len(np.unique(ys)) == 2:
                        values.append(function(ys, candidate[index]) - function(ys, reference[index]))
                samples[iteration] = np.mean(values)
            observed = observed_values.mean()
            p = min(1.0, 2 * min((samples <= 0).mean(), (samples >= 0).mean()))
            p = max(p, 1 / n_bootstrap)
            key = f"{track}_{tier}_{metric}"
            draws[key] = samples
            rows.append({"track": track, "tier": tier, "metric": metric, "n_experiments": len(experiments),
                         "delta": observed, "ci_low": np.quantile(samples, 0.025),
                         "ci_high": np.quantile(samples, 0.975), "p_two_sided": p,
                         "n_bootstrap": n_bootstrap, "resampling_unit": "window within experiment; macro mean"})
    return pd.DataFrame(rows), draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True,
                    help="name=/path/to/run; repeat for validation-only candidate selection")
    ap.add_argument("--reference", required=True, help="name of strongest reference configuration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--select", action="store_true", help="select hyperparameters by validation tail AUPRC")
    args = ap.parse_args()
    configs = {}
    for item in args.config:
        name, sep, path = item.partition("=")
        if not sep or not name or name in configs:
            raise ValueError("Each --config must be a unique name=path")
        configs[name] = path
    if args.reference not in configs:
        raise ValueError("Reference configuration is missing")
    os.makedirs(args.out, exist_ok=False)
    if args.select:
        validation, selected = select_configurations(configs)
        validation.to_csv(os.path.join(args.out, "validation_selection.tsv"), sep="\t", index=False)
    else:
        selected = {track: args.reference for track in read_predictions(configs[args.reference], "test").track.unique()}
    all_metrics, all_deltas = [], []
    decisions = []
    for track, candidate_name in selected.items():
        candidate = read_predictions(configs[candidate_name], "test")
        reference = read_predictions(configs[args.reference], "test")
        candidate = candidate[candidate.track == track]
        reference = reference[reference.track == track]
        deltas = paired_deltas(candidate, reference)
        keys = ["experiment", "fold", "track", "window_id", "occurrence"]
        joined = candidate[keys + ["label", "score", "tier"]].merge(
            reference[keys + ["label", "score"]], on=keys,
            suffixes=("_candidate", "_reference"), validate="one_to_one")
        boot, draws = bootstrap(joined, args.bootstrap, args.seed)
        deltas.to_csv(os.path.join(args.out, f"paired_deltas_{track}.tsv"), sep="\t", index=False)
        boot.to_csv(os.path.join(args.out, f"bootstrap_{track}.tsv"), sep="\t", index=False)
        np.savez_compressed(os.path.join(args.out, f"bootstrap_draws_{track}.npz"), **draws)
        metrics_frame = group_metrics(candidate)
        metrics_frame["config"] = candidate_name
        all_metrics.append(metrics_frame)
        all_deltas.append(deltas)
        tail = deltas[(deltas.tier == "tail") & (deltas.metric == "auprc")].delta.mean()
        head = deltas[deltas.tier == "head"].groupby("metric").delta.mean()
        decisions.append({"track": track, "selected_config": candidate_name,
                          "selected_by": "validation_tail_auprc" if args.select else "none",
                          "tail_delta_auprc": float(tail),
                          "head_delta_auroc": float(head.get("auroc", np.nan)),
                          "head_delta_auprc": float(head.get("auprc", np.nan)),
                          "m1b_validation_gate": "requires separate multi-seed validation stability review",
                          "final_victory": False,
                          "note": "Single-config evaluation is not M1f final victory; at least two T1 slices and all controls are required"})
    pd.concat(all_metrics).to_csv(os.path.join(args.out, "candidate_metrics.tsv"), sep="\t", index=False)
    with open(os.path.join(args.out, "gate_decision.json"), "w") as handle:
        json.dump({"selection": selected, "decisions": decisions,
                   "acceptance": {"tail_improvement_required": True, "head_auroc_max_loss": 0.01,
                                  "head_auprc_max_loss": 0.02, "final_m1f_requires_t1_slices": 2}},
           handle, indent=2)
    print(json.dumps({"selected": selected, "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
