#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def read_score(path):
    frame = pd.read_csv(path, usecols=["hgvs_pro", "score"], dtype={"hgvs_pro": str})
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if frame["hgvs_pro"].duplicated().any():
        raise ValueError(f"duplicate HGVS in {path}")
    return frame.set_index("hgvs_pro")["score"]


def top_decile(frame, column):
    ranked = frame.sort_values([column, "hgvs_pro"], ascending=[False, True], kind="mergesort")
    return ranked.head(math.ceil(len(ranked) * 0.1))


def correlations(frame, source, target):
    paired = frame[[source, target]].dropna()
    result = {
        "n_paired": int(len(paired)),
        "pearson": float(pearsonr(paired[source], paired[target]).statistic),
        "spearman": float(spearmanr(paired[source], paired[target]).statistic),
    }
    for label, column in [("source_top10", source), ("target_top10", target)]:
        tail = top_decile(paired.reset_index(), column)
        result[f"{label}_n"] = int(len(tail))
        result[f"{label}_spearman"] = float(spearmanr(tail[source], tail[target]).statistic)
    return result


def percentile_interval(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(args.protocol).read_text())
    condition_files = protocol["data"]["conditions"]
    conditions = list(condition_files)
    sources = protocol["directions"]["sources"]
    targets = protocol["directions"]["targets"]
    budgets = protocol["pools"]["budgets"]
    replicates = protocol["pools"]["replicates_per_position_and_budget"]

    scores = {name: read_score(data_dir / filename) for name, filename in condition_files.items()}
    unselected = read_score(data_dir / protocol["data"]["unselected_file"])
    reference_ids = set(scores[conditions[0]].index)
    if any(set(score.index) != reference_ids for score in scores.values()) or set(unselected.index) != reference_ids:
        raise ValueError("condition HGVS sets differ")
    frame = pd.DataFrame(scores)
    frame["unselected"] = unselected
    frame["hgvs_pro"] = frame.index
    frame["position"] = frame["hgvs_pro"].str.extract(r"p\.[A-Za-z*]{3}(\d+)", expand=False).astype(int)
    if frame["position"].isna().any() or frame["hgvs_pro"].isna().any():
        raise ValueError("position parsing failed")

    positions = np.array(sorted(frame["position"].unique()))
    split_seed = protocol["split"]["seed"]
    ordered = sorted(positions, key=lambda p: hashlib.sha256(f"scenario_tem1_v1|split|{split_seed}|{p}".encode()).hexdigest())
    n_train = int(len(ordered) * protocol["split"]["train_fraction"])
    n_dev = int(len(ordered) * protocol["split"]["dev_fraction"])
    split_map = {p: "train" for p in ordered[:n_train]}
    split_map.update({p: "dev" for p in ordered[n_train:n_train + n_dev]})
    split_map.update({p: "test" for p in ordered[n_train + n_dev:]})
    frame["split"] = frame["position"].map(split_map)
    split_frame = frame[frame["split"] == args.split].copy()
    split_positions = sorted(split_frame["position"].unique())
    complete_mask = frame[conditions].notna().all(axis=1)
    complete = frame[complete_mask & (frame["split"] == args.split)].copy()

    audit = {
        "all_variants": int(len(frame)),
        "all_positions": int(len(positions)),
        "split_counts": {key: int(sum(frame["split"] == key)) for key in ["train", "dev", "test"]},
        "split_position_counts": {key: int(sum(frame.loc[frame["split"] == key, "position"].nunique() for _ in [0])) for key in ["train", "dev", "test"]},
        "requested_split": args.split,
        "requested_split_variants": int(len(split_frame)),
        "requested_split_positions": int(len(split_positions)),
        "complete_six_condition_variants_in_split": int(len(complete)),
        "complete_six_condition_positions_in_split": int(complete["position"].nunique()),
        "missing_by_condition_in_split": {key: int(split_frame[key].isna().sum()) for key in conditions},
        "same_variant_set_across_seven_conditions": True,
        "position_disjoint_from_other_splits": True,
    }

    correlation_rows = []
    for source in sources:
        for target in targets:
            if source == target:
                continue
            row = {"split": args.split, "source": source, "target": target}
            row.update(correlations(split_frame, source, target))
            correlation_rows.append(row)
    correlation_frame = pd.DataFrame(correlation_rows)
    correlation_frame.to_csv(output_dir / "correlations.tsv", sep="\t", index=False)

    membership_frames = []
    pool_frames = []
    pool_seed = protocol["pools"]["seed"]
    for position in sorted(complete["position"].unique()):
        variants = sorted(complete.loc[complete["position"] == position, "hgvs_pro"])
        for budget in budgets:
            if len(variants) < budget:
                continue
            for replicate in range(replicates):
                rng = np.random.default_rng(stable_seed(f"scenario_tem1_v1|pool|{pool_seed}|{position}|{budget}|{replicate}"))
                chosen = sorted(rng.choice(variants, size=budget, replace=False).tolist())
                pool_id = f"p{position}_N{budget}_r{replicate:02d}"
                pool_frames.append({"pool_id": pool_id, "position": position, "N": budget, "replicate": replicate})
                membership_frames.append(pd.DataFrame({"pool_id": pool_id, "position": position, "N": budget, "replicate": replicate, "hgvs_pro": chosen}))
    pools = pd.DataFrame(pool_frames)
    membership = pd.concat(membership_frames, ignore_index=True)
    membership.to_csv(output_dir / "pool_membership.tsv.gz", sep="\t", index=False, compression="gzip")

    directions = [(source, target) for source in sources for target in targets if source != target]
    if len(directions) != protocol["directions"]["expected_direction_count"]:
        raise ValueError("direction count differs from frozen protocol")
    selection_rows = []
    position_effect_rows = []
    metric_names = ["bon_regret", "random_n_regret", "delta_bon_minus_random", "turnover_proxy_vs_oracle", "turnover_source_oracle_vs_target_oracle"]
    effects = {(source, target, budget): {} for source, target in directions for budget in budgets}
    joined = membership.set_index("hgvs_pro").join(frame[conditions + ["unselected"]])
    for pool in pool_frames:
        pool_id = pool["pool_id"]
        position = pool["position"]
        budget = pool["N"]
        current = joined.loc[joined["pool_id"] == pool_id]
        if len(current) != budget or current[conditions].isna().any().any():
            raise ValueError(f"invalid fixed pool {pool_id}")
        for source, target in directions:
            proxy_pick = min(current.index, key=lambda x: (-current.loc[x, source], x))
            target_oracle = min(current.index, key=lambda x: (-current.loc[x, target], x))
            source_oracle = min(current.index, key=lambda x: (-current.loc[x, source], x))
            target_values = current[target].to_numpy(dtype=float)
            oracle_value = float(current.loc[target_oracle, target])
            bon = oracle_value - float(current.loc[proxy_pick, target])
            random_n = oracle_value - float(np.mean(target_values))
            row = {
                "pool_id": pool_id, "position": position, "N": budget, "replicate": pool["replicate"],
                "source": source, "target": target, "proxy_pick": proxy_pick,
                "target_oracle": target_oracle, "source_oracle": source_oracle,
                "bon_regret": bon, "random_n_regret": random_n,
                "delta_bon_minus_random": bon - random_n,
                "turnover_proxy_vs_oracle": float(proxy_pick != target_oracle),
                "turnover_source_oracle_vs_target_oracle": float(source_oracle != target_oracle),
            }
            selection_rows.append(row)
            effects[(source, target, budget)].setdefault(position, []).append([row[name] for name in metric_names])
    selections = pd.DataFrame(selection_rows)
    selections.to_csv(output_dir / "selections.tsv.gz", sep="\t", index=False, compression="gzip")

    for (source, target, budget), by_position in effects.items():
        for position, values in by_position.items():
            means = np.mean(np.asarray(values), axis=0)
            position_effect_rows.append({"position": position, "N": budget, "source": source, "target": target, "n_pools": len(values), **dict(zip(metric_names, means))})
    position_effects = pd.DataFrame(position_effect_rows)
    position_effects.to_csv(output_dir / "position_effects.tsv.gz", sep="\t", index=False, compression="gzip")

    rng = np.random.default_rng(protocol["bootstrap"]["seed"])
    n_bootstrap = protocol["bootstrap"]["resamples"]
    bootstrap_positions_by_budget = {
        budget: sorted(p for p in complete["position"].unique() if sum(complete["position"] == p) >= budget)
        for budget in budgets
    }
    position_draws = {}
    for budget in budgets:
        eligible = bootstrap_positions_by_budget[budget]
        draws = rng.integers(0, len(eligible), size=(n_bootstrap, len(eligible)))
        position_draws[budget] = draws
        np.save(output_dir / f"bootstrap_position_indices_N{budget}.npy", draws)
    bootstrap_arrays = {}
    direction_rows = []
    for source, target in directions:
        for budget in budgets:
            key = (source, target, budget)
            available = sorted(effects[key])
            if available != bootstrap_positions_by_budget[budget]:
                raise ValueError("budget-specific bootstrap positions differ from pool availability")
            matrix = np.asarray([np.mean(effects[key][p], axis=0) for p in available])
            point = matrix.mean(axis=0)
            sampled = matrix[position_draws[budget]].mean(axis=1)
            for j, metric in enumerate(metric_names):
                bootstrap_arrays[f"{source}__{target}__N{budget}__{metric}"] = sampled[:, j]
            row = {"split": args.split, "source": source, "target": target, "N": budget, "n_positions": len(available), "n_pools": int(len(selections[(selections.source == source) & (selections.target == target) & (selections.N == budget)]))}
            for j, metric in enumerate(metric_names):
                row[metric] = float(point[j])
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = percentile_interval(sampled[:, j])
            row["cross_condition_direction"] = "amp_to_cefotaxime" if target == "cefotaxime0.15" else ("higher_amp" if conditions.index(target) > conditions.index(source) else "lower_amp")
            direction_rows.append(row)
    direction_results = pd.DataFrame(direction_rows)
    direction_results.to_csv(output_dir / "direction_results.tsv", sep="\t", index=False)

    overall_rows = []
    for budget in budgets:
        sampled_metrics = {metric: [] for metric in metric_names}
        point_metrics = {metric: [] for metric in metric_names}
        for source, target in directions:
            key = f"{source}__{target}__N{budget}"
            for metric in metric_names:
                sampled_metrics[metric].append(bootstrap_arrays[f"{key}__{metric}"])
                point_metrics[metric].append(float(direction_results.loc[(direction_results.source == source) & (direction_results.target == target) & (direction_results.N == budget), metric].iloc[0]))
        row = {"split": args.split, "N": budget, "n_directions": len(directions)}
        for metric in metric_names:
            samples = np.mean(sampled_metrics[metric], axis=0)
            row[metric] = float(np.mean(point_metrics[metric]))
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = percentile_interval(samples)
            bootstrap_arrays[f"overall__N{budget}__{metric}"] = samples
        subset = direction_results[direction_results.N == budget]
        row["directions_negative_delta"] = int((subset.delta_bon_minus_random < 0).sum())
        row["directions_positive_delta"] = int((subset.delta_bon_minus_random > 0).sum())
        row["amp_to_cef_directions_negative_delta"] = int((subset[subset.target == "cefotaxime0.15"].delta_bon_minus_random < 0).sum())
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(output_dir / "overall_results.tsv", sep="\t", index=False)
    np.savez_compressed(output_dir / "bootstrap_effects.npz", **bootstrap_arrays)

    run = {
        "protocol_id": protocol["protocol_id"], "split": args.split,
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(args.protocol), "code_sha256": sha256_file(__file__),
        "data_audit": audit, "n_directions": len(directions), "n_selections": len(selections),
        "test_execution": args.split == "test", "model_fitting": False,
        "regret_units": protocol["data"]["score_semantics"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run, indent=2) + "\n")
    (output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"split": args.split, "directions": len(directions), "selections": len(selections), "overall": overall_rows}, indent=2))


if __name__ == "__main__":
    main()
