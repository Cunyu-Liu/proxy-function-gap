#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FEATURES = ["gc", "uAUG", "longest_poly_u", "support"]
SCORE_FEATURES = FEATURES + ["proxy_mean"]


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def bootstrap_seed(base, analysis, context, cap):
    digest = sha256_text(f"{base}|{analysis}|{context}|{cap}")
    return int(digest[:8], 16)


def longest_poly_u(sequence):
    best = current = 0
    for base in sequence:
        current = current + 1 if base == "T" else 0
        best = max(best, current)
    return best


def load_pairs(config):
    pool_root = Path(config["pool_root"])
    analysis_root = Path(config["analysis_root"])
    pairs = []
    candidates_by_context = {}
    for context in config["contexts"]:
        candidates_by_context[context] = {}
        for pool_index in range(16):
            stem = f"castillo_{context}_{pool_index:03d}"
            score_path = analysis_root / context / "cnn_scores" / f"{stem}.json"
            evaluation_path = analysis_root / context / "evaluation" / f"{stem}.json"
            candidate_records = read_json(score_path)["records"]
            candidate_map = {r["sequence_id"]: r for r in candidate_records}
            candidates_by_context[context].update(candidate_map)
            with gzip.open(pool_root / context / "evaluator" / "labels.tsv.gz", "rt") as handle:
                labels = {r["sequence_id"]: float(r["measured_rl"]) for r in csv.DictReader(handle, delimiter="\t")}
            pool_labels = {sequence_id: labels[sequence_id] for sequence_id in candidate_map}
            pool_maximum = max(pool_labels.values())
            evaluation = read_json(evaluation_path)["records"]
            lookup = {(r["proposal_cap"], r["seed"], r["strategy"]): r for r in evaluation}
            for cap in config["caps"]:
                for seed in config["seeds"]:
                    random_record = lookup[(cap, seed, "random")]
                    bon_record = lookup[(cap, seed, "bon")]
                    random_candidate = candidate_map[random_record["selected_id"]]
                    bon_candidate = candidate_map[bon_record["selected_id"]]
                    random_mrl = pool_labels[random_record["selected_id"]]
                    bon_mrl = pool_labels[bon_record["selected_id"]]
                    random_regret = pool_maximum - random_mrl
                    bon_regret = pool_maximum - bon_mrl
                    assert abs(random_regret - random_record["regret"]) < 1e-9
                    assert abs(bon_regret - bon_record["regret"]) < 1e-9
                    row = {
                        "context": context,
                        "pool_id": random_record["pool_id"],
                        "pool_index": pool_index,
                        "seed": seed,
                        "cap": cap,
                        "random_sequence_id": random_record["selected_id"],
                        "bon_sequence_id": bon_record["selected_id"],
                        "random_mrl": random_mrl,
                        "bon_mrl": bon_mrl,
                        "delta_regret": bon_regret - random_regret,
                    }
                    for feature in SCORE_FEATURES:
                        row[f"random_{feature}"] = float(random_candidate[feature])
                        row[f"bon_{feature}"] = float(bon_candidate[feature])
                        row[f"delta_{feature}"] = row[f"bon_{feature}"] - row[f"random_{feature}"]
                    pairs.append(row)
    return pairs, candidates_by_context


def pool_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["pool_id"]].append(row)
    return [groups[key] for key in sorted(groups)]


def interval(values):
    if not values:
        return [None, None]
    low, high = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return [float(low), float(high)]


def bootstrap_p(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    lower = (1 + int(np.sum(values <= 0))) / (len(values) + 1)
    upper = (1 + int(np.sum(values >= 0))) / (len(values) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def unadjusted_bootstrap(groups, base_seed, analysis, context, cap):
    pool_means = np.asarray([np.mean([r["delta_regret"] for r in group]) for group in groups])
    effect = float(np.mean(pool_means))
    if np.all(pool_means == 0):
        return effect, [0.0, 0.0], 1.0, []
    rng = np.random.RandomState(bootstrap_seed(base_seed, analysis, context, cap))
    replicates = []
    for _ in range(10000):
        indices = rng.randint(0, len(pool_means), len(pool_means))
        replicates.append(float(np.mean(pool_means[indices])))
    return effect, interval(replicates), bootstrap_p(replicates), replicates


def adjusted_fit(groups):
    if not groups:
        return None
    pool_count = len(groups)
    row_count = sum(len(group) for group in groups)
    x = np.zeros((row_count, pool_count + len(FEATURES)))
    y = np.empty(row_count)
    position = 0
    for pool_position, group in enumerate(groups):
        for row in group:
            x[position, pool_position] = 1.0
            for feature_position, feature in enumerate(FEATURES):
                x[position, pool_count + feature_position] = row[f"delta_{feature}"]
            y[position] = row["delta_regret"]
            position += 1
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError(f"rank-deficient paired regression: rank={rank}, columns={x.shape[1]}")
    return float(np.mean(coefficients[:pool_count]))


def adjusted_bootstrap(groups, base_seed, context, cap):
    if all(row["delta_regret"] == 0 and all(row[f"delta_{feature}"] == 0 for feature in FEATURES) for group in groups for row in group):
        return 0.0, [0.0, 0.0], 1.0, []
    effect = adjusted_fit(groups)
    rng = np.random.RandomState(bootstrap_seed(base_seed, "adjusted", context, cap))
    replicates = []
    for _ in range(10000):
        indices = rng.randint(0, len(groups), len(groups))
        sampled = [groups[index] for index in indices]
        replicates.append(adjusted_fit(sampled))
    return effect, interval(replicates), bootstrap_p(replicates), replicates


def add_multiplicity(rows):
    for prefix in ["unadjusted", "adjusted"]:
        p_values = [row[f"{prefix}_p"] for row in rows]
        order = sorted(range(len(rows)), key=lambda index: (p_values[index], index))
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, min(1.0, (len(rows) - rank) * p_values[index]))
            rows[index][f"{prefix}_holm_p"] = running
        for row in rows:
            row[f"{prefix}_bonferroni_p"] = min(1.0, len(rows) * row[f"{prefix}_p"])


def clean_result(row):
    return {key: value for key, value in row.items() if not key.endswith("_replicates")}


def primary_analysis(pairs, config, output):
    results = []
    pool_rows = []
    feature_rows = []
    for context in config["contexts"]:
        for cap in config["caps"]:
            selected = [r for r in pairs if r["context"] == context and r["cap"] == cap]
            groups = pool_groups(selected)
            unadjusted = unadjusted_bootstrap(groups, config["bootstrap"]["seed"], "unadjusted", context, cap)
            adjusted = adjusted_bootstrap(groups, config["bootstrap"]["seed"], context, cap)
            result = {
                "context": context,
                "cap": cap,
                "n_pairs": len(selected),
                "n_pools": len(groups),
                "unadjusted_effect": unadjusted[0],
                "unadjusted_ci_low": unadjusted[1][0],
                "unadjusted_ci_high": unadjusted[1][1],
                "unadjusted_p": unadjusted[2],
                "adjusted_effect": adjusted[0],
                "adjusted_ci_low": adjusted[1][0],
                "adjusted_ci_high": adjusted[1][1],
                "adjusted_p": adjusted[2],
            }
            results.append(result)
            for group in groups:
                pool_rows.append({
                    "context": context,
                    "cap": cap,
                    "pool_id": group[0]["pool_id"],
                    "n_pairs": len(group),
                    "mean_delta_regret": float(np.mean([r["delta_regret"] for r in group])),
                    **{f"mean_delta_{feature}": float(np.mean([r[f"delta_{feature}"] for r in group])) for feature in FEATURES},
                })
            for feature in FEATURES:
                feature_rows.append({
                    "context": context,
                    "cap": cap,
                    "feature": feature,
                    "bon_mean": float(np.mean([r[f"bon_{feature}"] for r in selected])),
                    "random_mean": float(np.mean([r[f"random_{feature}"] for r in selected])),
                    "paired_delta": float(np.mean([r[f"delta_{feature}"] for r in selected])),
                })
    add_multiplicity(results)
    write_tsv(output / "results" / "primary_paired.tsv", [clean_result(r) for r in results])
    write_tsv(output / "results" / "pool_effects.tsv", pool_rows)
    write_tsv(output / "results" / "selection_feature_contrasts.tsv", feature_rows)
    return results


def family_analysis(pairs, config, output):
    cutoff = config["poly_u_q75_nearest_rank"]
    all_results = []
    for family in ["no_uAUG", "no_polyU_upper_quartile"]:
        results = []
        for context in config["contexts"]:
            for cap in config["caps"]:
                candidates = [r for r in pairs if r["context"] == context and r["cap"] == cap]
                if family == "no_uAUG":
                    selected = [r for r in candidates if not r["bon_uAUG"] and not r["random_uAUG"]]
                else:
                    selected = [r for r in candidates if r["bon_longest_poly_u"] < cutoff and r["random_longest_poly_u"] < cutoff]
                groups = pool_groups(selected)
                estimate = unadjusted_bootstrap(groups, config["bootstrap"]["seed"], family, context, cap)
                results.append({
                    "family": family,
                    "context": context,
                    "cap": cap,
                    "n_pairs": len(selected),
                    "n_pools": len(groups),
                    "effect": estimate[0],
                    "ci_low": estimate[1][0],
                    "ci_high": estimate[1][1],
                    "p": estimate[2],
                })
        for prefix in [""]:
            p_values = [row["p"] for row in results]
            if any(value is None for value in p_values):
                raise ValueError(f"missing p-value in {family}")
            order = sorted(range(len(results)), key=lambda index: (p_values[index], index))
            running = 0.0
            for rank, index in enumerate(order):
                running = max(running, min(1.0, (len(results) - rank) * p_values[index]))
                results[index]["holm_p"] = running
            for row in results:
                row["bonferroni_p"] = min(1.0, len(results) * row["p"])
        all_results.extend(results)
    write_tsv(output / "results" / "sequence_family_sensitivity.tsv", all_results)
    return all_results


def choose_score_match(bon_id, candidate_records):
    bon = next(r for r in candidate_records if r["sequence_id"] == bon_id)
    candidates = [r for r in candidate_records if r["sequence_id"] != bon_id]
    return min(candidates, key=lambda r: (abs(float(r["proxy_mean"]) - float(bon["proxy_mean"])), sha256_text(r["sequence_id"])))


def score_matched_analysis(config, output):
    analysis_root = Path(config["analysis_root"])
    rows = []
    for context in config["contexts"]:
        for pool_index in range(16):
            stem = f"castillo_{context}_{pool_index:03d}"
            candidate_records = read_json(analysis_root / context / "cnn_scores" / f"{stem}.json")["records"]
            evaluation = read_json(analysis_root / context / "evaluation" / f"{stem}.json")["records"]
            bon_records = [r for r in evaluation if r["proposal_cap"] == 16 and r["strategy"] == "bon"]
            assert len(bon_records) == 5
            for bon_record in bon_records:
                matched = choose_score_match(bon_record["selected_id"], candidate_records)
                bon = next(r for r in candidate_records if r["sequence_id"] == bon_record["selected_id"])
                row = {"context": context, "pool_id": bon_record["pool_id"], "seed": bon_record["seed"]}
                for feature in SCORE_FEATURES:
                    row[f"bon_{feature}"] = float(bon[feature])
                    row[f"matched_{feature}"] = float(matched[feature])
                    row[f"delta_{feature}"] = row[f"bon_{feature}"] - row[f"matched_{feature}"]
                rows.append(row)
    write_tsv(output / "derived" / "score_matched_feature_pairs.tsv", rows)
    results = []
    for context in config["contexts"]:
        context_rows = [r for r in rows if r["context"] == context]
        for feature in SCORE_FEATURES:
            groups = pool_groups(context_rows)
            means = np.asarray([np.mean([r[f"delta_{feature}"] for r in group]) for group in groups])
            rng = np.random.RandomState(bootstrap_seed(config["bootstrap"]["seed"], f"score_match_{feature}", context, 16))
            replicates = [float(np.mean(means[rng.randint(0, len(means), len(means))])) for _ in range(10000)]
            ci = interval(replicates)
            results.append({
                "context": context,
                "feature": feature,
                "n_pairs": len(context_rows),
                "n_pools": len(groups),
                "bon_mean": float(np.mean([r[f"bon_{feature}"] for r in context_rows])),
                "matched_mean": float(np.mean([r[f"matched_{feature}"] for r in context_rows])),
                "paired_delta": float(np.mean([r[f"delta_{feature}"] for r in context_rows])),
                "ci_low": ci[0],
                "ci_high": ci[1],
            })
    write_tsv(output / "results" / "score_matched_features.tsv", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_json(Path(args.frozen))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    pairs, _ = load_pairs(config)
    write_tsv(output / "derived" / "paired_rows_remote_only.tsv", pairs)
    primary = primary_analysis(pairs, config, output)
    family = family_analysis(pairs, config, output)
    score = score_matched_analysis(config, output)
    summary = {
        "schema_version": "1.0",
        "analysis": config["analysis"],
        "status": config["status"],
        "new_blind_test": False,
        "causal_claim": False,
        "poly_u_q75": config["poly_u_q75_nearest_rank"],
        "primary_comparisons": len(primary),
        "family_comparisons": len(family),
        "score_matched_contrasts": len(score),
        "primary": primary,
        "sequence_family_sensitivity": family,
        "score_matched_features": score,
    }
    write_json(output / "results" / "summary.json", summary)
    print(json.dumps({"status": "complete", "pairs": len(pairs), "primary": len(primary), "family": len(family), "score_matched": len(score)}))


if __name__ == "__main__":
    main()
