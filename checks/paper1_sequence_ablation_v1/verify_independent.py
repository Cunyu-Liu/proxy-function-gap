#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

FEATURES = ["gc", "uAUG", "longest_poly_u", "support"]


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def numeric_rows(rows):
    output = []
    text = {"context", "pool_id", "random_sequence_id", "bon_sequence_id", "family", "feature"}
    for row in rows:
        output.append({key: (value if key in text else float(value)) for key, value in row.items()})
    return output


def seed(base, label, context, cap):
    value = hashlib.sha256(f"{base}|{label}|{context}|{cap}".encode()).hexdigest()
    return int(value[:8], 16)


def quantiles(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def p_value(values):
    a = (1 + sum(x <= 0 for x in values)) / (len(values) + 1)
    b = (1 + sum(x >= 0 for x in values)) / (len(values) + 1)
    return min(1.0, 2.0 * min(a, b))


def group(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["pool_id"]].append(row)
    return [result[key] for key in sorted(result)]


def mean_boot(groups, base, label, context, cap):
    values = np.array([np.mean([row["delta_regret"] for row in rows]) for rows in groups])
    if np.all(values == 0):
        return 0.0, [0.0, 0.0], 1.0
    generator = np.random.RandomState(seed(base, label, context, cap))
    draws = np.empty(10000)
    for index in range(10000):
        draws[index] = np.mean(values[generator.randint(0, len(values), len(values))])
    return float(np.mean(values)), quantiles(draws), p_value(draws)


def regression(groups):
    # Independently construct a design with one indicator per resampled cluster.
    n = sum(map(len, groups))
    design = np.zeros((n, len(groups) + 4))
    response = np.zeros(n)
    i = 0
    for j, rows in enumerate(groups):
        for row in rows:
            design[i, j] = 1
            design[i, len(groups):] = [row["delta_" + feature] for feature in FEATURES]
            response[i] = row["delta_regret"]
            i += 1
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    return float(sum(beta[:len(groups)]) / len(groups))


def adjusted_boot(groups, base, context, cap):
    if all(row["delta_regret"] == 0 and all(row["delta_" + feature] == 0 for feature in FEATURES) for rows in groups for row in rows):
        return 0.0, [0.0, 0.0], 1.0
    generator = np.random.RandomState(seed(base, "adjusted", context, cap))
    draws = np.empty(10000)
    for index in range(10000):
        selected = generator.randint(0, len(groups), len(groups))
        draws[index] = regression([groups[j] for j in selected])
    return regression(groups), quantiles(draws), p_value(draws)


def corrections(p):
    order = sorted(range(len(p)), key=lambda i: (p[i], i))
    holm = [0.0] * len(p)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(p) - rank) * p[index]))
        holm[index] = previous
    return holm, [min(1.0, len(p) * value) for value in p]


def close(actual, expected):
    if actual is None or expected is None:
        return actual == expected
    return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.frozen).read_text())
    root = Path(args.analysis)
    pairs = numeric_rows(read_tsv(root / "derived" / "paired_rows_remote_only.tsv"))
    expected_primary = numeric_rows(read_tsv(root / "results" / "primary_paired.tsv"))
    mismatches = []
    maximum_difference = 0.0
    comparisons = 0

    def compare(actual, expected, label):
        nonlocal maximum_difference, comparisons
        comparisons += 1
        if not close(actual, expected):
            mismatches.append({"field": label, "actual": actual, "expected": expected})
        elif actual is not None:
            maximum_difference = max(maximum_difference, abs(float(actual) - float(expected)))

    for context in config["contexts"]:
        for cap in config["caps"]:
            selected = [row for row in pairs if row["context"] == context and row["cap"] == cap]
            groups = group(selected)
            unadjusted = mean_boot(groups, config["bootstrap"]["seed"], "unadjusted", context, cap)
            adjusted = adjusted_boot(groups, config["bootstrap"]["seed"], context, cap)
            target = next(row for row in expected_primary if row["context"] == context and row["cap"] == cap)
            unadjusted_values = [unadjusted[0], unadjusted[1][0], unadjusted[1][1], unadjusted[2]]
            adjusted_values = [adjusted[0], adjusted[1][0], adjusted[1][1], adjusted[2]]
            for name, value in zip(["effect", "ci_low", "ci_high", "p"], unadjusted_values):
                compare(value, target["unadjusted_" + name], f"{context}/{cap}/unadjusted_{name}")
            for name, value in zip(["effect", "ci_low", "ci_high", "p"], adjusted_values):
                compare(value, target["adjusted_" + name], f"{context}/{cap}/adjusted_{name}")
    for prefix in ["unadjusted", "adjusted"]:
        holm, bonferroni = corrections([row[prefix + "_p"] for row in expected_primary])
        for row, h, b in zip(expected_primary, holm, bonferroni):
            compare(h, row[prefix + "_holm_p"], f"{row['context']}/{row['cap']}/{prefix}_holm")
            compare(b, row[prefix + "_bonferroni_p"], f"{row['context']}/{row['cap']}/{prefix}_bonferroni")

    expected_family = numeric_rows(read_tsv(root / "results" / "sequence_family_sensitivity.tsv"))
    cutoff = config["poly_u_q75_nearest_rank"]
    for family in ["no_uAUG", "no_polyU_upper_quartile"]:
        family_rows = []
        for context in config["contexts"]:
            for cap in config["caps"]:
                candidates = [row for row in pairs if row["context"] == context and row["cap"] == cap]
                if family == "no_uAUG":
                    selected = [row for row in candidates if not row["bon_uAUG"] and not row["random_uAUG"]]
                else:
                    selected = [row for row in candidates if row["bon_longest_poly_u"] < cutoff and row["random_longest_poly_u"] < cutoff]
                estimate = mean_boot(group(selected), config["bootstrap"]["seed"], family, context, cap)
                target = next(row for row in expected_family if row["family"] == family and row["context"] == context and row["cap"] == cap)
                family_rows.append(target)
                compare(len(selected), target["n_pairs"], f"{family}/{context}/{cap}/n_pairs")
                compare(len(group(selected)), target["n_pools"], f"{family}/{context}/{cap}/n_pools")
                estimate_values = [estimate[0], estimate[1][0], estimate[1][1], estimate[2]]
                for name, value in zip(["effect", "ci_low", "ci_high", "p"], estimate_values):
                    compare(value, target[name], f"{family}/{context}/{cap}/{name}")
        holm, bonferroni = corrections([row["p"] for row in family_rows])
        for row, h, b in zip(family_rows, holm, bonferroni):
            compare(h, row["holm_p"], f"{row['family']}/{row['context']}/{row['cap']}/holm")
            compare(b, row["bonferroni_p"], f"{row['family']}/{row['context']}/{row['cap']}/bonferroni")

    expected_score = numeric_rows(read_tsv(root / "results" / "score_matched_features.tsv"))
    analysis_root = Path(config["analysis_root"])
    for context in config["contexts"]:
        reconstructed = []
        for pool_index in range(16):
            stem = f"castillo_{context}_{pool_index:03d}"
            candidates = json.loads((analysis_root / context / "cnn_scores" / f"{stem}.json").read_text())["records"]
            records = json.loads((analysis_root / context / "evaluation" / f"{stem}.json").read_text())["records"]
            bon_records = [row for row in records if row["proposal_cap"] == 16 and row["strategy"] == "bon"]
            for selected in bon_records:
                bon = next(row for row in candidates if row["sequence_id"] == selected["selected_id"])
                eligible = [row for row in candidates if row["sequence_id"] != bon["sequence_id"]]
                matched = min(eligible, key=lambda row: (abs(row["proxy_mean"] - bon["proxy_mean"]), hashlib.sha256(row["sequence_id"].encode()).hexdigest()))
                reconstructed.append({"pool_id": selected["pool_id"], **{feature: float(bon[feature]) - float(matched[feature]) for feature in FEATURES + ["proxy_mean"]}})
        for feature in FEATURES + ["proxy_mean"]:
            groups = group(reconstructed)
            means = np.array([np.mean([row[feature] for row in rows]) for rows in groups])
            generator = np.random.RandomState(seed(config["bootstrap"]["seed"], "score_match_" + feature, context, 16))
            draws = [np.mean(means[generator.randint(0, len(means), len(means))]) for _ in range(10000)]
            ci = quantiles(draws)
            target = next(row for row in expected_score if row["context"] == context and row["feature"] == feature)
            compare(float(np.mean(means)), target["paired_delta"], f"score/{context}/{feature}/delta")
            compare(ci[0], target["ci_low"], f"score/{context}/{feature}/ci_low")
            compare(ci[1], target["ci_high"], f"score/{context}/{feature}/ci_high")

    report = {
        "round": 2,
        "pass": not mismatches,
        "implementation": "separate loader, design construction, bootstrap and multiplicity implementation; no import of primary analysis module",
        "comparisons": comparisons,
        "maximum_absolute_difference": maximum_difference,
        "mismatches": mismatches,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
