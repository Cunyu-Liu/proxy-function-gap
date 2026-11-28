import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

MUTANT_RE = re.compile(r"^([A-Z])([0-9]+)([A-Z])$")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def parse_mutant(mutant):
    tokens = []
    for token in mutant.split(":"):
        match = MUTANT_RE.fullmatch(token)
        if match is None:
            raise ValueError(f"Invalid mutant token: {mutant}")
        tokens.append((match.group(1), int(match.group(2)), match.group(3)))
    return tokens


def load_labeled_csv(path):
    mutants, labels = [], []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            mutants.append(row["mutant"])
            labels.append(float(row["DMS_score"]))
    return np.asarray(mutants), np.asarray(labels, dtype=np.float64)


def load_unlabeled_csv(path):
    with Path(path).open(newline="") as handle:
        return np.asarray([row["mutant"] for row in csv.DictReader(handle)])


def tie_key(seed, mutant):
    return hashlib.sha256(f"gb1-bon-tie-v1:{seed}:{mutant}".encode()).hexdigest()


def bon_prefix(mutants, scores, seed, maximum_budget):
    order = sorted(range(len(mutants)), key=lambda i: (-float(scores[i]), tie_key(seed, str(mutants[i]))))
    return [str(mutants[i]) for i in order[:maximum_budget]]


def exact_random_expected_max(labels, budget):
    values = np.sort(np.asarray(labels, dtype=np.float64))
    n = len(values)
    denominator = math.comb(n, budget)
    weights = np.zeros(n, dtype=np.float64)
    previous = 0
    for i in range(budget - 1, n):
        current = math.comb(i + 1, budget)
        weights[i] = (current - previous) / denominator
        previous = current
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("RandomN order-statistic weights do not sum to one")
    return float(np.dot(weights, values))


def rankdata_average(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return None
    return float(spearmanr(y_true, y_pred).statistic)


def regression_metrics(y_true, y_pred, floor):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    censored = y_true <= floor + 1e-12
    result = {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "spearman": spearman(y_true, y_pred),
        "censored_fraction": float(censored.mean()),
    }
    for name, mask in (("censored", censored), ("uncensored", ~censored)):
        result[name] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(y_true[mask] - y_pred[mask]))) if mask.any() else None,
            "spearman": spearman(y_true[mask], y_pred[mask]) if mask.sum() > 1 else None,
        }
    return result


def pool_metrics(mutants, labels, predictions, seed, budgets, floor):
    label_map = {str(mutant): float(label) for mutant, label in zip(mutants, labels)}
    prediction_map = {str(mutant): float(score) for mutant, score in zip(mutants, predictions)}
    if len(label_map) != len(mutants) or len(prediction_map) != len(mutants):
        raise RuntimeError("Duplicate pool identifiers")
    ordered = bon_prefix(mutants, predictions, seed, max(budgets))
    oracle = max(label_map.values())
    sorted_labels = np.sort(labels)[::-1]
    top16_threshold = sorted_labels[min(16, len(sorted_labels)) - 1]
    top_decile_threshold = sorted_labels[max(0, math.ceil(0.1 * len(labels)) - 1)]
    output = {}
    for budget in budgets:
        selected = ordered[:budget]
        selected_labels = np.asarray([label_map[mutant] for mutant in selected])
        best = float(selected_labels.max())
        random_max = exact_random_expected_max(labels, budget)
        output[str(budget)] = {
            "budget": int(budget),
            "oracle_lnw": float(oracle),
            "bon_max_lnw": best,
            "random_expected_max_lnw": random_max,
            "bon_regret": float(oracle - best),
            "random_regret": float(oracle - random_max),
            "delta_regret": float(random_max - best),
            "hit_at_budget": bool(best == oracle),
            "selected_mean_lnw": float(selected_labels.mean()),
            "top16_recall": float(np.sum(selected_labels >= top16_threshold) / max(1, int(np.sum(labels >= top16_threshold)))),
            "top_decile_recall": float(np.sum(selected_labels >= top_decile_threshold) / max(1, int(np.sum(labels >= top_decile_threshold)))),
            "bon_selected_censored_fraction": float(np.mean(selected_labels <= floor + 1e-12)),
        }
    return output
