from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (101, 202, 303, 404, 505)
QUANTILES = (0.50, 0.75, 0.90)
N_BOOT = 10000
BOOTSTRAP_SEED = 20261125
MIN_ACCEPTED_DECISIONS = 8
MIN_ACCEPTED_POOLS = 3


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def threshold_at_quantile(decisions, quantile):
    risks = np.asarray([row["risk"] for row in decisions], dtype=np.float64)
    return float(np.quantile(risks, quantile, method="linear"))


def select_quantile(dev_rows):
    candidates = []
    for quantile in QUANTILES:
        threshold = threshold_at_quantile(dev_rows, quantile)
        summary = summarize(dev_rows, threshold)
        candidates.append({"quantile": quantile, "threshold": threshold, "dev": summary})
    selected = max(candidates, key=lambda row: (row["dev"]["regret_reduction"], row["dev"]["coverage"], -row["quantile"]))
    return selected, candidates


def summarize(decisions, threshold, n_boot=N_BOOT, seed=BOOTSTRAP_SEED):
    if not decisions:
        raise ValueError("empty decisions")
    risks = np.asarray([row["risk"] for row in decisions], dtype=np.float64)
    regrets = np.asarray([row["regret"] for row in decisions], dtype=np.float64)
    pools = np.asarray([row["pool_id"] for row in decisions])
    accepted = risks <= threshold
    if not accepted.any():
        accepted_mean = None
        reduction = None
    else:
        accepted_mean = float(regrets[accepted].mean())
        reduction = float(regrets.mean() - accepted_mean)
    unique_pools = sorted(set(pools.tolist()))
    indices_by_pool = [np.flatnonzero(pools == pool) for pool in unique_pools]
    rng = np.random.default_rng(seed)
    reductions = []
    accepted_means = []
    coverages = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(unique_pools), size=len(unique_pools))
        indices = np.concatenate([indices_by_pool[index] for index in sampled])
        mask = accepted[indices]
        if not mask.any():
            continue
        full_value = float(regrets[indices].mean())
        accepted_value = float(regrets[indices][mask].mean())
        reductions.append(full_value - accepted_value)
        accepted_means.append(accepted_value)
        coverages.append(float(mask.mean()))

    def interval(values):
        if not values:
            return None
        return [float(value) for value in np.percentile(values, [2.5, 97.5])]

    return {
        "threshold": float(threshold),
        "decisions": len(decisions),
        "pools": len(unique_pools),
        "accepted_decisions": int(accepted.sum()),
        "accepted_pools": len(set(pools[accepted].tolist())),
        "coverage": float(accepted.mean()),
        "full_bon_mean_regret": float(regrets.mean()),
        "accepted_mean_regret": accepted_mean,
        "regret_reduction": reduction,
        "regret_reduction_95ci": interval(reductions),
        "accepted_mean_regret_95ci": interval(accepted_means),
        "coverage_95ci": interval(coverages),
        "bootstrap": {"unit": "pool", "rounds": n_boot, "seed": seed, "valid_rounds": len(reductions)},
    }


def calibration_gate(summary):
    checks = {
        "minimum_accepted_decisions": summary["accepted_decisions"] >= MIN_ACCEPTED_DECISIONS,
        "minimum_accepted_pools": summary["accepted_pools"] >= MIN_ACCEPTED_POOLS,
        "positive_point_reduction": summary["regret_reduction"] is not None and summary["regret_reduction"] > 0,
        "reduction_ci_excludes_zero": summary["regret_reduction_95ci"] is not None and summary["regret_reduction_95ci"][0] > 0,
    }
    return {"pass": all(checks.values()), "checks": checks}


def finite_row(row):
    for key in ("risk", "regret", "proxy_mean", "selected_measured_rl", "pool_best_measured_rl"):
        if key in row and row[key] is not None and not math.isfinite(float(row[key])):
            raise ValueError(f"nonfinite {key}")
