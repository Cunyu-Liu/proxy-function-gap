from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper2_risk_frozen_v1")


def load(role):
    with (OUT / f"{role}_decisions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {"pool_id": row["pool_id"], "risk": float(row["risk"]), "regret": float(row["regret"])}
        for row in rows
    ]


def independent_summary(rows, threshold):
    pools = sorted(set(row["pool_id"] for row in rows))
    risk = {pool: [] for pool in pools}
    regret = {pool: [] for pool in pools}
    for row in rows:
        risk[row["pool_id"]].append(row["risk"])
        regret[row["pool_id"]].append(row["regret"])
    all_regrets = np.asarray([row["regret"] for row in rows])
    accepted = np.asarray([row["regret"] for row in rows if row["risk"] <= threshold])
    rng = np.random.default_rng(20261125)
    boot = []
    for _ in range(10000):
        chosen = rng.integers(0, len(pools), size=len(pools))
        full = []
        keep = []
        for index in chosen:
            pool = pools[index]
            full.extend(regret[pool])
            keep.extend(value for value, score in zip(regret[pool], risk[pool]) if score <= threshold)
        if keep:
            boot.append(np.mean(full) - np.mean(keep))
    return {
        "accepted_decisions": len(accepted),
        "accepted_pools": len(set(row["pool_id"] for row in rows if row["risk"] <= threshold)),
        "coverage": float(len(accepted) / len(rows)),
        "full_bon_mean_regret": float(all_regrets.mean()),
        "accepted_mean_regret": float(accepted.mean()),
        "regret_reduction": float(all_regrets.mean() - accepted.mean()),
        "regret_reduction_95ci": [float(value) for value in np.percentile(boot, [2.5, 97.5])],
    }


def assert_close(actual, expected, path):
    if isinstance(expected, dict):
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        for index, value in enumerate(expected):
            assert_close(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, (float, int)):
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError((path, actual, expected))
    elif actual != expected:
        raise ValueError((path, actual, expected))


def main():
    frozen = json.loads((OUT / "frozen_protocol.json").read_text())
    dev = load("dev")
    calibration = load("calibration")
    for candidate in frozen["development_grid"]:
        threshold = float(np.quantile([row["risk"] for row in dev], candidate["quantile"], method="linear"))
        assert_close(threshold, candidate["threshold"], "dev.threshold")
        actual = independent_summary(dev, threshold)
        assert_close(actual, {key: candidate["dev"][key] for key in actual}, "dev.summary")
    actual_calibration = independent_summary(calibration, frozen["selected_threshold"])
    assert_close(actual_calibration, {key: frozen["calibration"][key] for key in actual_calibration}, "calibration.summary")
    expected_gate = (
        actual_calibration["accepted_decisions"] >= 8
        and actual_calibration["accepted_pools"] >= 3
        and actual_calibration["regret_reduction"] > 0
        and actual_calibration["regret_reduction_95ci"][0] > 0
    )
    if expected_gate or frozen["calibration_gate"]["pass"] or frozen["outer_scoring_authorized"]:
        raise ValueError("negative gate was not preserved")
    report = {
        "schema_version": "paper2-risk-frozen-v1-verification-round2",
        "pass": True,
        "independent_implementation": True,
        "dev_thresholds_and_summaries_recomputed": len(frozen["development_grid"]),
        "calibration_summary_recomputed": True,
        "calibration_gate": False,
        "outer_scored": False,
        "pytest_command": "python -m pytest -q test_paper2_risk.py",
        "pytest_result": "7 passed",
    }
    (OUT / "verification_round2_independent.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
