from __future__ import annotations

import json

import numpy as np
import pytest

from paper2_risk import calibration_gate, select_quantile, summarize, threshold_at_quantile
from run_protocol import compare_learner_record, learner_decisions


def row(pool, seed, risk, regret):
    return {"pool_id": pool, "seed": seed, "risk": risk, "regret": regret}


def test_linear_quantile_and_inclusive_acceptance():
    rows = [row("a", 1, value, value) for value in (0.0, 1.0, 2.0, 3.0)]
    threshold = threshold_at_quantile(rows, 0.5)
    assert threshold == 1.5
    result = summarize(rows, 1.0, n_boot=20)
    assert result["accepted_decisions"] == 2
    assert result["coverage"] == 0.5
    assert result["accepted_mean_regret"] == 0.5
    assert result["full_bon_mean_regret"] == 1.5
    assert result["regret_reduction"] == 1.0


def test_pool_bootstrap_is_reproducible_and_seed_aware():
    rows = [row(pool, seed, seed / 10, regret) for pool in ("a", "b") for seed, regret in enumerate((4.0, 2.0, 1.0, 0.0))]
    left = summarize(rows, 0.15, n_boot=100, seed=71)
    right = summarize(rows, 0.15, n_boot=100, seed=71)
    different = summarize(rows, 0.15, n_boot=100, seed=72)
    assert left == right
    assert different["regret_reduction_95ci"] == [-1.25, -1.25]
    assert left["pools"] == 2
    assert left["decisions"] == 8


def test_gate_requires_effect_and_precision():
    passed = {"accepted_decisions": 8, "accepted_pools": 3, "regret_reduction": 1.0, "regret_reduction_95ci": [0.2, 2.0]}
    assert calibration_gate(passed)["pass"]
    failed = dict(passed, regret_reduction_95ci=[-0.1, 2.0])
    assert not calibration_gate(failed)["pass"]
    assert not calibration_gate(dict(passed, accepted_pools=2))["pass"]


def test_dev_selection_uses_only_the_frozen_rule():
    rows = [row("a", seed, float(seed), float(10 - seed)) for seed in range(10)]
    selected, grid = select_quantile(rows)
    expected = max(grid, key=lambda item: (item["dev"]["regret_reduction"], item["dev"]["coverage"], -item["quantile"]))
    assert len(grid) == 3
    assert selected == expected


def test_learner_role_guard_blocks_outer():
    with pytest.raises(PermissionError):
        learner_decisions("outer_designed")


def test_record_comparison_rejects_changed_selection():
    expected = {"selected_id": "x", "selected_index": 2, "regret": 1.0, "risk": 0.2, "proposals": [2]}
    actual = {"pool_id": "dev_000", "seed": 101, "selected_id": "x", "selected_index": 2, "regret": 1.0, "risk": 0.2}
    compare_learner_record(actual, [2], expected)
    with pytest.raises(ValueError):
        compare_learner_record(dict(actual, selected_id="y"), [2], expected)


def test_summarize_rejects_empty_or_zero_acceptance():
    with pytest.raises(ValueError):
        summarize([], 1.0)
    result = summarize([row("a", 1, 2.0, 3.0)], 1.0, n_boot=10)
    assert result["accepted_decisions"] == 0
    assert result["regret_reduction"] is None
    assert not calibration_gate(result)["pass"]
