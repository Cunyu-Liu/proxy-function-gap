import json
import random
from pathlib import Path

import numpy as np
import pytest

from checks.paper1_budget_matched_v1.run_analysis import (
    add_multiplicity,
    bootstrap_stats,
    choose_bon_position,
    decompose,
    draw_nested_prefix,
    random_n_position,
)


def test_nested_prefix_matches_frozen_random_float_algorithm():
    def expected(seed, n):
        rng = random.Random(seed)
        return [min(int(value * 64), 63) for value in [rng.random() for _ in range(n)]]

    for seed in (101, 202, 303, 404, 505):
        prefix = draw_nested_prefix(64, 16, seed)
        assert prefix == expected(seed, 16)
        assert prefix[:4] == expected(seed, 4)


def test_bon_uses_first_maximum_position_including_duplicates():
    assert choose_bon_position([4, 7, 4, 9], {4: 2.0, 7: 3.0, 9: 3.0}) == 1
    assert choose_bon_position([4, 4, 4], {4: 2.0}) == 0


def test_random_n_uses_frozen_pcg64_child_and_stays_in_prefix():
    for key in [(0, 0, 0, 0), (3, 15, 4, 4), (2, 7, 2, 3)]:
        expected = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(20261125, spawn_key=key))
        ).integers(0, 16)
        assert random_n_position(20261125, key, 16) == expected
        assert random_n_position(20261125, key, 16) == expected
    assert random_n_position(20261125, (0, 0, 0, 0), 1) == 0


def test_regret_decomposition_is_exact():
    bon, random_n, random1 = 1.25, 2.5, 4.0
    original, ranking, budget = decompose(bon, random_n, random1)
    assert original == pytest.approx(bon - random1)
    assert ranking == pytest.approx(bon - random_n)
    assert budget == pytest.approx(random_n - random1)
    assert original == pytest.approx(ranking + budget)


def test_structural_zero_has_degenerate_interval_and_p_one():
    result = bootstrap_stats(np.zeros(16), 123, 10000)
    assert result["effect"] == 0
    assert result["nominal_95_ci"] == [0, 0]
    assert result["bonferroni_20_ci"] == [0, 0]
    assert result["p"] == 1


def test_complete_all20_holm_and_bonferroni_include_cap1():
    rows = [{"p": 1.0, "cap": 1} for _ in range(4)]
    rows += [{"p": value, "cap": cap} for value, cap in zip([0.001, 0.002, 0.01, 0.04] * 4, [2, 4, 8, 16] * 4)]
    add_multiplicity(rows)
    assert len(rows) == 20
    assert all(row["holm_p"] == 1 and row["bonferroni_p"] == 1 for row in rows if row["cap"] == 1)
    ordered = sorted(rows, key=lambda row: row["p"])
    adjusted = [row["holm_p"] for row in ordered]
    assert adjusted == sorted(adjusted)
    assert min(row["holm_p"] for row in rows) == pytest.approx(0.02)
    assert all(row["bonferroni_p"] == pytest.approx(min(1, 20 * row["p"])) for row in rows)


def test_frozen_config_retains_full_grid_and_forbids_model_execution():
    config = json.loads((Path(__file__).parents[2] / "checks/paper1_budget_matched_v1/frozen_config.json").read_text())
    assert config["pool_count"] == 16
    assert config["caps"] == [1, 2, 4, 8, 16]
    assert config["seeds"] == [101, 202, 303, 404, 505]
    assert len(config["contexts"]) == 4
    assert config["family_size"] == 20
    assert config["include_cap1_structural_zero"] is True
    assert config["new_blind_test"] is False
    assert config["execution"] == "CPU_only_no_training_no_inference"
