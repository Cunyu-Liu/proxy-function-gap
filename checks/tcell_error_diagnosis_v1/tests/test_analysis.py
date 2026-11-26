import random

import numpy as np

from run_analysis import bootstrap, draw_index, proposal_prefix, random_n_position


def test_mse_and_constant_offset_preserve_order():
    prediction = np.array([1.0, 3.0, 2.0])
    measured = np.array([2.0, 2.0, 4.0])
    error = prediction - measured
    assert np.isclose(np.mean(error ** 2), np.mean(error) ** 2 + np.var(error))
    assert np.array_equal(np.argsort(-prediction), np.argsort(-(prediction + 1000)))


def test_regret_decomposition():
    full_oracle = 12.0
    prefix_oracle = 9.0
    selected = 7.0
    assert full_oracle - selected == (full_oracle - prefix_oracle) + (prefix_oracle - selected)


def test_prefix_is_nested_and_random_n_is_in_prefix():
    prefix = proposal_prefix(64, 101)
    assert len(prefix) == 16
    assert all(0 <= index < 64 for index in prefix)
    for cap in (1, 2, 4, 8, 16):
        position = random_n_position("defined_tcell_r1", 0, 0, (1, 2, 4, 8, 16).index(cap), cap)
        assert 0 <= position < cap
        assert prefix[position] in prefix[:cap]


def test_draw_index_floating_fallback():
    assert draw_index(random.Random(1), [0.0, 1.0]) == 1


def test_structural_zero_bootstrap_does_not_draw():
    result = bootstrap(np.zeros(16), 20261126)
    assert result == {"effect": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0}


def test_tukey_rule_detects_fixed_outlier():
    values = np.array([0.0, 0.1, 0.2, 0.3, 5.0])
    q1, q3 = np.quantile(values, [0.25, 0.75])
    threshold = q3 + 1.5 * (q3 - q1)
    assert np.array_equal(values > threshold, np.array([False, False, False, False, True]))
