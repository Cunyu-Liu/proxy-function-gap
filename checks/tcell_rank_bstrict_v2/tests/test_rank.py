import json
from pathlib import Path

import numpy as np
import pytest

from rank import cap_regret, cluster_bootstrap, fixed_tail, fit_pairwise_newton, pair_arrays, pairwise_loss_numpy


def test_fixed_tail_ignores_labels_and_pair_weights_balance_pools():
    pools = np.arange(128).reshape(2, 64)
    p = np.arange(128, dtype=np.float64)
    y = np.zeros(128, dtype=np.float64)
    y[pools[0, -16:]] = np.arange(16)
    y[pools[1, -16:]] = np.arange(16)[::-1]
    tail = fixed_tail(p, pools)
    expected = pools[:, -16:][:, ::-1]
    assert np.array_equal(tail, expected)
    z = np.column_stack((p, np.ones(len(p))))
    pairs = pair_arrays(y, p, z, tail)
    totals = np.bincount(pairs["pool_ids"], weights=pairs["weights"], minlength=2)
    np.testing.assert_allclose(totals, [0.5, 0.5], rtol=0, atol=1e-15)
    assert pairs["weights"].sum() == pytest.approx(1.0)
    changed = fixed_tail(p, pools)
    y[:] = 100
    np.testing.assert_array_equal(changed, tail)


def test_pairwise_loss_uses_final_score_order_and_zero_strength_is_baseline_only():
    p = np.array([0.0, 1.0])
    y = np.array([1.0, 0.0])
    pools = np.array([[0, 1]])
    tail = fixed_tail(p, pools, 2)
    z = np.array([[1.0], [0.0]])
    pairs = pair_arrays(y, p, z, tail)
    assert pairwise_loss_numpy(pairs, np.array([0.0]), 1.0) == pytest.approx(np.log1p(np.exp(1.0)))
    assert pairwise_loss_numpy(pairs, np.array([3.0]), 1.0) < 0.2
    assert pairwise_loss_numpy(pairs, np.array([100.0]), 1.0, strength=0.0) == pytest.approx(np.log1p(np.exp(1.0)))


def test_cap_regret_indexes_proposal_labels_before_argmax():
    p = np.array([0.9, 0.8, 0.1, 0.0])
    y = np.array([0.0, 4.0, 10.0, 2.0])
    pools = np.arange(4).reshape(1, 4)
    proposals = np.array([[[0, 1]]])
    regret, selected = cap_regret(p, y, pools, proposals, cap=2)
    assert selected[0, 0] == 0
    assert regret[0, 0] == 10.0
    corrected, corrected_selected = cap_regret(np.array([0.0, 1.0, 0.0, 0.0]), y, pools, proposals, cap=2)
    assert corrected_selected[0, 0] == 1
    assert corrected[0, 0] == 6.0


def test_cluster_bootstrap_resamples_pool_means_not_seeds():
    values = np.array([[1.0, 3.0], [2.0, 4.0]])
    result = cluster_bootstrap(values, 17, 1000)
    assert result["effect"] == pytest.approx(2.5)
    assert result["clusters"] == 2
    assert result["ci_low"] >= 2.0
    assert result["ci_high"] <= 3.0


def test_newton_cuda_float64_converges_without_iteration_cap():
    config = json.loads(Path(__file__).with_name("rank_frozen.json").read_text())
    rng = np.random.default_rng(7)
    pools = np.arange(1600, dtype=np.int64).reshape(25, 64)
    p = rng.normal(size=1600)
    z = rng.normal(size=(1600, 5))
    truth = np.array([0.2, -0.1, 0.05, 0.1, -0.05])
    score = p + z @ truth
    y = score + rng.normal(scale=0.1, size=1600)
    tail = fixed_tail(p, pools)
    pairs = pair_arrays(y, p, z, tail)
    beta, report = fit_pairwise_newton(pairs, 1e-4, config)
    assert report["converged"] and report["finite"]
    assert report["device"] == "cuda:0" and report["dtype"] == "torch.float64"
    assert report["scientific_iteration_cap"] is None
    assert report["gradient_inf"] <= 1e-5
    assert report["final_hessian_condition"] < 1e12
    assert np.linalg.norm(beta - truth) < 0.3


def test_newton_derivatives_include_deployment_scale():
    config = json.loads(Path(__file__).with_name("rank_frozen.json").read_text())
    pairs = {
        "labels": np.array([-1.0]), "weights": np.array([1.0]),
        "p_diff": np.array([1.0]), "z_diff": np.array([[1.0]]),
    }
    scale = config["sample_train_sd"]
    lam = 1.0
    low, high = -100.0, 100.0
    for _ in range(200):
        middle = (low + high) / 2
        derivative = scale / (1.0 + np.exp(-(1.0 + scale * middle))) + 2 * lam * middle
        if derivative > 0:
            high = middle
        else:
            low = middle
    expected = (low + high) / 2
    beta, report = fit_pairwise_newton(pairs, lam, config)
    assert report["converged"]
    assert beta[0] == pytest.approx(expected, abs=1e-6)


def test_failed_auxiliary_gate_has_no_target_permission(tmp_path):
    gate = {"all_directions_pass": False, "target_evaluation_permitted": False}
    path = tmp_path / "auxiliary_gate.json"
    path.write_text(json.dumps(gate))
    assert not json.loads(path.read_text())["all_directions_pass"]
    assert not (tmp_path / "auxiliary_gate_passed.json").exists()
