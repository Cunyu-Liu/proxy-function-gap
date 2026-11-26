import random

import numpy as np
import pytest

from proxy_function_gap.mechanism import sequence_features
from proxy_function_gap.residual import HalfIndex, cap16_regret, cluster_bootstrap, fit_ridge_cuda, packed_halves, proposal_indices, simple_features


def test_simple_features_match_frozen_definitions():
    rng = random.Random(19)
    sequences = ["".join(rng.choice("ACGT") for _ in range(50)) for _ in range(25)] + ["ATG" + "A" * 20 + "TTTTT" + "C" * 22]
    values = simple_features(np.asarray(sequences, dtype="S50"), np.arange(len(sequences), dtype=float))
    for index, sequence in enumerate(sequences):
        expected = sequence_features(sequence)
        assert values[index, 0] == index
        assert values[index, 1] == index ** 2
        assert values[index, 2] == expected["gc"]
        assert bool(values[index, 3]) == expected["uAUG"]
        assert values[index, 4] == expected["longest_poly_u"]
        assert values[index, 5] == 1


def test_half_index_exact_and_h1():
    index = HalfIndex()
    sequence = "ACGT" * 12 + "AC"
    index.add(sequence)
    changed = sequence[:25] + ("A" if sequence[25] != "A" else "C") + sequence[26:]
    assert index.match(sequence) == (True, False)
    assert index.match(changed) == (False, True)
    assert index.match("A" * 50) == (False, False)


def test_packed_halves_recover_exact_matches():
    values = np.asarray(["A" * 25 + "C" * 25, "A" * 25 + "G" * 25, "T" * 50], dtype="S50")
    packed = packed_halves(values)
    assert packed[0, 0] == packed[1, 0]
    assert packed[0, 1] != packed[1, 1]
    assert len(np.unique(packed, axis=0)) == 3


def test_cap16_regret_and_seed_reproducibility():
    scores = np.arange(64, dtype=float)
    labels = np.arange(64, dtype=float)[::-1].copy()
    pools = np.arange(64)[None, :]
    proposals = np.asarray([proposal_indices(64, seed, 16) for seed in (101, 202)])[None, ...]
    first = cap16_regret(scores, labels, pools, proposals)
    second = cap16_regret(scores, labels, pools, proposals)
    assert first["mean_regret"] == second["mean_regret"]
    assert np.array_equal(first["pool_seed_regret"], second["pool_seed_regret"])
    assert np.all(first["pool_seed_regret"] >= 0)


def test_cap16_uses_proposal_position_to_index_proposed_labels():
    scores = np.asarray([10.0, 9.0, 8.0, 7.0])
    labels = np.asarray([0.0, 1.0, 20.0, 4.0])
    pools = np.arange(4)[None, :]
    proposals = np.asarray([[[1, 0, 1, 0]]])
    result = cap16_regret(scores, labels, pools, proposals)
    assert result["pool_seed_regret"][0, 0] == 20.0


def test_affine_positive_slope_preserves_order_and_regret():
    rng = np.random.default_rng(5)
    p = rng.normal(size=64)
    y = rng.normal(size=64)
    pools = np.arange(64)[None, :]
    proposals = np.asarray([proposal_indices(64, seed, 16) for seed in (101, 202)])[None, ...]
    affine = 1.7 * p - 3.2
    assert np.array_equal(np.argsort(-p, kind="stable"), np.argsort(-affine, kind="stable"))
    assert cap16_regret(p, y, pools, proposals)["mean_regret"] == cap16_regret(affine, y, pools, proposals)["mean_regret"]


def test_cuda_ridge_recovers_fixed_coefficients_and_records_condition():
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for formal solver")
    rng = np.random.default_rng(23)
    x = rng.normal(size=(512, 5))
    x = np.column_stack((x, np.ones(len(x))))
    beta = np.asarray([0.2, -0.1, 0.4, 0.05, -0.3, 0.7])
    y = x @ beta
    fitted, report = fit_ridge_cuda(x, y, 1e-6)
    assert np.allclose(fitted, beta, rtol=1e-6, atol=1e-6)
    assert report["finite"] and report["converged"]
    assert report["device"] == "cuda:0" and report["dtype"] == "torch.float64"
    assert np.isfinite(report["condition_number"])


def test_cluster_bootstrap_zero_effect():
    result = cluster_bootstrap(np.zeros(16), 20261124, 100)
    assert result["effect"] == 0
    assert result["ci_low"] == result["ci_high"] == 0
    assert result["p_two_sided"] == 1
