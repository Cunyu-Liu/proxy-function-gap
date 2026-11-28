from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from upts import cap16_regret, member_std, spearman, top16_by_score, upts_score


def test_member_std_ddof1():
    members = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    std = member_std(members)
    assert std[0] == 1.0
    assert std[1] == 0.0


def test_kappa0_identity():
    rng = np.random.default_rng(0)
    p = rng.normal(size=128).astype(np.float32)
    std = member_std(rng.normal(size=(128, 3)))
    assert np.array_equal(upts_score(p, std, 0.0), p.astype(np.float64))


def test_first_occurrence_tie():
    scores = np.array([5.0, 5.0, 1.0, 0.0] * 16, dtype=np.float64)
    labels = np.array([0.0, 9.0, 0.0, 0.0] * 16, dtype=np.float64)
    pools = np.arange(64).reshape(1, 64)
    proposals = np.array([[[0, 1, 2, 3]]])
    result = cap16_regret(scores, labels, pools, proposals)
    assert result["pool_seed_regret"][0, 0] == 9.0


def test_upts_demotes_high_std():
    p = np.array([7.0, 6.9])
    std = np.array([1.0, 0.0])
    score = upts_score(p, std, 1.0)
    assert int(np.argmax(score)) == 1


def test_top16_tie_break_by_index():
    scores = np.full(64, 5.0)
    scores[60:] = [10.0, 9.0, 8.0, 7.0]
    idx = top16_by_score(scores)
    assert idx[:4].tolist() == [60, 61, 62, 63]
    assert idx[4:].tolist() == list(range(12))


def test_spearman_known():
    assert abs(spearman([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [8, 6, 4, 2]) + 1.0) < 1e-12
    assert spearman([1, 1, 1], [1, 2, 3]) is None


def test_cap16_matches_residual_v1():
    sys.path.insert(0, "/home/proxy-function-gap/src")
    sys.path.insert(0, "/home/proxy-function-gap/checks/tcell_residual_bstrict_v1/src")
    from proxy_function_gap.residual import cap16_regret as residual_cap16
    rng = np.random.default_rng(7)
    scores = rng.normal(size=128)
    labels = rng.normal(size=128)
    pools = np.arange(128).reshape(2, 64)
    proposals = rng.integers(0, 64, size=(2, 5, 16))
    a = cap16_regret(scores, labels, pools, proposals)
    b = residual_cap16(scores, labels, pools, proposals)
    assert a["mean_regret"] == b["mean_regret"]
    assert np.array_equal(a["pool_seed_regret"], b["pool_seed_regret"])