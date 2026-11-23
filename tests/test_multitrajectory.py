import random

import pytest

from proxy_function_gap.budget import Budget, BudgetExceeded, ScoreCache
from proxy_function_gap.pool import CandidatePool
from proxy_function_gap.selection import select_trajectories


def pool(probs=(0.5, 0.5)):
    return CandidatePool(("a", "b"), ("A", "C"), probs)


def scorer(values):
    return lambda candidate: values[candidate.index]


def test_multi_trajectory_shares_totals_and_computes_rates():
    result = select_trajectories(
        pool(), "fp-rej", 2, 3, 2, random.Random(4), verifier=scorer((1.0, 1.0))
    )
    assert result.calls["proposals"] == 3
    assert result.calls["verifier_item_forwards"] == 3
    assert result.calls["accepted_proposals"] == 3
    assert result.acceptance_rate == 1.0
    assert result.fallback_rate == 0.0
    assert result.coverage == 1.0
    assert len(result.selected_ids) == len(set(result.selected_ids))


def test_multi_all_zero_abstains_with_coverage_and_fallback_rates():
    result = select_trajectories(
        pool(), "fp-rej", 2, 3, 2, random.Random(4), verifier=scorer((0.0, 0.0)), qhat=lambda: 0.8
    )
    assert result.abstain
    assert result.abstain_reason == "all_trajectories_abstained"
    assert result.calls["proposals"] == 6
    assert result.calls["verifier_item_forwards"] == 6
    assert result.calls["qhat_item_forwards"] == 3
    assert result.calls["fallbacks"] == 3
    assert result.fallback_rate == 1.0
    assert result.coverage == 0.0


def test_multi_trajectory_does_not_reset_item_budget():
    calls = []

    def verifier(candidate):
        calls.append(candidate.sequence_id)
        return 0.0

    with pytest.raises(BudgetExceeded, match="verifier_item_forwards"):
        select_trajectories(
            pool((1.0, 0.0)),
            "fp-rej",
            2,
            3,
            1,
            random.Random(4),
            budget_limits=Budget(max_verifier_item_forwards=3),
            verifier=verifier,
        )
    assert len(calls) == 3


def test_multi_trajectory_shares_explicit_cache():
    cache = ScoreCache()
    result = select_trajectories(
        pool((1.0, 0.0)),
        "fp-rej",
        2,
        3,
        1,
        random.Random(4),
        verifier=scorer((0.0, 0.0)),
        cache=cache,
    )
    assert result.calls["verifier_item_forwards"] == 1
    assert result.calls["verifier_cache_misses"] == 1
    assert result.calls["verifier_cache_hits"] == 5


def test_per_trajectory_counts_are_deltas_not_cumulative_snapshots():
    result = select_trajectories(
        pool((1.0, 0.0)), "fp-rej", 2, 3, 1, random.Random(4), verifier=scorer((0.0, 0.0))
    )
    assert [item.calls["verifier_item_forwards"] for item in result.trajectory_results] == [2, 2, 2]
    assert result.calls["verifier_item_forwards"] == 6


def test_multi_top_k_is_identity_vote_without_replacement():
    result = select_trajectories(pool(), "random", 1, 3, 2, random.Random(4))
    assert len(result.selected_ids) <= 2
    assert len(result.selected_ids) == len(set(result.selected_ids))
    assert sum(result.selected_scores) == 3


def test_fp_tilt_multi_cache_avoids_repeat_pool_forward():
    kwargs = dict(
        pool=pool(),
        strategy="fp-tilt",
        proposal_cap=2,
        trajectories=2,
        top_k=1,
        alpha=2,
        verifier=scorer((0.5, 0.25)),
    )
    uncached = select_trajectories(rng=random.Random(4), **kwargs)
    cached = select_trajectories(rng=random.Random(4), cache=ScoreCache(), **kwargs)
    assert uncached.calls["verifier_item_forwards"] == 4
    assert cached.calls["verifier_item_forwards"] == 2
    assert cached.calls["verifier_cache_hits"] == 2


def test_multi_q_gate_counts_q_once_per_trajectory_without_proposals():
    result = select_trajectories(
        pool(),
        "greedy",
        2,
        3,
        1,
        random.Random(4),
        verifier=scorer((1.0, 1.0)),
        qhat=lambda: 0.1,
        abstain_below_q=0.5,
    )
    assert result.calls["qhat_item_forwards"] == 3
    assert result.calls["proposals"] == 0
    assert result.calls["verifier_item_forwards"] == 0
    assert result.coverage == 0.0


def test_invalid_trajectory_count_is_rejected():
    with pytest.raises(ValueError, match="trajectories"):
        select_trajectories(pool(), "random", 1, 0, 1, random.Random(4))
