import random

import pytest

from proxy_function_gap.budget import BatchScorer, Budget, BudgetExceeded, ScoreCache
from proxy_function_gap.pool import CandidatePool
from proxy_function_gap.selection import select


def one_candidate_pool():
    return CandidatePool(("only",), ("A",), (1.0,))


def test_actual_item_forward_budget_by_strategy():
    expected = {
        "random": (1, 0, 0, 0),
        "bon": (4, 4, 0, 0),
        "wsc": (4, 0, 4, 0),
        "greedy": (4, 0, 4, 0),
        "fp-tilt": (0, 0, 1, 0),
        "fp-rej": (4, 0, 4, 1),
    }
    for strategy, counts in expected.items():
        result = select(
            one_candidate_pool(),
            strategy,
            4,
            random.Random(5),
            proxy=lambda _: 0.0,
            verifier=lambda _: 0.0,
            qhat=lambda: 1.0,
        )
        actual = (
            result.calls["proposals"],
            result.calls["proxy_item_forwards"],
            result.calls["verifier_item_forwards"],
            result.calls["qhat_item_forwards"],
        )
        assert actual == counts, strategy
        assert "proxy_calls" not in result.calls


def test_same_cap_does_not_mean_same_actual_cost():
    pool = one_candidate_pool()
    bon = select(pool, "bon", 10, random.Random(5), proxy=lambda _: 0.0)
    rej = select(pool, "fp-rej", 10, random.Random(5), verifier=lambda _: 1.0)
    assert bon.calls["proxy_item_forwards"] == 10
    assert rej.calls["verifier_item_forwards"] == 1
    assert bon.calls["proposals"] == 10
    assert rej.calls["proposals"] == 1


def test_duplicate_proposals_are_not_cached_by_default():
    calls = []

    def verifier(candidate):
        calls.append(candidate.sequence_id)
        return 0.0

    result = select(one_candidate_pool(), "fp-rej", 5, random.Random(5), verifier=verifier)
    assert result.proposed_indices == (0, 0, 0, 0, 0)
    assert len(calls) == 5
    assert result.calls["verifier_item_forwards"] == 5
    assert result.calls["verifier_cache_hits"] == 0


def test_explicit_cache_charges_only_new_identity():
    calls = []

    def verifier(candidate):
        calls.append(candidate.sequence_id)
        return 0.0

    result = select(
        one_candidate_pool(), "fp-rej", 5, random.Random(5), verifier=verifier, cache=ScoreCache()
    )
    assert len(calls) == 1
    assert result.calls["verifier_item_forwards"] == 1
    assert result.calls["verifier_cache_misses"] == 1
    assert result.calls["verifier_cache_hits"] == 4


def test_ensemble_charges_every_member_and_returns_mean():
    members = (lambda _: 0.0, lambda _: 0.5, lambda _: 1.0)
    result = select(one_candidate_pool(), "wsc", 4, random.Random(5), verifier=members)
    assert result.calls["verifier_item_forwards"] == 12
    assert result.proposed_weights == pytest.approx((0.5, 0.5, 0.5, 0.5))


def test_batch_scorer_charges_items_not_batch_invocations():
    batches = []

    def batch(candidates):
        batches.append(len(candidates))
        return [0.2] * len(candidates)

    result = select(
        one_candidate_pool(), "bon", 4, random.Random(5), proxy=BatchScorer(batch)
    )
    assert batches == [4]
    assert result.calls["proxy_item_forwards"] == 4
    assert result.calls["proxy_scorer_batches"] == 1


def test_batch_ensemble_charges_members_times_items():
    batch = BatchScorer(lambda candidates: [0.2] * len(candidates))
    result = select(one_candidate_pool(), "bon", 3, random.Random(5), proxy=(batch, batch))
    assert result.calls["proxy_item_forwards"] == 6
    assert result.calls["proxy_scorer_batches"] == 2


def test_batch_cache_coalesces_duplicate_items():
    batches = []

    def batch(candidates):
        batches.append(len(candidates))
        return [0.0] * len(candidates)

    result = select(
        one_candidate_pool(),
        "wsc",
        4,
        random.Random(5),
        verifier=BatchScorer(batch),
        cache=ScoreCache(),
    )
    assert batches == [1]
    assert result.calls["verifier_item_forwards"] == 1
    assert result.calls["verifier_cache_misses"] == 1
    assert result.calls["verifier_cache_hits"] == 3


def test_cache_stores_ensemble_mean_not_individual_member_forwards():
    members = (lambda _: 0.0, lambda _: 1.0)
    result = select(
        one_candidate_pool(), "wsc", 4, random.Random(5), verifier=members, cache=ScoreCache()
    )
    assert result.calls["verifier_item_forwards"] == 2
    assert result.calls["verifier_cache_hits"] == 3
    assert result.proposed_weights == pytest.approx((0.5, 0.5, 0.5, 0.5))


def test_item_budget_stops_atomic_batch_before_scorer_call():
    calls = []

    def batch(candidates):
        calls.append(len(candidates))
        return [0.0] * len(candidates)

    with pytest.raises(BudgetExceeded, match="proxy_item_forwards"):
        select(
            one_candidate_pool(),
            "bon",
            4,
            random.Random(5),
            budget_limits=Budget(max_proxy_item_forwards=3),
            proxy=BatchScorer(batch),
        )
    assert calls == []


def test_batch_output_length_is_validated():
    with pytest.raises(ValueError, match="output length"):
        select(one_candidate_pool(), "bon", 2, random.Random(5), proxy=BatchScorer(lambda _: []))


def test_non_bon_strategies_have_no_hidden_proxy_forward():
    def forbidden_proxy(_):
        raise AssertionError("hidden proxy call")

    for strategy in ("wsc", "greedy", "fp-tilt", "fp-rej"):
        result = select(
            one_candidate_pool(),
            strategy,
            2,
            random.Random(5),
            proxy=forbidden_proxy,
            verifier=lambda _: 0.0,
        )
        assert result.calls["proxy_item_forwards"] == 0


def test_alpha_zero_wsc_skips_verifier_forward():
    result = select(one_candidate_pool(), "wsc", 4, random.Random(5), alpha=0)
    assert result.selected_id == "only"
    assert result.selected_scores == (4.0,)
    assert result.calls["verifier_item_forwards"] == 0


def test_top_k_deduplication_does_not_erase_already_incurred_cost():
    result = select(one_candidate_pool(), "wsc", 4, random.Random(5), verifier=lambda _: 1.0, top_k=1)
    assert result.selected_ids == ("only",)
    assert result.calls["verifier_item_forwards"] == 4


def test_cache_is_scoped_by_score_kind():
    cache = ScoreCache()
    pool = one_candidate_pool()
    select(pool, "bon", 1, random.Random(5), proxy=lambda _: 0.5, cache=cache)
    result = select(pool, "wsc", 1, random.Random(5), verifier=lambda _: 0.5, cache=cache)
    assert result.calls["verifier_item_forwards"] == 1
    assert result.calls["verifier_cache_misses"] == 1


def test_empty_ensemble_and_invalid_budgets_are_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        select(one_candidate_pool(), "bon", 1, random.Random(5), proxy=())
    with pytest.raises(ValueError, match="positive integers"):
        Budget(max_proxy_item_forwards=0)
