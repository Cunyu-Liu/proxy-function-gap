import random

import pytest

from proxy_function_gap.pool import CandidatePool
from proxy_function_gap.selection import Budget, BudgetExceeded, Strategy, select


def one_candidate_pool():
    return CandidatePool(("only",), ("A",), (1.0,))


@pytest.mark.parametrize(
    "strategy, expected",
    [
        ("random", {"proposals": 1, "proxy_calls": 0, "verifier_calls": 0, "qhat_calls": 0}),
        ("bon", {"proposals": 4, "proxy_calls": 4, "verifier_calls": 0, "qhat_calls": 0}),
        ("wsc", {"proposals": 4, "proxy_calls": 0, "verifier_calls": 4, "qhat_calls": 0}),
        ("greedy", {"proposals": 4, "proxy_calls": 0, "verifier_calls": 4, "qhat_calls": 1}),
        ("v-only", {"proposals": 4, "proxy_calls": 0, "verifier_calls": 4, "qhat_calls": 0}),
        ("joint", {"proposals": 4, "proxy_calls": 0, "verifier_calls": 4, "qhat_calls": 1}),
    ],
)
def test_every_scoring_invocation_is_counted_by_model(strategy, expected):
    result = select(
        one_candidate_pool(),
        strategy,
        Budget(4),
        random.Random(5),
        proxy=lambda _: 0.0,
        verifier=lambda _: 0.0,
        qhat=lambda: 1.0,
    )
    for field, count in expected.items():
        assert result.calls[field] == count
    if strategy in ("v-only", "joint"):
        assert result.calls["fallbacks"] == 1


@pytest.mark.parametrize("strategy", ["v-only", "joint"])
def test_early_acceptance_reports_actual_not_capital_budget(strategy):
    result = select(
        one_candidate_pool(),
        strategy,
        Budget(10),
        random.Random(5),
        verifier=lambda _: 1.0,
        qhat=lambda: 1.0,
    )
    assert result.calls["proposals"] == 1
    assert result.calls["verifier_calls"] == 1
    assert result.calls["accepted_proposals"] == 1


def test_duplicate_candidates_are_not_cached_into_zero_cost_calls():
    calls = []

    def verifier(candidate):
        calls.append(candidate.sequence_id)
        return 0.0

    result = select(one_candidate_pool(), "v-only", Budget(5), random.Random(5), verifier=verifier)
    assert result.proposed_indices == (0, 0, 0, 0, 0)
    assert len(calls) == 5
    assert result.calls["verifier_calls"] == 5


def test_model_budget_limit_stops_before_an_extra_scorer_call():
    calls = []

    def verifier(candidate):
        calls.append(candidate.sequence_id)
        return 0.0

    with pytest.raises(BudgetExceeded, match="verifier_calls"):
        select(
            one_candidate_pool(),
            "v-only",
            Budget(5, max_verifier_calls=2),
            random.Random(5),
            verifier=verifier,
        )
    assert len(calls) == 2


@pytest.mark.parametrize("strategy", ["wsc", "greedy", "v-only", "joint"])
def test_non_bon_strategies_have_no_hidden_proxy_calls(strategy):
    def forbidden_proxy(_):
        raise AssertionError("hidden proxy call")

    result = select(
        one_candidate_pool(),
        strategy,
        Budget(2),
        random.Random(5),
        proxy=forbidden_proxy,
        verifier=lambda _: 0.0,
        qhat=lambda: 1.0,
    )
    assert result.calls["proxy_calls"] == 0


def test_non_positive_budget_is_rejected():
    with pytest.raises(ValueError, match="positive integers"):
        Budget(0)
