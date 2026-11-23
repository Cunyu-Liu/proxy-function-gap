import math
import random

import pytest

from proxy_function_gap.pool import CandidatePool
from proxy_function_gap.selection import Budget, Strategy, _draw_index, _weighted_position, select


def pool(probs=(0.5, 0.5), feasible=(True, True)):
    return CandidatePool(("a", "b"), ("A", "C"), probs, feasible)


def scorer(values):
    return lambda candidate: values[candidate.index]


def test_random_never_proposes_masked_candidate():
    for seed in range(10):
        result = select(pool(feasible=(True, False)), "random", Budget(4), random.Random(seed))
        assert result.selected_id == "a"
        assert result.calls["proposals"] == 1
        assert result.calls["proxy_calls"] == 0
        assert result.calls["verifier_calls"] == 0


def test_bon_returns_best_proxy_among_actual_proposals():
    result = select(
        pool(), "bon", Budget(8), random.Random(3), proxy=scorer((3.0, 1.0))
    )
    assert result.selected_id == "a"
    assert result.calls["proposals"] == 8
    assert result.calls["proxy_calls"] == 8
    assert result.selected_index in result.proposed_indices


def test_wsc_uses_posthoc_verifier_weights():
    result = select(
        pool(), "wsc", Budget(12), random.Random(7), alpha=2, verifier=scorer((0.0, 1.0))
    )
    assert result.selected_id == "b"
    assert result.calls["verifier_calls"] == 12
    assert not result.used_fallback


def test_greedy_returns_max_joint_weight():
    result = select(
        pool(),
        "greedy",
        Budget(12),
        random.Random(7),
        alpha=2,
        verifier=scorer((0.2, 0.9)),
        qhat=lambda: 0.5,
    )
    assert result.selected_id == "b"
    assert max(result.proposed_weights) == pytest.approx(0.45 ** 2)
    assert result.calls["qhat_calls"] == 1


def test_v_only_accepts_unit_weight_on_first_proposal():
    result = select(
        pool((1.0, 0.0)),
        "v-only",
        Budget(5),
        random.Random(1),
        verifier=scorer((1.0, 0.0)),
    )
    assert result.selected_id == "a"
    assert result.accepted
    assert result.calls["accepted_proposals"] == 1
    assert result.calls["proposals"] == 1
    assert result.calls["verifier_calls"] == 1


def test_joint_zero_q_reaches_documented_uniform_fallback():
    result = select(
        pool((1.0, 0.0)),
        "joint",
        Budget(3),
        random.Random(1),
        verifier=scorer((1.0, 0.0)),
        qhat=lambda: 0.0,
    )
    assert result.selected_id == "a"
    assert result.used_fallback
    assert result.calls["fallbacks"] == 1
    assert result.calls["proposals"] == 3
    assert result.calls["qhat_calls"] == 1
    assert result.proposed_weights == (0.0, 0.0, 0.0)


def test_alpha_zero_restores_unit_acceptance_weights():
    result = select(
        pool(), "v-only", Budget(5), random.Random(1), alpha=0, verifier=scorer((0.0, 0.0))
    )
    assert result.accepted
    assert result.proposed_weights == (1.0,)
    assert result.calls["proposals"] == 1


def test_same_seed_is_reproducible():
    arguments = dict(
        pool=pool(),
        strategy="joint",
        budget_limits=Budget(8),
        alpha=5,
        verifier=scorer((0.3, 0.8)),
        qhat=lambda: 0.7,
    )
    first = select(rng=random.Random(99), **arguments)
    second = select(rng=random.Random(99), **arguments)
    assert first == second


def test_infeasible_high_score_is_never_scored():
    calls = []

    def tracking_verifier(candidate):
        calls.append(candidate.sequence_id)
        return 1.0

    result = select(
        pool(feasible=(True, False)),
        "v-only",
        Budget(5),
        random.Random(2),
        verifier=tracking_verifier,
    )
    assert result.selected_id == "a"
    assert calls == ["a"]


def test_invalid_scores_and_missing_scorers_are_rejected():
    with pytest.raises(ValueError, match="expected finite"):
        select(pool(), "bon", Budget(1), random.Random(1), proxy=lambda _: float("inf"))
    with pytest.raises(ValueError, match=r"expected \[0, 1\]"):
        select(pool(), "v-only", Budget(1), random.Random(1), verifier=lambda _: 1.1)
    with pytest.raises(ValueError, match="requires verifier"):
        select(pool(), "v-only", Budget(1), random.Random(1))
    with pytest.raises(ValueError, match="requires qhat"):
        select(pool(), "joint", Budget(1), random.Random(1), verifier=scorer((0.5, 0.5)))


def test_selected_candidate_always_came_from_proposals():
    for strategy in (Strategy.RANDOM, Strategy.BON, Strategy.WSC, Strategy.GREEDY, Strategy.V_ONLY, Strategy.JOINT):
        result = select(
            pool(),
            strategy,
            Budget(6),
            random.Random(11),
            alpha=2,
            proxy=scorer((0.1, 0.9)),
            verifier=scorer((0.4, 0.7)),
            qhat=lambda: 0.8,
        )
        assert result.selected_index in result.proposed_indices


class NearOneRandom:
    def random(self):
        return math.nextafter(1.0, 0.0)


def test_rounding_fallback_never_returns_zero_mass_tail():
    probabilities = [0.1] * 10 + [0.0]
    assert math.fsum(probabilities) == 1.0
    assert _draw_index(probabilities, NearOneRandom()) == 9
    assert _weighted_position(probabilities, NearOneRandom()) == 9


def test_greedy_pool_level_q_does_not_change_ranking():
    arguments = dict(
        pool=pool(),
        strategy="greedy",
        budget_limits=Budget(8),
        rng=random.Random(23),
        alpha=2,
        verifier=scorer((0.2, 0.9)),
    )
    low_q = select(qhat=lambda: 0.2, **arguments)
    arguments["rng"] = random.Random(23)
    high_q = select(qhat=lambda: 0.9, **arguments)
    assert low_q.selected_index == high_q.selected_index
    assert low_q.proposed_indices == high_q.proposed_indices


def test_wsc_all_zero_weights_use_cached_uniform_fallback():
    result = select(
        pool(), "wsc", Budget(4), random.Random(23), verifier=scorer((0.0, 0.0))
    )
    assert result.selected_index in result.proposed_indices
    assert result.proposed_weights == (0.0, 0.0, 0.0, 0.0)
    assert not result.used_fallback
