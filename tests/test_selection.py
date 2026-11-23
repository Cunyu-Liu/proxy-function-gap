import math
import random

import pytest

from proxy_function_gap.pool import CandidatePool
from proxy_function_gap.selection import Strategy, _draw_index, select


def pool(probs=(0.5, 0.5), feasible=None):
    ids = tuple(chr(ord("a") + index) for index in range(len(probs)))
    return CandidatePool(ids, tuple("A" for _ in probs), probs, feasible)


def scorer(values):
    return lambda candidate: values[candidate.index]


class SequenceRandom:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)

    def shuffle(self, values):
        return None


def test_random_never_proposes_masked_candidate_and_has_zero_forwards():
    result = select(pool(feasible=(True, False)), "random", 4, random.Random(1))
    assert result.selected_id == "a"
    assert result.calls["proposals"] == 1
    assert result.calls["proxy_item_forwards"] == 0
    assert result.calls["verifier_item_forwards"] == 0
    assert result.calls["qhat_item_forwards"] == 0


def test_random_top_k_is_without_replacement():
    result = select(pool((1 / 3, 1 / 3, 1 / 3)), "random", 3, random.Random(1), top_k=3)
    assert len(result.selected_ids) == 3
    assert len(set(result.selected_ids)) == 3


def test_bon_ranks_unique_identities_by_proxy():
    result = select(pool(), "bon", 8, random.Random(3), proxy=scorer((3.0, 1.0)), top_k=2)
    assert result.selected_ids == ("a", "b")
    assert result.calls["proxy_item_forwards"] == 8
    assert set(result.selected_indices) <= set(result.proposed_indices)


def test_wsc_is_identity_weighted_vote_not_weighted_resampling():
    rng = SequenceRandom((0.1, 0.1, 0.9))
    result = select(pool(), "wsc", 3, rng, alpha=1, verifier=scorer((0.5, 0.9)))
    assert result.proposed_indices == (0, 0, 1)
    assert result.selected_id == "a"
    assert result.selected_scores == (pytest.approx(1.0),)


def test_wsc_top_k_contains_each_identity_once():
    rng = SequenceRandom((0.1, 0.1, 0.9))
    result = select(pool(), "wsc", 3, rng, verifier=scorer((0.5, 0.9)), top_k=2)
    assert result.selected_ids == ("a", "b")
    assert result.selected_scores == pytest.approx((1.0, 0.9))


def test_greedy_is_verifier_reward_argmax_and_ignores_q_and_alpha():
    for alpha in (0, 1, 20):
        result = select(
            pool(),
            "greedy",
            3,
            SequenceRandom((0.1, 0.1, 0.9)),
            alpha=alpha,
            verifier=scorer((0.5, 0.9)),
            qhat=lambda: 0.01,
        )
        assert result.selected_id == "b"
        assert result.selected_scores == (pytest.approx(0.9),)
        assert result.calls["qhat_item_forwards"] == 0


def test_fp_tilt_is_analytic_and_q_does_not_change_selection():
    arguments = dict(
        pool=pool(),
        strategy="fp-tilt",
        proposal_cap=3,
        alpha=2,
        verifier=scorer((0.5, 0.25)),
    )
    first = select(rng=random.Random(2), qhat=lambda: 0.0, **arguments)
    second = select(rng=random.Random(2), qhat=lambda: 1.0, **arguments)
    assert first.selected_id == "a"
    assert first.selected_scores == (pytest.approx(0.8),)
    assert first.selected_indices == second.selected_indices
    assert first.calls["qhat_item_forwards"] == 0
    assert second.calls["qhat_item_forwards"] == 0
    assert first.calls["proposals"] == 0


def test_fp_tilt_constant_v_collapses_to_base_ranking():
    result = select(pool((0.2, 0.8)), "fp-tilt", 3, random.Random(2), verifier=scorer((0.4, 0.4)))
    assert result.selected_id == "b"
    assert result.selected_scores == (pytest.approx(0.8),)


@pytest.mark.parametrize("strategy", ["wsc", "greedy", "fp-tilt"])
def test_all_zero_guidance_defaults_to_explicit_abstention(strategy):
    result = select(pool(), strategy, 3, random.Random(2), verifier=scorer((0.0, 0.0)))
    assert result.abstain
    assert result.abstain_reason == "zero_guidance_weight"
    assert result.selected_indices == ()
    assert result.coverage == 0.0


@pytest.mark.parametrize("strategy", ["wsc", "greedy", "fp-tilt"])
def test_all_zero_guidance_uniform_fallback_must_be_explicit(strategy):
    result = select(
        pool(),
        strategy,
        3,
        random.Random(2),
        verifier=scorer((0.0, 0.0)),
        zero_weight_policy="uniform",
    )
    assert not result.abstain
    assert result.used_fallback
    assert result.selected_id in ("a", "b")


def test_fp_rej_q_zero_defaults_to_abstention_after_cap():
    result = select(
        pool(),
        "fp-rej",
        3,
        random.Random(2),
        verifier=scorer((0.8, 0.8)),
        qhat=lambda: 0.0,
    )
    assert result.abstain
    assert result.calls["proposals"] == 3
    assert result.calls["rejected_proposals"] == 3
    assert result.calls["fallbacks"] == 1
    assert result.fallback_rate == 1.0
    assert result.acceptance_rate == 0.0


def test_fp_rej_q_zero_uniform_is_explicit():
    result = select(
        pool((1.0, 0.0)),
        "fp-rej",
        3,
        random.Random(2),
        verifier=scorer((0.8, 0.8)),
        qhat=lambda: 0.0,
        zero_weight_policy="uniform",
    )
    assert result.selected_id == "a"
    assert result.used_fallback
    assert result.calls["fallbacks"] == 1


@pytest.mark.parametrize(
    "aggregation, expected",
    [("product", 0.25 ** 2), ("min", 0.5 ** 2), ("mean", 0.5 ** 2)],
)
def test_fp_rej_uses_requested_aggregation(aggregation, expected):
    result = select(
        pool((1.0, 0.0)),
        "fp-rej",
        1,
        SequenceRandom((0.1, 0.9, 0.1)),
        alpha=2,
        aggregation=aggregation,
        verifier=scorer((0.5, 0.5)),
        qhat=lambda: 0.5,
        zero_weight_policy="uniform",
    )
    assert result.proposed_weights == (pytest.approx(expected),)


def test_single_proposal_collapse_and_acceptance_rates():
    accepted = select(pool((1.0, 0.0)), "fp-rej", 1, random.Random(1), verifier=scorer((1.0, 0.0)))
    assert accepted.accepted
    assert accepted.calls["proposals"] == 1
    assert accepted.acceptance_rate == 1.0
    rejected = select(pool((1.0, 0.0)), "fp-rej", 1, random.Random(1), verifier=scorer((0.0, 0.0)))
    assert rejected.abstain
    assert rejected.calls["proposals"] == 1
    assert rejected.fallback_rate == 1.0


def test_q_coverage_gate_abstains_before_candidate_scoring():
    result = select(
        pool(),
        "greedy",
        3,
        random.Random(1),
        verifier=lambda _: (_ for _ in ()).throw(AssertionError("hidden V call")),
        qhat=lambda: 0.2,
        abstain_below_q=0.5,
    )
    assert result.abstain_reason == "qhat_below_threshold"
    assert result.calls["qhat_item_forwards"] == 1
    assert result.calls["proposals"] == 0
    assert result.calls["verifier_item_forwards"] == 0


def test_single_fp_rej_top_k_requires_multi_trajectory_interface():
    with pytest.raises(ValueError, match="select_trajectories"):
        select(pool(), "fp-rej", 3, random.Random(1), top_k=2, verifier=scorer((1.0, 1.0)))


def test_same_seed_is_reproducible():
    arguments = dict(
        pool=pool(),
        strategy="fp-rej",
        proposal_cap=8,
        alpha=5,
        verifier=scorer((0.3, 0.8)),
        qhat=lambda: 0.7,
    )
    assert select(rng=random.Random(99), **arguments) == select(rng=random.Random(99), **arguments)


def test_invalid_scores_are_rejected():
    with pytest.raises(ValueError, match="expected finite"):
        select(pool(), "bon", 1, random.Random(1), proxy=lambda _: float("inf"))
    with pytest.raises(ValueError, match=r"expected \[0, 1\]"):
        select(pool(), "fp-rej", 1, random.Random(1), verifier=lambda _: 1.1)


class NearOneRandom:
    def random(self):
        return math.nextafter(1.0, 0.0)


def test_rounding_fallback_never_returns_zero_mass_tail():
    probabilities = [0.1] * 10 + [0.0]
    assert math.fsum(probabilities) == 1.0
    assert _draw_index(probabilities, NearOneRandom()) == 9


def test_fp_tilt_alpha_zero_skips_verifier_and_q_forward():
    result = select(
        pool((0.2, 0.8)),
        "fp-tilt",
        3,
        random.Random(2),
        alpha=0,
        qhat=lambda: 0.0,
    )
    assert result.selected_id == "b"
    assert result.selected_scores == (pytest.approx(0.8),)
    assert result.calls["verifier_item_forwards"] == 0
    assert result.calls["qhat_item_forwards"] == 0


@pytest.mark.parametrize("strategy", ["bon", "wsc", "greedy"])
def test_batch_rankers_top_k_never_duplicate_identity(strategy):
    result = select(
        pool(),
        strategy,
        6,
        random.Random(3),
        top_k=2,
        proxy=scorer((3.0, 1.0)),
        verifier=scorer((0.5, 0.9)),
    )
    assert len(result.selected_ids) == 2
    assert len(set(result.selected_ids)) == 2


def test_constant_unit_v_fp_rej_collapses_to_first_base_proposal():
    unit = select(pool(), "fp-rej", 3, random.Random(9), verifier=scorer((1.0, 1.0)))
    alpha_zero = select(pool(), "fp-rej", 3, random.Random(9), alpha=0)
    assert unit.selected_index == alpha_zero.selected_index
    assert unit.calls["proposals"] == 1
    assert alpha_zero.calls["proposals"] == 1
