import math

import pytest

from proxy_function_gap.pool import CandidatePool


def make_pool(feasible=(True, True, True)):
    return CandidatePool(("a", "b", "c"), ("A", "C", "G"), (0.2, 0.3, 0.5), feasible)


def test_masked_proposal_is_renormalized():
    assert make_pool((True, True, False)).masked_proposal_probs() == pytest.approx((0.4, 0.6, 0.0))


def test_exact_tilt_matches_manual_calculation():
    distribution = make_pool((True, True, False)).exact_tilted_distribution((0.5, 0.25, 1.0), 2)
    assert distribution == pytest.approx((8 / 11, 3 / 11, 0.0))
    assert math.fsum(distribution) == pytest.approx(1.0)


def test_alpha_zero_recovers_masked_base():
    pool = make_pool((True, False, True))
    assert pool.exact_tilted_distribution((0.0, 1.0, 0.0), 0, qhat=0.0) == pytest.approx(
        pool.masked_proposal_probs()
    )


def test_pool_level_q_cancels_in_exact_joint_tilt():
    pool = make_pool()
    v_only = pool.exact_tilted_distribution((0.2, 0.5, 0.8), 3)
    low_q = pool.exact_tilted_distribution((0.2, 0.5, 0.8), 3, qhat=0.2)
    high_q = pool.exact_tilted_distribution((0.2, 0.5, 0.8), 3, qhat=0.9)
    assert low_q == pytest.approx(v_only)
    assert high_q == pytest.approx(v_only)


def test_larger_alpha_moves_mass_toward_higher_verifier_score():
    pool = CandidatePool(("low", "high"), ("A", "C"), (0.5, 0.5))
    low_alpha = pool.exact_tilted_distribution((0.2, 0.8), 1)
    high_alpha = pool.exact_tilted_distribution((0.2, 0.8), 5)
    assert high_alpha[1] > low_alpha[1] > 0.5


def test_zero_tilt_normalizer_is_an_error():
    with pytest.raises(ValueError, match="zero normalizer"):
        make_pool().exact_tilted_distribution((0.0, 0.0, 0.0), 1)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"proposal_probs": (0.2, 0.2, 0.2)}, "sum to 1"),
        ({"proposal_probs": (0.2, -0.2, 1.0)}, "non-negative"),
        ({"proposal_probs": (0.2, float("nan"), 0.8)}, "finite"),
    ],
)
def test_invalid_probabilities_are_rejected(kwargs, message):
    arguments = {"sequence_ids": ("a", "b", "c"), "sequences": ("A", "C", "G")}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        CandidatePool(**arguments)


def test_duplicate_ids_and_no_feasible_mass_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        CandidatePool(("a", "a"), ("A", "C"), (0.5, 0.5))
    with pytest.raises(ValueError, match="feasible"):
        make_pool((False, False, False))


def test_invalid_tilt_inputs_are_rejected():
    pool = make_pool()
    with pytest.raises(ValueError, match="alpha"):
        pool.exact_tilted_distribution((0.2, 0.3, 0.4), -1)
    with pytest.raises(ValueError, match="verifier"):
        pool.exact_tilted_distribution((0.2, 1.3, 0.4), 1)
    with pytest.raises(ValueError, match="qhat"):
        pool.exact_tilted_distribution((0.2, 0.3, 0.4), 1, qhat=-0.1)
