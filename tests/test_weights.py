import random

import pytest

from proxy_function_gap.budget import Aggregation, guidance_weight
from proxy_function_gap.pool import CandidatePool
from proxy_function_gap.selection import select


def test_product_applies_alpha_once_to_v_times_q():
    assert guidance_weight(0.3, 0.4, 2, "product") == pytest.approx((0.3 * 0.4) ** 2)
    assert guidance_weight(0.3, 0.4, 2, "product") != pytest.approx((0.3 ** 2) * 0.4)


def test_min_and_mean_apply_alpha_after_aggregation():
    assert guidance_weight(0.3, 0.4, 2, "min") == pytest.approx(0.3 ** 2)
    assert guidance_weight(0.3, 0.4, 2, "mean") == pytest.approx(0.35 ** 2)


def test_v_only_ignores_aggregation():
    for aggregation in Aggregation:
        assert guidance_weight(0.3, None, 2, aggregation) == pytest.approx(0.09)


@pytest.mark.parametrize("aggregation", Aggregation)
@pytest.mark.parametrize("v,q", [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.3, 0.7)])
def test_alpha_zero_is_exactly_unit_weight(aggregation, v, q):
    assert guidance_weight(v, q, 0, aggregation) == 1.0


@pytest.mark.parametrize("aggregation", ["product", "min"])
def test_zero_q_gives_zero_joint_weight_for_product_and_min(aggregation):
    assert guidance_weight(0.8, 0.0, 3, aggregation) == 0.0


def test_zero_q_mean_remains_nonzero():
    assert guidance_weight(0.8, 0.0, 3, "mean") == pytest.approx(0.4 ** 3)


@pytest.mark.parametrize("aggregation", Aggregation)
def test_unit_v_and_q_give_unit_weight(aggregation):
    assert guidance_weight(1.0, 1.0, 20, aggregation) == 1.0


def test_invalid_weight_inputs_are_rejected():
    with pytest.raises(ValueError):
        guidance_weight(-0.1, 0.5, 1)
    with pytest.raises(ValueError):
        guidance_weight(0.5, 1.1, 1)
    with pytest.raises(ValueError):
        guidance_weight(0.5, 0.5, -1)


def test_fp_rej_alpha_zero_accepts_without_v_or_q_forward():
    pool = CandidatePool(("a",), ("A",), (1.0,))
    result = select(pool, "fp-rej", 3, random.Random(1), alpha=0, qhat=lambda: 0.0)
    assert result.selected_id == "a"
    assert result.accepted
    assert result.proposed_weights == (1.0,)
    assert result.calls["verifier_item_forwards"] == 0
    assert result.calls["qhat_item_forwards"] == 0


def test_fp_rej_w_one_never_enters_fallback_division():
    pool = CandidatePool(("a",), ("A",), (1.0,))
    result = select(
        pool,
        "fp-rej",
        3,
        random.Random(1),
        alpha=2,
        verifier=lambda _: 1.0,
        qhat=lambda: 1.0,
    )
    assert result.accepted
    assert result.calls["proposals"] == 1
    assert result.calls["fallbacks"] == 0
    assert result.acceptance_rate == 1.0
