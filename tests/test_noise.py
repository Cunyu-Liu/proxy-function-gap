import math
import random

import pytest

from proxy_function_gap.noise import add_probability_logit_noise, probability_logit


@pytest.mark.parametrize("probability", [0.0, 0.2, 0.5, 1.0])
def test_zero_sigma_preserves_probability_exactly_including_endpoints(probability):
    class ForbiddenRandom:
        def gauss(self, *_):
            raise AssertionError("zero noise must not draw")

    assert add_probability_logit_noise(probability, 0, ForbiddenRandom()) == probability


def test_logit_reference_values_and_clipped_endpoints():
    assert probability_logit(0.5) == 0.0
    assert probability_logit(0.0) == pytest.approx(math.log(1e-6 / (1 - 1e-6)))
    assert math.isfinite(probability_logit(1.0))


def test_noise_matches_manual_logit_calculation():
    rng = random.Random(7)
    expected_rng = random.Random(7)
    epsilon = expected_rng.gauss(0.0, 0.3)
    expected = 1 / (1 + math.exp(-(probability_logit(0.2) + epsilon)))
    assert add_probability_logit_noise(0.2, 0.3, rng) == pytest.approx(expected)


def test_same_seed_is_reproducible():
    first = add_probability_logit_noise(0.3, 1.0, random.Random(99))
    second = add_probability_logit_noise(0.3, 1.0, random.Random(99))
    assert first == second


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_noisy_endpoints_remain_bounded(probability):
    value = add_probability_logit_noise(probability, 2.0, random.Random(99))
    assert 0 < value < 1


def test_invalid_noise_inputs_are_rejected():
    with pytest.raises(ValueError):
        add_probability_logit_noise(-0.1, 1.0, random.Random(1))
    with pytest.raises(ValueError):
        add_probability_logit_noise(0.5, -1.0, random.Random(1))
    with pytest.raises(ValueError):
        probability_logit(0.5, clip=0.0)
