import math
import random


def probability_logit(probability: float, clip: float = 1e-6) -> float:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if not math.isfinite(clip) or not 0 < clip < 0.5:
        raise ValueError("clip must be in (0, 0.5)")
    clipped = min(max(probability, clip), 1.0 - clip)
    return math.log(clipped / (1.0 - clipped))


def add_probability_logit_noise(
    probability: float, sigma: float, rng: random.Random, clip: float = 1e-6
) -> float:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")
    if sigma == 0:
        return float(probability)
    logit = probability_logit(probability, clip) + rng.gauss(0.0, sigma)
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)
