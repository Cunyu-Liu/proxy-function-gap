import random
from dataclasses import dataclass
from typing import Tuple

from .constraints import UTRConstraints, utr_constraint_mask
from .pool import CandidatePool


@dataclass(frozen=True)
class SyntheticCase:
    seed: int
    pool: CandidatePool
    synthetic_truth: Tuple[float, ...]
    structured_bias: Tuple[float, ...]
    gaussian_noise: Tuple[float, ...]
    proxy_values: Tuple[float, ...]
    verifier_values: Tuple[float, ...]
    qhat: float
    constraints: UTRConstraints

    def proxy(self, candidate) -> float:
        return self.proxy_values[candidate.index]

    def verifier(self, candidate) -> float:
        return self.verifier_values[candidate.index]

    def pool_qhat(self) -> float:
        return self.qhat


def make_known_error_case(seed: int = 314159) -> SyntheticCase:
    rng = random.Random(seed)
    ids = ("decoy", "best", "middle", "masked-uaug")
    sequences = ("AAAAAA", "CCCCCC", "GGGGGG", "AUGAAA")
    proposal_probs = (0.25, 0.25, 0.25, 0.25)
    constraints = UTRConstraints(min_length=6, max_length=6, forbid_uaug=True, max_poly_u=3)
    feasible = utr_constraint_mask(sequences, constraints)
    pool = CandidatePool(ids, sequences, proposal_probs, feasible)
    truth = (0.10, 1.00, 0.40, 0.20)
    bias = (2.50, 0.00, 0.00, 4.00)
    noise = tuple(rng.gauss(0.0, 0.01) for _ in ids)
    proxy = tuple(y + b + epsilon for y, b, epsilon in zip(truth, bias, noise))
    verifier = (0.05, 0.95, 0.60, 1.00)
    return SyntheticCase(
        seed=seed,
        pool=pool,
        synthetic_truth=truth,
        structured_bias=bias,
        gaussian_noise=noise,
        proxy_values=proxy,
        verifier_values=verifier,
        qhat=0.80,
        constraints=constraints,
    )
