import random
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from .constraints import UTRConstraints, utr_constraint_mask
from .pool import CandidatePool


class SyntheticScenario(str, Enum):
    ADVERSARIAL = "adversarial"
    INDEPENDENT = "independent"
    ORACLE = "oracle"


@dataclass(frozen=True)
class SyntheticCase:
    seed: int
    scenario: SyntheticScenario
    pool: CandidatePool
    synthetic_truth: Tuple[float, ...]
    proxy_signal: Tuple[float, ...]
    structured_bias: Tuple[float, ...]
    synthetic_proxy_noise: Tuple[float, ...]
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


def make_known_error_case(
    seed: int = 314159, scenario: SyntheticScenario = SyntheticScenario.ADVERSARIAL
) -> SyntheticCase:
    scenario = SyntheticScenario(scenario)
    rng = random.Random(seed)
    ids = ("decoy", "best", "middle", "masked-uaug")
    sequences = ("AAAAAA", "CCCCCC", "GGGGGG", "AUGAAA")
    proposal_probs = (0.25, 0.25, 0.25, 0.25)
    constraints = UTRConstraints(min_length=6, max_length=6, forbid_uaug=True, max_poly_u=3)
    feasible = utr_constraint_mask(sequences, constraints)
    pool = CandidatePool(ids, sequences, proposal_probs, feasible)
    truth = (0.10, 1.00, 0.40, 0.20)
    if scenario == SyntheticScenario.ADVERSARIAL:
        proxy_signal = truth
        bias = (2.50, 0.00, 0.00, 4.00)
        noise = tuple(rng.gauss(0.0, 0.01) for _ in ids)
        verifier = (0.05, 0.95, 0.60, 1.00)
    elif scenario == SyntheticScenario.INDEPENDENT:
        proxy_signal = (0.80, 0.15, 0.65, 0.10)
        bias = (0.00, 0.00, 0.00, 0.00)
        noise = tuple(rng.gauss(0.0, 0.01) for _ in ids)
        verifier = (0.80, 0.20, 0.70, 1.00)
    else:
        proxy_signal = truth
        bias = (0.00, 0.00, 0.00, 0.00)
        noise = (0.00, 0.00, 0.00, 0.00)
        verifier = truth
    proxy = tuple(signal + b + epsilon for signal, b, epsilon in zip(proxy_signal, bias, noise))
    return SyntheticCase(
        seed=seed,
        scenario=scenario,
        pool=pool,
        synthetic_truth=truth,
        proxy_signal=proxy_signal,
        structured_bias=bias,
        synthetic_proxy_noise=noise,
        proxy_values=proxy,
        verifier_values=verifier,
        qhat=0.80,
        constraints=constraints,
    )
