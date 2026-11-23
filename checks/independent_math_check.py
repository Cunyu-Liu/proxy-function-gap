import itertools
import json
import math
import random
from fractions import Fraction

from proxy_function_gap.pool import CandidatePool
from proxy_function_gap.selection import Budget, select


def reference_bounded_distribution(probabilities, weights, cap):
    result = [0.0 for _ in probabilities]
    for proposals in itertools.product(range(len(probabilities)), repeat=cap):
        path_probability = 1.0
        all_rejected_probability = 1.0
        for index in proposals:
            path_probability *= probabilities[index]
            all_rejected_probability *= 1.0 - weights[index]
        for position, index in enumerate(proposals):
            prefix_rejection = math.prod(1.0 - weights[previous] for previous in proposals[:position])
            acceptance_probability = path_probability * prefix_rejection * weights[index]
            result[index] += acceptance_probability
        odds = [weight / (1.0 - weight) for weight in (weights[index] for index in proposals)]
        normalizer = math.fsum(odds)
        for position, index in enumerate(proposals):
            fallback_share = 1.0 / len(proposals) if normalizer == 0 else odds[position] / normalizer
            result[index] += path_probability * all_rejected_probability * fallback_share
    return result


def main():
    pool = CandidatePool(
        ("a", "b", "masked"),
        ("A", "C", "G"),
        (1 / 6, 1 / 3, 1 / 2),
        (True, True, False),
    )
    exact = pool.exact_tilted_distribution((0.5, 0.25, 1.0), 2, qhat=0.2)
    exact_reference = (Fraction(2, 3), Fraction(1, 3), Fraction(0))
    exact_error = max(abs(float(reference) - actual) for reference, actual in zip(exact_reference, exact))
    assert exact_error < 1e-12, exact_error
    assert math.fsum(exact) == 1.0
    assert exact[2] == 0.0

    bounded_pool = CandidatePool(("a", "b"), ("A", "C"), (0.6, 0.4))
    probabilities = (0.6, 0.4)
    weights = (0.5, 0.25)
    reference = reference_bounded_distribution(probabilities, weights, 2)
    assert math.fsum(reference) == 1.0
    counts = [0, 0]
    samples = 10000
    rng = random.Random(20261123)
    for _ in range(samples):
        selected = select(
            bounded_pool,
            "v-only",
            Budget(2),
            rng,
            verifier=lambda candidate: weights[candidate.index],
        )
        counts[selected.selected_index] += 1
    empirical = [count / samples for count in counts]
    empirical_error = max(abs(actual - expected) for actual, expected in zip(empirical, reference))
    assert empirical_error < 0.025, (empirical, reference)

    print(
        json.dumps(
            {
                "status": "PASS",
                "exact_reference": [str(value) for value in exact_reference],
                "exact_max_abs_error": exact_error,
                "bounded_reference": reference,
                "bounded_empirical": empirical,
                "bounded_max_abs_error": empirical_error,
                "samples": samples,
                "seed": 20261123,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
