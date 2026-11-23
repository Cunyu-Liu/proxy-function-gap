import itertools
import json
import math
import random
from fractions import Fraction

from proxy_function_gap.budget import guidance_weight
from proxy_function_gap.pool import CandidatePool
from proxy_function_gap.selection import select


def reference_bounded_distribution(probabilities, weights, cap):
    result = [0.0 for _ in probabilities]
    accepted_mass = 0.0
    fallback_mass = 0.0
    path_mass = 0.0
    expected_proposals = 0.0
    proposal_acceptance_rate = 0.0
    for proposals in itertools.product(range(len(probabilities)), repeat=cap):
        path_probability = math.prod(probabilities[index] for index in proposals)
        path_mass += path_probability
        prefix_rejection = 1.0
        for position, index in enumerate(proposals):
            acceptance_probability = path_probability * prefix_rejection * weights[index]
            result[index] += acceptance_probability
            accepted_mass += acceptance_probability
            proposal_acceptance_rate += acceptance_probability / (position + 1)
            expected_proposals += acceptance_probability * (position + 1)
            prefix_rejection *= 1.0 - weights[index]
        path_fallback = path_probability * prefix_rejection
        fallback_mass += path_fallback
        expected_proposals += path_fallback * cap
        odds = [weights[index] / (1.0 - weights[index]) for index in proposals]
        normalizer = math.fsum(odds)
        if normalizer > 0:
            for position, index in enumerate(proposals):
                result[index] += path_fallback * odds[position] / normalizer
    return {
        "distribution": result,
        "accepted_mass": accepted_mass,
        "fallback_mass": fallback_mass,
        "path_mass": path_mass,
        "expected_proposals": expected_proposals,
        "proposal_acceptance_rate": proposal_acceptance_rate,
    }


def main():
    pool = CandidatePool(
        ("a", "b", "masked"),
        ("A", "C", "G"),
        (1 / 6, 1 / 3, 1 / 2),
        (True, True, False),
    )
    exact = pool.exact_tilted_distribution((0.5, 0.25, 1.0), 2)
    exact_reference = (Fraction(2, 3), Fraction(1, 3), Fraction(0))
    exact_error = max(abs(float(reference) - actual) for reference, actual in zip(exact_reference, exact))
    assert exact_error < 1e-12, exact_error
    assert math.fsum(exact) == 1.0
    assert exact[2] == 0.0

    assert guidance_weight(0.3, 0.4, 2, "product") == (0.3 * 0.4) ** 2
    assert guidance_weight(0.3, 0.4, 2, "min") == min(0.3, 0.4) ** 2
    assert guidance_weight(0.3, 0.4, 2, "mean") == ((0.3 + 0.4) / 2) ** 2

    bounded_pool = CandidatePool(("a", "b"), ("A", "C"), (0.6, 0.4))
    probabilities = (0.6, 0.4)
    weights = (0.5, 0.25)
    samples = 6000
    cap_reports = []
    for cap in (1, 2, 3):
        reference = reference_bounded_distribution(probabilities, weights, cap)
        assert abs(reference["path_mass"] - 1.0) < 1e-12
        assert abs(reference["accepted_mass"] + reference["fallback_mass"] - 1.0) < 1e-12
        assert abs(math.fsum(reference["distribution"]) - 1.0) < 1e-12
        z = sum(p * w for p, w in zip(probabilities, weights))
        assert abs(reference["accepted_mass"] - (1 - (1 - z) ** cap)) < 1e-12
        assert abs(reference["fallback_mass"] - (1 - z) ** cap) < 1e-12

        rng = random.Random(20261123 + cap)
        counts = [0, 0]
        accepted = 0
        fallbacks = 0
        proposal_acceptance_rates = []
        for _ in range(samples):
            selected = select(
                bounded_pool,
                "fp-rej",
                cap,
                rng,
                verifier=lambda candidate: weights[candidate.index],
            )
            if not selected.abstain:
                counts[selected.selected_index] += 1
            accepted += selected.accepted
            fallbacks += selected.calls["fallbacks"]
            proposal_acceptance_rates.append(selected.acceptance_rate)
        empirical = [count / samples for count in counts]
        empirical_error = max(
            abs(actual - expected) for actual, expected in zip(empirical, reference["distribution"])
        )
        accepted_rate = accepted / samples
        fallback_rate = fallbacks / samples
        proposal_acceptance_rate = math.fsum(proposal_acceptance_rates) / samples
        assert empirical_error < 0.025, (cap, empirical, reference["distribution"])
        assert abs(accepted_rate - reference["accepted_mass"]) < 0.02
        assert abs(fallback_rate - reference["fallback_mass"]) < 0.02
        assert abs(proposal_acceptance_rate - reference["proposal_acceptance_rate"]) < 0.02
        cap_reports.append(
            {
                "cap": cap,
                "ordered_paths": 2 ** cap,
                "reference": reference,
                "empirical_distribution": empirical,
                "distribution_max_abs_error": empirical_error,
                "empirical_trajectory_acceptance_rate": accepted_rate,
                "empirical_fallback_rate": fallback_rate,
                "empirical_proposal_acceptance_rate": proposal_acceptance_rate,
            }
        )

    constant_reports = []
    for cap in (1, 2, 3):
        reference = reference_bounded_distribution(probabilities, (0.4, 0.4), cap)
        error = max(abs(actual - expected) for actual, expected in zip(reference["distribution"], probabilities))
        assert error < 1e-12, (cap, reference["distribution"])
        assert abs(reference["accepted_mass"] - (1 - 0.6 ** cap)) < 1e-12
        constant_reports.append({"cap": cap, "distribution": reference["distribution"], "max_abs_error": error})

    print(
        json.dumps(
            {
                "status": "PASS",
                "exact_reference": [str(value) for value in exact_reference],
                "exact_max_abs_error": exact_error,
                "samples_per_cap": samples,
                "seed_base": 20261123,
                "caps": cap_reports,
                "constant_v": constant_reports,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
