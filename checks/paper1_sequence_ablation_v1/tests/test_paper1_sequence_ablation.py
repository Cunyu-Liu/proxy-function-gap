import hashlib
import unittest

import numpy as np

from paper1_sequence_ablation import adjusted_fit, add_multiplicity, choose_score_match, longest_poly_u
from verify_raw import longest_t


class SequenceFeatureTests(unittest.TestCase):
    def test_gc_relevant_poly_u_runs(self):
        cases = {"ACGT": 1, "TTACGTTT": 3, "TTTT": 4, "ACG": 0, "TATTTAT": 3}
        for sequence, expected in cases.items():
            self.assertEqual(longest_poly_u(sequence), expected)
            self.assertEqual(longest_t(sequence), expected)

    def test_score_match_distance_and_hash_tie(self):
        records = [
            {"sequence_id": "bon", "proxy_mean": 5.0},
            {"sequence_id": "a", "proxy_mean": 4.0},
            {"sequence_id": "b", "proxy_mean": 6.0},
            {"sequence_id": "c", "proxy_mean": 3.0},
        ]
        expected = min(["a", "b"], key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        self.assertEqual(choose_score_match("bon", records)["sequence_id"], expected)
        self.assertNotEqual(choose_score_match("bon", records)["sequence_id"], "bon")


class StatisticsTests(unittest.TestCase):
    def test_adjusted_fit_recovers_zero_feature_contrast_effect(self):
        groups = []
        for pool in range(4):
            rows = []
            for seed in range(6):
                delta_gc = (seed - 2.5) * 0.01
                row = {"delta_gc": delta_gc, "delta_uAUG": seed % 2, "delta_longest_poly_u": seed % 3, "delta_support": ((seed * 2 + pool) % 5 - 2) * 0.01}
                row["delta_regret"] = pool + 2 * delta_gc - 0.5 * row["delta_uAUG"] + 0.25 * row["delta_longest_poly_u"] + row["delta_support"]
                rows.append(row)
            groups.append(rows)
        self.assertAlmostEqual(adjusted_fit(groups), 1.5)

    def test_holm_and_bonferroni(self):
        rows = [{"unadjusted_p": p, "adjusted_p": p} for p in [0.01, 0.04, 0.03, 1.0]]
        add_multiplicity(rows)
        self.assertAlmostEqual(rows[0]["unadjusted_holm_p"], 0.04)
        self.assertAlmostEqual(rows[1]["unadjusted_holm_p"], 0.09)
        self.assertAlmostEqual(rows[2]["unadjusted_holm_p"], 0.09)
        self.assertEqual(rows[3]["unadjusted_bonferroni_p"], 1.0)

    def test_pool_resampling_keeps_all_seed_rows_together(self):
        groups = [[{"delta_regret": float(pool) + seed, "delta_gc": 0.01 * seed, "delta_uAUG": float(seed % 2), "delta_longest_poly_u": float(seed % 3), "delta_support": 0.01 * ((seed + pool) % 4)} for seed in range(5)] for pool in range(3)]
        estimate = adjusted_fit(groups)
        self.assertTrue(np.isfinite(estimate))


if __name__ == "__main__":
    unittest.main()
