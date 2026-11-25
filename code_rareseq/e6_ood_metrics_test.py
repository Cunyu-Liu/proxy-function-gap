#!/usr/bin/env python3
import unittest

import numpy as np

from e6_grimm_baselines import OOD_METRIC_DEFINITION, ood_statistics


class OodStatisticsTest(unittest.TestCase):
    def test_fpr95_is_not_ood_tpr_at_known_fpr5(self):
        known = np.r_[np.zeros(99), 1.0]
        unknown = np.linspace(0.1, 1.0, 10)
        y = np.r_[np.zeros(len(known), dtype=int), np.ones(len(unknown), dtype=int)]
        score = np.r_[known, unknown]
        result = ood_statistics(y, score)
        self.assertAlmostEqual(result["fpr95"], 0.01, places=12)
        self.assertAlmostEqual(result["ood_tpr_at_known_fpr5"], 1.0, places=12)
        self.assertEqual(result["metric_definition"], OOD_METRIC_DEFINITION)

    def test_aurc_accepts_low_ood_scores_first(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        result = ood_statistics(y, score)
        reversed_result = ood_statistics(y, score[::-1])
        self.assertAlmostEqual(result["aurc_ood_error"], 0.14583333333333333, places=12)
        self.assertAlmostEqual(reversed_result["aurc_ood_error"], 0.6041666666666666, places=12)
        self.assertLess(result["aurc_ood_error"], reversed_result["aurc_ood_error"])

    def test_perfect_ood_ranking_has_low_aurc(self):
        y = np.array([0, 0, 0, 1, 1])
        score = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = ood_statistics(y, score)
        reversed_result = ood_statistics(y, score[::-1])
        self.assertEqual(result["auroc_ood"], 1.0)
        self.assertEqual(result["fpr95"], 0.0)
        self.assertAlmostEqual(result["aurc_ood_error"], 0.09, places=12)
        self.assertLess(result["aurc_ood_error"], reversed_result["aurc_ood_error"])

    def test_rejects_nonfinite_or_single_class_input(self):
        with self.assertRaises(ValueError):
            ood_statistics(np.array([0, 1]), np.array([0.0, np.nan]))
        with self.assertRaises(ValueError):
            ood_statistics(np.array([0, 0]), np.array([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
