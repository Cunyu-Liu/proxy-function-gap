#!/usr/bin/env python3
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from e5_rbp_holdout import assigned_train_mask, aurc, fpr95, ood_tpr_at_known_fpr5


class E5OodMetricsTest(unittest.TestCase):
    def test_energy_high_confidence_known_is_not_ood(self):
        logit = np.array([-8.0, 8.0, -0.1, 0.1])
        y_ood = np.array([0, 0, 1, 1])
        energy = -np.abs(logit)
        self.assertEqual(roc_auc_score(y_ood, energy), 1.0)
        self.assertEqual(roc_auc_score(y_ood, -energy), 0.0)

    def test_fpr95_is_not_ood_tpr_at_known_fpr5(self):
        known = np.r_[np.zeros(99), 1.0]
        unknown = np.linspace(0.1, 1.0, 10)
        y_ood = np.r_[np.zeros(len(known), dtype=int), np.ones(len(unknown), dtype=int)]
        score = np.r_[known, unknown]
        self.assertAlmostEqual(fpr95(y_ood, score), 0.01, places=12)
        self.assertAlmostEqual(ood_tpr_at_known_fpr5(y_ood, score), 1.0, places=12)

    def test_aurc_accepts_low_uncertainty_first(self):
        y = np.array([0, 0, 0, 0])
        probability = np.array([0.1, 0.2, 0.8, 0.9])
        unknown = np.array([0.1, 0.2, 0.8, 0.9])
        forward = aurc(y, probability, unknown)
        reversed_result = aurc(y, probability, unknown[::-1])
        self.assertAlmostEqual(forward, 0.14583333333333333, places=12)
        self.assertAlmostEqual(reversed_result, 0.6041666666666666, places=12)
        self.assertLess(forward, reversed_result)

    def test_unassigned_train_positives_are_removed(self):
        class Loader:
            fold_by_wid = {"a": 2, "b": -1, "c": 3, "d": -1, "e": 4}

        ids = np.array(["a", "b", "c", "d", "e"])
        labels = np.array([1, 1, 0, 0, 1])
        mask = assigned_train_mask(Loader(), ids, labels)
        self.assertEqual(ids[mask].tolist(), ["a", "c", "d", "e"])


if __name__ == "__main__":
    unittest.main()
