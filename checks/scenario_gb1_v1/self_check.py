#!/usr/bin/env python3
import itertools
import math
import unittest

import numpy as np

from gb1_common import bon_prefix, exact_random_expected_max, pool_metrics
from train_dev import feature_vocabulary, make_features


class ContractTests(unittest.TestCase):
    def test_exact_random_expected_max_against_enumeration(self):
        labels = np.asarray([-4.605170185988092, -2.0, -2.0, 0.5, 3.0])
        for budget in (1, 2, 4, 5):
            expected = np.mean([max(labels[list(indices)]) for indices in itertools.combinations(range(len(labels)), budget)])
            self.assertAlmostEqual(exact_random_expected_max(labels, budget), expected, places=12)

    def test_tie_break_is_seed_dependent_and_nested(self):
        mutants = np.asarray(["A228C:A229D", "A228D:A229C", "A228E:A229F", "A228F:A229E"])
        scores = np.ones(4)
        first = bon_prefix(mutants, scores, 11, 4)
        second = bon_prefix(mutants, scores, 23, 4)
        self.assertEqual(len(set(first)), 4)
        self.assertNotEqual(first, second)
        self.assertEqual(first[:2], bon_prefix(mutants, scores, 11, 2))

    def test_regret_sign(self):
        mutants = np.asarray(["A228C:A229D", "A228D:A229C", "A228E:A229F", "A228F:A229E"])
        labels = np.asarray([3.0, 2.0, 1.0, 0.0])
        predictions = labels.copy()
        result = pool_metrics(mutants, labels, predictions, 11, [1, 2], -4.605170185988092)
        self.assertEqual(result["1"]["bon_regret"], 0.0)
        self.assertLess(result["1"]["delta_regret"], 0.0)
        self.assertTrue(result["1"]["hit_at_budget"])
    def test_substitution_features_generalize_to_blocked_positions(self):
        vocabulary = feature_vocabulary(["A228C"] * 741)
        self.assertEqual(len(vocabulary), 380)
        train_features = make_features(["A228C"], vocabulary)
        test_features = make_features(["A230C"], vocabulary)
        np.testing.assert_array_equal(train_features, test_features)


if __name__ == "__main__":
    unittest.main()
