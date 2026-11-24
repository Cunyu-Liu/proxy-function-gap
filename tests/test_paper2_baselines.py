import unittest

import numpy as np

from proxy_function_gap.paper2_baselines import permute_features, rank_candidates


class Paper2BaselineTest(unittest.TestCase):
    def test_distance_and_lcb_ranking(self):
        mean = np.asarray([3.0, 2.0, 1.0])
        std = np.asarray([0.0, 2.0, 0.0])
        distance = np.asarray([0.0, 0.0, 0.5])
        sequences = ("AAAA", "AAAT", "TTTT")
        self.assertEqual(rank_candidates("lcb", [0, 1, 2], mean, std, distance, sequences, lambda_=1.0), [0, 2, 1])
        self.assertEqual(rank_candidates("distance", [0, 1, 2], mean, std, distance, sequences, lambda_=4.0), [0, 1, 2])

    def test_diversity_lcb_sequential_ranking(self):
        mean = np.asarray([3.0, 2.9, 2.0])
        std = np.zeros(3)
        distance = np.zeros(3)
        sequences = ("AAAA", "AAAT", "TTTT")
        ranking = rank_candidates("diversity_lcb", [0, 1, 2], mean, std, distance, sequences, lambda_=0.0, gamma=2.0)
        self.assertEqual(ranking, [0, 2, 1])

    def test_permutation_is_deterministic_and_method_specific(self):
        mean = np.arange(5, dtype=float)
        std = np.arange(5, dtype=float) + 10
        distance = np.arange(5, dtype=float) + 20
        first = permute_features("distance", mean, std, distance, "pool", 101, 1)
        second = permute_features("distance", mean, std, distance, "pool", 101, 1)
        np.testing.assert_array_equal(first[0], mean)
        np.testing.assert_array_equal(first[1], std)
        np.testing.assert_array_equal(first[2], second[2])
        self.assertFalse(np.array_equal(first[2], distance))


if __name__ == "__main__":
    unittest.main()
