import json
import unittest

import numpy as np

from evaluate_once import evaluate_method
from prepare import low_order_role, mutation_order


class ScenarioTests(unittest.TestCase):
    def test_order_and_split_are_deterministic(self):
        self.assertEqual(mutation_order("A1C:D2E"), 2)
        self.assertEqual(low_order_role("A1C"), low_order_role("A1C"))

    def test_regret_direction(self):
        config = {"evaluation": {"budgets": [1, 2], "random_replicates_per_pool_budget": 8, "random_seed": 7, "bootstrap_pool_resamples": 100, "bootstrap_seed": 9, "top_tail_fractions": [0.1]}}
        ids = np.asarray([f"p{p}_{i}" for p in range(2) for i in range(10)])
        pools = np.asarray([f"pool_{p}" for p in range(2) for _ in range(10)])
        orders = np.full(20, 5)
        y = np.asarray([float(i) for _ in range(2) for i in range(10)])
        good, _, _ = evaluate_method("good", ids, pools, orders, y.copy(), y, config)
        bad, _, _ = evaluate_method("bad", ids, pools, orders, -y.copy(), y, config)
        self.assertTrue(all(r["mean_bon_minus_random_regret"] <= 0 for r in good["primary"]))
        self.assertTrue(all(r["mean_bon_minus_random_regret"] >= 0 for r in bad["primary"]))


if __name__ == "__main__":
    unittest.main()
