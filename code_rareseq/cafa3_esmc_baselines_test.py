import unittest

import numpy as np
from scipy import sparse

from cafa3_esmc_baselines import (ancestor_closure, frequency_bucket, macro_auprc,
                                  metric_at_threshold, validation_thresholds)


class Cafa3BaselineMetricTest(unittest.TestCase):
    def setUp(self):
        self.y = sparse.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.uint8))
        self.p = np.array([[0.8, 0.2], [0.7, 0.6]], dtype=np.float32)
        self.ic = np.array([1.0, 2.0], dtype=np.float32)

    def test_protein_metric_and_smin(self):
        result = metric_at_threshold(self.y, self.p, self.ic, 0.5)
        self.assertAlmostEqual(result["fmax_at_val_threshold"], 5 / 6)
        self.assertAlmostEqual(result["coverage"], 1.0)
        self.assertAlmostEqual(result["ru"], 0.0)
        self.assertAlmostEqual(result["mi"], 0.5)
        self.assertAlmostEqual(result["smin_at_val_threshold"], 0.5)

    def test_macro_auprc_excludes_undefined_and_masks_inactive_logits(self):
        y = sparse.csr_matrix(np.array([[1, 0, 0], [1, 0, 1]], dtype=np.uint8))
        logits = np.log(np.array([[0.8, 0.9, 0.9], [0.7, 0.8, 0.8]]) /
                        np.array([[0.2, 0.1, 0.1], [0.3, 0.2, 0.2]])).astype(np.float32)
        values, table = macro_auprc(y, logits, np.array([2, 0, 4]), active=np.array([True, False, True]),
                                    logits=True, return_table=True)
        self.assertTrue(np.isnan(table.loc[1, "auprc"]))
        self.assertEqual(values["overall"][1], 2)
        self.assertAlmostEqual(table.loc[1, "n_positive"], 0)
        self.assertGreater(values["overall"][0], 0.5)

    def test_nan_is_not_accepted_as_a_negative_or_probability(self):
        invalid = self.p.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            metric_at_threshold(self.y, invalid, self.ic, 0.5)

    def test_validation_threshold_grid_is_fixed(self):
        fmax, smin, grid = validation_thresholds(self.y, self.p, self.ic)
        self.assertEqual(len(grid), 100)
        self.assertEqual(grid.threshold.iloc[0], 0.01)
        self.assertEqual(grid.threshold.iloc[-1], 1.0)
        self.assertTrue(0.01 <= fmax <= 1.0)
        self.assertTrue(0.01 <= smin <= 1.0)

    def test_frequency_boundaries_and_ontology_cycle(self):
        expected = {0: "0", 1: "1-4", 4: "1-4", 5: "5-19", 19: "5-19",
                    20: "20-99", 99: "20-99", 100: "100-499", 499: "100-499", 500: ">=500"}
        self.assertEqual({key: frequency_bucket(key) for key in expected}, expected)
        with self.assertRaises(ValueError):
            ancestor_closure({"A": ["B"], "B": ["A"]})("A")


if __name__ == "__main__":
    unittest.main()
