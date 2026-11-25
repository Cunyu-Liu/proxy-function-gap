#!/usr/bin/env python3
import os
import tempfile
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from e5_holdout_ext_summary import compare_pair_across_seeds
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


class E5CrossSeedAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {"score_rows": 3, "tasks": [{
            "experiment": "RBP_K562", "role": "heldout_test",
            "validation_n": 2, "validation_ids_sha256": "val", "validation_positives": 1,
            "test_n": 3, "test_ids_sha256": "test", "test_positives": 1}]}
        self.arrays = {"window_id": np.array(["a", "b", "c"]),
                       "occurrence": np.array([0, 0, 0]),
                       "sample_role": np.array(["known", "known", "heldout"]),
                       "class_label": np.array([0, 1, 1], dtype=np.int8),
                       "y_ood": np.array([0, 0, 1], dtype=np.int8),
                       "logit": np.array([-1.0, 2.0, 0.5]),
                       "probability": np.array([0.2, 0.8, 0.6]),
                       "msp": np.array([-0.8, -0.8, -0.6]),
                       "energy": np.array([-1.0, -2.0, -0.5]),
                       "deep_knn": np.array([-0.9, -0.7, -0.4]),
                       "max_cosine": np.array([0.9, 0.7, 0.4]),
                       "metric_version": np.int64(2)}

    def write_pair(self, directory, name, arrays=None):
        path = os.path.join(directory, name)
        np.savez(path, **(self.arrays if arrays is None else arrays))
        return path

    def test_validation_or_test_task_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            left = self.write_pair(directory, "left.npz")
            right = self.write_pair(directory, "right.npz")
            for key in ("validation_ids_sha256", "test_ids_sha256", "test_positives"):
                changed = {"score_rows": self.manifest["score_rows"],
                           "tasks": [{**self.manifest["tasks"][0], key: "changed"}]}
                with self.assertRaises(ValueError):
                    compare_pair_across_seeds(self.manifest, changed, left, right)

    def test_score_rows_and_npz_key_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            left = self.write_pair(directory, "left.npz")
            right = self.write_pair(directory, "right.npz")
            changed_rows = {**self.manifest, "score_rows": 4}
            with self.assertRaises(ValueError):
                compare_pair_across_seeds(self.manifest, changed_rows, left, right)
            missing_key = {key: value for key, value in self.arrays.items() if key != "energy"}
            with self.assertRaises(ValueError):
                compare_pair_across_seeds(self.manifest, self.manifest, left,
                                          self.write_pair(directory, "missing.npz", missing_key))

    def test_npz_label_change_is_rejected_but_score_change_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            left = self.write_pair(directory, "left.npz")
            changed_labels = {**self.arrays, "class_label": np.array([0, 0, 1], dtype=np.int8)}
            with self.assertRaises(ValueError):
                compare_pair_across_seeds(self.manifest, self.manifest, left,
                                          self.write_pair(directory, "labels.npz", changed_labels))
            changed_scores = {**self.arrays, "energy": np.array([-2.0, -3.0, -1.0])}
            compare_pair_across_seeds(self.manifest, self.manifest, left,
                                      self.write_pair(directory, "scores.npz", changed_scores))


if __name__ == "__main__":
    unittest.main()
