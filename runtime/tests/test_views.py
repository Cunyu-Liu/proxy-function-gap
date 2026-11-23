import csv
import tempfile
import unittest
from pathlib import Path

from sample_runtime.data import attach_predictions, load_design_rows
from sample_runtime.provenance import DEFAULT_MODEL_ROLE_MANIFEST
from sample_runtime.views import FORBIDDEN_LEARNER_FIELDS, generate_views, load_model_roles, permission_negative_tests, validate_views


class ViewTests(unittest.TestCase):
    def test_model_specific_views_and_permission_negatives(self):
        step, target = load_design_rows()
        step = attach_predictions(step[:2], {"original": [0.1, 0.2], "retrained": [0.3, 0.4]})
        target = attach_predictions(target[:2], {"original": [0.5, 0.6], "retrained": [0.7, 0.8]})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            generate_views(output, step, target, DEFAULT_MODEL_ROLE_MANIFEST)
            result = validate_views(output, step, target, DEFAULT_MODEL_ROLE_MANIFEST)
            self.assertEqual(result["view_rows_asserted"], 16)
            self.assertFalse(result["retrained_blind"])
            for model in ("original", "retrained"):
                for dataset in ("step", "target"):
                    with (output / "views" / model / f"{dataset}_learner.csv").open() as handle:
                        fields = csv.DictReader(handle).fieldnames
                    self.assertTrue(FORBIDDEN_LEARNER_FIELDS.isdisjoint(fields))
        negatives = permission_negative_tests()
        self.assertTrue(negatives["all_passed"])
        self.assertGreaterEqual(len(negatives["negative_tests"]), 6)

    def test_role_manifest_separates_original_and_retrained(self):
        roles = load_model_roles(DEFAULT_MODEL_ROLE_MANIFEST)["models"]
        self.assertTrue(roles["original"]["blind"])
        self.assertFalse(roles["retrained"]["blind"])
        self.assertTrue(roles["retrained"]["lineage_status"].startswith("unknown"))


if __name__ == "__main__":
    unittest.main()
