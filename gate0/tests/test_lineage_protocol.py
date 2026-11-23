import unittest

from gate0_sample2019.lineage_protocol import ModelSpecificEvaluator, forbidden_assignment


class LineagePermissionTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = ModelSpecificEvaluator({"original_e1_context": "eGFP|unmod|designed_single_assay"})

    def test_original_fixed_pool_scope_only(self):
        self.assertTrue(self.evaluator.outer_permission("original_evolution", "eGFP|unmod|designed_single_assay"))
        self.assertFalse(self.evaluator.outer_permission("original_evolution", "eGFP|unmod|designed_single_assay", is_step=True))
        self.assertFalse(self.evaluator.outer_permission("original_evolution", "eGFP|unmod|2"))

    def test_retrained_has_no_outer(self):
        self.assertFalse(self.evaluator.outer_permission("retrained_evolution", "eGFP|unmod|designed_single_assay"))
        self.assertTrue(forbidden_assignment("retrained_evolution", "historical_measured_state_outer", "design_pool", "eGFP|unmod|designed_single_assay"))

    def test_control_cannot_be_design_or_outer(self):
        self.assertTrue(forbidden_assignment("original_evolution", "historical_measured_state_outer", "control_nonexact", "eGFP|unmod|designed_single_assay"))
        self.assertTrue(forbidden_assignment("original_evolution", "development", "control_nonexact", "eGFP|unmod|designed_single_assay"))

    def test_metrics_are_disabled(self):
        with self.assertRaises(PermissionError):
            self.evaluator.evaluate_outer()


if __name__ == "__main__":
    unittest.main()
