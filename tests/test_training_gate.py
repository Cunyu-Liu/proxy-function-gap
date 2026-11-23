import unittest

from gate0_sample2019.training_gate import TrainingGate


class TrainingGateTests(unittest.TestCase):
    def test_fresh_training_start(self):
        manifest = {"formal_training_start_authorized": True, "initialization": "fresh_random", "official_weights": False, "distillation": False}
        self.assertTrue(TrainingGate.training_start_permission(manifest))
        manifest["official_weights"] = True
        self.assertFalse(TrainingGate.training_start_permission(manifest))

    def test_e1_requires_all_evidence(self):
        manifest = {"E1_enabled": True}
        self.assertFalse(TrainingGate.e1_permission(manifest))
        manifest.update({"model_hash": "m", "epoch": 5, "convergence": "CONVERGED", "V_calibration": "CALIBRATED", "Q_calibration": "CALIBRATED", "configuration_frozen": True})
        self.assertTrue(TrainingGate.e1_permission(manifest))
        manifest["configuration_frozen"] = False
        self.assertFalse(TrainingGate.e1_permission(manifest))

    def test_forbidden_roles_and_outer(self):
        for role in TrainingGate.FORBIDDEN_ROLES:
            with self.assertRaises(PermissionError):
                TrainingGate.assert_role_allowed(role)
        with self.assertRaises(PermissionError):
            TrainingGate.assert_learner_outer_access()


if __name__ == "__main__":
    unittest.main()
