class TrainingGate:
    FORBIDDEN_ROLES = {"official_seen_test", "official_boundary_ambiguous", "scaler_seen_quarantine", "control_nonexact", "quarantine_unsupported_context", "quarantine_retrain_lineage_unknown"}

    @staticmethod
    def training_start_permission(manifest):
        return bool(manifest.get("formal_training_start_authorized") and manifest.get("initialization") == "fresh_random" and not manifest.get("official_weights") and not manifest.get("distillation"))

    @staticmethod
    def e1_permission(manifest):
        required = [manifest.get("model_hash"), manifest.get("epoch"), manifest.get("convergence") == "CONVERGED", manifest.get("V_calibration") == "CALIBRATED", manifest.get("Q_calibration") == "CALIBRATED", manifest.get("configuration_frozen") is True]
        return bool(all(required) and manifest.get("E1_enabled") is True)

    @staticmethod
    def assert_learner_outer_access():
        raise PermissionError("Outer labels are Evaluator-only")

    @staticmethod
    def assert_role_allowed(role):
        if role in TrainingGate.FORBIDDEN_ROLES:
            raise PermissionError(f"Forbidden role for fresh proxy: {role}")
        return True
