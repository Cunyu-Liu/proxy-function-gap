class ModelSpecificEvaluator:
    def __init__(self, manifest):
        self.manifest = manifest

    def outer_permission(self, model_lineage, context, is_step=False):
        if model_lineage == "retrained_evolution":
            return False
        if model_lineage != "original_evolution":
            return False
        if is_step:
            return False
        return context == self.manifest["original_e1_context"] if "original_e1_context" in self.manifest else context == "eGFP|unmod|designed_single_assay"

    def evaluate_outer(self, *args, **kwargs):
        raise PermissionError("Outer metrics remain disabled for both model lineages")


def forbidden_assignment(model_lineage, role, pool_class, context, is_step=False):
    if role == "historical_measured_state_outer" and model_lineage == "retrained_evolution":
        return True
    if role == "historical_measured_state_outer" and pool_class == "control_nonexact":
        return True
    if role == "historical_measured_state_outer" and is_step:
        return False
    if pool_class == "control_nonexact" and role != "control_nonexact":
        return True
    return False
