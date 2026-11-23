import csv
import json
from pathlib import Path

from .data import connection_key, write_rows
from .model import sha256_file

BASE_VIEW_FIELDS = (
    "raw_row_id", "sequence_id", "sequence_hash", "source_row_number", "raw_index",
    "library", "id", "generation", "sparse_step", "target_mrl_info1", "utr",
)
MODEL_VIEW_FIELDS = ("model_raw_output", "model_pred_mrl")
LEARNER_FIELDS = BASE_VIEW_FIELDS + MODEL_VIEW_FIELDS
EVALUATOR_FIELDS = BASE_VIEW_FIELDS + ("measured_rl", "total_reads", "published_pred_info2") + MODEL_VIEW_FIELDS
FORBIDDEN_LEARNER_FIELDS = frozenset(("measured_rl", "total_reads", "published_pred_info2"))


def load_model_roles(path):
    return validate_model_roles(json.loads(Path(path).read_text()))


def validate_model_roles(manifest):
    if manifest.get("schema_version") != "1.0" or set(manifest.get("models", {})) != {"original", "retrained"}:
        raise ValueError("Role manifest must define exactly original and retrained")
    for name, role in manifest["models"].items():
        if not isinstance(role.get("blind"), bool) or not role.get("role") or not role.get("lineage_status"):
            raise ValueError(f"Incomplete role for {name}")
        if role["lineage_status"].startswith("unknown") and role["blind"]:
            raise ValueError(f"Unknown lineage cannot be marked blind: {name}")
        if not role.get("learner_view_allowed") or not role.get("evaluator_view_allowed"):
            raise ValueError(f"Both model-specific views are required for {name}")
    return manifest


def assert_learner_fields(fields):
    fields = tuple(fields)
    forbidden = FORBIDDEN_LEARNER_FIELDS.intersection(fields)
    if forbidden:
        raise PermissionError(f"Learner fields contain labels or measurement metadata: {sorted(forbidden)}")
    if set(fields) != set(LEARNER_FIELDS):
        raise PermissionError("Learner fields do not match the strict allowlist")


def _view_row(source, model_name, role):
    fields = LEARNER_FIELDS if role == "learner" else EVALUATOR_FIELDS
    row = {field: source.get(field, "") for field in BASE_VIEW_FIELDS}
    if role == "evaluator":
        for field in FORBIDDEN_LEARNER_FIELDS:
            row[field] = source.get(field, "")
    row["model_raw_output"] = source[f"{model_name}_raw_output"]
    row["model_pred_mrl"] = source[f"{model_name}_pred_mrl"]
    if set(row) != set(fields):
        raise ValueError("View row schema mismatch")
    if role == "learner":
        assert_learner_fields(row)
    return row


def generate_views(output_dir, step_rows, target_rows, role_manifest_path):
    roles = load_model_roles(role_manifest_path)
    manifest = {
        "schema_version": "1.0",
        "model_role_manifest_sha256": sha256_file(role_manifest_path),
        "label_holders": {field: "Evaluator" for field in sorted(FORBIDDEN_LEARNER_FIELDS)},
        "target_info2_usage": "Evaluator-only published prediction for numerical validation; never a blind feature",
        "views": {},
    }
    for model_name, model_role in roles["models"].items():
        manifest["views"][model_name] = {**model_role, "datasets": {}}
        for dataset, rows in (("step", step_rows), ("target", target_rows)):
            manifest["views"][model_name]["datasets"][dataset] = {}
            for role, fields in (("learner", LEARNER_FIELDS), ("evaluator", EVALUATOR_FIELDS)):
                relative = Path("views") / model_name / f"{dataset}_{role}.csv"
                view_rows = [_view_row(row, model_name, role) for row in rows]
                (output_dir / relative).parent.mkdir(parents=True, exist_ok=True)
                write_rows(output_dir / relative, view_rows, fields)
                manifest["views"][model_name]["datasets"][dataset][role] = {
                    "path": str(relative), "columns": list(fields), "rows": len(view_rows),
                }
    (output_dir / "view_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_views(output_dir, step_rows, target_rows, role_manifest_path):
    roles = load_model_roles(role_manifest_path)
    checked = 0
    for model_name in roles["models"]:
        for dataset, source_rows in (("step", step_rows), ("target", target_rows)):
            for role, fields in (("learner", LEARNER_FIELDS), ("evaluator", EVALUATOR_FIELDS)):
                path = output_dir / "views" / model_name / f"{dataset}_{role}.csv"
                with path.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    if tuple(reader.fieldnames) != fields:
                        raise PermissionError(f"View columns mismatch: {path}")
                    rows = list(reader)
                if len(rows) != len(source_rows):
                    raise ValueError(f"View row conservation failed: {path}")
                for source, view in zip(source_rows, rows):
                    assert connection_key(source) == connection_key(view)
                    assert source["raw_row_id"] == view["raw_row_id"]
                    assert source["sequence_id"] == view["sequence_id"]
                    assert source["sequence_hash"] == view["sequence_hash"]
                    assert source[f"{model_name}_raw_output"] == view["model_raw_output"]
                    assert source[f"{model_name}_pred_mrl"] == view["model_pred_mrl"]
                    if role == "learner":
                        assert_learner_fields(view)
                    else:
                        assert source["measured_rl"] == view["measured_rl"]
                        assert source["total_reads"] == view["total_reads"]
                checked += len(rows)
    return {"view_rows_asserted": checked, "learner_forbidden_fields": sorted(FORBIDDEN_LEARNER_FIELDS), "original_blind": roles["models"]["original"]["blind"], "retrained_blind": roles["models"]["retrained"]["blind"], "retrained_lineage_status": roles["models"]["retrained"]["lineage_status"], "model_specific_views": True}


def permission_negative_tests():
    results = []
    for field in sorted(FORBIDDEN_LEARNER_FIELDS):
        try:
            assert_learner_fields(LEARNER_FIELDS + (field,))
        except PermissionError:
            results.append({"test": f"learner_rejects_{field}", "pass": True})
        else:
            raise ValueError(f"Learner accepted forbidden field {field}")
    invalid = {"schema_version": "1.0", "models": {
        "original": {"role": "x", "lineage_status": "unknown_retrain_membership", "blind": True, "learner_view_allowed": True, "evaluator_view_allowed": True},
        "retrained": {"role": "y", "lineage_status": "unknown_retrain_membership", "blind": False, "learner_view_allowed": True, "evaluator_view_allowed": True},
    }}
    try:
        validate_model_roles(invalid)
    except ValueError:
        results.append({"test": "unknown_lineage_blind_rejected", "pass": True})
    else:
        raise ValueError("Unknown lineage blind claim was accepted")
    try:
        validate_model_roles({"schema_version": "1.0", "models": {}})
    except ValueError:
        results.append({"test": "missing_model_roles_rejected", "pass": True})
    else:
        raise ValueError("Missing model roles were accepted")
    try:
        assert_learner_fields(BASE_VIEW_FIELDS)
    except PermissionError:
        results.append({"test": "learner_missing_model_output_rejected", "pass": True})
    else:
        raise ValueError("Learner view without model output was accepted")
    return {"negative_tests": results, "all_passed": all(item["pass"] for item in results)}
