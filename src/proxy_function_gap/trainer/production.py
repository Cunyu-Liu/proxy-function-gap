import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoint import atomic_json_write
from .data import ROLES, _read_source_ids, canonical_row_order_hash, sha256_file


DEFAULT_EVIDENCE = {
    "split_v1": "/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/derived/gate0_split_v1/split_manifest.json",
    "split_v2": "/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/derived/gate0_split_v2/split_manifest_v2.json",
    "parent_pack": "/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/derived/gate0_fresh_proxy_v1/pack_manifest.json",
    "input_manifest": "/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/data_manifest.json",
}


def _json(path):
    return json.loads(Path(path).read_text())


def _require_hash(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}")
    return actual


def prepare_production_pack(production_manifest_path, output_dir, evidence_paths=None, expected_source_commit=None):
    production_path = Path(production_manifest_path).resolve()
    root = production_path.parent.parent
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    evidence = dict(DEFAULT_EVIDENCE if evidence_paths is None else evidence_paths)
    production = _json(production_path)
    child_path = root / "child_manifest.json"
    integration_path = root / "integration_manifest.json"
    final_path = root / "final_validation.json"
    verification_path = root / "verification_summary.json"
    child = _json(child_path)
    integration = _json(integration_path)
    final = _json(final_path)
    verification = _json(verification_path)
    if expected_source_commit is not None and child["git_commit"] != expected_source_commit:
        raise PermissionError("fresh child source commit mismatch")
    if child["production_data_manifest"] != str(production_path) or integration["production_data_manifest"] != str(production_path):
        raise PermissionError("production manifest is not the authorized selected build")
    if not child["formal_training_start_authorized"] or not integration["formal_training_start_authorized"]:
        raise PermissionError("fresh child training is not authorized")
    if not final["pass"] or not all(verification["rounds"].values()) or not verification["formal_training_start_authorized"]:
        raise PermissionError("fresh child verification has not passed")
    for value in (child, integration, verification, final, production):
        if value.get("synthetic") is not False or value.get("outer_metrics_computed", False) is not False:
            raise PermissionError("production or outer-metric state is invalid")
    if child["child_split_version"] != "fresh-proxy-cnn-split-v2" or child["child_lineage_version"] != "fresh-proxy-cnn-lineage-v2":
        raise PermissionError("fresh child v2 identity mismatch")
    if child["initialization"] != "fresh_random" or child["official_weights"] or child["distillation"]:
        raise PermissionError("training must be fresh with no official weights or distillation")
    if child["E1_enabled"] or production["E1_enabled"] or production["outer_learner_package_created"]:
        raise PermissionError("E1 and outer learner packaging must remain disabled")

    _require_hash(evidence["split_v1"], production["split_v1_manifest_sha256"])
    _require_hash(evidence["split_v2"], production["split_v2_manifest_sha256"])
    _require_hash(evidence["parent_pack"], production["pack_manifest_sha256"])
    _require_hash(evidence["input_manifest"], child["data_manifest_sha256"])
    _require_hash(child["learner_allowlist"], production["learner_allowlist_sha256"])
    input_manifest = _json(evidence["input_manifest"])
    input_hashes = {item["name"]: item["sha256"] for item in input_manifest["files"]}
    if production["source_file_hashes"] != input_hashes:
        raise ValueError("raw source-file hashes differ from the input manifest")
    split = _json(evidence["split_v2"])
    allowlist = _json(child["learner_allowlist"])
    if split["split_version"] != "sample2019-split-v2" or allowlist["child_split_version"] != child["child_split_version"]:
        raise PermissionError("split lineage mismatch")
    if allowlist["outer_access"] or allowlist["official_seen_control_quarantine_access"]:
        raise PermissionError("learner outer access must remain disabled")

    output_pack = output / "training_pack.npz"
    arrays = {}
    role_sources = {}
    role_counts = {}
    order_hashes = {}
    for role in ROLES:
        item = production["production_files"][role]
        source_npz = Path(item["path"])
        _require_hash(source_npz, item["sha256"])
        source_tsv = Path(production["source_tsv"][role])
        _require_hash(source_tsv, production["source_tsv_hashes"][role])
        allow_role = allowlist["feature_label_sources"][role]
        if Path(allow_role["path"]) != source_tsv or allow_role["sha256"] != production["source_tsv_hashes"][role]:
            raise PermissionError(f"allowlist/source mismatch for {role}")
        source_ids = _read_source_ids(source_tsv, "row_id")
        with np.load(source_npz, allow_pickle=False) as pack:
            if bool(pack["synthetic"]) or pack["utr50_onehot"].shape[1:] != (50, 4):
                raise ValueError(f"invalid production NPZ for {role}")
            row_ids = [str(value) for value in pack["row_id"].tolist()]
            if row_ids != source_ids:
                raise ValueError(f"physical row-order mismatch for {role}")
            order_hash = canonical_row_order_hash(row_ids)
            if order_hash != production["canonical_row_id_hashes"][role]:
                raise ValueError(f"production canonical row-id hash mismatch for {role}")
            if any(str(value) != role for value in pack["role"].tolist()):
                raise ValueError(f"role field mismatch for {role}")
            if any(str(value) != production["pack_version"] for value in pack["split_version"].tolist()):
                raise ValueError(f"NPZ split version mismatch for {role}")
            if any(str(value) != production["normalization_version"] for value in pack["normalization_version"].tolist()):
                raise ValueError(f"normalization mismatch for {role}")
            utrs = [str(value) for value in pack["utr50"].tolist()]
            sequence_hashes = pack["sequence_hash"].tolist()
            if any(hashlib.sha256(utr.encode()).hexdigest() != str(sequence_hashes[index]) for index, utr in enumerate(utrs)):
                raise ValueError(f"sequence hash mismatch for {role}")
            arrays[f"x_{role}"] = pack["utr50_onehot"].transpose(0, 2, 1).astype(np.float32)
            arrays[f"y_{role}"] = pack["measured_rl"].astype(np.float32)
            arrays[f"row_id_{role}"] = np.asarray(row_ids)
        role_counts[role] = len(row_ids)
        order_hashes[role] = order_hash
        role_sources[role] = {
            "path": str(source_tsv),
            "sha256": production["source_tsv_hashes"][role],
            "row_id_column": "row_id",
            "physical_order_required": True,
        }
    expected_counts = {
        "train": child["train_scaler_rows"],
        "dev": child["dev_rows"],
        "calibration": child["calibration_rows"],
    }
    if role_counts != expected_counts:
        raise ValueError("child role counts do not match production NPZ")
    temporary = output / "training_pack.tmp.npz"
    np.savez(temporary, **arrays)
    temporary.replace(output_pack)
    pack_hash = sha256_file(output_pack)
    trainer_allowlist = {
        "feature_roles": {
            role: {"path": role_sources[role]["path"], "columns": ["row_id", "utr50"]} for role in ROLES
        },
        "label_roles": {
            role: {"path": role_sources[role]["path"], "columns": ["row_id", "measured_rl"]} for role in ROLES
        },
        "parameter_edit_roles": ["dev", "calibration"],
    }
    trainer_allowlist_path = output / "learner_allowlist.json"
    atomic_json_write(trainer_allowlist, trainer_allowlist_path)
    manifest = {
        "schema_version": "2.0",
        "split_version": split["split_version"],
        "split_hash": split["split_hash"],
        "model_lineage": "original_evolution",
        "role_counts": role_counts,
        "pack": {"path": str(output_pack), "sha256": pack_hash},
        "split_manifest": {"path": evidence["split_v2"], "sha256": production["split_v2_manifest_sha256"]},
        "learner_allowlist": {"path": str(trainer_allowlist_path), "sha256": sha256_file(trainer_allowlist_path)},
        "source_learner_allowlist": {"path": child["learner_allowlist"], "sha256": production["learner_allowlist_sha256"]},
        "role_sources": role_sources,
        "row_order_sha256": order_hashes,
        "training_authorized": True,
        "synthetic": False,
        "fresh_child_v2": True,
        "physical_row_order_verified": True,
        "child_manifest": {"path": str(child_path), "sha256": sha256_file(child_path)},
        "integration_manifest": {"path": str(integration_path), "sha256": sha256_file(integration_path)},
        "final_validation": {"path": str(final_path), "sha256": sha256_file(final_path)},
    }
    manifest_path = output / "data_manifest.json"
    atomic_json_write(manifest, manifest_path)
    return manifest_path
