import hashlib
import json
from pathlib import Path

import numpy as np

from .data import ROLES, sha256_file


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_synthetic_pack(output_dir, count_per_role=8):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261123)
    arrays = {}
    counts = {}
    for role_index, role in enumerate(ROLES):
        x = np.zeros((count_per_role, 4, 50), dtype=np.float32)
        positions = np.arange(50)
        channels = rng.integers(0, 4, size=(count_per_role, 50))
        x[np.arange(count_per_role)[:, None], channels, positions] = 1.0
        y = np.zeros(count_per_role, dtype=np.float32)
        arrays[f"x_{role}"] = x
        arrays[f"y_{role}"] = y
        counts[role] = count_per_role
    pack_path = output / "synthetic_pack.npz"
    np.savez(pack_path, **arrays)
    split_hash = hashlib.sha256(b"synthetic-sample2019-split-v2").hexdigest()
    split = {
        "split_version": "sample2019-split-v2",
        "split_hash": split_hash,
        "outer_metrics_computed": False,
        "role_counts": {"original_evolution": dict(counts)},
    }
    split_path = output / "split_manifest_v2.json"
    _write_json(split_path, split)
    allowlist = {
        "feature_roles": {role: {"path": str(pack_path), "columns": ["x"]} for role in ROLES},
        "label_roles": {role: {"path": str(pack_path), "columns": ["y"]} for role in ROLES},
        "parameter_edit_roles": ["dev", "calibration"],
    }
    allowlist_path = output / "learner_allowlist.json"
    _write_json(allowlist_path, allowlist)
    manifest = {
        "schema_version": "1.0",
        "split_version": "sample2019-split-v2",
        "split_hash": split_hash,
        "model_lineage": "original_evolution",
        "role_counts": counts,
        "pack": {"path": str(pack_path), "sha256": sha256_file(pack_path)},
        "split_manifest": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "learner_allowlist": {"path": str(allowlist_path), "sha256": sha256_file(allowlist_path)},
        "training_authorized": False,
        "synthetic": True,
    }
    manifest_path = output / "data_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path
