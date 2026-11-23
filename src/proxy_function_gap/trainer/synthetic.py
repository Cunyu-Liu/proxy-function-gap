import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import ROLES, canonical_row_order_hash, sha256_file


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class ZeroParameterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        return inputs[:, 0, 0] * 0 + self.weight


def make_synthetic_pack(output_dir, count_per_role=8):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261123)
    arrays = {}
    counts = {}
    role_sources = {}
    order_hashes = {}
    for role in ROLES:
        x = np.zeros((count_per_role, 4, 50), dtype=np.float32)
        positions = np.arange(50)
        channels = rng.integers(0, 4, size=(count_per_role, 50))
        x[np.arange(count_per_role)[:, None], channels, positions] = 1.0
        y = np.zeros(count_per_role, dtype=np.float32)
        row_ids = tuple(f"synthetic-{role}-{index:04d}" for index in range(count_per_role))
        arrays[f"x_{role}"] = x
        arrays[f"y_{role}"] = y
        arrays[f"row_id_{role}"] = np.asarray(row_ids)
        counts[role] = count_per_role
        order_hashes[role] = canonical_row_order_hash(row_ids)
        source_path = output / f"{role}.tsv.gz"
        with gzip.open(source_path, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["row_id", "measured_rl"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for row_id in reversed(row_ids):
                writer.writerow({"row_id": row_id, "measured_rl": "0.0"})
        role_sources[role] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "row_id_column": "row_id",
        }
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
        "feature_roles": {
            role: {"path": role_sources[role]["path"], "columns": ["row_id"]} for role in ROLES
        },
        "label_roles": {
            role: {"path": role_sources[role]["path"], "columns": ["row_id", "measured_rl"]} for role in ROLES
        },
        "parameter_edit_roles": ["dev", "calibration"],
    }
    allowlist_path = output / "learner_allowlist.json"
    _write_json(allowlist_path, allowlist)
    manifest = {
        "schema_version": "2.0",
        "split_version": "sample2019-split-v2",
        "split_hash": split_hash,
        "model_lineage": "original_evolution",
        "role_counts": counts,
        "pack": {"path": str(pack_path), "sha256": sha256_file(pack_path)},
        "split_manifest": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "learner_allowlist": {"path": str(allowlist_path), "sha256": sha256_file(allowlist_path)},
        "role_sources": role_sources,
        "row_order_sha256": order_hashes,
        "training_authorized": False,
        "synthetic": True,
    }
    manifest_path = output / "data_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path
