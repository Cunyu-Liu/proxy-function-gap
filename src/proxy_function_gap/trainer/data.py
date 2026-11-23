import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


ROLES = ("train", "dev", "calibration")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PreflightReport:
    manifest_path: Path
    pack_path: Path
    split_version: str
    split_hash: str
    model_lineage: str
    role_counts: Dict[str, int]
    data_pack_sha256: str
    split_manifest_sha256: str
    learner_allowlist_sha256: str
    training_authorized: bool
    synthetic: bool

    def to_dict(self):
        return {
            "manifest_path": str(self.manifest_path),
            "pack_path": str(self.pack_path),
            "split_version": self.split_version,
            "split_hash": self.split_hash,
            "model_lineage": self.model_lineage,
            "role_counts": dict(self.role_counts),
            "data_pack_sha256": self.data_pack_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "learner_allowlist_sha256": self.learner_allowlist_sha256,
            "training_authorized": self.training_authorized,
            "synthetic": self.synthetic,
            "ready_for_formal_training": self.training_authorized and not self.synthetic,
        }


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    scale: float
    raw_std: float

    @classmethod
    def fit(cls, train_labels: np.ndarray) -> "TargetScaler":
        values = np.asarray(train_labels, dtype=np.float64)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("train labels must be a non-empty finite vector")
        mean = float(values.mean())
        raw_std = float(values.std(ddof=0))
        return cls(mean=mean, scale=raw_std if raw_std > 0 else 1.0, raw_std=raw_std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean

    def to_dict(self):
        return {"mean": self.mean, "scale": self.scale, "raw_std": self.raw_std, "ddof": 0, "fit_role": "train"}


def _evidence(manifest, key):
    item = manifest[key]
    path = Path(item["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != item["sha256"]:
        raise ValueError(f"evidence hash mismatch for {key}")
    return path, actual


def _read_pack(path: Path, expected_counts: Dict[str, int]):
    arrays = {}
    with np.load(path, allow_pickle=False) as pack:
        expected_keys = {f"{kind}_{role}" for role in ROLES for kind in ("x", "y")}
        if set(pack.files) != expected_keys:
            raise ValueError("pack must contain exactly x/y for train/dev/calibration")
        for role in ROLES:
            x = pack[f"x_{role}"]
            y = pack[f"y_{role}"]
            if x.ndim != 3 or x.shape[1:] != (4, 50):
                raise ValueError(f"x_{role} must have shape (n, 4, 50)")
            if y.ndim != 1 or y.shape[0] != x.shape[0] or not x.shape[0]:
                raise ValueError(f"invalid labels for {role}")
            if not np.issubdtype(x.dtype, np.floating) or not np.issubdtype(y.dtype, np.floating):
                raise ValueError("pack arrays must use floating dtypes")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("pack arrays must be finite")
            if not np.all((x == 0) | (x == 1)) or not np.all(x.sum(axis=1) == 1):
                raise ValueError("pack inputs must be one-hot 50nt sequences")
            if expected_counts.get(role) != x.shape[0]:
                raise ValueError(f"role count mismatch for {role}")
            arrays[role] = (x.astype(np.float32, copy=False), y.astype(np.float32, copy=False))
    return arrays


def preflight(manifest_path) -> PreflightReport:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    pack_path, pack_hash = _evidence(manifest, "pack")
    split_path, split_hash_file = _evidence(manifest, "split_manifest")
    allowlist_path, allowlist_hash = _evidence(manifest, "learner_allowlist")
    split = json.loads(split_path.read_text())
    allowlist = json.loads(allowlist_path.read_text())
    if manifest["split_version"] != "sample2019-split-v2" or split["split_version"] != manifest["split_version"]:
        raise PermissionError("trainer roles must come from sample2019-split-v2")
    if manifest["split_hash"] != split["split_hash"]:
        raise PermissionError("split hash mismatch")
    lineage = manifest["model_lineage"]
    if lineage != "original_evolution":
        raise PermissionError("fresh trainer requires original_evolution split-v2 roles")
    if split.get("outer_metrics_computed") is not False:
        raise PermissionError("split outer-metric state is not fail-closed")
    role_counts = split["role_counts"][lineage]
    if any(role not in role_counts for role in ROLES):
        raise PermissionError("split-v2 is missing a required trainer role")
    for key in ("feature_roles", "label_roles"):
        roles = allowlist.get(key, {})
        if any(role not in roles for role in ROLES) or any("outer" in role for role in roles):
            raise PermissionError(f"invalid Learner {key} permissions")
        if any(not Path(roles[role]["path"]).is_file() for role in ROLES):
            raise PermissionError(f"Learner {key} contains a missing role path")
    if not {"dev", "calibration"}.issubset(set(allowlist.get("parameter_edit_roles", []))):
        raise PermissionError("dev/calibration parameter origin is not allowed")
    if any(manifest["role_counts"][role] > role_counts[role] for role in ROLES):
        raise ValueError("pack role count exceeds split-v2 role count")
    arrays = _read_pack(pack_path, manifest["role_counts"])
    return PreflightReport(
        manifest_path=manifest_path,
        pack_path=pack_path,
        split_version=manifest["split_version"],
        split_hash=manifest["split_hash"],
        model_lineage=lineage,
        role_counts={role: arrays[role][0].shape[0] for role in ROLES},
        data_pack_sha256=pack_hash,
        split_manifest_sha256=split_hash_file,
        learner_allowlist_sha256=allowlist_hash,
        training_authorized=manifest["training_authorized"] is True,
        synthetic=manifest["synthetic"] is True,
    )


def load_roles(report: PreflightReport):
    if sha256_file(report.pack_path) != report.data_pack_sha256:
        raise ValueError("data pack changed after preflight")
    return _read_pack(report.pack_path, report.role_counts)


def encode_50nt(sequences):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    result = np.zeros((len(sequences), 4, 50), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        if len(sequence) != 50 or set(sequence) - set(mapping):
            raise ValueError("expected exactly 50 A/C/G/T bases")
        for column, base in enumerate(sequence):
            result[row, mapping[base], column] = 1.0
    return result
