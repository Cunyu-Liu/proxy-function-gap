import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


ROLES = ("train", "dev", "calibration")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_row_order_hash(row_ids) -> str:
    ids = sorted(str(row_id) for row_id in row_ids)
    if not ids or any(not row_id for row_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("row_id values must be non-empty and unique")
    payload = "\n".join(ids) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    source_sha256: Dict[str, str]
    row_order_sha256: Dict[str, str]
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
            "source_sha256": dict(self.source_sha256),
            "row_order_sha256": dict(self.row_order_sha256),
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


def _read_source_ids(path: Path, column: str):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if column not in (reader.fieldnames or []):
            raise ValueError(f"missing {column} in {path}")
        ids = [row[column] for row in reader]
    canonical_row_order_hash(ids)
    return ids


def _read_pack(path: Path, expected_counts: Dict[str, int]):
    arrays = {}
    order_hashes = {}
    with np.load(path, allow_pickle=False) as pack:
        expected_keys = {f"{kind}_{role}" for role in ROLES for kind in ("x", "y", "row_id")}
        if set(pack.files) != expected_keys:
            raise ValueError("pack must contain exactly x/y/row_id for train/dev/calibration")
        for role in ROLES:
            x = pack[f"x_{role}"]
            y = pack[f"y_{role}"]
            row_ids = [str(value) for value in pack[f"row_id_{role}"].tolist()]
            if x.ndim != 3 or x.shape[1:] != (4, 50):
                raise ValueError(f"x_{role} must have shape (n, 4, 50)")
            if y.ndim != 1 or y.shape[0] != x.shape[0] or not x.shape[0] or len(row_ids) != x.shape[0]:
                raise ValueError(f"invalid labels or row IDs for {role}")
            if not np.issubdtype(x.dtype, np.floating) or not np.issubdtype(y.dtype, np.floating):
                raise ValueError("pack arrays must use floating dtypes")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("pack arrays must be finite")
            if not np.all((x == 0) | (x == 1)) or not np.all(x.sum(axis=1) == 1):
                raise ValueError("pack inputs must be one-hot 50nt sequences")
            if expected_counts.get(role) != x.shape[0]:
                raise ValueError(f"role count mismatch for {role}")
            order_hashes[role] = canonical_row_order_hash(row_ids)
            arrays[role] = (x.astype(np.float32, copy=False), y.astype(np.float32, copy=False), tuple(row_ids))
    return arrays, order_hashes


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
    split_role_counts = split["role_counts"][lineage]
    if any(role not in split_role_counts for role in ROLES):
        raise PermissionError("split-v2 is missing a required trainer role")
    for key in ("feature_roles", "label_roles"):
        roles = allowlist.get(key, {})
        if any(role not in roles for role in ROLES) or any("outer" in role for role in roles):
            raise PermissionError(f"invalid Learner {key} permissions")
        if any(not Path(roles[role]["path"]).is_file() for role in ROLES):
            raise PermissionError(f"Learner {key} contains a missing role path")
    if not {"dev", "calibration"}.issubset(set(allowlist.get("parameter_edit_roles", []))):
        raise PermissionError("dev/calibration parameter origin is not allowed")
    if any(manifest["role_counts"][role] > split_role_counts[role] for role in ROLES):
        raise ValueError("pack role count exceeds split-v2 role count")

    arrays, order_hashes = _read_pack(pack_path, manifest["role_counts"])
    source_hashes = {}
    for role in ROLES:
        source_path, source_hash = _evidence(manifest["role_sources"], role)
        source_hashes[role] = source_hash
        source_column = manifest["role_sources"][role].get("row_id_column", "row_id")
        source_ids = _read_source_ids(source_path, source_column)
        pack_ids = arrays[role][2]
        if len(source_ids) != len(pack_ids) or set(source_ids) != set(pack_ids):
            raise ValueError(f"row_id set/count mismatch for {role}")
        source_order_hash = canonical_row_order_hash(source_ids)
        if source_order_hash != order_hashes[role]:
            raise ValueError(f"canonical row-order hash mismatch for {role}")
        expected_order_hash = manifest["row_order_sha256"][role]
        if expected_order_hash != order_hashes[role]:
            raise ValueError(f"manifest row-order hash mismatch for {role}")
        label_path = Path(allowlist["label_roles"][role]["path"]).resolve()
        if label_path != source_path:
            raise PermissionError(f"label allowlist path does not match fresh TSV for {role}")
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
        source_sha256=source_hashes,
        row_order_sha256=order_hashes,
        training_authorized=manifest["training_authorized"] is True,
        synthetic=manifest["synthetic"] is True,
    )


def load_roles(report: PreflightReport):
    if sha256_file(report.pack_path) != report.data_pack_sha256:
        raise ValueError("data pack changed after preflight")
    arrays, order_hashes = _read_pack(report.pack_path, report.role_counts)
    if order_hashes != report.row_order_sha256:
        raise ValueError("row-order identity changed after preflight")
    return arrays


def encode_50nt(sequences):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    result = np.zeros((len(sequences), 4, 50), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        if len(sequence) != 50 or set(sequence) - set(mapping):
            raise ValueError("expected exactly 50 A/C/G/T bases")
        for column, base in enumerate(sequence):
            result[row, mapping[base], column] = 1.0
    return result
