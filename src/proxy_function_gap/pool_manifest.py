import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .trainer.data import canonical_row_order_hash, sha256_file


REQUIRED_CANDIDATE_COLUMNS = ("row_id", "sequence_id", "utr50", "group_id", "proxy_mean", "proxy_std", "proxy_range", "support")
PRODUCTION_FIELDS = {
    "source_pool_manifest", "source_pool_file", "source_pool_final_validation", "vault_references",
    "pool_hash", "support_definition", "support_train_source",
}


@dataclass(frozen=True)
class PoolData:
    manifest_path: Path
    manifest_sha256: str
    pool_id: str
    pool_role: str
    synthetic: bool
    row_ids: tuple
    sequence_ids: tuple
    utr50: tuple
    group_ids: tuple
    proxy_mean: np.ndarray
    proxy_std: np.ndarray
    proxy_range: np.ndarray
    support: np.ndarray
    measured: np.ndarray
    proposal_probs: np.ndarray
    member_predictions: np.ndarray
    source_manifests: tuple

    def __len__(self):
        return len(self.row_ids)


def _rows(path):
    with gzip.open(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _jsonl_rows(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _evidence(item):
    path = Path(item["path"]).resolve()
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise ValueError(f"evidence hash mismatch: {path}")
    return path


def _read_label_source(path):
    ids = []
    values = []
    for row in _rows(path):
        ids.append(row["row_id"])
        values.append(float(row["measured_rl"]))
    if len(set(ids)) != len(ids) or not np.isfinite(values).all():
        raise ValueError("label source requires unique row_id and finite measured values")
    return ids, np.asarray(values, dtype=np.float64)


def _verify_production(manifest, candidate_rows, label_path, prediction_manifest, prediction_ids):
    from .production_pool import SUPPORT_DEFINITION, train_support

    role = manifest["pool_role"]
    source_pool_path = _evidence(manifest["source_pool_manifest"])
    source_pool_file = _evidence(manifest["source_pool_file"])
    final_validation_path = _evidence(manifest["source_pool_final_validation"])
    vault_path = _evidence(manifest["vault_references"])
    source = json.loads(source_pool_path.read_text())
    if source["synthetic"] is not False or source["E1_enabled"] is not False:
        raise PermissionError("invalid fixed-pool production boundary")
    if source["source_split_hash"] != manifest["split_hash"] or source["pool_hashes"].get(manifest["pool_id"]) != manifest["pool_hash"]:
        raise ValueError("fixed-pool split or pool hash mismatch")
    expected_pool_file = (source_pool_path.parent / source["files"][role]).resolve()
    if source_pool_file != expected_pool_file:
        raise PermissionError("candidate source is not the frozen role pool")
    validation = json.loads(final_validation_path.read_text())
    if final_validation_path != (source_pool_path.parent.parent / "final_validation.json").resolve() or validation.get("pass") is not True or validation.get("E1_enabled") is not False:
        raise PermissionError("fixed-pool final validation is missing or invalid")

    production_path = Path(source["source_production_manifest"]).resolve()
    if sha256_file(production_path) != source["source_production_manifest_sha256"]:
        raise ValueError("source production manifest hash mismatch")
    production = json.loads(production_path.read_text())
    if label_path != Path(production["source_tsv"][role]).resolve() or manifest["label_source"]["sha256"] != production["source_tsv_hashes"][role]:
        raise PermissionError("label source is not the frozen Learner role source")
    vault = json.loads(vault_path.read_text())
    if vault_path != (source_pool_path.parent / "evaluator" / "vault_references.json").resolve() or vault[role] != {"holder": "Learner", "path": str(label_path), "sha256": production["source_tsv_hashes"][role]}:
        raise PermissionError("Learner vault binding mismatch")
    if prediction_manifest.get("prediction_units") != "measured_rl" or prediction_manifest.get("inverse_target_transform") is not True:
        raise ValueError("production predictions must use measured_rl units with an inverse target transform")
    if prediction_manifest.get("synthetic") is not False or prediction_manifest["source_sha256"][role] != production["source_tsv_hashes"][role]:
        raise ValueError("inference provenance does not match the frozen role source")
    if canonical_row_order_hash(prediction_ids) != prediction_manifest["row_order_sha256"][role]:
        raise ValueError("full prediction row-order hash mismatch")

    source_rows = [row for row in _jsonl_rows(source_pool_file) if row["pool_id"] == manifest["pool_id"]]
    if len(source_rows) != len(candidate_rows):
        raise ValueError("frozen pool and candidate counts differ")
    for source_row, candidate in zip(source_rows, candidate_rows):
        if source_row["pool_id"] != manifest["pool_id"] or source_row["pool_role"] != role or source_row["pool_hash"] != manifest["pool_hash"] or source_row["source_split_hash"] != manifest["split_hash"] or source_row["synthetic"] is not False or source_row["independent_evidence"] is not True:
            raise PermissionError("frozen source-row role or independence mismatch")
        expected = {"row_id": source_row["row_id"], "sequence_id": source_row["sequence_id"], "utr50": source_row["utr50"], "group_id": source_row["cluster_id"]}
        if any(candidate[key] != value for key, value in expected.items()):
            raise ValueError("candidate identity differs from frozen source pool")
    if not np.allclose(np.asarray(manifest.get("proposal_probs", [])), np.asarray([float(row["q_value"]) for row in source_rows]), rtol=0, atol=1e-15):
        raise ValueError("proposal probabilities differ from frozen q values")

    train_path = _evidence(manifest["support_train_source"])
    if train_path != Path(production["production_files"]["train"]["path"]).resolve() or manifest["support_train_source"]["sha256"] != production["production_files"]["train"]["sha256"]:
        raise PermissionError("support reference is not the frozen fresh train source")
    if manifest["support_definition"] != SUPPORT_DEFINITION:
        raise ValueError("unknown production support definition")
    with np.load(train_path, allow_pickle=False) as train:
        expected_support = train_support([row["utr50"] for row in source_rows], train["utr50"].tolist())
    actual_support = np.asarray([float(row["support"]) for row in candidate_rows])
    if not np.allclose(actual_support, expected_support, rtol=0, atol=1e-12):
        raise ValueError("candidate support differs from frozen train-only definition")


def load_pool_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    required = {
        "schema_version", "pool_id", "pool_role", "split_version", "split_hash", "candidate_file",
        "prediction_manifest", "label_source", "row_order_sha256", "candidate_count", "synthetic",
    }
    allowed = required | {"proposal_probs"} | PRODUCTION_FIELDS
    if set(manifest) - allowed:
        raise ValueError(f"unknown pool manifest fields: {sorted(set(manifest) - allowed)}")
    if manifest["schema_version"] != "1.0" or manifest["pool_role"] not in ("dev", "calibration"):
        raise PermissionError("pool role must be dev or calibration; outer is not implemented")
    if manifest["split_version"] != "sample2019-split-v2" or not manifest["split_hash"]:
        raise PermissionError("invalid pool split identity")
    synthetic = manifest["synthetic"] is True
    if not synthetic and not PRODUCTION_FIELDS <= set(manifest):
        raise PermissionError("production pools require complete frozen-pool provenance")
    candidate_path = _evidence(manifest["candidate_file"])
    prediction_manifest_path = _evidence(manifest["prediction_manifest"])
    label_path = _evidence(manifest["label_source"])
    prediction_manifest = json.loads(prediction_manifest_path.read_text())
    if prediction_manifest["labels_in_output"] or prediction_manifest["outer_labels_read"] or prediction_manifest["outer_metrics_computed"]:
        raise PermissionError("prediction manifest violates no-label/no-outer boundary")
    prediction_path = prediction_manifest_path.parent / f"{manifest['pool_role']}_predictions.npz"
    if sha256_file(prediction_path) != prediction_manifest["files"][f"{manifest['pool_role']}_predictions.npz"]["sha256"]:
        raise ValueError("prediction file hash mismatch")

    candidate_rows = list(_rows(candidate_path))
    if len(candidate_rows) != manifest["candidate_count"] or not candidate_rows:
        raise ValueError("candidate count mismatch")
    if tuple(candidate_rows[0].keys()) != REQUIRED_CANDIDATE_COLUMNS:
        raise ValueError("candidate TSV must contain exactly the declared schema and no labels")
    row_ids = tuple(row["row_id"] for row in candidate_rows)
    if len(set(row_ids)) != len(row_ids) or canonical_row_order_hash(row_ids) != manifest["row_order_sha256"]:
        raise ValueError("candidate row-order hash mismatch")
    label_ids, measured = _read_label_source(label_path)
    if not set(row_ids) <= set(label_ids):
        raise ValueError("candidate row_ids are not present in the allowed label source")
    label_by_id = dict(zip(label_ids, measured))
    measured = np.asarray([label_by_id[row_id] for row_id in row_ids], dtype=np.float64)
    with np.load(prediction_path, allow_pickle=False) as predictions:
        if set(predictions.files) != {"row_id", "ensemble_mean", "ensemble_std", "ensemble_range", "pred_seed_11", "pred_seed_23", "pred_seed_37"}:
            raise ValueError("unexpected prediction columns")
        prediction_ids = [str(value) for value in predictions["row_id"].tolist()]
        if len(set(prediction_ids)) != len(prediction_ids) or not set(row_ids) <= set(prediction_ids):
            raise ValueError("prediction row_id coverage mismatch")
        indices = {row_id: index for index, row_id in enumerate(prediction_ids)}
        selected = np.asarray([indices[row_id] for row_id in row_ids])
        member_predictions = np.stack([predictions[f"pred_seed_{seed}"][selected] for seed in (11, 23, 37)])
        for name, key in (("proxy_mean", "ensemble_mean"), ("proxy_std", "ensemble_std"), ("proxy_range", "ensemble_range")):
            actual = np.asarray([float(row[name]) for row in candidate_rows])
            if not np.allclose(actual, predictions[key][selected], rtol=0, atol=1e-12):
                raise ValueError(f"candidate {name} differs from prediction NPZ")
    support = np.asarray([float(row["support"]) for row in candidate_rows])
    if not np.isfinite(support).all() or np.any(support < 0):
        raise ValueError("support must be finite and non-negative")
    proposal = np.asarray(manifest.get("proposal_probs", [1.0 / len(row_ids)] * len(row_ids)), dtype=np.float64)
    if len(proposal) != len(row_ids) or np.any(proposal < 0) or not np.isclose(proposal.sum(), 1.0):
        raise ValueError("invalid proposal distribution")
    if not synthetic:
        _verify_production(manifest, candidate_rows, label_path, prediction_manifest, prediction_ids)
    return PoolData(
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        pool_id=manifest["pool_id"],
        pool_role=manifest["pool_role"],
        synthetic=synthetic,
        row_ids=row_ids,
        sequence_ids=tuple(row["sequence_id"] for row in candidate_rows),
        utr50=tuple(row["utr50"] for row in candidate_rows),
        group_ids=tuple(row["group_id"] for row in candidate_rows),
        proxy_mean=np.asarray([float(row["proxy_mean"]) for row in candidate_rows]),
        proxy_std=np.asarray([float(row["proxy_std"]) for row in candidate_rows]),
        proxy_range=np.asarray([float(row["proxy_range"]) for row in candidate_rows]),
        support=support,
        measured=measured,
        proposal_probs=proposal,
        member_predictions=member_predictions,
        source_manifests=((path, sha256_file(path)),),
    )


def combine_pools(pools, pool_id):
    if not pools or len({(pool.pool_role, pool.synthetic) for pool in pools}) != 1:
        raise ValueError("combined pools must share one role and synthetic status")
    row_ids = tuple(row_id for pool in pools for row_id in pool.row_ids)
    sequence_ids = tuple(value for pool in pools for value in pool.sequence_ids)
    group_ids = tuple(value for pool in pools for value in pool.group_ids)
    group_sets = [set(pool.group_ids) for pool in pools]
    all_groups = [value for values in group_sets for value in values]
    conflicts = {
        "row": sorted({value for value in row_ids if row_ids.count(value) > 1}),
        "sequence": sorted({value for value in sequence_ids if sequence_ids.count(value) > 1}),
        "group": sorted({value for value in set(all_groups) if sum(value in values for values in group_sets) > 1}),
    }
    if any(conflicts.values()):
        raise ValueError(f"combined pools must be disjoint by row, sequence and group: {conflicts}")
    count = len(row_ids)
    sources = tuple(source for pool in pools for source in pool.source_manifests)
    identity = json.dumps([[str(path), sha] for path, sha in sources], separators=(",", ":"))
    import hashlib
    return PoolData(
        manifest_path=None,
        manifest_sha256=hashlib.sha256(identity.encode()).hexdigest(),
        pool_id=pool_id,
        pool_role=pools[0].pool_role,
        synthetic=pools[0].synthetic,
        row_ids=row_ids,
        sequence_ids=sequence_ids,
        utr50=tuple(value for pool in pools for value in pool.utr50),
        group_ids=group_ids,
        proxy_mean=np.concatenate([pool.proxy_mean for pool in pools]),
        proxy_std=np.concatenate([pool.proxy_std for pool in pools]),
        proxy_range=np.concatenate([pool.proxy_range for pool in pools]),
        support=np.concatenate([pool.support for pool in pools]),
        measured=np.concatenate([pool.measured for pool in pools]),
        proposal_probs=np.full(count, 1.0 / count),
        member_predictions=np.concatenate([pool.member_predictions for pool in pools], axis=1),
        source_manifests=sources,
    )
