import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np

from .pool_manifest import load_pool_manifest
from .trainer.checkpoint import atomic_json_write
from .trainer.data import canonical_row_order_hash, sha256_file


SUPPORT_DEFINITION = "1 - minimum Hamming distance to any fresh train UTR50 / 50"


def _evidence(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def _jsonl_rows(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_candidates(path, rows):
    fields = ["row_id", "sequence_id", "utr50", "group_id", "proxy_mean", "proxy_std", "proxy_range", "support"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    temporary.replace(path)


def train_support(sequences, train_utr50):
    train_utr50 = list(train_utr50)
    if not train_utr50 or any(len(value) != 50 for value in train_utr50):
        raise ValueError("support reference must contain nonempty fixed UTR50 sequences")
    reference = np.frombuffer("".join(train_utr50).encode("ascii"), dtype=np.uint8).reshape(-1, 50)
    values = []
    for sequence in sequences:
        if len(sequence) != 50:
            raise ValueError("candidate support requires UTR50")
        candidate = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        distance = int(np.count_nonzero(reference != candidate, axis=1).min())
        values.append(1.0 - distance / 50.0)
    return np.asarray(values, dtype=np.float64)


def build_production_pool(source_pool_manifest, pool_id, prediction_manifest, output_dir):
    source_pool_manifest = Path(source_pool_manifest).resolve()
    prediction_manifest_path = Path(prediction_manifest).resolve()
    output = Path(output_dir)
    source = json.loads(source_pool_manifest.read_text())
    if source["synthetic"] is not False or source["E1_enabled"] is not False:
        raise PermissionError("only immutable non-synthetic pre-E1 pools are allowed")
    if pool_id not in source["pool_hashes"]:
        raise ValueError(f"unknown fixed pool: {pool_id}")
    role = "calibration" if pool_id.startswith("calibration_") else "dev" if pool_id.startswith("dev_") else None
    if role is None:
        raise PermissionError("production adapter only builds dev/calibration pools")
    final_validation = source_pool_manifest.parent.parent / "final_validation.json"
    validation = json.loads(final_validation.read_text())
    if validation.get("pass") is not True or validation.get("E1_enabled") is not False:
        raise PermissionError("fixed-pool delivery is not validated")
    source_pool_file = source_pool_manifest.parent / source["files"][role]
    rows = [row for row in _jsonl_rows(source_pool_file) if row["pool_id"] == pool_id]
    if not rows:
        raise ValueError("fixed-pool source identity mismatch")

    production_manifest_path = Path(source["source_production_manifest"])
    if sha256_file(production_manifest_path) != source["source_production_manifest_sha256"]:
        raise ValueError("source production manifest hash mismatch")
    production = json.loads(production_manifest_path.read_text())
    label_path = Path(production["source_tsv"][role]).resolve()
    if sha256_file(label_path) != production["source_tsv_hashes"][role]:
        raise ValueError("learner label-source hash mismatch")
    vault_path = source_pool_manifest.parent / "evaluator" / "vault_references.json"
    vault = json.loads(vault_path.read_text())
    if vault[role]["holder"] != "Learner" or Path(vault[role]["path"]).resolve() != label_path or vault[role]["sha256"] != production["source_tsv_hashes"][role]:
        raise PermissionError("label source is not the frozen Learner role source")

    inference = json.loads(prediction_manifest_path.read_text())
    if inference["synthetic"] is not False or inference["outer_labels_read"] or inference["outer_metrics_computed"]:
        raise PermissionError("invalid production inference boundary")
    if inference.get("prediction_units") != "measured_rl" or inference.get("inverse_target_transform") is not True:
        raise ValueError("production predictions must be inverse-transformed to measured_rl units")
    if inference["source_sha256"][role] != production["source_tsv_hashes"][role]:
        raise ValueError("inference and pool label source differ")
    prediction_path = prediction_manifest_path.parent / f"{role}_predictions.npz"
    if sha256_file(prediction_path) != inference["files"][prediction_path.name]["sha256"]:
        raise ValueError("prediction file hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as predictions:
        prediction_ids = [str(value) for value in predictions["row_id"].tolist()]
        if len(set(prediction_ids)) != len(prediction_ids):
            raise ValueError("duplicate prediction row_id")
        if canonical_row_order_hash(prediction_ids) != inference["row_order_sha256"][role]:
            raise ValueError("prediction role order hash mismatch")
        indices = {row_id: index for index, row_id in enumerate(prediction_ids)}
        missing = [row["row_id"] for row in rows if row["row_id"] not in indices]
        if missing:
            raise ValueError("fixed-pool rows missing from inference")
        selected = np.asarray([indices[row["row_id"]] for row in rows])
        means = predictions["ensemble_mean"][selected]
        stds = predictions["ensemble_std"][selected]
        ranges = predictions["ensemble_range"][selected]

    train_path = Path(production["production_files"]["train"]["path"])
    if sha256_file(train_path) != production["production_files"]["train"]["sha256"]:
        raise ValueError("support train-source hash mismatch")
    with np.load(train_path, allow_pickle=False) as train:
        support = train_support([row["utr50"] for row in rows], train["utr50"].tolist())
    candidate_rows = [
        {
            "row_id": row["row_id"],
            "sequence_id": row["sequence_id"],
            "utr50": row["utr50"],
            "group_id": row["cluster_id"],
            "proxy_mean": repr(float(means[index])),
            "proxy_std": repr(float(stds[index])),
            "proxy_range": repr(float(ranges[index])),
            "support": repr(float(support[index])),
        }
        for index, row in enumerate(rows)
    ]
    output.mkdir(parents=True, exist_ok=False)
    candidate_path = output / "candidates.tsv.gz"
    _write_candidates(candidate_path, candidate_rows)
    proposal = [float(row["q_value"]) for row in rows]
    manifest = {
        "schema_version": "1.0",
        "pool_id": pool_id,
        "pool_role": role,
        "split_version": "sample2019-split-v2",
        "split_hash": source["source_split_hash"],
        "candidate_file": _evidence(candidate_path),
        "prediction_manifest": _evidence(prediction_manifest_path),
        "label_source": _evidence(label_path),
        "row_order_sha256": canonical_row_order_hash([row["row_id"] for row in rows]),
        "candidate_count": len(rows),
        "synthetic": False,
        "proposal_probs": proposal,
        "source_pool_manifest": _evidence(source_pool_manifest),
        "source_pool_file": _evidence(source_pool_file),
        "source_pool_final_validation": _evidence(final_validation),
        "vault_references": _evidence(vault_path),
        "pool_hash": source["pool_hashes"][pool_id],
        "support_definition": SUPPORT_DEFINITION,
        "support_train_source": _evidence(train_path),
    }
    manifest_path = output / "pool_manifest.json"
    atomic_json_write(manifest, manifest_path)
    pool = load_pool_manifest(manifest_path)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "pool_id": pool.pool_id,
        "role": pool.pool_role,
        "rows": len(pool),
        "synthetic": False,
        "outer_labels_read": False,
        "outer_metrics_computed": False,
        "E1_enabled": False,
    }
