import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from proxy_function_gap.pool_manifest import load_pool_manifest
from proxy_function_gap.trainer.data import canonical_row_order_hash, sha256_file


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_tsv(path, fieldnames, rows):
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def pool_factory(tmp_path):
    def build(pool_id, role="dev", count=12):
        root = tmp_path / pool_id
        root.mkdir()
        row_ids = tuple(f"{pool_id}-row-{index:03d}" for index in range(count))
        sequence_ids = tuple(f"{pool_id}-seq-{index:03d}" for index in range(count))
        group_ids = tuple(f"{pool_id}-group-{index % 3}" for index in range(count))
        mean = np.asarray([0.2, 1.0, 0.5, 0.8] * (count // 4), dtype=np.float64)
        measured = np.asarray([0.1, 0.9, 0.4, 0.85] * (count // 4), dtype=np.float64)
        members = np.stack([mean - 0.1, mean, mean + 0.1])
        prediction_path = root / f"{role}_predictions.npz"
        np.savez(
            prediction_path,
            row_id=np.asarray(row_ids),
            ensemble_mean=mean,
            ensemble_std=members.std(axis=0, ddof=1),
            ensemble_range=members.max(axis=0) - members.min(axis=0),
            pred_seed_11=members[0],
            pred_seed_23=members[1],
            pred_seed_37=members[2],
        )
        prediction_manifest = {
            "schema_version": "1.0",
            "labels_in_output": False,
            "outer_labels_read": False,
            "outer_metrics_computed": False,
            "files": {prediction_path.name: {"sha256": sha256_file(prediction_path), "rows": count}},
        }
        prediction_manifest_path = root / "prediction_manifest.json"
        write_json(prediction_manifest_path, prediction_manifest)
        support = np.linspace(0.1, 1.0, count)
        candidate_path = root / "candidates.tsv.gz"
        write_tsv(
            candidate_path,
            ["row_id", "sequence_id", "utr50", "group_id", "proxy_mean", "proxy_std", "proxy_range", "support"],
            [
                {
                    "row_id": row_ids[index],
                    "sequence_id": sequence_ids[index],
                    "utr50": "A" * 50,
                    "group_id": group_ids[index],
                    "proxy_mean": repr(float(mean[index])),
                    "proxy_std": repr(float(members[:, index].std(ddof=1))),
                    "proxy_range": repr(float(members[:, index].max() - members[:, index].min())),
                    "support": repr(float(support[index])),
                }
                for index in range(count)
            ],
        )
        label_path = root / "labels.tsv.gz"
        write_tsv(
            label_path,
            ["row_id", "measured_rl"],
            [{"row_id": row_ids[index], "measured_rl": repr(float(measured[index]))} for index in range(count)],
        )
        manifest = {
            "schema_version": "1.0",
            "pool_id": pool_id,
            "pool_role": role,
            "split_version": "sample2019-split-v2",
            "split_hash": "synthetic-split-hash",
            "candidate_file": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "prediction_manifest": {"path": str(prediction_manifest_path), "sha256": sha256_file(prediction_manifest_path)},
            "label_source": {"path": str(label_path), "sha256": sha256_file(label_path)},
            "row_order_sha256": canonical_row_order_hash(row_ids),
            "candidate_count": count,
            "synthetic": True,
            "proposal_probs": [1.0 / count] * count,
        }
        manifest_path = root / "pool_manifest.json"
        write_json(manifest_path, manifest)
        return load_pool_manifest(manifest_path), manifest_path, manifest

    return build


@pytest.fixture
def member_scorers():
    def make(pool):
        return tuple(
            lambda candidate, member=member: pool.member_predictions[member, candidate.index]
            for member in range(3)
        )
    return make
