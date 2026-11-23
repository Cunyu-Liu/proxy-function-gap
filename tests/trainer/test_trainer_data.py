import csv
import gzip
import json

import numpy as np
import pytest

from proxy_function_gap.trainer.data import TargetScaler, encode_50nt, load_roles, preflight, sha256_file
from proxy_function_gap.trainer.synthetic import make_synthetic_pack


def test_preflight_checks_exact_roles_hashes_and_permission(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    report = preflight(manifest_path)
    assert report.split_version == "sample2019-split-v2"
    assert report.role_counts == {"train": 8, "dev": 8, "calibration": 8}
    assert report.data_pack_sha256 == sha256_file(report.pack_path)
    assert set(report.source_sha256) == {"train", "dev", "calibration"}
    assert set(report.row_order_sha256) == {"train", "dev", "calibration"}
    assert all(len(value) == 64 for value in report.row_order_sha256.values())
    assert not report.training_authorized
    assert report.synthetic
    assert not report.to_dict()["ready_for_formal_training"]
    roles = load_roles(report)
    assert set(roles) == {"train", "dev", "calibration"}
    assert len(roles["train"][2]) == 8


def test_production_manifest_with_false_synthetic_is_ready_when_authorized(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    manifest = json.loads(manifest_path.read_text())
    manifest["synthetic"] = False
    manifest["training_authorized"] = True
    manifest_path.write_text(json.dumps(manifest))
    report = preflight(manifest_path)
    assert not report.synthetic
    assert report.training_authorized
    assert report.to_dict()["ready_for_formal_training"]


def test_pack_hash_tampering_is_rejected(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    report = preflight(manifest_path)
    with np.load(report.pack_path) as pack:
        arrays = {name: pack[name] for name in pack.files}
    arrays["x_outer"] = arrays["x_train"]
    np.savez(report.pack_path, **arrays)
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        preflight(manifest_path)
    with pytest.raises(ValueError, match="changed after preflight"):
        load_roles(report)


def test_row_id_set_mismatch_is_rejected_even_with_same_count(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    manifest = json.loads(manifest_path.read_text())
    source = manifest["role_sources"]["train"]
    with gzip.open(source["path"], "rt", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows[0]["row_id"] = "synthetic-train-missing"
    with gzip.open(source["path"], "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "measured_rl"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source["sha256"] = sha256_file(source["path"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row_id set/count mismatch for train"):
        preflight(manifest_path)


def test_source_hash_and_manifest_order_hash_are_enforced(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    manifest = json.loads(manifest_path.read_text())
    source = manifest["role_sources"]["train"]
    with gzip.open(source["path"], "rt", newline="") as handle:
        content = handle.read()
    with gzip.open(source["path"], "wt", newline="") as handle:
        handle.write(content + "\n")
    with pytest.raises(ValueError, match="evidence hash mismatch for train"):
        preflight(manifest_path)


def test_manifest_order_hash_mismatch_is_rejected(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    manifest = json.loads(manifest_path.read_text())
    manifest["row_order_sha256"]["train"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest row-order hash mismatch for train"):
        preflight(manifest_path)


def test_outer_allowlist_permission_is_rejected_even_with_updated_hash(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    manifest = json.loads(manifest_path.read_text())
    allowlist_path = manifest["learner_allowlist"]["path"]
    allowlist = json.loads(open(allowlist_path).read())
    allowlist["label_roles"]["outer"] = {"path": manifest["pack"]["path"], "columns": ["y"]}
    with open(allowlist_path, "w") as handle:
        json.dump(allowlist, handle)
    manifest["learner_allowlist"]["sha256"] = sha256_file(allowlist_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PermissionError, match="invalid Learner"):
        preflight(manifest_path)


def test_split_hash_mismatch_is_rejected(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    manifest = json.loads(manifest_path.read_text())
    manifest["split_hash"] = "invalid"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PermissionError, match="split hash mismatch"):
        preflight(manifest_path)


def test_scaler_uses_train_labels_only_and_handles_constant_targets():
    train = np.array([1.0, 3.0], dtype=np.float32)
    scaler = TargetScaler.fit(train)
    assert scaler.mean == 2.0
    assert scaler.raw_std == 1.0
    assert scaler.transform(np.array([3.0])) == pytest.approx([1.0])
    assert scaler.inverse(np.array([1.0])) == pytest.approx([3.0])
    constant = TargetScaler.fit(np.zeros(3))
    assert constant.raw_std == 0
    assert constant.scale == 1
    assert constant.to_dict()["fit_role"] == "train"


def test_encode_50nt_has_channels_first_shape_and_validates_alphabet():
    encoded = encode_50nt(["A" * 50, "C" * 50])
    assert encoded.shape == (2, 4, 50)
    assert encoded[0, 0].sum() == 50
    assert encoded[1, 1].sum() == 50
    with pytest.raises(ValueError):
        encode_50nt(["A" * 49])
    with pytest.raises(ValueError):
        encode_50nt(["N" * 50])


def test_pack_role_count_cannot_exceed_split_v2_count(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    manifest = json.loads(manifest_path.read_text())
    manifest["role_counts"]["train"] = 9
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exceeds split-v2"):
        preflight(manifest_path)


def test_allowlist_role_path_must_exist(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    manifest = json.loads(manifest_path.read_text())
    allowlist_path = manifest["learner_allowlist"]["path"]
    allowlist = json.loads(open(allowlist_path).read())
    allowlist["label_roles"]["train"]["path"] = str(tmp_path / "missing.tsv.gz")
    with open(allowlist_path, "w") as handle:
        json.dump(allowlist, handle)
    manifest["learner_allowlist"]["sha256"] = sha256_file(allowlist_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PermissionError, match="missing role path"):
        preflight(manifest_path)


def test_physical_order_requirement_rejects_reversed_source_tsv(tmp_path):
    manifest_path = make_synthetic_pack(tmp_path / "pack")
    manifest = json.loads(manifest_path.read_text())
    manifest["role_sources"]["train"]["physical_order_required"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="physical row-order mismatch for train"):
        preflight(manifest_path)
