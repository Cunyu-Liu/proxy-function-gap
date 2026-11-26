from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from adaptation import (
    ANALYSIS_ID, V1_ANALYSIS_ID, HalfIndex, allocate_exact, component_identity,
    component_pools, digest, h1_components, make_pools, make_proposals,
    save_json, state_dict_hash,
)


def read_sample_index(config):
    index = HalfIndex()
    rows = 0
    hashes = {}
    for path in config["sample_roles"]:
        hashes[path] = digest(path)
        if hashes[path] != config["sample_sha256"][path]:
            raise ValueError("Sample role hash mismatch")
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                value = row["utr50"]
                if len(value) != 50 or set(value) - set("ACGT"):
                    raise ValueError("invalid Sample utr50")
                index.add(value)
                rows += 1
    return index, {"rows": rows, "unique_sequences": len(index.exact), "sha256": hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if config["analysis_id"] != ANALYSIS_ID:
        raise ValueError("wrong analysis")
    root = Path(config["artifact_root"])
    v1 = Path(config["v1_artifact_root"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    shutil.copyfile(config_path, root / "configs" / "listnet_frozen.json")
    config_hash = digest(config_path)
    workspace = Path(__file__).resolve().parent
    source_files = [
        "adaptation.py", "prepare.py", "train.py", "select_models.py",
        "freeze_predictions.py", "evaluate.py", "independent_verify.py",
        "delivery_verify.py", "export_delivery.py", "tests/test_listnet.py",
    ]
    save_json(root / "verification" / "prefreeze.json", {
        "analysis_id": ANALYSIS_ID,
        "config_sha256": config_hash,
        "code_sha256": {name: digest(workspace / name) for name in source_files},
        "test_v2_mrl_read": False,
        "target_evaluation_count": 0,
    })

    raw_hashes = {name: digest(path) for name, path in config["raw"].items()}
    if raw_hashes != config["raw_sha256"]:
        raise ValueError("raw target hash mismatch")
    v1_prepare = json.loads((v1 / "prepare_manifest.json").read_text())
    if v1_prepare["status"] != "PASS" or v1_prepare["analysis_id"] != V1_ANALYSIS_ID:
        raise ValueError("v1 preparation identity mismatch")
    sequences = np.load(v1 / "data" / "sequences.npy", allow_pickle=False)
    v1_split = np.load(v1 / "data" / "split.npy", allow_pickle=False)
    if digest(v1 / "data" / "sequences.npy") != v1_prepare["files"]["sequences.npy"] or digest(v1 / "data" / "split.npy") != v1_prepare["files"]["split.npy"]:
        raise ValueError("v1 sequence/split hash mismatch")
    train_indices = np.load(v1 / "data" / "train_indices.npy", allow_pickle=False)
    dev_indices = np.load(v1 / "data" / "dev_indices.npy", allow_pickle=False)
    v1_test_indices = np.load(v1 / "data" / "test_indices.npy", allow_pickle=False)
    train_labels = np.load(v1 / "data" / "train_dev_mean_labels.npy", allow_pickle=False)
    if len(train_indices) != 50000 or len(dev_indices) != 10000 or len(v1_test_indices) != 1024:
        raise ValueError("v1 role size mismatch")
    if not np.isfinite(train_labels[train_indices]).all() or not np.isfinite(train_labels[dev_indices]).all() or not np.isnan(train_labels[v1_test_indices]).all():
        raise ValueError("v1 train/dev label role mismatch")

    components, h1_edges = h1_components(sequences)
    if len(components) != v1_prepare["components"] or h1_edges != v1_prepare["h1_edges"]:
        raise ValueError("v1 component reconstruction mismatch")
    identities, order_keys = [], []
    eligible = []
    sample_index, sample_report = read_sample_index(config)
    sample_exact = sample_h1 = 0
    excluded_v1_components = 0
    for members in components:
        identity, order_key, minimum = component_identity(sequences, members)
        identities.append(identity)
        order_keys.append((order_key, minimum))
        v1_roles = {int(v1_split[index]) for index in members if v1_split[index] >= 0}
        if len(v1_roles) > 1:
            raise ValueError("v1 component crosses roles")
        excluded = bool(v1_roles)
        excluded_v1_components += int(excluded)
        clean = not excluded
        if clean:
            for index in members:
                exact, h1 = sample_index.match(sequences[index].decode("ascii"))
                sample_exact += int(exact)
                sample_h1 += int(h1)
                if exact or h1:
                    clean = False
                    break
        eligible.append(clean)

    eligible_components = [index for index in range(len(components)) if eligible[index]]
    chosen_local = allocate_exact(
        [components[index] for index in eligible_components],
        [order_keys[index] for index in eligible_components],
        config["split"]["test_unique_sequences"],
    )
    test_components = [eligible_components[index] for index in chosen_local]
    test_set = set(test_components)
    pools = component_pools(
        [components[index] for index in test_components],
        [order_keys[index] for index in test_components],
        config["split"]["test_pools"],
        config["split"]["test_pool_size"],
    )
    test_indices = np.asarray(sorted(pools.ravel().tolist()), dtype=np.int64)
    if len(test_indices) != 1024 or len(set(test_indices.tolist())) != 1024:
        raise ValueError("test-v2 size mismatch")
    if set(test_indices.tolist()) & set(np.flatnonzero(v1_split >= 0).tolist()):
        raise ValueError("test-v2 overlaps a v1 role")
    component_of = np.full(len(sequences), -1, dtype=np.int64)
    for component, members in enumerate(components):
        for index in members:
            component_of[index] = component
    for row in pools:
        for component in set(component_of[row].tolist()):
            if not set(components[component]) <= set(row.tolist()):
                raise ValueError("incomplete component inside test pool")
    for left in range(len(pools)):
        for right in range(left + 1, len(pools)):
            if set(component_of[pools[left]].tolist()) & set(component_of[pools[right]].tolist()):
                raise ValueError("exact/H1 component crosses test-v2 pools")

    (root / "data").mkdir()
    np.save(root / "data" / "sequences.npy", sequences)
    np.save(root / "data" / "v1_split.npy", v1_split)
    np.save(root / "data" / "component.npy", component_of)
    np.save(root / "data" / "train_50000_indices.npy", train_indices)
    np.save(root / "data" / "dev_indices.npy", dev_indices)
    np.save(root / "data" / "test_v2_indices.npy", test_indices)
    np.save(root / "data" / "test_v2_pools.npy", pools)
    train_pools = make_pools(train_indices, sequences, "train", 50000)
    dev_pools = make_pools(dev_indices, sequences, "dev", 10000)
    test_proposals = make_proposals(16, 64, config["selection"]["proposal_seeds"], "test_v2")
    dev_proposals = make_proposals(int(np.sum((dev_pools >= 0).all(axis=1))), 64, config["selection"]["proposal_seeds"], "dev", analysis_id=V1_ANALYSIS_ID)
    np.save(root / "data" / "train_pools.npy", train_pools)
    np.save(root / "data" / "dev_pools.npy", dev_pools)
    np.save(root / "data" / "dev_proposals.npy", dev_proposals)
    np.save(root / "data" / "test_v2_proposals.npy", test_proposals)
    np.save(root / "data" / "train_dev_mean_labels.npy", train_labels)

    inference = json.loads(Path(config["model"]["inference_config"]).read_text())
    members = []
    for member in inference["members"]:
        if digest(member["checkpoint"]) != member["sha256"]:
            raise ValueError("original CNN checkpoint hash mismatch")
        checkpoint = torch.load(member["checkpoint"], map_location="cpu", weights_only=False)
        model_hash = hashlib.sha256(b"FreshSampleCNN" + state_dict_hash(checkpoint["model_state"]).encode()).hexdigest()
        if model_hash != member["model_hash"]:
            raise ValueError("original CNN model hash mismatch")
        members.append(member)

    data_names = [
        "sequences.npy", "v1_split.npy", "component.npy", "train_50000_indices.npy",
        "dev_indices.npy", "test_v2_indices.npy", "test_v2_pools.npy",
        "train_pools.npy", "dev_pools.npy", "dev_proposals.npy",
        "test_v2_proposals.npy", "train_dev_mean_labels.npy",
    ]
    report = {
        "status": "PASS", "analysis_id": ANALYSIS_ID, "config_sha256": config_hash,
        "v1_artifact_root": str(v1), "v1_prepare_manifest_sha256": digest(v1 / "prepare_manifest.json"),
        "v1_training_freeze_sha256": digest(v1 / "TRAINING_FROZEN.json"),
        "raw_sha256": raw_hashes, "raw_hashes_pass": True,
        "raw_target_label_columns_accessed": False,
        "common_target_sequence_keys": len(sequences), "h1_edges": h1_edges,
        "components": len(components), "component_size_max": max(map(len, components)),
        "sample": sample_report, "sample_exact_matches_among_otherwise_eligible": sample_exact,
        "sample_h1_only_matches_among_otherwise_eligible": sample_h1,
        "excluded_v1_components": excluded_v1_components,
        "eligible_components_after_v1_and_sample_exclusion": len(eligible_components),
        "test_components": len(test_components),
        "counts": {"train": len(train_indices), "dev": len(dev_indices), "test_v2": len(test_indices)},
        "test_v2_disjoint_from_v1_train_dev_test": True,
        "test_v2_to_v1_exact_h1": 0, "cross_role_exact_h1": 0,
        "test_sample_exact": 0, "test_sample_h1": 0,
        "cross_test_pool_exact_h1": 0,
        "repeat_linked_test": True, "test_pools": pools.shape[0], "test_pool_size": pools.shape[1],
        "complete_train_pools": int(np.sum((train_pools >= 0).all(axis=1))),
        "complete_dev_pools": int(np.sum((dev_pools >= 0).all(axis=1))),
        "train_dev_groups_reuse_v1": True,
        "train_dev_label": "v1 frozen arithmetic mean of r1/r2",
        "original_cnn_members": members,
        "files": {name: digest(root / "data" / name) for name in data_names},
        "train_dev_mrl_read_from_v1_frozen_array": True,
        "test_v2_mrl_read": False, "test_mrl_read": False,
        "target_evaluation_count": 0,
    }
    save_json(root / "SPLIT_FROZEN.json", {key: value for key, value in report.items() if key not in ("original_cnn_members", "train_dev_mrl_read_from_v1_frozen_array")})
    save_json(root / "prepare_manifest.json", report)
    save_json(root / "verification" / "round1_preflight.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("sample", "original_cnn_members")}, indent=2))


if __name__ == "__main__":
    main()
