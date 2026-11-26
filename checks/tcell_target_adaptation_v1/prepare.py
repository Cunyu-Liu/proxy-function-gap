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
    ANALYSIS_ID, HalfIndex, allocate_exact, component_identity, digest,
    h1_components, make_pools, make_proposals, save_json, state_dict_hash,
    valid_sequence,
)


def sequence(row):
    return row.get("UTR") or row["utr"]


def read_target_keys(path, expected):
    values = []
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            value = sequence(row)
            if valid_sequence(value):
                values.append(value)
    if len(values) != expected["valid_unique_50nt"] or len(set(values)) != len(values):
        raise ValueError("target sequence-key mapping mismatch")
    return set(values), {"valid_unique_50nt": len(values), "label_columns_accessed": False}


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
                if not valid_sequence(value):
                    raise ValueError("invalid Sample utr50")
                index.add(value)
                rows += 1
    return index, {"rows": rows, "unique_sequences": len(index.exact), "sha256": hashes}


def read_train_dev_labels(config, sequences, split):
    labeled = (split == 0) | (split == 1)
    wanted = {sequences[index].decode("ascii") for index in np.flatnonzero(labeled)}
    if len(wanted) != config["split"]["train_unique_sequences"] + config["split"]["dev_unique_sequences"]:
        raise ValueError("train/dev wanted count mismatch")
    labels = {}
    reports = {}
    for repeat in ("defined_tcell_r1", "defined_tcell_r2"):
        current = {}
        with gzip.open(config["raw"][repeat], "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                value = sequence(row)
                if value in wanted:
                    label = float(row["rl"])
                    if not np.isfinite(label):
                        raise ValueError("nonfinite train/dev MRL")
                    current[value] = label
        if set(current) != wanted:
            raise ValueError("incomplete train/dev target labels")
        labels[repeat] = current
        reports[repeat] = {"train_dev_rows": len(current), "test_rows_read": 0}
    result = np.full(len(sequences), np.nan, dtype=np.float64)
    for index in np.flatnonzero(labeled):
        value = sequences[index].decode("ascii")
        result[index] = (labels["defined_tcell_r1"][value] + labels["defined_tcell_r2"][value]) / 2
    if not np.isfinite(result[labeled]).all() or not np.isnan(result[~labeled]).all():
        raise ValueError("mean-label role mismatch")
    return result, reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if config["analysis_id"] != ANALYSIS_ID:
        raise ValueError("wrong analysis")
    root = Path(config["artifact_root"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    shutil.copyfile(config_path, root / "configs" / "adaptation_frozen.json")
    config_hash = digest(config_path)
    workspace = Path(__file__).resolve().parent
    source_files = ["adaptation.py", "prepare.py", "train.py", "select_models.py", "evaluate.py", "independent_verify.py", "delivery_verify.py", "export_delivery.py", "tests/test_adaptation.py"]
    save_json(root / "verification" / "prefreeze.json", {"analysis_id": ANALYSIS_ID, "config_sha256": config_hash, "code_sha256": {name: digest(workspace / name) for name in source_files}, "target_mrl_read": False, "target_evaluation_count": 0})

    raw_hashes = {name: digest(path) for name, path in config["raw"].items()}
    if raw_hashes != config["raw_sha256"]:
        raise ValueError("raw target hash mismatch")
    target_sets, target_reports = {}, {}
    for repeat in ("defined_tcell_r1", "defined_tcell_r2"):
        target_sets[repeat], target_reports[repeat] = read_target_keys(config["raw"][repeat], config["raw_expected"][repeat])
    common = target_sets["defined_tcell_r1"] & target_sets["defined_tcell_r2"]
    if len(common) != config["raw_expected"]["common_valid_unique_50nt"]:
        raise ValueError("common target count mismatch")
    common = sorted(common, key=lambda value: hashlib.sha256(f"{ANALYSIS_ID}|sequence-order|{value}".encode()).digest())
    sequences = np.asarray(common, dtype="S50")

    components, h1_edges = h1_components(sequences)
    identities, order_keys, minimums = [], [], []
    for members in components:
        identity, order_key, minimum = component_identity(sequences, members)
        identities.append(identity)
        order_keys.append((order_key, minimum))
    component_order = sorted(range(len(components)), key=lambda index: order_keys[index])
    rank = {component: position for position, component in enumerate(component_order)}
    components = [components[index] for index in component_order]
    identities = [identities[index] for index in component_order]
    order_keys = [order_keys[index] for index in component_order]

    sample_index, sample_report = read_sample_index(config)
    sample_clean = [True] * len(components)
    sample_exact = sample_h1 = 0
    for component, members in enumerate(components):
        for index in members:
            exact, h1 = sample_index.match(common[index])
            sample_exact += int(exact)
            sample_h1 += int(h1)
            if exact or h1:
                sample_clean[component] = False
                break

    test_components = allocate_exact(components, order_keys, config["split"]["test_unique_sequences"], sample_clean)
    remaining = [index for index in range(len(components)) if index not in set(test_components)]
    dev_components = allocate_exact([components[index] for index in remaining], [order_keys[index] for index in remaining], config["split"]["dev_unique_sequences"])
    dev_components = [remaining[index] for index in dev_components]
    dev_set = set(dev_components)
    remaining = [index for index in remaining if index not in dev_set]
    train_components = allocate_exact([components[index] for index in remaining], [order_keys[index] for index in remaining], config["split"]["train_unique_sequences"])
    train_components = [remaining[index] for index in train_components]

    split = np.full(len(sequences), -1, dtype=np.int8)
    component_assignment = np.full(len(sequences), -1, dtype=np.int64)
    for role_value, chosen in ((2, test_components), (1, dev_components), (0, train_components)):
        for component in chosen:
            for index in components[component]:
                split[index] = role_value
                component_assignment[index] = component
    if not set(split.tolist()) <= {-1, 0, 1, 2}:
        raise ValueError("invalid selected role")
    counts = {role: int(np.sum(split == value)) for role, value in (("train", 0), ("dev", 1), ("test", 2))}
    expected_counts = {"train": 50000, "dev": 10000, "test": 1024}
    if counts != expected_counts:
        raise ValueError("split count mismatch")
    for component in range(len(components)):
        values = {int(split[index]) for index in components[component] if split[index] >= 0}
        if len(values) > 1:
            raise ValueError("component crosses roles")
    if not all(sample_clean[component] for component in test_components):
        raise ValueError("Sample-contaminated test component")

    train_components_set = set(train_components)
    nested_components = sorted(train_components, key=lambda component: hashlib.sha256(f"{ANALYSIS_ID}|nested|{identities[component]}".encode()).hexdigest())
    nested = {}
    for budget in config["split"]["nested_train_budgets"]:
        local = allocate_exact([components[index] for index in nested_components], [hashlib.sha256(f"{ANALYSIS_ID}|nested|{identities[index]}".encode()).hexdigest() for index in nested_components], budget)
        chosen = [nested_components[index] for index in local]
        indices = sorted(index for component in chosen for index in components[component])
        nested[str(budget)] = indices
    for smaller, larger in ((1000, 5000), (5000, 20000), (20000, 50000)):
        if not set(nested[str(smaller)]) <= set(nested[str(larger)]):
            raise ValueError("nested budget subset failure")

    (root / "data").mkdir()
    np.save(root / "data" / "sequences.npy", sequences)
    np.save(root / "data" / "split.npy", split)
    np.save(root / "data" / "component.npy", component_assignment)
    for role, value in (("train", 0), ("dev", 1), ("test", 2)):
        np.save(root / "data" / f"{role}_indices.npy", np.flatnonzero(split == value).astype(np.int64))
    for budget, indices in nested.items():
        np.save(root / "data" / f"train_{budget}_indices.npy", np.asarray(indices, dtype=np.int64))
    pools = make_pools(np.flatnonzero(split == 2), sequences, "test", 1024)
    proposals = make_proposals(16, 64, config["selection"]["proposal_seeds"], "test")
    np.save(root / "data" / "test_pools.npy", pools)
    np.save(root / "data" / "test_proposals.npy", proposals)
    split_manifest = {
        "status": "PASS", "analysis_id": ANALYSIS_ID, "config_sha256": config_hash,
        "raw_sha256": raw_hashes, "raw_hashes_pass": True, "targets": target_reports,
        "common_target_sequence_keys": len(common), "h1_edges": h1_edges,
        "components": len(components), "component_size_max": max(map(len, components)),
        "sample": sample_report, "sample_exact_matches": sample_exact, "sample_h1_only_matches": sample_h1,
        "sample_clean_components": int(sum(sample_clean)), "counts": counts,
        "test_components": len(test_components), "dev_components": len(dev_components), "train_components": len(train_components),
        "nested_counts": {key: len(value) for key, value in nested.items()},
        "test_sample_exact": 0, "test_sample_h1": 0, "cross_role_exact_h1": 0,
        "repeat_linked_test": True, "test_pools": pools.shape[0], "test_pool_size": pools.shape[1],
        "files": {name: digest(root / "data" / name) for name in ("sequences.npy", "split.npy", "component.npy", "train_indices.npy", "dev_indices.npy", "test_indices.npy", "test_pools.npy", "test_proposals.npy")},
        "target_mrl_read": False, "test_mrl_read": False, "target_evaluation_count": 0,
    }
    save_json(root / "SPLIT_FROZEN.json", split_manifest)
    labels, label_reports = read_train_dev_labels(config, sequences, split)
    np.save(root / "data" / "train_dev_mean_labels.npy", labels)
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
    split_manifest["files"]["train_dev_mean_labels.npy"] = digest(root / "data" / "train_dev_mean_labels.npy")
    report = dict(split_manifest)
    report.update({"label_reports": label_reports, "train_dev_label": "arithmetic mean of r1/r2", "original_cnn_members": members, "target_mrl_read": True, "train_dev_mrl_read": True, "test_mrl_read": False, "target_evaluation_count": 0, "target_evaluation_permitted_after_training_freeze": True})
    save_json(root / "prepare_manifest.json", report)
    save_json(root / "verification" / "round1_preflight.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("sample", "original_cnn_members")}, indent=2))


if __name__ == "__main__":
    main()
