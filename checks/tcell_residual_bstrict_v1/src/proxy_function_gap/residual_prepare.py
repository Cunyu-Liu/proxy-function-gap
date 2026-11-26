from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np

from .residual import HalfIndex, cnn_predict, decode_sequence, digest, packed_halves, save_json, sequence_feature_check


def sequence(row):
    return row.get("UTR") or row["utr"]


def raw_hash_expectations(config):
    result = {}
    for context in config["raw"]:
        manifest_path = Path(config["pool_root"]) / context / "pool_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        result[context] = manifest["raw_file_sha256"]
    return result


def read_target_sequences(config):
    output = {}
    reports = {}
    for context in ("defined_tcell_r1", "defined_tcell_r2"):
        path = Path(config["raw"][context])
        sequences = []
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                value = sequence(row)
                if len(value) == 50 and not set(value) - set("ACGT"):
                    sequences.append(value)
        expected = config["raw_expected"][context]
        if len(sequences) != expected["valid_unique_50nt"] or len(set(sequences)) != len(sequences):
            raise ValueError(f"target unlabeled mapping mismatch: {context}")
        output[context] = sequences
        reports[context] = {"valid_unique_50nt": len(sequences), "label_columns_accessed": False}
    return output, reports


def read_sample_index(config):
    index = HalfIndex()
    hashes = {}
    rows = 0
    for path in config["sample_roles"]:
        hashes[path] = digest(path)
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                index.add(row["utr50"])
                rows += 1
    return index, {"rows": rows, "unique_sequences": len(index.exact), "sha256": hashes}


def process_auxiliary(context, config, target_indices, sample_index, out):
    path = Path(config["raw"][context])
    retained_sequences = []
    retained_labels = []
    raw_rows = valid_rows = duplicate_valid_rows = 0
    excluded = {"target_exact_union": 0, "target_h1_only_union": 0, "sample_exact_or_h1": 0}
    per_target = {name: {"exact": 0, "h1_only": 0} for name in target_indices}
    seen = set()
    started = time.perf_counter()
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_rows += 1
            value = sequence(row)
            if len(value) != 50 or set(value) - set("ACGT") or not row["rl"]:
                continue
            label = float(row["rl"])
            if not np.isfinite(label):
                continue
            valid_rows += 1
            if value in seen:
                duplicate_valid_rows += 1
            seen.add(value)
            target_match = False
            any_target_exact = False
            for name, index in target_indices.items():
                exact, h1 = index.match(value)
                per_target[name]["exact"] += int(exact)
                per_target[name]["h1_only"] += int(h1)
                target_match |= exact or h1
                any_target_exact |= exact
            sample_exact, sample_h1 = sample_index.match(value)
            excluded["target_exact_union"] += int(any_target_exact)
            excluded["target_h1_only_union"] += int(target_match and not any_target_exact)
            excluded["sample_exact_or_h1"] += int(sample_exact or sample_h1)
            if target_match or sample_exact or sample_h1:
                continue
            retained_sequences.append(value)
            retained_labels.append(label)
    expected = config["raw_expected"][context]
    if raw_rows != expected["rows"] or valid_rows != expected["valid_unique_50nt"] or len(seen) != valid_rows or duplicate_valid_rows != 0:
        raise ValueError(f"auxiliary raw mapping/duplicate aggregation mismatch: {context}: rows={raw_rows}, valid={valid_rows}, unique={len(seen)}, duplicates={duplicate_valid_rows}")
    seq = np.asarray(retained_sequences, dtype="S50")
    y = np.asarray(retained_labels, dtype=np.float64)
    del retained_sequences, retained_labels, seen
    p, members, inference = cnn_predict(seq, config["inference_config"])
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "sequences.npy", seq)
    np.save(out / "labels.npy", y)
    np.save(out / "cnn_predictions.npy", p)
    np.save(out / "member_predictions.npy", members)
    report = {
        "context": context, "raw_file": str(path), "raw_sha256": digest(path), "raw_rows": raw_rows,
        "valid_rows": valid_rows, "valid_unique_rows": len(seen) if False else valid_rows,
        "duplicate_valid_rows_aggregated": duplicate_valid_rows, "excluded": excluded,
        "excluded_by_target": per_target, "retained_rows": len(seq), "retained_unique": len(np.unique(seq)),
        "inference": inference, "elapsed_seconds": time.perf_counter() - started,
        "target_mrl_read": False, "sample_mrl_used": False,
        "files": {name: digest(out / name) for name in ("sequences.npy", "labels.npy", "cnn_predictions.npy", "member_predictions.npy")},
    }
    save_json(out / "prepare_manifest.json", report)
    return report


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, value):
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def split_context(context, config, source_out):
    seq = np.load(source_out / "sequences.npy", allow_pickle=False)
    packed = packed_halves(seq)
    uf = UnionFind(len(seq))
    for side in range(2):
        order = np.argsort(packed[:, side], kind="stable")
        sorted_keys = packed[order, side]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        ends = np.r_[starts[1:], len(seq)]
        for start, end in zip(starts, ends):
            if end - start > 1:
                first = int(order[start])
                for position in range(start + 1, end):
                    uf.union(first, int(order[position]))
    roots = np.asarray([uf.find(i) for i in range(len(seq))], dtype=np.int32)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    rep0 = np.full(len(unique_roots), np.iinfo(np.uint64).max, dtype=np.uint64)
    rep1 = np.full(len(unique_roots), np.iinfo(np.uint64).max, dtype=np.uint64)
    for row_index, component_index in enumerate(inverse):
        left, right = packed[row_index]
        if (left < rep0[component_index]) or (left == rep0[component_index] and right < rep1[component_index]):
            rep0[component_index], rep1[component_index] = left, right
    import hashlib
    split_by_component = np.empty(len(unique_roots), dtype=np.uint8)
    for index, (left, right) in enumerate(zip(rep0, rep1)):
        payload = f"{context}|{int(left):016x}{int(right):016x}"
        bucket = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % 10
        split_by_component[index] = 0 if bucket < 8 else (1 if bucket == 8 else 2)
    split = split_by_component[inverse]
    np.save(source_out / "split.npy", split)
    role_names = ("train", "dev", "test")
    roles = {}
    for role_index, role in enumerate(role_names):
        mask = split == role_index
        roots_in_role = np.unique(roots[mask])
        roles[role] = {"rows": int(mask.sum()), "components": len(roots_in_role)}
        if roles[role]["rows"] < config["split"]["minimum_rows_per_role"] or roles[role]["components"] < config["split"]["minimum_components_per_role"]:
            raise ValueError(f"insufficient strict {context} {role}")
    for side in range(2):
        order = np.argsort(packed[:, side], kind="stable")
        keys = packed[order, side]
        ordered_split = split[order]
        starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        if any(len(np.unique(ordered_split[start:end])) != 1 for start, end in zip(starts, np.r_[starts[1:], len(seq)])):
            raise ValueError(f"H1 split leakage: {context} half={side}")
    report = {
        "context": context, "rows": len(seq), "components": len(unique_roots), "roles": roles,
        "method": config["split"], "split_sha256": digest(source_out / "split.npy"),
        "exact_h1_component_cross_role": 0, "target_mrl_read": False,
    }
    save_json(source_out / "split_manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config["artifact_root"])
    out.mkdir(parents=True, exist_ok=False)
    sequence_feature_check()
    expected_hashes = raw_hash_expectations(config)
    for context, path in config["raw"].items():
        actual = digest(path)
        if actual != expected_hashes[context]:
            raise ValueError(f"raw hash mismatch: {context}")
    targets, target_reports = read_target_sequences(config)
    target_indices = {name: HalfIndex() for name in targets}
    for name, values in targets.items():
        for value in values:
            target_indices[name].add(value)
    sample_index, sample_report = read_sample_index(config)
    save_json(out / "raw_preflight.json", {"raw_sha256": expected_hashes, "raw_hashes_pass": True, "targets": target_reports, "sample": sample_report, "target_mrl_read": False})
    reports = {}
    for context in ("n50_hek293t_r1", "defined_hepg2_r1"):
        reports[context] = {"prepare": process_auxiliary(context, config, target_indices, sample_index, out / "auxiliary" / context)}
        reports[context]["split"] = split_context(context, config, out / "auxiliary" / context)
        save_json(out / "auxiliary_prepare_summary.json", reports)


if __name__ == "__main__":
    main()
