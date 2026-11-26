from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from anchor import (
    ANALYSIS_ID, HalfIndex, digest, isolate_h1, make_proposals, save_json,
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


def match_hepg2(path, target_keys, sample_index, expected):
    labels = {}
    sample_exact = 0
    sample_h1 = 0
    raw_rows = 0
    valid_rows = 0
    seen = set()
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_rows += 1
            value = sequence(row)
            if not valid_sequence(value) or not row["rl"]:
                continue
            label = float(row["rl"])
            if not np.isfinite(label):
                continue
            valid_rows += 1
            if value in seen:
                raise ValueError("unexpected duplicate HepG2 valid sequence")
            seen.add(value)
            if value not in target_keys:
                continue
            exact, h1 = sample_index.match(value)
            sample_exact += int(exact)
            sample_h1 += int(h1)
            if not exact and not h1:
                labels[value] = label
    if raw_rows != expected["rows"] or valid_rows != expected["valid_unique_50nt"] or len(seen) != valid_rows:
        raise ValueError("HepG2 raw mapping/count mismatch")
    return labels, {
        "raw_rows": raw_rows, "valid_unique_50nt": valid_rows,
        "exact_shared_with_both_target_repeats_before_sample_exclusion": len(labels) + sample_exact + sample_h1,
        "excluded_sample_exact": sample_exact, "excluded_sample_h1_only": sample_h1,
    }


def verify_isolation(sequences, sample_index):
    index = HalfIndex()
    for value in sequences:
        exact, h1 = index.match(value)
        if exact or h1:
            raise ValueError("within/cross-pool exact/H1 isolation failure")
        sample_exact, sample_h1 = sample_index.match(value)
        if sample_exact or sample_h1:
            raise ValueError("Sample exact/H1 isolation failure")
        index.add(value)
    return {"within_and_cross_pool_exact": 0, "within_and_cross_pool_h1": 0, "sample_exact": 0, "sample_h1": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if config["analysis_id"] != ANALYSIS_ID:
        raise ValueError("wrong frozen analysis")
    out = Path(config["artifact_root"])
    out.mkdir(parents=True, exist_ok=False)
    config_hash = digest(config_path)
    (out / "configs").mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, out / "configs" / "anchor_frozen.json")
    if digest(out / "configs" / "anchor_frozen.json") != config_hash:
        raise ValueError("serialized frozen config differs")
    source_files = ["anchor.py", "prepare.py", "evaluate.py", "tests/test_anchor.py"]
    workspace = Path(__file__).resolve().parent
    save_json(out / "verification" / "prefreeze.json", {
        "analysis_id": ANALYSIS_ID, "config_sha256": config_hash,
        "code_sha256": {name: digest(workspace / name) for name in source_files},
        "target_mrl_read": False, "target_evaluation_count": 0,
    })

    raw_hashes = {name: digest(path) for name, path in config["raw"].items()}
    if raw_hashes != config["raw_sha256"]:
        raise ValueError("raw input hash mismatch")
    target_sets = {}
    target_reports = {}
    for name in ("defined_tcell_r1", "defined_tcell_r2"):
        target_sets[name], target_reports[name] = read_target_keys(config["raw"][name], config["raw_expected"][name])
    common_targets = target_sets["defined_tcell_r1"] & target_sets["defined_tcell_r2"]
    sample_index, sample_report = read_sample_index(config)
    hepg2_labels, hepg2_report = match_hepg2(
        config["raw"]["defined_hepg2_r1"], common_targets, sample_index,
        config["raw_expected"]["defined_hepg2_r1"],
    )
    eligible = set(hepg2_labels)
    minimum = config["eligibility"]["minimum_eligible_after_sample_exclusion"]
    if len(eligible) < minimum:
        raise ValueError(f"insufficient eligible sequences: {len(eligible)} < {minimum}")
    required = config["pools"]["unique_sequences_total"]
    retained = isolate_h1(eligible, required)
    isolation = verify_isolation(retained, sample_index)
    sequences = np.asarray(retained, dtype="S50")
    hepg2 = np.asarray([hepg2_labels[value] for value in retained], dtype=np.float64)
    pools = np.arange(required, dtype=np.int64).reshape(config["pools"]["count"], config["pools"]["size"])
    proposals = make_proposals(len(pools), pools.shape[1], config["selection"]["proposal_seeds"])
    out.mkdir(exist_ok=True)
    np.save(out / "sequences.npy", sequences)
    np.save(out / "hepg2_labels.npy", hepg2)
    np.save(out / "pools.npy", pools)
    np.save(out / "proposals.npy", proposals)

    sys.path.insert(0, str(Path(config["historical_source_root"]) / "src"))
    from proxy_function_gap.residual import cnn_predict
    cnn, members, inference = cnn_predict(sequences, config["inference_config"])
    if not inference["actual_cuda"] or inference["cpu_fallback"] or not np.all(np.isfinite(cnn)):
        raise RuntimeError("CUDA inference gate failed")
    if any(row["gpu"].get("index", 0) != 0 for row in inference["members"]):
        raise RuntimeError("CNN inference was not on cuda:0")
    np.save(out / "cnn_predictions.npy", cnn)
    np.save(out / "cnn_member_predictions.npy", members)
    report = {
        "status": "PASS", "analysis_id": ANALYSIS_ID, "config_sha256": config_hash,
        "raw_sha256": raw_hashes, "raw_hashes_pass": True,
        "targets": target_reports, "common_target_sequence_keys": len(common_targets),
        "sample": sample_report, "hepg2": hepg2_report,
        "eligible_after_sample_exclusion": len(eligible),
        "retained_h1_isolated": len(retained), "isolation": isolation,
        "pool_count": len(pools), "pool_size": pools.shape[1],
        "repeat_linked": True, "same_sequence_pool_membership": True,
        "inference": inference,
        "files": {name: digest(out / name) for name in (
            "sequences.npy", "hepg2_labels.npy", "pools.npy", "proposals.npy",
            "cnn_predictions.npy", "cnn_member_predictions.npy",
        )},
        "target_mrl_read": False, "target_label_columns_accessed": False,
        "target_evaluation_count": 0, "target_evaluation_permitted": True,
    }
    save_json(out / "prepare_manifest.json", report)
    save_json(out / "verification" / "round1_preflight.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "inference"}, indent=2))


if __name__ == "__main__":
    main()
