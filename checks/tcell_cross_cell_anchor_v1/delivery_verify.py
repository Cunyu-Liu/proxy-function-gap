from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    config_hash = digest(config_path)
    prefreeze = json.loads((root / "verification" / "prefreeze.json").read_text())
    prepare = json.loads((root / "prepare_manifest.json").read_text())
    results = json.loads((root / "target" / "results.json").read_text())
    assert prefreeze["config_sha256"] == config_hash == prepare["config_sha256"] == results["config_sha256"]
    assert digest(root / "configs" / "anchor_frozen.json") == config_hash
    assert all(digest(path) == config["raw_sha256"][name] for name, path in config["raw"].items())
    assert all(digest(path) == config["sample_sha256"][path] for path in config["sample_roles"])
    assert all(digest(root / name) == expected for name, expected in prepare["files"].items())
    assert prepare["target_mrl_read"] is False and prepare["target_evaluation_count"] == 0
    assert prepare["eligible_after_sample_exclusion"] == 48748
    assert prepare["retained_h1_isolated"] == 1024
    assert set(prepare["isolation"].values()) == {0}
    assert prepare["repeat_linked"] and prepare["same_sequence_pool_membership"]
    assert prepare["inference"]["actual_cuda"] and not prepare["inference"]["cpu_fallback"]
    for member in prepare["inference"]["members"]:
        assert member["gpu"]["actual_device"] == "cuda:0"
        assert member["gpu"]["gpu_capability"] == [12, 0]
        assert member["gpu"]["torch_cuda_version"] == "12.8"
        assert member["gpu"]["deterministic_algorithms"]
        assert not member["gpu"]["tf32_enabled"]
    sequences = np.load(root / "sequences.npy", allow_pickle=False)
    pools = np.load(root / "pools.npy", allow_pickle=False)
    assert len(sequences) == 1024 and len(np.unique(sequences)) == 1024
    assert pools.shape == (16, 64) and np.array_equal(np.sort(pools.ravel()), np.arange(1024))
    started = root / "target" / "ONE_EVALUATION_STARTED.json"
    completed = root / "target" / "ONE_EVALUATION_COMPLETED.json"
    assert started.is_file() and completed.is_file()
    completion = json.loads(completed.read_text())
    assert completion["evaluation_count"] == results["evaluation_count"] == 1
    assert completion["results_sha256"] == digest(root / "target" / "results.json")
    assert results["primary"]["success"] is False
    assert results["primary"]["combined_anchor_minus_cnn"]["effect"] > 0
    assert results["sequence_independent"] is False and results["new_blind_test"] is False
    independent = json.loads((root / "verification" / "round2_independent.json").read_text())
    assert independent["status"] == "PASS" and independent["max_absolute_numeric_difference"] == 0
    failed_summary = {
        "target_protocol_changes": 0,
        "records": [
            "prepare attempt 1: serialized config hash mismatch before raw sequence/label processing; partial artifact retained in workspace",
            "prepare attempt 1 network retries: artifact directory already existed; no raw sequence/label processing",
            "prepare attempt 2: CUBLAS_WORKSPACE_CONFIG missing after label-blind pools were built; no target labels read; partial artifact retained",
            "prepare attempt 3: CUDA device executed but older toolkit emitted sm_120 compatibility warning; no target labels read; artifact retained and final inference rerun with CUDA 12.8",
            "synthetic tests initially had two incorrect fixture expectations and then exposed a fusion allocation bug; all fixed before prefreeze/target evaluation",
        ],
    }
    save_json(root / "failed_attempts" / "summary.json", failed_summary)
    report = {
        "status": "PASS", "round": 3, "config_sha256": config_hash,
        "raw_and_sample_hashes_pass": True, "prepared_array_hashes_pass": True,
        "repeat_linked_unique_pool_audit": True, "isolation_pass": True,
        "cuda_0_sm120_compatible_toolkit_pass": True,
        "evaluation_count": 1, "one_shot_sentinels_pass": True,
        "independent_round2_pass": True,
        "negative_result_preserved": True,
        "paired_cross_cell_retrospective_boundary_preserved": True,
        "post_target_protocol_changes": 0,
    }
    save_json(root / "verification" / "round3_delivery.json", report)
    manifest = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json" and "export" not in path.relative_to(root).parts:
            manifest[str(path.relative_to(root))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    save_json(root / "artifact_manifest.json", {"analysis_id": config["analysis_id"], "files": manifest})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
