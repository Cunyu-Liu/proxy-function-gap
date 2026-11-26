from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from backbone import HalfIndex, digest, make_proposals, save_json


def complete_pools(sequences, context, split, pool_size=64):
    keyed = sorted((hashlib.sha256(f"{context}|{split}|{sequence.decode('ascii')}".encode()).digest(), index) for index, sequence in enumerate(sequences))
    count = len(keyed) // pool_size
    if count < 1:
        raise ValueError("insufficient complete auxiliary pools")
    return np.asarray([index for _, index in keyed[:count * pool_size]], dtype=np.int64).reshape(count, pool_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["artifact_root"])
    prepare = json.loads((root / "prepare_sample_manifest.json").read_text())
    if prepare["status"] != "PASS" or prepare["target_mrl_read"]:
        raise RuntimeError("valid label-blind Sample preparation required")
    sample_path = root / "data" / "extended_train_dev_sequences.npy"
    if digest(sample_path) != prepare["files"]["data/extended_train_dev_sequences.npy"]:
        raise ValueError("extended Sample reference changed")
    sample_index = HalfIndex()
    for value in np.load(sample_path, allow_pickle=False):
        sample_index.add(value.decode("ascii"))
    report = {"status": "PASS", "analysis_id": config["analysis_id"], "config_sha256": digest(args.config), "sample_reference_rows": len(sample_index.exact), "contexts": {}, "target_mrl_read": False, "target_label_columns_accessed": False, "target_evaluation_count": 0}
    for context in config["auxiliary"]["contexts"]:
        source = Path(config["auxiliary"]["input_root"]) / context
        old_manifest = json.loads((source / "prepare_manifest.json").read_text())
        if old_manifest["target_mrl_read"] or not old_manifest["inference"]["actual_cuda"]:
            raise ValueError("old auxiliary isolation/CUDA failure")
        sequences = np.load(source / "sequences.npy", allow_pickle=False)
        labels = np.load(source / "labels.npy", allow_pickle=False).astype(np.float64)
        baseline = np.load(source / "cnn_predictions.npy", allow_pickle=False).astype(np.float64)
        split = np.load(source / "split.npy", allow_pickle=False)
        keep = np.ones(len(sequences), dtype=bool)
        excluded_exact = 0
        excluded_h1 = 0
        for index, value in enumerate(sequences):
            exact, h1 = sample_index.match(value.decode("ascii"))
            if exact or h1:
                keep[index] = False
                excluded_exact += int(exact)
                excluded_h1 += int(h1)
        context_report = {"input_rows": len(sequences), "retained_rows": int(keep.sum()), "excluded_extended_sample_exact": excluded_exact, "excluded_extended_sample_h1_only": excluded_h1, "roles": {}}
        for role, value in config["auxiliary"]["split_values"].items():
            selected = keep & (split == value)
            role_sequences = sequences[selected]
            role_labels = labels[selected]
            role_baseline = baseline[selected]
            pools = complete_pools(role_sequences, context, role)
            proposals = make_proposals(len(pools), 64, config["auxiliary"]["proposal_seeds"], namespace=f"{context}|{role}")
            role_root = root / "auxiliary" / context / role
            role_root.mkdir(parents=True, exist_ok=False)
            np.save(role_root / "sequences.npy", role_sequences)
            np.save(role_root / "labels.npy", role_labels)
            np.save(role_root / "baseline.npy", role_baseline)
            np.save(role_root / "pools.npy", pools)
            np.save(role_root / "proposals.npy", proposals)
            context_report["roles"][role] = {"rows": len(role_sequences), "complete_pools": len(pools), "train_sd": float(role_labels.std(ddof=0)) if role == "train" else None, "files": {name: digest(role_root / name) for name in ("sequences.npy", "labels.npy", "baseline.npy", "pools.npy", "proposals.npy")}}
        report["contexts"][context] = context_report
    save_json(root / "prepare_auxiliary_manifest.json", report)
    save_json(root / "verification" / "round1_auxiliary_prepare.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
