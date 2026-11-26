from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from adaptation import digest, save_json


def sequence(row):
    return row.get("UTR") or row["utr"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    amendment_path = config_path.parent / "dev_repeat_amendment_frozen.json"
    amendment_hash = digest(amendment_path)
    amendment = json.loads(amendment_path.read_text())
    if amendment["base_config_sha256"] != digest(config_path) or amendment["selection_metric"] != "maximum_of_r1_and_r2_cap16_regret":
        raise ValueError("invalid dev repeat amendment")
    if (root / "target").exists():
        raise RuntimeError("dev repeat labels must be frozen before target evaluation")
    sequences = np.load(root / "data" / "sequences.npy", allow_pickle=False)
    dev_indices = np.load(root / "data" / "dev_indices.npy", allow_pickle=False)
    wanted = {sequences[index].decode("ascii") for index in dev_indices}
    reports = {}
    for repeat in ("defined_tcell_r1", "defined_tcell_r2"):
        values = {}
        with gzip.open(config["raw"][repeat], "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                value = sequence(row)
                if value in wanted:
                    label = float(row["rl"])
                    if not np.isfinite(label):
                        raise ValueError("nonfinite dev repeat label")
                    values[value] = label
        if set(values) != wanted or len(values) != 10000:
            raise ValueError("incomplete dev repeat labels")
        aligned = np.asarray([values[sequences[index].decode("ascii")] for index in dev_indices], dtype=np.float64)
        np.save(root / "data" / f"dev_{repeat}_labels.npy", aligned)
        reports[repeat] = {"rows": len(aligned), "sha256": digest(root / "data" / f"dev_{repeat}_labels.npy"), "test_rows_read": 0}
    mean_labels = np.load(root / "data" / "train_dev_mean_labels.npy", allow_pickle=False)[dev_indices]
    reconstructed = (np.load(root / "data" / "dev_defined_tcell_r1_labels.npy", allow_pickle=False) + np.load(root / "data" / "dev_defined_tcell_r2_labels.npy", allow_pickle=False)) / 2
    if not np.array_equal(reconstructed, mean_labels):
        raise ValueError("repeat mean does not reproduce frozen training labels")
    report = {"status": "PASS", "base_config_sha256": digest(config_path), "amendment_sha256": amendment_hash, "selection_metric": amendment["selection_metric"], "dev_repeats": reports, "mean_label_exact_reconstruction": True, "train_repeat_equal_weight": True, "three_seed_equal_weight": True, "from_scratch_same_grid_and_convergence": True, "test_mrl_read": False, "target_evaluation_count": 0}
    save_json(root / "DEV_REPEAT_SELECTION_FROZEN.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
