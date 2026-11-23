import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--cross-dir", required=True)
    parser.add_argument("--cnn-scores-dir", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--round", required=True)
    args = parser.parse_args()
    errors = []
    with gzip.open(Path(args.pool_dir) / "evaluator/labels.tsv.gz", "rt", newline="") as handle:
        labels = {row["sequence_id"]: float(row["measured_rl"]) for row in csv.DictReader(handle, delimiter="\t")}
    for index in range(args.count):
        pool_id = f"castillo_{args.context}_{index:03d}"
        payload = json.loads((Path(args.cross_dir) / f"{pool_id}.json").read_text())
        kmer = {row["sequence_id"]: row for row in json.loads((Path(args.cross_dir) / f"{pool_id}_kmer.json").read_text())["records"]}
        if payload["labels_returned_to_learner"]:
            errors.append(f"labels:{pool_id}")
        for row in payload.get("records", []):
            pass
        cnn = {row["sequence_id"]: row for row in json.loads((Path(args.cnn_scores_dir) / f"{pool_id}.json").read_text())["records"]}
        fresh_error = np.asarray([abs(row["proxy_mean"] - labels[sequence_id]) for sequence_id, row in cnn.items()])
        kmer_error = np.asarray([abs(row["kmer_mean"] - labels[sequence_id]) for sequence_id, row in kmer.items()])
        gaps = np.asarray([abs(cnn[sequence_id]["proxy_mean"] - row["kmer_mean"]) for sequence_id, row in kmer.items()])
        cut = np.quantile(gaps, 0.25); high_cut = np.quantile(np.maximum(fresh_error, kmer_error), 0.75)
        low = gaps <= cut; high = np.maximum(fresh_error, kmer_error) >= high_cut
        checks = {
            "fresh_mean_abs_error": fresh_error.mean(), "kmer_mean_abs_error": kmer_error.mean(),
            "mean_architecture_gap": gaps.mean(), "architecture_error_correlation": np.corrcoef(fresh_error, kmer_error)[0, 1],
            "low_gap_high_error_count": (low & high).sum(),
            "low_gap_mean_max_error": np.maximum(fresh_error, kmer_error)[low].mean(),
            "other_gap_mean_max_error": np.maximum(fresh_error, kmer_error)[~low].mean(),
        }
        for name, expected in checks.items():
            if not np.isclose(payload[name], expected):
                errors.append(f"{name}:{pool_id}")
    report = {"round": args.round, "context": args.context, "pools": args.count, "pass": not errors, "errors": errors, "labels_returned_to_learner": False}
    (Path(args.cross_dir) / f"independent_validation_round{args.round}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if errors:
        raise ValueError("cross-architecture validation failed")


if __name__ == "__main__":
    main()
