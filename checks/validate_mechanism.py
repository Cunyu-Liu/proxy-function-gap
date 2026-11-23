import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--round", required=True)
    args = parser.parse_args()
    errors = []
    with gzip.open(Path(args.pool_dir) / "evaluator/labels.tsv.gz", "rt", newline="") as handle:
        labels = {row["sequence_id"]: float(row["measured_rl"]) for row in csv.DictReader(handle, delimiter="\t")}
    for index in range(args.count):
        pool_id = f"castillo_{args.context}_{index:03d}"
        scores_payload = json.loads((Path(args.scores_dir) / f"{pool_id}.json").read_text())
        assert scores_payload["metadata"]["actual_cuda_inference"] and not scores_payload["metadata"]["outer_labels_read"]
        scores = {row["sequence_id"]: row for row in scores_payload["records"]}
        selection = json.loads((Path(args.selection_dir) / f"{pool_id}.json").read_text())
        differences = []
        optimized = []
        matched = []
        for row in selection["records"]:
            if row["strategy"] != "bon" or row["proposal_cap"] != 16:
                continue
            selected = scores[row["selected_id"]]
            alternatives = [candidate for candidate in scores.values() if candidate["sequence_id"] != selected["sequence_id"]]
            other = min(alternatives, key=lambda candidate: (abs(candidate["proxy_mean"] - selected["proxy_mean"]), hashlib.sha256(candidate["sequence_id"].encode()).hexdigest()))
            selected_error = abs(selected["proxy_mean"] - labels[selected["sequence_id"]])
            matched_error = abs(other["proxy_mean"] - labels[other["sequence_id"]])
            differences.append(selected_error - matched_error); optimized.append(selected_error); matched.append(matched_error)
        report = json.loads((Path(args.evaluation_dir) / f"{pool_id}.json").read_text())
        if not np.isclose(report["score_matched"]["abs_error_difference"], np.mean(differences)) or not np.isclose(report["score_matched"]["optimized_mean_abs_error"], np.mean(optimized)) or not np.isclose(report["score_matched"]["matched_mean_abs_error"], np.mean(matched)):
            errors.append(f"score_match:{pool_id}")
        if report["labels_returned_to_learner"]:
            errors.append(f"label_boundary:{pool_id}")
    report = {"round": args.round, "context": args.context, "pools": args.count, "pass": not errors, "errors": errors, "labels_returned_to_learner": False}
    (Path(args.evaluation_dir) / f"independent_validation_round{args.round}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if errors:
        raise ValueError("mechanism validation failed")


if __name__ == "__main__":
    main()
