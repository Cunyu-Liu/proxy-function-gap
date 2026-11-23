import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--round", required=True)
    parser.add_argument("--expected", type=int, default=50)
    args = parser.parse_args()
    errors = []
    for index in range(args.count):
        pool_id = f"castillo_{args.context}_{index:03d}"
        with gzip.open(Path(args.pool_dir) / "evaluator/labels.tsv.gz", "rt", newline="") as handle:
            labels = {row["sequence_id"]: float(row["measured_rl"]) for row in csv.DictReader(handle, delimiter="\t")}
        with gzip.open(Path(args.pool_dir) / "learner" / f"{pool_id}.jsonl.gz", "rt") as handle:
            members = [json.loads(line) for line in handle if line.strip()]
        expected_max = max(labels[row["sequence_id"]] for row in members)
        selection = Path(args.selection_dir) / f"{pool_id}.json"
        payload = json.loads(selection.read_text())
        assert all(row["regret"] is None for row in payload["records"])
        evaluation = json.loads((Path(args.evaluation_dir) / f"{pool_id}.json").read_text())
        if evaluation["selection_sha256"] != sha256_file(selection) or evaluation["labels_returned_to_learner"] or len(evaluation["records"]) != args.expected:
            errors.append(f"provenance:{pool_id}")
        for actual, original in zip(evaluation["records"], payload["records"]):
            expected = expected_max - labels[original["selected_id"]]
            if not np.isclose(actual["regret"], expected):
                errors.append(f"regret:{pool_id}")
    report = {"round": args.round, "context": args.context, "pools": args.count, "pass": not errors, "errors": errors[:20], "labels_returned_to_learner": False}
    (Path(args.evaluation_dir) / f"independent_validation_round{args.round}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if errors:
        raise ValueError("Castillo evaluation validation failed")


if __name__ == "__main__":
    main()
