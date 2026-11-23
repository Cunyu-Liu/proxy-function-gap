import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-pool-root", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--round", required=True)
    args = parser.parse_args()
    fixed = Path(args.fixed_pool_root) / "build_a"
    source = json.loads((fixed / "pool_manifest.json").read_text())
    vault = json.loads((fixed / "evaluator" / "vault_references.json").read_text())[args.role]
    with gzip.open(vault["path"], "rt", newline="") as handle:
        labels = {}
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["measured_rl"]:
                labels.setdefault(row["sequence_id"], []).append(float(row["measured_rl"]))
    labels = {key: float(np.median(value)) for key, value in labels.items()}
    errors = []
    for index in range(args.count):
        pool_id = f"{args.role}_{index:03d}"
        with gzip.open(fixed / source["files"][args.role], "rt") as handle:
            members = [json.loads(line) for line in handle if line.strip() and json.loads(line)["pool_id"] == pool_id]
        expected_max = max(labels[row["sequence_id"]] for row in members)
        selection_path = Path(args.selection_dir) / f"{pool_id}.json"
        q_path = Path(args.selection_dir) / f"{pool_id}_q.json"
        learner = json.loads(selection_path.read_text())
        q = {(row["pool_id"], row["seed"]): row for row in json.loads(q_path.read_text())["decisions"]}
        evaluation_path = Path(args.evaluation_dir) / f"{pool_id}.json"
        payload = json.loads(evaluation_path.read_text())
        if payload["selection_sha256"] != sha256_file(selection_path) or payload["q_sha256"] != sha256_file(q_path) or payload["labels_returned_to_learner"]:
            errors.append(f"provenance:{pool_id}")
        if len(payload["records"]) != 875:
            errors.append(f"count:{pool_id}")
        for row in payload["records"]:
            original = next(item for item in learner["records"] if item["pool_id"] == row["pool_id"] and item["strategy"] == row["strategy"] and item["alpha"] == row["alpha"] and item["proposal_cap"] == row["proposal_cap"] and item["seed"] == row["seed"])
            expected_selector = None if original["abstain"] else expected_max - labels[original["selected_id"]]
            if (row["selector_regret"] is None) != (expected_selector is None) or (expected_selector is not None and not np.isclose(row["selector_regret"], expected_selector)):
                errors.append(f"selector_regret:{pool_id}")
            expected_abstain = q[(pool_id, row["seed"])]["abstain"] if row["q_applied"] else original["abstain"]
            expected_regret = None if expected_abstain else expected_selector
            if row["abstain"] != expected_abstain or (row["regret"] is None) != (expected_regret is None) or (expected_regret is not None and not np.isclose(row["regret"], expected_regret)):
                errors.append(f"final_regret:{pool_id}")
    report = {"round": args.round, "pool_role": args.role, "pools": args.count, "pass": not errors, "errors": errors[:20], "labels_returned_to_learner": False}
    output = Path(args.evaluation_dir) / f"independent_validation_round{args.round}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if errors:
        raise ValueError("outer evaluation independent validation failed")


if __name__ == "__main__":
    main()
