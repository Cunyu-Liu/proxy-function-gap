import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    first = Path(args.first)
    second = Path(args.second)
    errors = []
    sequences = []
    for index in range(args.count):
        pool_id = f"castillo_{args.context}_{index:03d}"
        for directory in (first, second):
            path = directory / "learner" / f"{pool_id}.jsonl.gz"
            with gzip.open(path, "rt") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if len(rows) != 64 or any("measured_rl" in row for row in rows) or any(row["pool_id"] != pool_id for row in rows):
                errors.append(f"learner_contract:{pool_id}")
            if abs(sum(row["q_value"] for row in rows) - 1) > 1e-12:
                errors.append(f"q_sum:{pool_id}")
        if sha256_file(first / "learner" / f"{pool_id}.jsonl.gz") != sha256_file(second / "learner" / f"{pool_id}.jsonl.gz"):
            errors.append(f"double_build:{pool_id}")
        with gzip.open(first / "learner" / f"{pool_id}.jsonl.gz", "rt") as handle:
            sequences.extend(json.loads(line)["utr50"] for line in handle if line.strip())
    if sha256_file(first / "evaluator/labels.tsv.gz") != sha256_file(second / "evaluator/labels.tsv.gz") or sha256_file(first / "pool_manifest.json") != sha256_file(second / "pool_manifest.json"):
        errors.append("double_build_metadata")
    half_index = defaultdict(list)
    for sequence in sequences:
        for position, key in ((0, sequence[:25]), (1, sequence[25:])):
            for other in half_index[(position, key)]:
                if sum(a != b for a, b in zip(sequence, other)) == 1:
                    errors.append("hamming1_leakage")
            half_index[(position, key)].append(sequence)
    report = {"pass": not errors, "context": args.context, "pools": args.count, "sequences": len(sequences), "checks": ["double_build_hashes", "learner_no_labels_and_q", "cross_pool_hamming1"], "errors": errors[:20], "labels_returned_to_learner": False}
    (first / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if errors:
        raise ValueError("Castillo pool validation failed")


if __name__ == "__main__":
    main()
