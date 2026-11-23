import argparse
import json
from pathlib import Path

from proxy_function_gap.castillo import evaluate_castillo_pool
from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected", type=int, default=50)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for index in range(args.count):
        pool_id = f"castillo_{args.context}_{index:03d}"
        selection = Path(args.selection_dir) / f"{pool_id}.json"
        target = output / f"{pool_id}.json"
        evaluate_castillo_pool(args.pool_dir, pool_id, selection, target)
        results.append({"pool_id": pool_id, "sha256": sha256_file(target), "records": args.expected})
    report = {"context": args.context, "pools": results, "outer_metrics_computed": True, "labels_returned_to_learner": False}
    (output / "evaluation_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"context": args.context, "pools": len(results), "labels_returned_to_learner": False}))


if __name__ == "__main__":
    main()
