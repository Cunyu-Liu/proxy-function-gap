import argparse
import json
from pathlib import Path

from proxy_function_gap.outer_evaluator import evaluate_outer_pool
from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-pool-root", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--role", choices=["outer_designed", "outer_random"], required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.fixed_pool_root) / "build_a" / "pool_manifest.json"
    results = []
    for index in range(args.count):
        pool_id = f"{args.role}_{index:03d}"
        selection = Path(args.selection_dir) / f"{pool_id}.json"
        decisions = Path(args.selection_dir) / f"{pool_id}_q.json"
        target = output / f"{pool_id}.json"
        result = evaluate_outer_pool(source, pool_id, selection, decisions, target)
        results.append({"pool_id": pool_id, "records": result["records"], "sha256": sha256_file(target)})
    manifest = {"pool_role": args.role, "pools": results, "outer_metrics_computed": True, "labels_returned_to_learner": False}
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pool_role": args.role, "pools": len(results), "labels_returned_to_learner": False}))


if __name__ == "__main__":
    main()
