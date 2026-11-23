import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from proxy_function_gap.castillo import load_castillo_pool
from proxy_function_gap.inference import EnsembleInferenceConfig
from proxy_function_gap.mechanism import score_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--train-reference", required=True)
    parser.add_argument("--inference-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUDA determinism configuration is required")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = EnsembleInferenceConfig.from_json(args.inference_config)

    def execute(index):
        pool_id = f"castillo_{args.context}_{index:03d}"
        pool = load_castillo_pool(args.pool_dir, pool_id, args.train_reference)
        return score_pool(pool, config, output / f"{pool_id}.json")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(execute, range(args.count)))
    report = {"pass": True, "context": args.context, "pools": results, "outer_labels_read": False}
    (output / "batch_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": True, "context": args.context, "pools": len(results)}))


if __name__ == "__main__":
    main()
