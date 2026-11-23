import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--train-reference", required=True)
    parser.add_argument("--inference-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    if environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUDA determinism configuration is required")

    def execute(index):
        pool_id = f"castillo_{args.context}_{index:03d}"
        target = output / f"{pool_id}.json"
        command = [args.python, "-m", "proxy_function_gap.pool_cli", "castillo-grid", "--pool-dir", args.pool_dir, "--pool-id", pool_id, "--train-reference", args.train_reference, "--inference-config", args.inference_config, "--output", str(target)]
        with (output / f"{pool_id}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
        payload = json.loads(target.read_text())
        assert len(payload["records"]) == 50 and all(row["regret"] is None for row in payload["records"])
        return {"pool_id": pool_id, "records": 50}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(execute, range(args.count)))
    report = {"pass": True, "context": args.context, "pools": results, "outer_labels_read": False}
    (output / "batch_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": True, "context": args.context, "pools": len(results), "records": sum(row["records"] for row in results)}))


if __name__ == "__main__":
    main()
