import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--mode", choices=("select", "evaluate"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-dir")
    parser.add_argument("--frozen-config")
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    if environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    label_path = None
    if args.mode == "evaluate" and args.role.startswith("outer_"):
        source = Path(json.loads(Path(args.config).read_text())["source_pool_manifest"])
        vault = json.loads((source.parent / "evaluator" / "vault_references.json").read_text())
        label_path = vault[args.role]["path"]

    def execute(index):
        pool_id = f"{args.role}_{index:03d}"
        target = output / f"{pool_id}.json"
        command = [args.python, "-m", "proxy_function_gap.paper2_cli", args.mode, "--config", args.config, "--role", args.role, "--index", str(index), "--output", str(target)]
        if args.mode == "select":
            command += ["--permutations", str(args.permutations)]
            if args.frozen_config:
                command += ["--frozen-config", args.frozen_config]
        else:
            command += ["--selection", str(Path(args.input_dir) / f"{pool_id}.json")]
            if label_path:
                command += ["--label-path", label_path]
        with (output / f"{pool_id}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
        payload = json.loads(target.read_text())
        if args.mode == "select" and not payload["actual_cuda_inference"]:
            raise ValueError(f"CUDA evidence missing for {pool_id}")
        return {"pool_id": pool_id, "records": len(payload["records"]), "actual_cuda_inference": payload["actual_cuda_inference"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        pools = list(executor.map(execute, range(args.count)))
    report = {"pass": True, "mode": args.mode, "role": args.role, "pools": pools, "records": sum(row["records"] for row in pools), "config": args.config, "frozen_config": args.frozen_config, "permutations": args.permutations}
    (output / "batch_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
