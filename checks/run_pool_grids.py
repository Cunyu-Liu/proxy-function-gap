import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--role", choices=["dev", "calibration", "outer_designed", "outer_random"], required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--inference-config", required=True)
    parser.add_argument("--v-artifacts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--q-artifacts")
    parser.add_argument("--frozen-config")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if args.role.startswith("outer"):
        source_pool_manifest = str(Path(args.pool_root) / "build_a" / "pool_manifest.json")
        manifests = [Path(source_pool_manifest)] * args.count
        if not args.q_artifacts or not args.frozen_config:
            raise ValueError("outer grids require Q artifacts and frozen dev config")
    else:
        manifests = [Path(args.pool_root) / f"{args.role}_{index:03d}" / "pool_manifest.json" for index in range(args.count)]
    if any(not path.is_file() for path in manifests):
        raise FileNotFoundError("one or more production pool manifests are missing")
    environment = dict(os.environ)
    if environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")

    def execute(item):
        index, manifest = item
        pool_id = f"{args.role}_{index:03d}"
        target = output / f"{pool_id}.json"
        if args.role.startswith("outer"):
            command = [
                args.python, "-m", "proxy_function_gap.pool_cli", "outer-grid",
                "--source-pool-manifest", str(manifest), "--pool-id", pool_id,
                "--inference-config", args.inference_config, "--v-artifacts", args.v_artifacts,
                "--q-artifacts", args.q_artifacts, "--frozen-config", args.frozen_config,
                "--output", str(target),
            ]
        else:
            command = [
                args.python, "-m", "proxy_function_gap.pool_cli", "production-grid",
                "--manifest", str(manifest), "--inference-config", args.inference_config,
                "--v-artifacts", args.v_artifacts, "--output", str(target),
            ]
        with (output / f"{pool_id}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
        payload = json.loads(target.read_text())
        if payload["execution_metadata"]["actual_cuda_inference"] is not True or len(payload["records"]) != 875:
            raise ValueError(f"invalid grid result for {pool_id}")
        return {"pool_id": pool_id, "records": len(payload["records"]), "sha256": payload["selection_config_sha256"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(execute, enumerate(manifests)))
    report = {"pass": True, "pools": results, "records": sum(item["records"] for item in results), "outer_labels_read": False, "scientific_result": False}
    (output / "batch_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": True, "pools": len(results), "records": report["records"]}))


if __name__ == "__main__":
    main()
