import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from proxy_function_gap.trainer.data import sha256_file


def _execute(command, log_path, environment):
    with log_path.open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if config["protocol"] != "strict_h1_retrospective_protocol_sensitivity" or config["sample_exclusion"]["hamming2"] != "predeclared_sensitivity_diagnostic_only_not_a_gate":
        raise ValueError("strict H1 protocol contract mismatch")
    root = Path(config["pool_root"])
    root.mkdir(parents=True, exist_ok=False)
    frozen = root / "frozen_config.json"
    shutil.copyfile(config_path, frozen)
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = config["cuda"]["cublas_workspace_config"]

    def build_context(item):
        context, raw_file = item
        first = root / context
        second = root / f"{context}_rebuild"
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        common = ["--raw-file", raw_file, "--context", context, "--sample-roles", *config["sample_roles"], "--train-reference", config["train_reference"]]
        _execute([sys.executable, "-m", "proxy_function_gap.pool_cli", "castillo-build", *common, "--output-dir", str(first)], logs / f"{context}_build.log", environment)
        _execute([sys.executable, "-m", "proxy_function_gap.pool_cli", "castillo-build", *common, "--output-dir", str(second)], logs / f"{context}_rebuild.log", environment)
        command = [sys.executable, str(Path(__file__).with_name("validate_castillo_pools.py")), "--first", str(first), "--second", str(second), "--context", context, "--sample-roles", *config["sample_roles"], "--count", str(config["pool_count"]), "--pool-size", str(config["pool_size"])]
        _execute(command, logs / f"{context}_validation.log", environment)
        validation = json.loads((first / "validation.json").read_text())
        if not validation["pass"]:
            raise ValueError(f"strict H1 validation failed for {context}")
        return {"context": context, "pass": True, "validation_sha256": sha256_file(first / "validation.json"), "manifest_sha256": sha256_file(first / "pool_manifest.json"), "h2_sensitivity_diagnostic": validation["h2_sensitivity_diagnostic"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        contexts = list(executor.map(build_context, config["contexts"].items()))
    report = {
        "pass": True,
        "protocol": config["protocol"],
        "evidence_boundary": config["evidence_boundary"],
        "frozen_config_sha256": sha256_file(frozen),
        "contexts": contexts,
        "old_directories_overwritten": False,
    }
    (root / "build_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
