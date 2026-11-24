import argparse
import json
from pathlib import Path

from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-config", required=True)
    parser.add_argument("--com-config", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--source-pool-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ensemble = json.loads(Path(args.ensemble_config).read_text())
    com = json.loads(Path(args.com_config).read_text())
    members = []
    for seed in com["seeds"]:
        run_dir = Path(com["output_root"]) / f"seed_{seed}"
        summary = json.loads((run_dir / "summary.json").read_text())
        if not summary["converged_dev_plateau"] or summary["status"] != "completed":
            raise ValueError(f"COM seed {seed} has not converged")
        checkpoint = run_dir / "best.pt"
        if sha256_file(checkpoint) != summary["best_checkpoint_sha256"]:
            raise ValueError(f"COM checkpoint hash mismatch for seed {seed}")
        members.append({"seed": seed, "checkpoint": str(checkpoint), "sha256": summary["best_checkpoint_sha256"], "model_hash": summary["best_model_hash"]})
    lambdas = [0.0, 0.5, 1.0, 2.0, 4.0]
    result = {
        "schema_version": "1.0",
        "pool_root": args.pool_root,
        "source_pool_manifest": args.source_pool_manifest,
        "cnn_members": ensemble["members"],
        "com_members": members,
        "caps": [1, 2, 4, 8, 16],
        "paired_seeds": [101, 202, 303, 404, 505],
        "memory_fraction": 0.02,
        "methods": {
            "bon": {"lambda": 0.0, "gamma": 0.0},
            "lcb": [{"lambda": value, "gamma": 0.0} for value in lambdas],
            "distance": [{"lambda": value, "gamma": 0.0} for value in lambdas + [8.0]],
            "com": {"lambda": 0.0, "gamma": 0.0},
            "diversity_lcb": [{"lambda": value, "gamma": gamma} for value in lambdas for gamma in lambdas],
        },
        "source_ensemble_config_sha256": sha256_file(args.ensemble_config),
        "source_com_config_sha256": sha256_file(args.com_config),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": args.output, "com_members": len(members), "dev_configurations": sum(len(value) if isinstance(value, list) else 1 for value in result["methods"].values())}))


if __name__ == "__main__":
    main()
