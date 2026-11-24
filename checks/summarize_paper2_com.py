import argparse
import json
from pathlib import Path

from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output_root"])
    seeds = []
    for seed in config["seeds"]:
        run_dir = root / f"seed_{seed}"
        summary = json.loads((run_dir / "summary.json").read_text())
        history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line.strip()]
        if len(history) != summary["total_epochs"] or sha256_file(run_dir / "history.jsonl") != summary["history_sha256"]:
            raise ValueError(f"history mismatch for seed {seed}")
        best = history[summary["best_epoch"] - 1]
        seeds.append({
            "seed": seed,
            "epochs": summary["total_epochs"],
            "best_epoch": summary["best_epoch"],
            "best_train_eval_loss": best["train_eval_loss"],
            "best_dev_eval_loss": best["dev_eval_loss"],
            "final_online_com_loss": history[-1]["train_online_com_loss"],
            "final_online_regression_mse": history[-1]["train_online_regression_mse"],
            "final_online_conservatism": history[-1]["train_online_conservatism"],
            "best_alpha": summary["best_alpha"],
            "final_plateau_epochs": history[-1]["plateau_epochs"],
            "converged_dev_plateau": summary["converged_dev_plateau"],
            "training_seconds": summary["total_training_seconds"],
            "gpu": summary["gpu"],
            "max_memory_allocated_bytes": summary["max_memory_allocated_bytes"],
            "best_model_hash": summary["best_model_hash"],
            "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
            "history_sha256": summary["history_sha256"],
            "code_commit": summary["code_commit"],
        })
    result = {
        "method": "COM-style conservative objective adaptation; not official COM UTR-ResNet",
        "config": config,
        "config_sha256": sha256_file(args.config),
        "seeds": seeds,
        "all_converged": all(row["converged_dev_plateau"] for row in seeds),
        "actual_parallel_cuda_workers": len(seeds),
        "scientific_result": False,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
