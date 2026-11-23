import argparse
import json
from pathlib import Path

from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    compare = json.loads(Path(args.compare).read_text())
    compared = {item["seed"]: item for item in compare["seeds"]}
    results = []
    for seed_dir in sorted(root.glob("seed_*")):
        history = [json.loads(line) for line in (seed_dir / "history.jsonl").read_text().splitlines()]
        summary = json.loads((seed_dir / "summary.json").read_text())
        best = min(history, key=lambda item: item["dev_eval_loss"])
        seed = summary["seed"]
        expected = compared[seed]
        result = {
            "seed": seed,
            "epochs": len(history),
            "best_epoch": best["epoch"],
            "final_train_online_loss": history[-1]["train_online_loss"],
            "final_train_eval_loss": history[-1]["train_eval_loss"],
            "final_dev_eval_loss": history[-1]["dev_eval_loss"],
            "best_train_eval_loss": best["train_eval_loss"],
            "best_dev_eval_loss": best["dev_eval_loss"],
            "final_plateau_epochs": history[-1]["plateau_epochs"],
            "converged_dev_plateau": summary["converged_dev_plateau"],
            "total_training_seconds": summary["total_training_seconds"],
            "max_memory_allocated_bytes": summary["max_memory_allocated_bytes"],
            "history_sha256": sha256_file(seed_dir / "history.jsonl"),
            "latest_checkpoint_sha256": sha256_file(seed_dir / "latest.pt"),
            "best_checkpoint_sha256": sha256_file(seed_dir / "best.pt"),
        }
        assert result["epochs"] == expected["total_epochs"]
        assert result["best_epoch"] == expected["best_epoch"]
        assert result["final_train_online_loss"] == expected["final_train_online_loss"]
        assert result["final_train_eval_loss"] == expected["final_train_eval_loss"]
        assert result["final_dev_eval_loss"] == expected["final_dev_eval_loss"]
        assert result["best_train_eval_loss"] == expected["best_train_eval_loss"]
        assert result["best_dev_eval_loss"] == expected["best_dev_eval_loss"]
        assert result["history_sha256"] == expected["history_sha256"]
        assert result["latest_checkpoint_sha256"] == expected["latest_checkpoint_sha256"]
        assert result["best_checkpoint_sha256"] == expected["best_checkpoint_sha256"]
        results.append(result)
    report = {
        "status": "PASS",
        "all_converged_dev_plateau": all(item["converged_dev_plateau"] for item in results),
        "outer_labels_read": False,
        "outer_metrics_computed": False,
        "seeds": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
