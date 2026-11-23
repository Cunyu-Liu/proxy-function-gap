import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from proxy_function_gap.trainer.checkpoint import load_checkpoint
from proxy_function_gap.trainer.config import TrainerConfig
from proxy_function_gap.trainer.cuda import configure_cuda
from proxy_function_gap.trainer.data import TargetScaler, load_roles, preflight, sha256_file
from proxy_function_gap.trainer.engine import _evaluate
from proxy_function_gap.trainer.model import FreshSampleCNN


def verify(config_path, run_dir, expected_code_commit, gpu_recheck=False):
    config = TrainerConfig.from_json(config_path)
    report = preflight(config.data_manifest)
    run_dir = Path(run_dir)
    results = []
    for seed, fraction in zip(config.seeds, config.memory_fractions):
        seed_dir = run_dir / f"seed_{seed}"
        summary = json.loads((seed_dir / "summary.json").read_text())
        history = [json.loads(line) for line in (seed_dir / "history.jsonl").read_text().splitlines()]
        latest = load_checkpoint(seed_dir / "latest.pt", torch.device("cpu"))
        best = load_checkpoint(seed_dir / "best.pt", torch.device("cpu"))
        assert summary["status"] == "completed"
        assert summary["converged_dev_plateau"] is True
        assert summary["stop_reason"] == "train_eval_and_dev_eval_plateau_min_delta_patience"
        assert summary["code_commit"] == expected_code_commit
        assert summary["frozen_config_sha256"] == config.frozen_config_sha256()
        assert summary["data_pack_sha256"] == report.data_pack_sha256
        assert summary["row_order_sha256"] == report.row_order_sha256
        assert summary["source_sha256"] == report.source_sha256
        assert summary["role_counts"] == report.role_counts
        assert summary["gpu"]["actual_device"] == f"cuda:{config.gpu_index}"
        assert summary["gpu"]["dtype"] == str(getattr(torch, config.dtype))
        assert summary["gpu"]["cpu_fallback"] is False
        assert summary["max_memory_allocated_bytes"] > 0
        assert summary["total_training_seconds"] > 0
        assert len(history) == summary["total_epochs"] == summary["history_records"]
        assert [item["epoch"] for item in history] == list(range(1, len(history) + 1))
        best_index = min(range(len(history)), key=lambda index: history[index]["dev_eval_loss"])
        assert history[best_index]["epoch"] == summary["best_epoch"]
        for item in history:
            for key in (
                "train_online_loss", "train_eval_loss", "dev_eval_loss", "pack_sha256",
                "row_order_sha256", "config_sha256", "frozen_config_sha256", "model_state_sha256",
            ):
                assert key in item
            assert all(math.isfinite(item[key]) for key in ("train_online_loss", "train_eval_loss", "dev_eval_loss"))
            assert item["pack_sha256"] == report.data_pack_sha256
            assert item["row_order_sha256"] == report.row_order_sha256
            assert item["config_sha256"] == summary["effective_config_sha256"]
            assert item["frozen_config_sha256"] == summary["frozen_config_sha256"]
            assert item["shuffle_seed"] == seed + item["epoch"]
            assert item["shuffle_epoch"] == item["epoch"]
        for item in history[-config.patience:]:
            assert not item["dev_absolute_best"]
            assert not item["train_significant_improvement"]
            assert not item["dev_significant_improvement"]
        assert latest["status"] == "completed"
        assert latest["epoch"] == len(history)
        assert latest["history"] == history
        assert latest["model_state_sha256"] == history[-1]["model_state_sha256"]
        assert latest["initial_model_hash"] == summary["initial_model_hash"]
        assert latest["code_identity"]["sha256"] == summary["code_identity_sha256"]
        assert latest["effective_config_sha256"] == summary["effective_config_sha256"]
        assert latest["row_order_sha256"] == summary["row_order_sha256"]
        assert latest["source_sha256"] == summary["source_sha256"]
        assert best["epoch"] == summary["best_epoch"]
        assert best["model_state_sha256"] == history[best_index]["model_state_sha256"]
        expected_best_hash = hashlib.sha256(b"FreshSampleCNN" + best["model_state_sha256"].encode()).hexdigest()
        assert expected_best_hash == summary["best_model_hash"]
        assert sha256_file(seed_dir / "latest.pt") == summary["latest_checkpoint_sha256"]
        assert sha256_file(seed_dir / "best.pt") == summary["best_checkpoint_sha256"]
        assert sha256_file(seed_dir / "history.jsonl") == summary["history_sha256"]
        result = {
            "seed": seed,
            "converged_dev_plateau": True,
            "total_epochs": len(history),
            "best_epoch": summary["best_epoch"],
            "final_train_online_loss": history[-1]["train_online_loss"],
            "final_train_eval_loss": history[-1]["train_eval_loss"],
            "final_dev_eval_loss": history[-1]["dev_eval_loss"],
            "best_train_eval_loss": history[best_index]["train_eval_loss"],
            "best_dev_eval_loss": history[best_index]["dev_eval_loss"],
            "total_training_seconds": summary["total_training_seconds"],
            "max_memory_allocated_bytes": summary["max_memory_allocated_bytes"],
            "gpu": summary["gpu"],
            "history_sha256": summary["history_sha256"],
            "latest_checkpoint_sha256": summary["latest_checkpoint_sha256"],
            "best_checkpoint_sha256": summary["best_checkpoint_sha256"],
            "initial_model_hash": summary["initial_model_hash"],
            "best_model_hash": summary["best_model_hash"],
        }
        if gpu_recheck:
            device, dtype, gpu = configure_cuda(config.gpu_index, fraction, config.dtype, seed)
            roles = load_roles(report)
            scaler = TargetScaler.fit(roles["train"][1])
            model = FreshSampleCNN()
            model.load_state_dict(best["model_state"])
            model.to(device=device, dtype=dtype)
            model.eval()
            train_y = scaler.transform(roles["train"][1])
            dev_y = scaler.transform(roles["dev"][1])
            train_loss = _evaluate(model, roles["train"][0], train_y, config.batch_size, device, dtype)
            dev_loss = _evaluate(model, roles["dev"][0], dev_y, config.batch_size, device, dtype)
            assert math.isclose(train_loss, history[best_index]["train_eval_loss"], rel_tol=1e-6, abs_tol=2e-6)
            assert math.isclose(dev_loss, history[best_index]["dev_eval_loss"], rel_tol=1e-6, abs_tol=2e-6)
            result["gpu_full_row_recheck"] = {
                "status": "PASS",
                "actual_device": str(device),
                "dtype": str(dtype),
                "train_rows": len(train_y),
                "dev_rows": len(dev_y),
                "train_eval_loss": train_loss,
                "dev_eval_loss": dev_loss,
            }
        results.append(result)
    return {
        "status": "PASS",
        "all_converged_dev_plateau": len(results) == len(config.seeds),
        "code_commit": expected_code_commit,
        "frozen_config_sha256": config.frozen_config_sha256(),
        "data_pack_sha256": report.data_pack_sha256,
        "row_order_sha256": report.row_order_sha256,
        "outer_labels_read": False,
        "outer_metrics_computed": False,
        "seeds": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--gpu-recheck", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(args.config, args.run_dir, args.expected_code_commit, args.gpu_recheck)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "all_converged": result["all_converged_dev_plateau"], "seeds": len(result["seeds"]), "gpu_recheck": args.gpu_recheck}))


if __name__ == "__main__":
    main()
