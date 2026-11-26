from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from backbone import DualPlateau, FreshSampleCNN, configure_cuda, context_scalers, digest, dynamic_objective, evaluate_pools, save_json, state_dict_hash


def load_pack(path, device):
    with np.load(path, allow_pickle=False) as pack:
        result = {name: pack[name] for name in pack.files}
    return {
        "codes": torch.from_numpy(result["codes"]).to(device),
        "labels_cpu": result["labels"],
        "labels": torch.from_numpy(result["labels"]).to(device=device, dtype=torch.float32),
        "pools": torch.from_numpy(result["pools"]).to(device),
        "pool_contexts": result["pool_contexts"].astype(np.int64),
        "pool_contexts_torch": torch.from_numpy(result["pool_contexts"].astype(np.int64)).to(device),
        "selection_pools": result["selection_pools"],
        "selection_pool_contexts": result["selection_pool_contexts"],
        "context_names": result["context_names"].tolist(),
        "path": str(path), "sha256": digest(path),
    }


def predict(model, pack, device):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(pack["codes"]), 4096):
            codes = pack["codes"][start:start + 4096]
            values = model(torch.nn.functional.one_hot(codes.long(), 4).permute(0, 2, 1).float())
            output.append(values.cpu().numpy())
    result = np.concatenate(output).astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite dev predictions")
    return result


def train_run(config, seed, variant, lambda_, data, output):
    started = time.perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = FreshSampleCNN().to(device=data["train"]["codes"].device, dtype=torch.float32)
    initial_hash = state_dict_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    plateau = DualPlateau(config["model"]["min_delta"], config["model"]["patience"])
    train_context_totals = np.bincount(data["train"]["pool_contexts"], minlength=len(data["train"]["context_names"]))
    history_path = output / "history.jsonl"
    output.mkdir(parents=True, exist_ok=False)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epoch = 0
    stop_reason = None
    with history_path.open("w") as history_handle:
        while True:
            epoch += 1
            epoch_started = time.perf_counter()
            model.train()
            permutation = torch.from_numpy(np.random.default_rng(seed + epoch).permutation(len(data["train"]["pools"]))).to(data["train"]["codes"].device)
            batch_pools = config["model"]["pools_per_batch"]
            for start in range(0, len(permutation), batch_pools):
                selected = permutation[start:start + batch_pools]
                pools = data["train"]["pools"][selected]
                pool_contexts = data["train"]["pool_contexts_torch"][selected]
                loss, _, _ = dynamic_objective(model, data["train"]["codes"], data["train"]["labels"], pools, pool_contexts, data["scalers"], lambda_, train_context_totals)
                if not math.isfinite(float(loss.item())):
                    raise FloatingPointError("nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
            train_metrics = evaluate_pools(model, data["train"]["codes"], data["train"]["labels"], data["train"]["pools"], data["train"]["pool_contexts"], data["scalers"], lambda_, batch_pools)
            dev_metrics = evaluate_pools(model, data["dev"]["codes"], data["dev"]["labels"], data["dev"]["pools"], data["dev"]["pool_contexts"], data["scalers"], lambda_, batch_pools)
            decision = plateau.update(epoch, train_metrics["loss"], dev_metrics["loss"])
            if decision["dev_best"]:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                torch.save({"status": "running", "epoch": epoch, "seed": seed, "variant": variant, "lambda": lambda_, "model_state": best_state, "model_state_sha256": state_dict_hash(best_state), "scalers": data["scalers"], "config_sha256": digest(config["config_path"])}, output / "best.pt")
            record = {"epoch": epoch, "seed": seed, "variant": variant, "lambda": lambda_, "train": train_metrics, "dev": dev_metrics, "best_epoch": best_epoch, "best_dev_loss": plateau.best_dev, "plateau_epochs": plateau.plateau_epochs, "dev_absolute_best": decision["dev_best"], "train_significant_improvement": decision["train_significant_improvement"], "dev_significant_improvement": decision["dev_significant_improvement"], "model_state_sha256": state_dict_hash(model.state_dict()), "epoch_seconds": time.perf_counter() - epoch_started, "elapsed_seconds": time.perf_counter() - started, "scientific_max_epoch": None}
            history_handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            history_handle.flush()
            if decision["stop"]:
                stop_reason = "train_eval_and_dev_eval_plateau_min_delta_patience"
                break
    model.load_state_dict(best_state)
    dev_predictions = predict(model, data["dev"], data["train"]["codes"].device)
    np.save(output / "dev_predictions.npy", dev_predictions)
    best_hash = state_dict_hash(best_state)
    summary = {"status": "completed", "seed": seed, "variant": variant, "lambda": lambda_, "total_epochs": epoch, "best_epoch": best_epoch, "stop_reason": stop_reason, "converged_dev_plateau": stop_reason is not None, "scientific_max_epoch": None, "initial_model_hash": initial_hash, "best_model_hash": best_hash, "best_checkpoint_sha256": digest(output / "best.pt"), "history_sha256": digest(history_path), "dev_predictions_sha256": digest(output / "dev_predictions.npy"), "history_records": epoch, "finite_losses": True, "gpu": data["gpu"], "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()), "total_training_seconds": time.perf_counter() - started, "scalers": data["scalers"], "train_data_sha256": data["train"]["sha256"], "dev_data_sha256": data["dev"]["sha256"], "context_names": data["train"]["context_names"], "target_mrl_read": False}
    save_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("variant", "lambda", "seed", "total_epochs", "best_epoch", "best_model_hash", "total_training_seconds")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--variant", choices=["main", "extended_mse_only", "original_mse_tail_rank", "all"], default="all")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    config["config_path"] = str(config_path)
    if args.seed not in config["model"]["seeds"]:
        raise ValueError("seed not frozen")
    prepare = json.loads((Path(config["artifact_root"]) / "prepare_sample_manifest.json").read_text())
    if prepare["status"] != "PASS" or prepare["target_mrl_read"]:
        raise RuntimeError("Sample preparation gate failed")
    device, gpu = configure_cuda(args.seed, config["model"]["memory_fraction_per_seed"])
    root = Path(config["artifact_root"])
    extended = {"train": load_pack(root / "data" / "extended_train.npz", device), "dev": load_pack(root / "data" / "extended_dev.npz", device)}
    extended["scalers"] = context_scalers(extended["train"]["labels_cpu"], extended["train"]["pool_contexts_torch"].cpu().numpy().repeat(64)[extended["train"]["pools"].cpu().numpy().reshape(-1) >= 0])
    original = {"train": load_pack(root / "data" / "original_train.npz", device), "dev": load_pack(root / "data" / "original_dev.npz", device)}
    original["scalers"] = context_scalers(original["train"]["labels_cpu"], original["train"]["pool_contexts_torch"].cpu().numpy().repeat(64)[original["train"]["pools"].cpu().numpy().reshape(-1) >= 0])
    for data in (extended, original):
        data["scalers"] = {int(key): value for key, value in data["scalers"].items()}
        data["gpu"] = gpu
    jobs = []
    if args.variant in ("main", "all"):
        jobs.extend(("main", value, extended) for value in config["model"]["lambda_grid"])
    if args.variant in ("extended_mse_only", "all"):
        jobs.append(("extended_mse_only", 0.0, extended))
    if args.variant in ("original_mse_tail_rank", "all"):
        jobs.extend(("original_mse_tail_rank", value, original) for value in config["model"]["lambda_grid"])
    summaries = []
    for variant, lambda_, data in jobs:
        name = variant if variant == "extended_mse_only" else f"{variant}_lambda_{lambda_:g}"
        output = root / "training" / name / f"seed_{args.seed}"
        summaries.append(train_run(config, args.seed, variant, lambda_, data, output))
    save_json(root / "training" / f"seed_{args.seed}_completed.json", {"status": "PASS", "seed": args.seed, "runs": summaries, "target_mrl_read": False})


if __name__ == "__main__":
    main()
