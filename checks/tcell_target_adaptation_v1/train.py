from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from adaptation import (
    DualPlateau, FreshSampleCNN, codes_to_one_hot, configure_cuda, digest,
    dynamic_objective, evaluate_pools, make_pools, run_name, save_json,
    state_dict_hash, target_scaler,
)


def predict(model, codes, indices):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(indices), 4096):
            values = model(codes_to_one_hot(codes[indices[start:start + 4096]]))
            output.append(values.cpu().numpy())
    result = np.concatenate(output).astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite dev predictions")
    return result


def train_run(config, seed, variant, budget, learning_rate, lambda_, data, original, output):
    started = time.perf_counter()
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = FreshSampleCNN()
    if variant == "from_scratch":
        initialization = "seeded_random"
    else:
        model.load_state_dict(copy.deepcopy(original["model_state"]))
        initialization = f"frozen_sample_cnn_seed_{seed}"
    model.to(device=data["codes"].device, dtype=torch.float32)
    initial_hash = state_dict_hash(model.state_dict())
    original_model_hash = hashlib.sha256(b"FreshSampleCNN" + initial_hash.encode()).hexdigest()
    if variant != "from_scratch" and original_model_hash != original["model_hash"]:
        raise ValueError("frozen initialization hash mismatch")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    plateau = DualPlateau(config["model"]["min_delta"], config["model"]["patience"])
    output.mkdir(parents=True, exist_ok=False)
    history_path = output / "history.jsonl"
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epoch = 0
    stop_reason = None
    mean, scale = data["scaler"]["mean"], data["scaler"]["scale"]
    with history_path.open("w") as history_handle:
        while True:
            epoch += 1
            epoch_started = time.perf_counter()
            model.train()
            permutation = torch.from_numpy(np.random.default_rng(seed + epoch).permutation(len(data["train_pools"]))).to(data["codes"].device)
            batch_pools = config["model"]["pools_per_batch"]
            for start in range(0, len(permutation), batch_pools):
                pools = data["train_pools"][permutation[start:start + batch_pools]]
                loss, _, _ = dynamic_objective(model, data["codes"], data["labels"], pools, mean, scale, lambda_)
                if not math.isfinite(float(loss.item())):
                    raise FloatingPointError("nonfinite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
            train_metrics = evaluate_pools(model, data["codes"], data["labels"], data["train_pools"], mean, scale, lambda_, batch_pools)
            dev_metrics = evaluate_pools(model, data["codes"], data["labels"], data["dev_pools"], mean, scale, lambda_, batch_pools)
            decision = plateau.update(epoch, train_metrics["loss"], dev_metrics["loss"])
            if decision["dev_best"]:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                torch.save({"status": "running", "epoch": epoch, "seed": seed, "variant": variant, "budget": budget, "learning_rate": learning_rate, "lambda": lambda_, "model_state": best_state, "model_state_sha256": state_dict_hash(best_state), "scaler": data["scaler"], "config_sha256": config["config_sha256"]}, output / "best.pt")
            record = {"epoch": epoch, "seed": seed, "variant": variant, "budget": budget, "learning_rate": learning_rate, "lambda": lambda_, "train": train_metrics, "dev": dev_metrics, "best_epoch": best_epoch, "best_dev_loss": plateau.best_dev, "plateau_epochs": plateau.plateau_epochs, "dev_absolute_best": decision["dev_best"], "train_significant_improvement": decision["train_significant_improvement"], "dev_significant_improvement": decision["dev_significant_improvement"], "model_state_sha256": state_dict_hash(model.state_dict()), "epoch_seconds": time.perf_counter() - epoch_started, "elapsed_seconds": time.perf_counter() - started, "scientific_max_epoch": None}
            history_handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            history_handle.flush()
            if decision["stop"]:
                stop_reason = "train_and_dev_plateau_with_absolute_dev_reset"
                break
    model.load_state_dict(best_state)
    dev_predictions = predict(model, data["codes"], data["dev_indices"])
    np.save(output / "dev_predictions.npy", dev_predictions)
    best_hash = state_dict_hash(best_state)
    summary = {"status": "completed", "seed": seed, "variant": variant, "budget": budget, "learning_rate": learning_rate, "lambda": lambda_, "total_epochs": epoch, "best_epoch": best_epoch, "stop_reason": stop_reason, "converged_dev_plateau": stop_reason is not None, "scientific_max_epoch": None, "initialization": initialization, "initial_model_hash": initial_hash, "best_model_hash": best_hash, "best_checkpoint_sha256": digest(output / "best.pt"), "history_sha256": digest(history_path), "dev_predictions_sha256": digest(output / "dev_predictions.npy"), "history_records": epoch, "finite_losses": True, "gpu": data["gpu"], "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()), "total_training_seconds": time.perf_counter() - started, "scaler": data["scaler"], "config_sha256": config["config_sha256"], "split_sha256": data["split_sha256"], "train_indices_sha256": data["train_indices_sha256"], "dev_indices_sha256": data["dev_indices_sha256"], "test_mrl_read": False, "test_inference": False}
    save_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("variant", "budget", "learning_rate", "lambda", "seed", "total_epochs", "best_epoch", "best_model_hash", "total_training_seconds")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stage", choices=["grid", "nested"], required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    config["config_sha256"] = digest(config_path)
    root = Path(config["artifact_root"])
    if args.seed not in config["model"]["seeds"]:
        raise ValueError("seed not frozen")
    prepare = json.loads((root / "prepare_manifest.json").read_text())
    split = json.loads((root / "SPLIT_FROZEN.json").read_text())
    if prepare["status"] != "PASS" or prepare["test_mrl_read"] or prepare["target_evaluation_count"] != 0 or split["config_sha256"] != config["config_sha256"]:
        raise RuntimeError("preparation/freeze gate failed")
    device, gpu = configure_cuda(args.seed, config["model"]["memory_fraction_per_seed"])
    sequences = np.load(root / "data" / "sequences.npy", allow_pickle=False)
    from adaptation import encode_codes
    codes_np = encode_codes(sequences)
    labels_np = np.load(root / "data" / "train_dev_mean_labels.npy", allow_pickle=False)
    dev_indices_np = np.load(root / "data" / "dev_indices.npy", allow_pickle=False)
    test_indices = set(np.load(root / "data" / "test_indices.npy", allow_pickle=False).tolist())
    train_dev_indices = np.flatnonzero(np.isfinite(labels_np))
    if test_indices & set(train_dev_indices.tolist()):
        raise ValueError("test index has training label")
    codes = torch.from_numpy(codes_np).to(device)
    labels = torch.from_numpy(np.nan_to_num(labels_np, nan=0.0)).to(device=device, dtype=torch.float32)
    dev_indices = torch.from_numpy(dev_indices_np).to(device)
    dev_pools = torch.from_numpy(make_pools(dev_indices_np, sequences, "dev", 10000)).to(device)
    inference = json.loads(Path(config["model"]["inference_config"]).read_text())
    member = next(row for row in inference["members"] if row["seed"] == args.seed)
    original_checkpoint = torch.load(member["checkpoint"], map_location="cpu", weights_only=False)
    original_state_hash = state_dict_hash(original_checkpoint["model_state"])
    if digest(member["checkpoint"]) != member["sha256"] or hashlib.sha256(b"FreshSampleCNN" + original_state_hash.encode()).hexdigest() != member["model_hash"]:
        raise ValueError("original member identity mismatch")
    original = {"model_state": original_checkpoint["model_state"], "model_hash": member["model_hash"]}

    jobs = []
    if args.stage == "grid":
        for variant in ("main", "from_scratch"):
            for learning_rate in config["model"]["learning_rate_grid"]:
                for lambda_ in config["model"]["lambda_grid"]:
                    jobs.append((variant, 50000, learning_rate, lambda_))
        for learning_rate in config["model"]["learning_rate_grid"]:
            jobs.append(("mse_only", 50000, learning_rate, 0.0))
    else:
        selection = json.loads((root / "selection" / "main.json").read_text())
        selected = selection["selected"]
        for budget in (1000, 5000, 20000):
            jobs.append(("nested", budget, selected["learning_rate"], selected["lambda"]))
    summaries = []
    for variant, budget, learning_rate, lambda_ in jobs:
        train_indices_np = np.load(root / "data" / f"train_{budget}_indices.npy", allow_pickle=False)
        scaler = target_scaler(labels_np[train_indices_np])
        train_pools = torch.from_numpy(make_pools(train_indices_np, sequences, "train", budget)).to(device)
        data = {"codes": codes, "labels": labels, "dev_indices": dev_indices, "dev_pools": dev_pools, "train_pools": train_pools, "scaler": scaler, "gpu": gpu, "split_sha256": digest(root / "SPLIT_FROZEN.json"), "train_indices_sha256": digest(root / "data" / f"train_{budget}_indices.npy"), "dev_indices_sha256": digest(root / "data" / "dev_indices.npy")}
        name = run_name(variant, budget, learning_rate, lambda_)
        summaries.append(train_run(config, args.seed, variant, budget, learning_rate, lambda_, data, original, root / "training" / name / f"seed_{args.seed}"))
    save_json(root / "training" / f"seed_{args.seed}_{args.stage}_completed.json", {"status": "PASS", "seed": args.seed, "stage": args.stage, "runs": summaries, "test_mrl_read": False, "test_inference": False})


if __name__ == "__main__":
    main()
