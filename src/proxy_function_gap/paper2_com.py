import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .trainer.checkpoint import atomic_json_write, atomic_torch_save, load_checkpoint
from .trainer.cuda import configure_cuda
from .trainer.data import TargetScaler, load_roles, preflight, sha256_file
from .trainer.engine import _code_commit, _epoch_batches, _evaluate, _parallel_results, _require_clean_repository
from .trainer.model import FreshSampleCNN, model_definition_hash, model_hash, state_dict_hash


@dataclass(frozen=True)
class COMConfig:
    data_manifest: Path
    output_root: Path
    seeds: tuple = (11, 23, 37)
    memory_fractions: tuple = (0.20, 0.20, 0.20)
    gpu_index: int = 0
    dtype: str = "float32"
    batch_size: int = 1024
    learning_rate: float = 0.001
    min_delta: float = 0.0001
    patience: int = 5
    particle_lr: float = 2.0
    particle_steps: int = 50
    soft_interpolation: float = 0.6
    initial_alpha: float = 0.1
    alpha_lr: float = 0.01
    overestimation_limit: float = 2.0

    def __post_init__(self):
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("at least three unique seeds are required")
        if len(self.memory_fractions) != len(self.seeds) or any(not 0 < value <= 1 for value in self.memory_fractions):
            raise ValueError("one valid memory fraction per seed is required")
        if sum(self.memory_fractions) > 1:
            raise ValueError("GPU memory fractions must sum to at most one")
        if self.batch_size <= 0 or self.particle_steps <= 0 or self.patience <= 0:
            raise ValueError("batch size, particle steps and patience must be positive")
        if min(self.learning_rate, self.particle_lr, self.initial_alpha, self.alpha_lr) <= 0:
            raise ValueError("learning rates and initial alpha must be positive")
        if self.min_delta < 0 or self.overestimation_limit < 0 or not 0 < self.soft_interpolation < 1:
            raise ValueError("invalid COM loss configuration")

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text())
        allowed = set(cls.__dataclass_fields__)
        if set(value) - allowed:
            raise ValueError(f"unknown COM config fields: {sorted(set(value) - allowed)}")
        value = dict(value)
        value["data_manifest"] = Path(value["data_manifest"])
        value["output_root"] = Path(value["output_root"])
        value["seeds"] = tuple(value.get("seeds", (11, 23, 37)))
        value["memory_fractions"] = tuple(value.get("memory_fractions", (0.20, 0.20, 0.20)))
        return cls(**value)

    def effective(self):
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "particle_lr": self.particle_lr,
            "particle_steps": self.particle_steps,
            "soft_interpolation": self.soft_interpolation,
            "initial_alpha": self.initial_alpha,
            "alpha_lr": self.alpha_lr,
            "overestimation_limit": self.overestimation_limit,
            "dtype": self.dtype,
            "architecture": "FreshSampleCNN",
            "negative_parameterization": "three_log_ratios_with_zero_reference",
        }

    def sha256(self):
        payload = json.dumps(self.effective(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def relaxed_inputs(one_hot):
    probabilities = torch.softmax(one_hot, dim=1)
    return probabilities


def adversarial_negatives(model, inputs, particle_lr, steps, clip):
    initial_logits = torch.log(clip * inputs + (1.0 - clip) / inputs.shape[1])
    reference = initial_logits[:, :1]
    logits = (initial_logits[:, 1:] - reference).detach()
    scaled_lr = particle_lr * math.sqrt(logits.shape[1] * logits.shape[2])
    for _ in range(steps):
        logits.requires_grad_(True)
        relaxed = torch.softmax(torch.cat((torch.zeros_like(reference), logits), dim=1), dim=1)
        scores = model(relaxed)
        gradient = torch.autograd.grad(scores.sum(), logits)[0]
        logits = (logits + scaled_lr * gradient).detach()
    relaxed = torch.softmax(torch.cat((torch.zeros_like(reference), logits), dim=1), dim=1)
    return relaxed.detach()


def conservative_loss(model, inputs, labels, log_alpha, config):
    positive = model(inputs)
    regression = torch.mean((positive - labels) ** 2)
    negatives = adversarial_negatives(model, inputs, config.particle_lr, config.particle_steps, config.soft_interpolation)
    negative = model(negatives)
    conservatism = negative - positive
    alpha = torch.nn.functional.softplus(log_alpha)
    model_loss = regression + alpha.detach() * conservatism.mean()
    alpha_loss = alpha * (config.overestimation_limit - conservatism.detach().mean())
    return model_loss + alpha_loss, regression.detach(), conservatism.detach().mean(), alpha.detach()


def _history_write(handle, record):
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def train_com_one(config, seed, memory_fraction, report):
    started = time.perf_counter()
    device, dtype, gpu = configure_cuda(config.gpu_index, memory_fraction, config.dtype, seed)
    roles = load_roles(report)
    scaler = TargetScaler.fit(roles["train"][1])
    model = FreshSampleCNN().to(device=device, dtype=dtype)
    log_alpha = torch.nn.Parameter(torch.tensor(math.log(config.initial_alpha), device=device, dtype=dtype))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
    train_x = roles["train"][0]
    train_y = scaler.transform(roles["train"][1]).astype(np.float64)
    dev_x = roles["dev"][0]
    dev_y = scaler.transform(roles["dev"][1]).astype(np.float64)
    run_dir = config.output_root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    history_path = run_dir / "history.jsonl"
    best_path = run_dir / "best.pt"
    latest_path = run_dir / "latest.pt"
    best_dev = float("inf")
    best_train = float("inf")
    best_epoch = 0
    plateau = 0
    best_state = copy.deepcopy(model.state_dict())
    best_log_alpha = log_alpha.detach().clone()
    history = []
    config_hash = config.sha256()
    definition_hash = model_definition_hash(model)
    initial_hash = model_hash(model)
    epoch = 0

    def checkpoint(status):
        return {
            "method": "paper2_com_style_adaptation",
            "status": status,
            "epoch": epoch,
            "seed": seed,
            "model_state": model.state_dict(),
            "best_model_state": best_state,
            "model_state_sha256": state_dict_hash(model.state_dict()),
            "model_definition_sha256": definition_hash,
            "initial_model_hash": initial_hash,
            "log_alpha": log_alpha.detach(),
            "best_log_alpha": best_log_alpha,
            "optimizer_state": optimizer.state_dict(),
            "alpha_optimizer_state": alpha_optimizer.state_dict(),
            "scaler": scaler.to_dict(),
            "history": history,
            "best_epoch": best_epoch,
            "plateau_epochs": plateau,
            "config": config.effective(),
            "config_sha256": config_hash,
            "data_pack_sha256": report.data_pack_sha256,
            "row_order_sha256": dict(report.row_order_sha256),
            "source_sha256": dict(report.source_sha256),
            "split_manifest_sha256": report.split_manifest_sha256,
            "learner_allowlist_sha256": report.learner_allowlist_sha256,
            "official_com_weights": False,
            "official_com_architecture": False,
        }

    stop_reason = None
    with history_path.open("w") as handle:
        while True:
            epoch += 1
            epoch_started = time.perf_counter()
            model.train()
            totals = np.zeros(4, dtype=np.float64)
            count = 0
            for inputs, labels in _epoch_batches(train_x, train_y, config.batch_size, epoch, seed, True):
                inputs = torch.from_numpy(inputs).to(device=device, dtype=dtype)
                labels = torch.from_numpy(labels).to(device=device, dtype=dtype)
                if not inputs.is_cuda or not labels.is_cuda:
                    raise RuntimeError("COM training requires CUDA inputs")
                optimizer.zero_grad(set_to_none=True)
                alpha_optimizer.zero_grad(set_to_none=True)
                loss, regression, conservatism, alpha = conservative_loss(model, inputs, labels, log_alpha, config)
                if not math.isfinite(float(loss.item())):
                    raise ValueError("non-finite COM training loss")
                loss.backward()
                optimizer.step()
                alpha_optimizer.step()
                batch = len(labels)
                totals += batch * np.asarray([float(loss.item()), float(regression.item()), float(conservatism.item()), float(alpha.item())])
                count += batch
            model.eval()
            train_eval = _evaluate(model, train_x, train_y, config.batch_size, device, dtype)
            dev_eval = _evaluate(model, dev_x, dev_y, config.batch_size, device, dtype)
            dev_best = dev_eval < best_dev
            train_significant = train_eval < best_train - config.min_delta
            dev_significant = dev_eval < best_dev - config.min_delta
            if dev_best:
                best_dev = dev_eval
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_log_alpha = log_alpha.detach().clone()
            best_train = min(best_train, train_eval)
            plateau = 0 if dev_best or train_significant or dev_significant else plateau + 1
            record = {
                "epoch": epoch,
                "train_online_com_loss": float(totals[0] / count),
                "train_online_regression_mse": float(totals[1] / count),
                "train_online_conservatism": float(totals[2] / count),
                "train_online_alpha": float(totals[3] / count),
                "train_eval_loss": train_eval,
                "dev_eval_loss": dev_eval,
                "best_dev_loss": best_dev,
                "best_epoch": best_epoch,
                "plateau_epochs": plateau,
                "train_significant_improvement": train_significant,
                "dev_significant_improvement": dev_significant,
                "dev_absolute_best": dev_best,
                "config_sha256": config_hash,
                "pack_sha256": report.data_pack_sha256,
                "row_order_sha256": dict(report.row_order_sha256),
                "model_state_sha256": state_dict_hash(model.state_dict()),
                "shuffle_seed": seed + epoch,
                "shuffle_epoch": epoch,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "total_elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)
            _history_write(handle, record)
            atomic_torch_save(checkpoint("running"), latest_path)
            if dev_best:
                atomic_torch_save(checkpoint("running"), best_path)
            if plateau >= config.patience:
                stop_reason = "train_eval_and_dev_eval_plateau_min_delta_patience"
                break
    atomic_torch_save(checkpoint("completed"), latest_path)
    model.load_state_dict(best_state)
    log_alpha.data.copy_(best_log_alpha)
    summary = {
        "method": "paper2_com_style_adaptation",
        "seed": seed,
        "status": "completed",
        "total_epochs": epoch,
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "converged_dev_plateau": stop_reason is not None,
        "total_training_seconds": time.perf_counter() - started,
        "best_dev_loss": best_dev,
        "best_train_loss": _evaluate(model, train_x, train_y, config.batch_size, device, dtype),
        "best_model_hash": model_hash(model),
        "best_log_alpha": float(best_log_alpha.item()),
        "best_alpha": float(torch.nn.functional.softplus(best_log_alpha).item()),
        "model_definition_sha256": definition_hash,
        "initial_model_hash": initial_hash,
        "config": config.effective(),
        "config_sha256": config_hash,
        "code_commit": _code_commit(),
        "data_pack_sha256": report.data_pack_sha256,
        "row_order_sha256": dict(report.row_order_sha256),
        "source_sha256": dict(report.source_sha256),
        "role_counts": report.role_counts,
        "scaler": scaler.to_dict(),
        "gpu": gpu,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "history_sha256": sha256_file(history_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "history_records": len(history),
        "official_com_weights": False,
        "official_com_architecture": False,
        "scientific_result": False,
    }
    atomic_json_write(summary, run_dir / "summary.json")
    return summary


def _worker(payload):
    return train_com_one(*payload)


def run_com_training(config):
    report = preflight(config.data_manifest)
    if not report.training_authorized or report.synthetic:
        raise PermissionError("COM training requires an authorized non-synthetic pack")
    _require_clean_repository()
    config.output_root.mkdir(parents=True, exist_ok=False)
    atomic_json_write({"config": config.effective(), "config_sha256": config.sha256(), "data_manifest": str(config.data_manifest), "code_commit": _code_commit()}, config.output_root / "frozen_config.json")
    return _parallel_results(_worker, [(config, seed, fraction, report) for seed, fraction in zip(config.seeds, config.memory_fractions)])


def verify_com_run(config):
    report = preflight(config.data_manifest)
    roles = load_roles(report)
    results = []
    for seed in config.seeds:
        run_dir = config.output_root / f"seed_{seed}"
        summary = json.loads((run_dir / "summary.json").read_text())
        history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line.strip()]
        if not summary["converged_dev_plateau"] or len(history) != summary["total_epochs"]:
            raise ValueError(f"invalid convergence history for seed {seed}")
        if history[-1]["plateau_epochs"] != config.patience or summary["best_epoch"] != max(history, key=lambda row: row["dev_eval_loss"])["epoch"]:
            raise ValueError(f"invalid plateau or best epoch for seed {seed}")
        if sha256_file(run_dir / "history.jsonl") != summary["history_sha256"] or sha256_file(run_dir / "best.pt") != summary["best_checkpoint_sha256"]:
            raise ValueError(f"artifact hash mismatch for seed {seed}")
        device, dtype, gpu = configure_cuda(config.gpu_index, 0.08, config.dtype, seed)
        checkpoint = load_checkpoint(run_dir / "best.pt", device)
        model = FreshSampleCNN().to(device=device, dtype=dtype)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        scaler = TargetScaler(**{key: checkpoint["scaler"][key] for key in ("mean", "scale", "raw_std")})
        dev_y = scaler.transform(roles["dev"][1]).astype(np.float64)
        actual_dev = _evaluate(model, roles["dev"][0], dev_y, config.batch_size, device, dtype)
        expected_dev = next(row["dev_eval_loss"] for row in history if row["epoch"] == summary["best_epoch"])
        if not math.isclose(actual_dev, expected_dev, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"CUDA dev loss mismatch for seed {seed}: {actual_dev} != {expected_dev}")
        results.append({"seed": seed, "actual_cuda_dev_loss": actual_dev, "expected_dev_loss": expected_dev, "gpu": gpu, "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device), "pass": True})
        del model
        torch.cuda.empty_cache()
    return {"pass": True, "seeds": results, "round_type": "independent_cuda_reload_recompute", "scientific_result": False}
