import copy
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .checkpoint import atomic_json_write, atomic_torch_save, load_checkpoint, trainer_code_identity
from .convergence import DualPlateau
from .cuda import configure_cuda, cuda_self_check
from .data import TargetScaler, load_roles, preflight, sha256_file
from .model import FreshSampleCNN, model_definition_hash, model_hash, state_dict_hash
from .synthetic import ZeroParameterModel


def _epoch_batches(x, y, batch_size, epoch, seed, shuffle):
    indices = np.random.default_rng(seed + epoch).permutation(len(x)) if shuffle else np.arange(len(x))
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        yield x[selected], y[selected]


def _forward(model, inputs, labels, device, dtype):
    inputs = torch.from_numpy(inputs).to(device=device, dtype=dtype)
    labels = torch.from_numpy(labels).to(device=device, dtype=dtype)
    if not inputs.is_cuda or not labels.is_cuda:
        raise RuntimeError("training inputs and labels must be CUDA tensors")
    return model(inputs), labels


def _evaluate(model, x, y, batch_size, device, dtype):
    total = 0.0
    count = 0
    with torch.no_grad():
        for inputs, labels in _epoch_batches(x, y, batch_size, 0, 0, False):
            outputs, labels = _forward(model, inputs, labels, device, dtype)
            loss = torch.mean((outputs - labels) ** 2)
            total += float(loss.item()) * len(labels)
            count += len(labels)
    return total / count


def _write_history(handle, record):
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def train_one(config, seed, memory_fraction, report, model_factory=None, resume_from=None):
    started = time.perf_counter()
    device, dtype, gpu_metadata = configure_cuda(config.gpu_index, memory_fraction, config.dtype, seed)
    roles = load_roles(report)
    scaler = TargetScaler.fit(roles["train"][1])
    model = (model_factory or FreshSampleCNN)()
    model.to(device=device, dtype=dtype)
    model_definition_sha256 = model_definition_hash(model)
    fresh_initialization_model_hash = model_hash(model)
    initial_model_hash = fresh_initialization_model_hash
    effective_config_sha256 = config.effective_config_sha256()
    code_identity = trainer_code_identity()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    plateau = DualPlateau(config.min_delta, config.patience)
    history = []
    start_epoch = 1
    best_model_state = copy.deepcopy(model.state_dict())
    resume_reference = resume_from or config.resume_for(seed)
    resume_path = None
    resume_checkpoint_sha256 = None
    if resume_reference is not None:
        resume_path, expected_sha256 = resume_reference
        resume_checkpoint_sha256 = sha256_file(resume_path)
        if resume_checkpoint_sha256 != expected_sha256:
            raise ValueError("resume checkpoint SHA256 mismatch")
        checkpoint = load_checkpoint(resume_path, device)
        if checkpoint.get("status") == "completed":
            raise ValueError("completed checkpoint cannot be resumed")
        expected_identities = {
            "seed": seed,
            "data_pack_sha256": report.data_pack_sha256,
            "row_order_sha256": report.row_order_sha256,
            "source_sha256": report.source_sha256,
            "split_manifest_sha256": report.split_manifest_sha256,
            "learner_allowlist_sha256": report.learner_allowlist_sha256,
            "code_identity": code_identity,
            "model_definition_sha256": model_definition_sha256,
            "effective_config_sha256": effective_config_sha256,
            "scaler": scaler.to_dict(),
        }
        for key, expected in expected_identities.items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"resume identity mismatch: {key}")
        initial_model_hash = checkpoint["initial_model_hash"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        plateau.load_state_dict(checkpoint["plateau"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_model_state = checkpoint["best_model_state"]

    run_dir = config.output_root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    history_path = run_dir / "history.jsonl"

    def checkpoint_payload(epoch, model_state, status):
        return {
            "status": status,
            "epoch": epoch,
            "seed": seed,
            "model_state": model_state,
            "model_state_sha256": state_dict_hash(model_state),
            "model_definition_sha256": model_definition_sha256,
            "initial_model_hash": initial_model_hash,
            "optimizer_state": optimizer.state_dict(),
            "scaler": scaler.to_dict(),
            "plateau": plateau.state_dict(),
            "history": history,
            "best_model_state": best_model_state,
            "data_pack_sha256": report.data_pack_sha256,
            "row_order_sha256": dict(report.row_order_sha256),
            "source_sha256": dict(report.source_sha256),
            "split_manifest_sha256": report.split_manifest_sha256,
            "learner_allowlist_sha256": report.learner_allowlist_sha256,
            "code_identity": code_identity,
            "effective_config_sha256": effective_config_sha256,
        }

    if resume_reference is not None:
        atomic_torch_save(checkpoint_payload(start_epoch - 1, best_model_state, "running"), best_path)

    train_x = roles["train"][0]
    train_y = scaler.transform(roles["train"][1]).astype(np.float64)
    dev_x = roles["dev"][0]
    dev_y = scaler.transform(roles["dev"][1]).astype(np.float64)
    stop_reason = None
    epoch = start_epoch - 1
    with history_path.open("w") as history_handle:
        for old_record in history:
            _write_history(history_handle, old_record)
        while True:
            epoch += 1
            epoch_started = time.perf_counter()
            model.train()
            total = 0.0
            count = 0
            for inputs, labels in _epoch_batches(train_x, train_y, config.batch_size, epoch, seed, True):
                outputs, labels = _forward(model, inputs, labels, device, dtype)
                loss = torch.mean((outputs - labels) ** 2)
                if not math.isfinite(float(loss.item())):
                    raise ValueError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(labels)
                count += len(labels)
            train_online_loss = total / count
            model.eval()
            train_eval_loss = _evaluate(model, train_x, train_y, config.batch_size, device, dtype)
            dev_eval_loss = _evaluate(model, dev_x, dev_y, config.batch_size, device, dtype)
            decision = plateau.update(epoch, train_eval_loss, dev_eval_loss)
            if decision["dev_best"]:
                best_model_state = copy.deepcopy(model.state_dict())
            record = {
                "epoch": epoch,
                "train_online_loss": train_online_loss,
                "train_eval_loss": train_eval_loss,
                "dev_eval_loss": dev_eval_loss,
                "best_dev_loss": plateau.best_dev,
                "best_epoch": plateau.best_epoch,
                "plateau_epochs": plateau.plateau_epochs,
                "train_significant_improvement": decision["train_significant_improvement"],
                "dev_significant_improvement": decision["dev_significant_improvement"],
                "dev_absolute_best": decision["dev_best"],
                "pack_sha256": report.data_pack_sha256,
                "row_order_sha256": dict(report.row_order_sha256),
                "config_sha256": effective_config_sha256,
                "model_state_sha256": state_dict_hash(model.state_dict()),
                "shuffle_seed": seed + epoch,
                "shuffle_epoch": epoch,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "total_elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)
            _write_history(history_handle, record)
            atomic_torch_save(checkpoint_payload(epoch, model.state_dict(), "running"), latest_path)
            if decision["dev_best"]:
                atomic_torch_save(checkpoint_payload(epoch, model.state_dict(), "running"), best_path)
            if decision["stop"]:
                stop_reason = "train_eval_and_dev_eval_plateau_min_delta_patience"
                break

    atomic_torch_save(checkpoint_payload(epoch, model.state_dict(), "completed"), latest_path)
    best_checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(best_checkpoint["model_state"])
    summary = {
        "seed": seed,
        "status": "completed",
        "total_epochs": epoch,
        "best_epoch": plateau.best_epoch,
        "stop_reason": stop_reason,
        "total_training_seconds": time.perf_counter() - started,
        "initial_model_hash": initial_model_hash,
        "fresh_initialization_model_hash": fresh_initialization_model_hash,
        "best_model_hash": model_hash(model),
        "model_definition_sha256": model_definition_sha256,
        "effective_config_sha256": effective_config_sha256,
        "code_identity_sha256": code_identity["sha256"],
        "data_pack_sha256": report.data_pack_sha256,
        "row_order_sha256": dict(report.row_order_sha256),
        "source_sha256": dict(report.source_sha256),
        "split_manifest_sha256": report.split_manifest_sha256,
        "learner_allowlist_sha256": report.learner_allowlist_sha256,
        "split_version": report.split_version,
        "split_hash": report.split_hash,
        "role_counts": report.role_counts,
        "scaler": scaler.to_dict(),
        "gpu": gpu_metadata,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "history_sha256": sha256_file(history_path),
        "latest_checkpoint_sha256": sha256_file(latest_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "history_records": len(history),
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "resume_checkpoint_sha256": resume_checkpoint_sha256,
        "synthetic_smoke": report.synthetic,
        "scientific_result": False,
    }
    atomic_json_write(summary, run_dir / "summary.json")
    return summary


def _train_worker(payload):
    config, seed, fraction, report = payload
    return train_one(config, seed, fraction, report)


def _synthetic_train_worker(payload):
    config, seed, fraction, report = payload
    return train_one(config, seed, fraction, report, model_factory=ZeroParameterModel)


def _cuda_worker(payload):
    gpu_index, fraction, dtype, seed = payload
    return cuda_self_check(gpu_index, fraction, dtype, seed)


def _parallel_results(worker, payloads):
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=context) as executor:
        futures = [executor.submit(worker, payload) for payload in payloads]
        return [future.result() for future in futures]


def run_parallel_training(config):
    report = preflight(config.data_manifest)
    if not report.training_authorized or report.synthetic:
        raise PermissionError("formal training requires explicit authorization and a non-synthetic pack")
    payloads = [
        (config, seed, fraction, report)
        for seed, fraction in zip(config.seeds, config.memory_fractions)
    ]
    return _parallel_results(_train_worker, payloads)


def run_synthetic_parallel_training(config):
    report = preflight(config.data_manifest)
    if not report.synthetic or report.training_authorized:
        raise PermissionError("synthetic smoke requires synthetic=true and training_authorized=false")
    payloads = [
        (config, seed, fraction, report)
        for seed, fraction in zip(config.seeds, config.memory_fractions)
    ]
    return _parallel_results(_synthetic_train_worker, payloads)


def run_parallel_cuda_self_checks(config):
    payloads = [
        (config.gpu_index, fraction, config.dtype, seed)
        for seed, fraction in zip(config.seeds, config.memory_fractions)
    ]
    return _parallel_results(_cuda_worker, payloads)
