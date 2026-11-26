#!/usr/bin/env python3
"""E1 RNA-FM LoRA/full 微调：validation-only 训练与冻结 checkpoint 的 test 评价。"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_g0_models import _fast_tokenizer_setup  # noqa: E402
from e1_frozen_baselines import (TaskLoader, metrics, prediction_rows, split_audit_rows,  # noqa: E402
                                 stable_seed)
from e1_locked_logger import E1RunLogger as RunLogger  # noqa: E402
from log_run import sha256_file  # noqa: E402

DATA_VERSION = "e0-v2-id90-strict3train-main"
TASK_KEYS = ("experiment", "rbp", "tier", "fold", "track")


def atomic_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_marker(path):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    os.replace(tmp, path)


def file_sha256(path):
    return sha256_file(path)


def save_checkpoint_compressed(directory, checkpoint):
    import torch
    raw_path = os.path.join(directory, "best.pt.tmp")
    compressed_path = os.path.join(directory, "best.pt.zlib")
    compressed_tmp = compressed_path + ".tmp"
    torch.save(checkpoint, raw_path)
    raw_sha256 = hashlib.sha256()
    compressed_sha256 = hashlib.sha256()
    compressor = zlib.compressobj(1)
    raw_bytes = 0
    compressed_bytes = 0
    with open(raw_path, "rb") as source, open(compressed_tmp, "wb") as target:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            raw_sha256.update(chunk)
            raw_bytes += len(chunk)
            packed = compressor.compress(chunk)
            target.write(packed)
            compressed_sha256.update(packed)
            compressed_bytes += len(packed)
        packed = compressor.flush()
        target.write(packed)
        compressed_sha256.update(packed)
        compressed_bytes += len(packed)
        target.flush()
        os.fsync(target.fileno())
    os.replace(compressed_tmp, compressed_path)
    roundtrip_sha256 = hashlib.sha256()
    roundtrip_bytes = 0
    decompressor = zlib.decompressobj()
    with open(compressed_path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            unpacked = decompressor.decompress(chunk)
            roundtrip_sha256.update(unpacked)
            roundtrip_bytes += len(unpacked)
    unpacked = decompressor.flush()
    roundtrip_sha256.update(unpacked)
    roundtrip_bytes += len(unpacked)
    if (roundtrip_bytes != raw_bytes or roundtrip_sha256.digest() != raw_sha256.digest()):
        raise RuntimeError("Checkpoint compression roundtrip failed")
    manifest = {"format": "torch.save FP32 weights + zlib-1 stream",
                "checkpoint": "best.pt.zlib", "raw_sha256": raw_sha256.hexdigest(),
                "compressed_sha256": compressed_sha256.hexdigest(),
                "raw_bytes": raw_bytes, "compressed_bytes": compressed_bytes,
                "compression_ratio": compressed_bytes / raw_bytes,
                "contains_optimizer_state": False, "contains_old_epoch_weights": False,
                "roundtrip_sha256_verified": True}
    atomic_json(os.path.join(directory, "checkpoint_manifest.json"), manifest)
    os.remove(raw_path)
    return manifest


def load_checkpoint_compressed(directory):
    import torch
    manifest_path = os.path.join(directory, "checkpoint_manifest.json")
    if not os.path.isfile(manifest_path):
        return torch.load(os.path.join(directory, "best.pt"), map_location="cpu", weights_only=True)
    manifest = json.load(open(manifest_path))
    compressed_path = os.path.join(directory, manifest["checkpoint"])
    if file_sha256(compressed_path) != manifest["compressed_sha256"]:
        raise RuntimeError("Compressed checkpoint SHA256 mismatch")
    decompressor = zlib.decompressobj()
    raw_sha256 = hashlib.sha256()
    fd, raw_path = tempfile.mkstemp(prefix="best.restore.", suffix=".pt", dir=directory)
    try:
        with os.fdopen(fd, "wb") as target, open(compressed_path, "rb") as source:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                unpacked = decompressor.decompress(chunk)
                target.write(unpacked)
                raw_sha256.update(unpacked)
            unpacked = decompressor.flush()
            target.write(unpacked)
            raw_sha256.update(unpacked)
        if raw_sha256.hexdigest() != manifest["raw_sha256"]:
            raise RuntimeError("Restored checkpoint SHA256 mismatch")
        return torch.load(raw_path, map_location="cpu", weights_only=True)
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


class SequenceLoader(TaskLoader):
    def __init__(self, args):
        super().__init__(args)
        self.cache = {}

    def sequences(self, name):
        if name not in self.cache:
            pos = pd.read_csv(os.path.join(self.args.windows_dir, f"{name}.windows.tsv"), sep="\t",
                              usecols=["window_id", "seq"])
            neg = pd.read_csv(os.path.join(self.args.neg_dir, f"{name}.negatives.tsv"), sep="\t",
                              usecols=["window_id", "seq"])
            both = pd.concat([pos, neg], ignore_index=True)
            self.cache[name] = dict(zip(both.window_id, both.seq))
        return self.cache[name]

    def arrays(self, task, include_test=False):
        sequences = self.sequences(task["experiment"])
        splits = task["splits"] if include_test else task["splits"][:2]
        return [(ids, [sequences[wid] for wid in ids], y) for ids, y in splits]


def encode(tokenizer, fast_encode, seqs, device):
    if fast_encode is not None:
        encoded = fast_encode(seqs)
    else:
        encoded = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=103)
    return {key: value.to(device, non_blocking=True) for key, value in encoded.items()}


def pooled_logits(model, head, encoded):
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
        return head(pooled).squeeze(1)


def evaluate_scores(model, head, tokenizer, fast_encode, seqs, batch, device):
    import torch
    model.eval()
    head.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            encoded = encode(tokenizer, fast_encode, seqs[i:i + batch], device)
            scores.append(pooled_logits(model, head, encoded).float().cpu().numpy())
    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite evaluation score")
    return result


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RNA-FM fine-tuning; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    x = torch.tensor([1.25, -2.0], device=device, requires_grad=True)
    y = torch.tensor([2.0, 3.0], device=device)
    before = x.detach().clone()
    probe = x * y
    observed = probe.detach().cpu().numpy()
    if not np.array_equal(observed, np.array([2.5, -6.0], dtype=np.float32)):
        raise RuntimeError(f"CUDA numeric probe failed: {observed}")
    probe.sum().backward()
    if not torch.equal(x.grad, y) or not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA backward probe failed")
    torch.optim.SGD([x], lr=0.1).step()
    if torch.equal(before, x.detach()) or not torch.isfinite(x).all():
        raise RuntimeError("CUDA optimizer probe failed")
    if torch.cuda.current_device() != 0 or probe.device.type != "cuda" or probe.device.index != 0:
        raise RuntimeError(f"CUDA device probe failed: {probe.device}")
    torch.cuda.synchronize(device)
    info = {"device": "cuda:0", "name": torch.cuda.get_device_name(device),
            "numeric_probe": observed.tolist(), "backward_probe": "passed",
            "optimizer_probe": "passed", "torch": str(torch.__version__)}
    print("DEVICE_PROBE " + json.dumps(info), flush=True)
    del x, y, before, probe
    torch.cuda.empty_cache()
    return device, info


def build_model(args, repo, device, lora_r=None):
    import torch
    from multimolecule import AutoModel
    model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device)
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        rank = int(lora_r if lora_r is not None else args.lora_r)
        config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=args.lora_dropout,
                            target_modules=["key", "value"], bias="none")
        model = get_peft_model(model, config)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    head = torch.nn.Sequential(torch.nn.Linear(model.config.hidden_size, args.hidden), torch.nn.GELU(),
                               torch.nn.Dropout(args.dropout), torch.nn.Linear(args.hidden, 1)).to(device)
    if next(model.parameters()).device.type != "cuda" or next(head.parameters()).device.type != "cuda":
        raise RuntimeError("Model or head is not on CUDA")
    return model, head


def encoder_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()
            if value.requires_grad}


def head_state(head):
    return {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}


def restore_state(model, head, checkpoint):
    incompatible = model.load_state_dict(checkpoint["encoder_state"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    head.load_state_dict(checkpoint["head_state"], strict=True)


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = -np.inf
        self.best_epoch = 0
        self.best_auroc = np.nan

    def update(self, epoch, auprc, auroc):
        improved = auprc > self.best_metric + self.min_delta
        if improved:
            self.best_metric = float(auprc)
            self.best_epoch = epoch
            self.best_auroc = float(auroc)
        return improved

    def should_stop(self, epoch):
        return epoch - self.best_epoch >= self.patience

    def info(self, stopped_epoch, max_epochs):
        if self.best_epoch == 0 or stopped_epoch >= max_epochs:
            raise RuntimeError(f"Training failsafe reached: best={self.best_epoch}, stopped={stopped_epoch}")
        if stopped_epoch != self.best_epoch + self.patience:
            raise RuntimeError("Early stopping did not exhaust patience")
        return {"best_epoch": self.best_epoch, "stopped_epoch": stopped_epoch, "early_stopped": True,
                "best_val_auprc": self.best_metric, "best_val_auroc": self.best_auroc,
                "patience": self.patience, "min_delta": self.min_delta}


def task_dir(out, task):
    return os.path.join(out, "tasks", f"{task['experiment']}.f{task['fold']}.{task['track']}")


def train_task(task, arrays, args, tokenizer, fast_encode, device, device_info):
    import torch
    import torch.nn.functional as F
    directory = task_dir(args.out, task)
    metadata_path = os.path.join(directory, "metadata.json")
    if os.path.isfile(metadata_path) and os.path.isfile(os.path.join(directory, ".complete")):
        return json.load(open(metadata_path))
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    seed = stable_seed(args.seed, task)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    model, head = build_model(args, args.model, device)
    encoder_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    opt = torch.optim.AdamW([{"params": encoder_params, "lr": args.encoder_lr},
                             {"params": head.parameters(), "lr": args.head_lr}],
                            weight_decay=args.weight_decay)
    (train_ids, train_seqs, ytr), (_, val_seqs, yva) = arrays
    y = torch.from_numpy(ytr.astype(np.float32)).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    stopping = EarlyStopping(args.patience, args.min_delta)
    best_encoder = None
    best_head = None
    history_path = os.path.join(directory, "history.jsonl")
    epoch = 0
    with open(history_path, "w", buffering=1) as history_handle:
        for epoch in range(1, args.epochs + 1):
            model.train()
            head.train()
            order = torch.randperm(len(y), device=device, generator=generator).cpu().numpy()
            total_loss = 0.0
            for i in range(0, len(order), args.batch):
                ii = order[i:i + args.batch]
                encoded = encode(tokenizer, fast_encode, [train_seqs[j] for j in ii], device)
                logits = pooled_logits(model, head, encoded)
                target = y[torch.from_numpy(ii).to(device)]
                target = target * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
                loss = F.binary_cross_entropy_with_logits(logits, target)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(encoder_params + list(head.parameters()), 1.0)
                if not torch.isfinite(grad_norm):
                    raise ValueError("Non-finite gradient norm")
                opt.step()
                total_loss += float(loss.detach()) * len(ii)
            val_score = evaluate_scores(model, head, tokenizer, fast_encode, val_seqs, args.batch, device)
            auroc, auprc = metrics(yva, val_score)
            train_loss = total_loss / len(order)
            if not np.isfinite([auroc, auprc, train_loss]).all():
                raise ValueError("Non-finite epoch metric")
            improved = stopping.update(epoch, auprc, auroc)
            row = {**{key: task[key] for key in TASK_KEYS}, "epoch": epoch,
                   "train_loss": train_loss, "val_auroc": auroc, "val_auprc": auprc,
                   "is_best": improved, "best_epoch_so_far": stopping.best_epoch,
                   "best_val_auprc_so_far": stopping.best_metric}
            history_handle.write(json.dumps(row) + "\n")
            print(f"  {task['experiment']} {task['track']} epoch={epoch} loss={train_loss:.6f} "
                  f"val_auprc={auprc:.6f} best={stopping.best_epoch}", flush=True)
            if improved:
                best_encoder = encoder_state(model)
                best_head = head_state(head)
            if stopping.should_stop(epoch):
                break
    info = stopping.info(epoch, args.epochs)
    if best_encoder is None or best_head is None:
        raise RuntimeError("No accepted best checkpoint")
    restore_state(model, head, {"encoder_state": best_encoder, "head_state": best_head})
    val_score = evaluate_scores(model, head, tokenizer, fast_encode, val_seqs, args.batch, device)
    val_auroc, val_auprc = metrics(yva, val_score)
    checkpoint = {"encoder_state": best_encoder, "head_state": best_head,
                  "task": {key: task[key] for key in
                           ("experiment", "rbp", "tier", "fold", "val_fold", "train_folds", "track")},
                  "seed": seed, "base_seed": args.seed, "mode": args.mode, "lora_r": args.lora_r,
                  "gradient_checkpointing": args.mode == "full",
                  "model": args.model, "hidden": args.hidden, "dropout": args.dropout,
                  "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
                  "split_audit": split_audit_rows(task), "training_device": device_info,
                  "fast_tokenizer_verified": fast_encode is not None, **info}
    checkpoint_manifest = save_checkpoint_compressed(directory, checkpoint)
    recovered = load_checkpoint_compressed(directory)
    if set(recovered["encoder_state"]) != set(best_encoder) or set(recovered["head_state"]) != set(best_head):
        raise RuntimeError("Recovered checkpoint keys differ")
    for name, value in best_encoder.items():
        if not torch.equal(value, recovered["encoder_state"][name]):
            raise RuntimeError(f"Recovered encoder tensor differs: {name}")
    for name, value in best_head.items():
        if not torch.equal(value, recovered["head_state"][name]):
            raise RuntimeError(f"Recovered head tensor differs: {name}")
    del recovered
    val_ids = arrays[1][0]
    prediction_path = os.path.join(directory, "validation_predictions.npz")
    atomic_npz(prediction_path, window_id=np.asarray(val_ids), label=yva, score=val_score)
    metadata = {**{key: task[key] for key in TASK_KEYS}, **info, "n_train": len(ytr),
                "n_eval": len(yva), "val_auroc": val_auroc, "val_auprc": val_auprc,
                "checkpoint": checkpoint_manifest["checkpoint"],
                "checkpoint_manifest": "checkpoint_manifest.json",
                "checkpoint_compressed_bytes": checkpoint_manifest["compressed_bytes"],
                "checkpoint_raw_sha256": checkpoint_manifest["raw_sha256"],
                "checkpoint_compressed_sha256": checkpoint_manifest["compressed_sha256"],
                "history": "history.jsonl", "validation_predictions": "validation_predictions.npz",
                "training_device": device_info, "stable_seed": seed}
    atomic_json(metadata_path, metadata)
    atomic_marker(os.path.join(directory, ".complete"))
    del model, head, opt, best_encoder, best_head, y, generator
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def preflight_storage(args, tasks):
    remaining = [task for task in tasks
                 if not os.path.isfile(os.path.join(task_dir(args.out, task), ".complete"))]
    per_task = 4_000_000 if args.mode == "lora" else 420_000_000
    prediction_bytes = sum(len(task["splits"][1][0]) for task in remaining) * 80
    required = ((len(remaining) * per_task + prediction_bytes + 20 * 1024**3)
                if remaining else 0)
    free = shutil.disk_usage(args.out).free
    report = {"remaining_tasks": len(remaining), "estimated_checkpoint_bytes_per_task": per_task,
              "checkpoint_format": "FP32 torch.save + zlib-1; planning ratio=1.0",
              "reserve_bytes": 20 * 1024**3, "required_bytes_including_reserve": required,
              "free_bytes": free, "passed": free >= required}
    atomic_json(os.path.join(args.out, "storage_preflight.json"), report)
    print("STORAGE_PREFLIGHT " + json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Insufficient /teams capacity for complete run; refusing partial matrix")


def aggregate_training(out, tasks):
    metadata_rows, histories, audits, predictions = [], [], [], []
    for task in tasks:
        directory = task_dir(out, task)
        metadata = json.load(open(os.path.join(directory, "metadata.json")))
        metadata_rows.append(metadata)
        audits.append(metadata.get("split_audit") or split_audit_rows(task))
        with open(os.path.join(directory, "history.jsonl")) as handle:
            histories.extend(json.loads(line) for line in handle if line.strip())
        data = np.load(os.path.join(directory, "validation_predictions.npz"))
        predictions.append(prediction_rows(task, "val", data["window_id"].tolist(), data["label"], data["score"]))
    pd.DataFrame(histories).to_csv(os.path.join(out, "validation_history.tsv"), sep="\t", index=False)
    pd.DataFrame(audits).to_csv(os.path.join(out, "split_audit.tsv"), sep="\t", index=False)
    results = pd.DataFrame([{**{key: row[key] for key in TASK_KEYS}, "split": "val",
                             **{key: row[key] for key in ("best_epoch", "stopped_epoch", "early_stopped",
                                                         "best_val_auprc", "best_val_auroc", "patience", "min_delta",
                                                         "n_train", "n_eval")},
                             "auroc": row["val_auroc"], "auprc": row["val_auprc"]}
                            for row in metadata_rows])
    results.to_csv(os.path.join(out, "finetune_validation_results.tsv"), sep="\t", index=False)
    results.groupby(["track", "split", "tier"])[["auroc", "auprc"]].mean().round(6).to_csv(
        os.path.join(out, "finetune_validation_summary.tsv"), sep="\t")
    pd.concat(predictions, ignore_index=True).to_csv(
        os.path.join(out, "finetune_val_predictions.tsv.gz"), sep="\t", index=False, compression="gzip")
    return results


def load_tokenizer(repo):
    from multimolecule import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    fast_encode, fast_ok = _fast_tokenizer_setup(tokenizer) if repo == "multimolecule/rnafm" else (None, False)
    return tokenizer, fast_encode, fast_ok


def cmd_train(args):
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E1", kind=f"finetune-train-{args.mode}")
    config = vars(args).copy()
    config["code_sha256"] = file_sha256(os.path.abspath(__file__))
    lg.start(config=config, seed=args.seed, data_version=args.data_version,
             inputs=[args.split_clstr, args.exps_file], code_version=config["code_sha256"])
    try:
        if args.data_version != DATA_VERSION:
            raise ValueError(f"Fine-tuning requires {DATA_VERSION}")
        if args.epochs != 1000 or args.patience != 20 or not np.isclose(args.min_delta, 1e-4):
            raise ValueError("Frozen convergence contract is epochs=1000, patience=20, min_delta=1e-4")
        device, device_info = require_cuda()
        loader = SequenceLoader(args)
        tasks = list(loader.tasks())
        preflight_storage(args, tasks)
        tokenizer, fast_encode, fast_ok = load_tokenizer(args.model)
        run_config = {"args": vars(args).copy(), "code_sha256": config["code_sha256"],
                      "cuda": device_info, "fast_tokenizer_verified": fast_ok,
                      "selection": "per-task validation AUPRC; test excluded",
                      "gradient_checkpointing": args.mode == "full",
                      "strict_train_folds": True, "data_version": args.data_version}
        atomic_json(os.path.join(args.out, "finetune_config.json"), run_config)
        for i, task in enumerate(tasks, 1):
            metadata = train_task(task, loader.arrays(task, include_test=False), args,
                                  tokenizer, fast_encode, device, device_info)
            print(f"TRAIN {i}/{len(tasks)} {task['experiment']} f{task['fold']} {task['track']} "
                  f"best={metadata['best_epoch']} stop={metadata['stopped_epoch']}", flush=True)
        results = aggregate_training(args.out, tasks)
        atomic_marker(os.path.join(args.out, ".train_complete"))
        summary = {"n_tasks": len(tasks), "best_epoch_min": int(results.best_epoch.min()),
                   "best_epoch_max": int(results.best_epoch.max()),
                   "stopped_epoch_max": int(results.stopped_epoch.max()),
                   "validation_only": True, "cuda": device_info}
        lg.finish(metrics=summary, outputs=[args.out])
        print("E1_FINETUNE_TRAIN_DONE " + json.dumps(summary), flush=True)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


def cmd_evaluate(args):
    import torch
    train_config_path = os.path.join(args.out, "finetune_config.json")
    train_config = json.load(open(train_config_path))
    if not os.path.isfile(os.path.join(args.out, ".train_complete")):
        raise RuntimeError("Training is not complete")
    train_args = SimpleNamespace(**train_config["args"])
    selection_sha256 = None
    if train_args.mode == "lora":
        if not args.selection:
            raise ValueError("LoRA evaluation requires frozen selection.json")
        selection = json.load(open(args.selection))
        selected_rank = int(selection["selected_rank"])
        if int(train_args.lora_r) != selected_rank:
            raise ValueError(f"Run rank {train_args.lora_r} != frozen rank {selected_rank}")
        selection_sha256 = file_sha256(args.selection)
    lg = RunLogger(args.log, stage="E1", kind=f"finetune-eval-{train_args.mode}")
    eval_config = {"train_config": train_config_path, "train_code_sha256": train_config["code_sha256"],
                   "selection": args.selection, "selection_sha256": selection_sha256}
    lg.start(config=eval_config, seed=train_args.seed, data_version=train_args.data_version,
             inputs=[train_config_path] + ([args.selection] if args.selection else []),
             code_version=file_sha256(os.path.abspath(__file__)))
    try:
        device, device_info = require_cuda()
        loader = SequenceLoader(train_args)
        tasks = list(loader.tasks())
        pending_test_bytes = sum(len(task["splits"][2][0]) for task in tasks
                                 if not os.path.isfile(os.path.join(
                                     args.out, "test_tasks",
                                     f"{task['experiment']}.f{task['fold']}.{task['track']}", "metadata.json"))) * 80
        free = shutil.disk_usage(args.out).free
        if free < pending_test_bytes + 5 * 1024**3:
            raise RuntimeError("Insufficient /teams capacity for complete test evaluation")
        tokenizer, fast_encode, fast_ok = load_tokenizer(train_args.model)
        if fast_ok != train_config["fast_tokenizer_verified"]:
            raise RuntimeError("Tokenizer verification changed between train and evaluation")
        test_root = os.path.join(args.out, "test_tasks")
        os.makedirs(test_root, exist_ok=True)
        for i, task in enumerate(tasks, 1):
            stem = f"{task['experiment']}.f{task['fold']}.{task['track']}"
            test_dir = os.path.join(test_root, stem)
            os.makedirs(test_dir, exist_ok=True)
            test_metadata_path = os.path.join(test_dir, "metadata.json")
            if not os.path.isfile(test_metadata_path):
                checkpoint = load_checkpoint_compressed(task_dir(args.out, task))
                model, head = build_model(train_args, train_args.model, device)
                restore_state(model, head, checkpoint)
                arrays = loader.arrays(task, include_test=True)
                val_score = evaluate_scores(model, head, tokenizer, fast_encode, arrays[1][1],
                                            train_args.batch, device)
                test_score = evaluate_scores(model, head, tokenizer, fast_encode, arrays[2][1],
                                             train_args.batch, device)
                train_metadata = json.load(open(os.path.join(task_dir(args.out, task), "metadata.json")))
                val_auroc, val_auprc = metrics(arrays[1][2], val_score)
                if not np.allclose([val_auroc, val_auprc],
                                   [train_metadata["val_auroc"], train_metadata["val_auprc"]], atol=1e-12):
                    raise RuntimeError("Checkpoint validation reproduction failed")
                test_auroc, test_auprc = metrics(arrays[2][2], test_score)
                test_prediction_path = os.path.join(test_dir, "test_predictions.npz")
                atomic_npz(test_prediction_path, window_id=np.asarray(arrays[2][0]),
                           label=arrays[2][2], score=test_score)
                checkpoint_manifest = json.load(open(os.path.join(task_dir(args.out, task),
                                                                  "checkpoint_manifest.json")))
                atomic_json(test_metadata_path, {"auroc": test_auroc, "auprc": test_auprc,
                                                 "n_eval": len(test_score), "split": "test",
                                                 "training_device": checkpoint["training_device"],
                                                 "evaluation_device": device_info,
                                                 "checkpoint_validation_reproduced": True,
                                                 "checkpoint_used_for_test": True,
                                                 "checkpoint_raw_sha256": checkpoint_manifest["raw_sha256"],
                                                 "checkpoint_compressed_sha256": checkpoint_manifest["compressed_sha256"],
                                                 "test_predictions_sha256": file_sha256(test_prediction_path)})
                del model, head, checkpoint
                gc.collect()
                torch.cuda.empty_cache()
            print(f"EVALUATE {i}/{len(tasks)} {task['experiment']} {task['track']}", flush=True)
        validation = pd.read_csv(os.path.join(args.out, "finetune_validation_results.tsv"), sep="\t")
        rows, predictions = [], []
        for task in tasks:
            stem = f"{task['experiment']}.f{task['fold']}.{task['track']}"
            metadata = json.load(open(os.path.join(test_root, stem, "metadata.json")))
            rows.append({**{key: task[key] for key in TASK_KEYS}, **metadata})
            data = np.load(os.path.join(test_root, stem, "test_predictions.npz"))
            predictions.append(prediction_rows(task, "test", data["window_id"].tolist(),
                                               data["label"], data["score"]))
        test = pd.DataFrame(rows)
        results = pd.concat([validation, test], ignore_index=True)
        results.to_csv(os.path.join(args.out, "finetune_results.tsv"), sep="\t", index=False)
        results.groupby(["track", "split", "tier"])[["auroc", "auprc"]].mean().round(6).to_csv(
            os.path.join(args.out, "finetune_summary.tsv"), sep="\t")
        pd.concat(predictions, ignore_index=True).to_csv(
            os.path.join(args.out, "finetune_test_predictions.tsv.gz"), sep="\t", index=False,
            compression="gzip")
        atomic_json(os.path.join(args.out, "finetune_eval_config.json"), eval_config)
        atomic_marker(os.path.join(args.out, ".eval_complete"))
        summary = {"n_tasks": len(tasks), "selection_sha256": selection_sha256,
                   "test_macro": {track: {"auroc": float(group.auroc.mean()),
                                          "auprc": float(group.auprc.mean())}
                                  for track, group in test.groupby("track")}}
        lg.finish(metrics=summary, outputs=[args.out])
        print("E1_FINETUNE_EVAL_DONE " + json.dumps(summary), flush=True)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


def add_data_arguments(parser):
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--neg-dir", required=True)
    parser.add_argument("--positives-audit", required=True)
    parser.add_argument("--split-clstr", required=True)
    parser.add_argument("--exps-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="multimolecule/rnafm")
    parser.add_argument("--mode", choices=["lora", "full"], required=True)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--tracks", default="neg1c,neg2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--encoder-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--data-version", default=DATA_VERSION)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    add_data_arguments(train)
    train.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--selection")
    evaluate.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    else:
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
