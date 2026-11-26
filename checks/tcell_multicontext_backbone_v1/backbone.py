from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def valid_sequence(value):
    return len(value) == 50 and set(value) <= set("ACGT")


class HalfIndex:
    def __init__(self):
        self.exact = set()
        self.halves = set()

    def add(self, sequence):
        self.exact.add(sequence)
        self.halves.add((0, sequence[:25]))
        self.halves.add((1, sequence[25:]))

    def match(self, sequence):
        if sequence in self.exact:
            return True, False
        left = (0, sequence[:25]) in self.halves
        right = (1, sequence[25:]) in self.halves
        if not (left or right):
            return False, False
        h1 = False
        if left:
            for position in range(25, 50):
                for base in "ACGT":
                    candidate = sequence[:position] + base + sequence[position + 1:]
                    if candidate != sequence and candidate in self.exact:
                        h1 = True
                        break
                if h1:
                    break
        if not h1 and right:
            for position in range(25):
                for base in "ACGT":
                    candidate = sequence[:position] + base + sequence[position + 1:]
                    if candidate != sequence and candidate in self.exact:
                        h1 = True
                        break
                if h1:
                    break
        return False, h1


class FreshSampleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5), nn.ReLU(), nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5), nn.ReLU(),
        )
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, inputs):
        return self.regressor(self.features(inputs).mean(dim=2)).squeeze(1)


def state_dict_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().numpy()
        value.update(name.encode())
        value.update(str(array.dtype).encode())
        value.update(str(array.shape).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def codes_to_one_hot(codes):
    return torch.nn.functional.one_hot(codes.long(), num_classes=4).permute(0, 2, 1).float()


def encode_codes(sequences):
    mapping = np.full(256, 255, dtype=np.uint8)
    for index, base in enumerate("ACGT"):
        mapping[ord(base)] = index
    output = np.empty((len(sequences), 50), dtype=np.uint8)
    for row, sequence in enumerate(sequences):
        if not valid_sequence(sequence):
            raise ValueError("expected exactly 50 A/C/G/T bases")
        output[row] = mapping[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    return output


def context_scalers(labels, contexts):
    result = {}
    for context in np.unique(contexts):
        values = np.asarray(labels[contexts == context], dtype=np.float64)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError("invalid context train labels")
        scale = float(values.std(ddof=0))
        if scale <= 0:
            raise ValueError("zero context train SD")
        result[int(context)] = {"mean": float(values.mean()), "scale": scale, "ddof": 0, "fit_role": "context_train_only"}
    return result


def make_pools(sequences, context, split, pool_size=64):
    keyed = sorted((hashlib.sha256(f"{context}|{split}|{sequence}".encode()).digest(), index) for index, sequence in enumerate(sequences))
    pools = []
    for start in range(0, len(keyed), pool_size):
        row = [index for _, index in keyed[start:start + pool_size]]
        row.extend([-1] * (pool_size - len(row)))
        pools.append(row)
    if not pools:
        raise ValueError("empty context role")
    return np.asarray(pools, dtype=np.int64)


def make_proposals(pool_count, pool_size, seeds, cap=16, namespace="auxiliary"):
    result = np.empty((pool_count, len(seeds), cap), dtype=np.int64)
    for pool in range(pool_count):
        for seed_index, seed in enumerate(seeds):
            rng = random.Random(f"{namespace}|{pool}|{seed}")
            result[pool, seed_index] = [min(pool_size - 1, int(rng.random() * pool_size)) for _ in range(cap)]
    return result


def cap_regret(scores, labels, pools, proposals, cap=16):
    prefix = proposals[:, :, :cap]
    pool_scores = scores[pools]
    pool_labels = labels[pools]
    proposed_scores = np.take_along_axis(pool_scores[:, None, :], prefix, axis=2)
    proposed_labels = np.take_along_axis(pool_labels[:, None, :], prefix, axis=2)
    selected = np.argmax(proposed_scores, axis=2)
    selected_labels = np.take_along_axis(proposed_labels, selected[..., None], axis=2)[..., 0]
    return pool_labels.max(axis=1)[:, None] - selected_labels, selected


def cluster_bootstrap(pool_seed_values, seed, replicates=10000):
    values = np.asarray(pool_seed_values, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("finite pool-by-seed matrix required")
    clusters = values.mean(axis=1)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    draws = clusters[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((np.sum(draws >= 0) + 1) / (replicates + 1), (np.sum(draws <= 0) + 1) / (replicates + 1))
    return {"effect": float(clusters.mean()), "ci_low": float(low), "ci_high": float(high), "p_two_sided": float(min(1.0, p)), "clusters": len(clusters), "replicates": replicates, "seed": seed}


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


@dataclass
class DualPlateau:
    min_delta: float
    patience: int
    best_train: float = math.inf
    best_dev: float = math.inf
    best_epoch: int = 0
    plateau_epochs: int = 0

    def update(self, epoch, train_loss, dev_loss):
        if not math.isfinite(train_loss) or not math.isfinite(dev_loss):
            raise FloatingPointError("nonfinite plateau input")
        train_significant = train_loss < self.best_train - self.min_delta
        dev_significant = dev_loss < self.best_dev - self.min_delta
        dev_best = dev_loss < self.best_dev
        self.best_train = min(self.best_train, train_loss)
        if dev_best:
            self.best_dev = dev_loss
            self.best_epoch = epoch
        self.plateau_epochs = 0 if dev_best or train_significant or dev_significant else self.plateau_epochs + 1
        return {"stop": self.plateau_epochs >= self.patience, "dev_best": dev_best, "train_significant_improvement": train_significant, "dev_significant_improvement": dev_significant}


def dynamic_objective(model, codes, labels, pools, pool_contexts, scalers, lambda_, context_totals):
    valid = pools >= 0
    rows = pools[valid]
    outputs = model(codes_to_one_hot(codes[rows]))
    raw_labels = labels[rows]
    width = pools.shape[1]
    means = torch.as_tensor([scalers[int(context)]["mean"] for context in pool_contexts], device=outputs.device, dtype=outputs.dtype)
    scales = torch.as_tensor([scalers[int(context)]["scale"] for context in pool_contexts], device=outputs.device, dtype=outputs.dtype)
    standardized = (raw_labels - means.repeat_interleave(width)[valid.reshape(-1)]) / scales.repeat_interleave(width)[valid.reshape(-1)]
    errors = torch.zeros(pools.shape, device=outputs.device, dtype=outputs.dtype)
    errors[valid] = outputs - standardized
    full_labels = torch.zeros(pools.shape, device=outputs.device, dtype=outputs.dtype)
    full_labels[valid] = standardized
    full_outputs = torch.zeros(pools.shape, device=outputs.device, dtype=outputs.dtype)
    full_outputs[valid] = outputs
    context_count = len(scalers)
    context_sizes = torch.as_tensor(context_totals, device=outputs.device, dtype=outputs.dtype)
    pool_contexts_tensor = torch.as_tensor(pool_contexts, device=outputs.device, dtype=torch.long)
    pool_weights = context_sizes.sum() / (context_count * context_sizes[pool_contexts_tensor])
    mse = torch.sum((errors * errors).sum(dim=1) / valid.sum(dim=1) * pool_weights) / len(pools)
    if lambda_ == 0:
        return mse, mse.detach(), torch.zeros((), device=outputs.device, dtype=outputs.dtype)
    scores = full_outputs
    complete = valid.all(dim=1)
    tail_size = min(16, width)
    tail = torch.argsort(scores.detach(), dim=1, descending=True, stable=True)[:, :tail_size]
    tail_scores = torch.gather(scores, 1, tail)
    tail_labels = torch.gather(full_labels, 1, tail)
    left, right = torch.triu_indices(tail_size, tail_size, offset=1, device=outputs.device)
    differences = tail_scores[:, left] - tail_scores[:, right]
    signs = torch.sign(tail_labels[:, left] - tail_labels[:, right])
    nontied = signs != 0
    logistic = torch.nn.functional.softplus(-signs * differences) * nontied
    per_pool = logistic.sum(dim=1) / nontied.sum(dim=1).clamp(min=1)
    per_pool = torch.where(complete & (nontied.sum(dim=1) > 0), per_pool, torch.zeros_like(per_pool))
    pair = torch.sum(per_pool * pool_weights) / len(pools)
    return mse + lambda_ * pair, mse.detach(), pair.detach()


def evaluate_pools(model, codes, labels, pools, pool_contexts, scalers, lambda_, batch_pools=16):
    model.eval()
    mse_total = 0.0
    pair_total = 0.0
    context_totals = np.bincount(pool_contexts, minlength=len(scalers))
    with torch.no_grad():
        for start in range(0, len(pools), batch_pools):
            end = min(len(pools), start + batch_pools)
            _, mse, pair = dynamic_objective(model, codes, labels, pools[start:end], pool_contexts[start:end], scalers, lambda_, context_totals)
            weight = (end - start) / len(pools)
            mse_total += float(mse.item()) * weight
            pair_total += float(pair.item()) * weight
    total = mse_total + lambda_ * pair_total
    if not np.isfinite([mse_total, pair_total, total]).all():
        raise FloatingPointError("nonfinite evaluation")
    return {"loss": total, "mse": mse_total, "pairwise": pair_total}


def configure_cuda(seed=0, fraction=0.28):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; silent CPU fallback forbidden")
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    properties = torch.cuda.get_device_properties(0)
    return torch.device("cuda:0"), {"index": 0, "name": properties.name, "total_memory": properties.total_memory, "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": "float32", "actual_cuda": True, "cpu_fallback": False, "memory_fraction": fraction}
