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

ANALYSIS_ID = "tcell_target_adaptation_v1"


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


def hamming1(left, right):
    return sum(a != b for a, b in zip(left, right)) == 1


class HalfIndex:
    def __init__(self):
        self.exact = set()
        self.halves = {}

    def add(self, sequence):
        if sequence in self.exact:
            return
        self.exact.add(sequence)
        for side, half in ((0, sequence[:25]), (1, sequence[25:])):
            self.halves.setdefault((side, half), []).append(sequence)

    def match(self, sequence):
        if sequence in self.exact:
            return True, False
        for side, half in ((0, sequence[:25]), (1, sequence[25:])):
            for other in self.halves.get((side, half), ()):
                if hamming1(sequence, other):
                    return False, True
        return False, False


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def h1_components(sequences):
    sequences = [value.decode("ascii") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in sequences]
    lookup = {value: index for index, value in enumerate(sequences)}
    if len(lookup) != len(sequences):
        raise ValueError("duplicate sequence")
    halves = {}
    for index, value in enumerate(sequences):
        halves.setdefault((0, value[:25]), []).append(index)
        halves.setdefault((1, value[25:]), []).append(index)
    union = UnionFind(len(sequences))
    edges = 0
    for group in halves.values():
        if len(group) < 2:
            continue
        for offset, left in enumerate(group):
            for right in group[offset + 1:]:
                if hamming1(sequences[left], sequences[right]):
                    union.union(left, right)
                    edges += 1
    components = {}
    for index in range(len(sequences)):
        components.setdefault(union.find(index), []).append(index)
    return list(components.values()), edges


def component_identity(sequences, members):
    values = sorted(sequences[index].decode("ascii") if isinstance(sequences[index], (bytes, np.bytes_)) else str(sequences[index]) for index in members)
    joined = "".join(values).encode("ascii")
    identity = hashlib.sha256(joined).hexdigest()
    order_key = hashlib.sha256(f"{ANALYSIS_ID}|component|{identity}".encode()).hexdigest()
    return identity, order_key, values[0]


def allocate_exact(components, component_keys, budget, eligible=None):
    ordered = sorted(range(len(components)), key=lambda index: component_keys[index])
    chosen = []
    count = 0
    for index in ordered:
        if eligible is not None and not eligible[index]:
            continue
        size = len(components[index])
        if size <= budget - count:
            chosen.append(index)
            count += size
            if count == budget:
                return chosen
    raise ValueError(f"exact component allocation failed: {count}/{budget}")


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


def run_name(variant, budget, learning_rate, lambda_):
    prefix = "main" if variant in ("main", "nested") else variant
    return f"{prefix}_{budget}_lr_{learning_rate:g}_lambda_{lambda_:g}"


def state_dict_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu()
        value.update(name.encode())
        value.update(str(array.dtype).encode())
        value.update(str(tuple(array.shape)).encode())
        value.update(array.numpy().tobytes())
    return value.hexdigest()


def encode_codes(sequences):
    mapping = np.full(256, 255, dtype=np.uint8)
    for index, base in enumerate("ACGT"):
        mapping[ord(base)] = index
    values = np.asarray(sequences, dtype="S50")
    if values.ndim != 1 or not len(values):
        raise ValueError("nonempty sequence vector required")
    codes = mapping[values.view(np.uint8).reshape(len(values), 50)]
    if np.any(codes == 255):
        raise ValueError("non-ACGT sequence")
    return codes.astype(np.uint8, copy=False)


def codes_to_one_hot(codes):
    return torch.nn.functional.one_hot(codes.long(), num_classes=4).permute(0, 2, 1).float()


def make_pools(global_indices, sequences, role, budget, pool_size=64):
    keyed = sorted((hashlib.sha256(f"{ANALYSIS_ID}|pools|{role}|{budget}|{sequences[index].decode('ascii')}".encode()).digest(), int(index)) for index in global_indices)
    rows = []
    for start in range(0, len(keyed), pool_size):
        row = [index for _, index in keyed[start:start + pool_size]]
        row.extend([-1] * (pool_size - len(row)))
        rows.append(row)
    if not rows:
        raise ValueError("empty pool role")
    return np.asarray(rows, dtype=np.int64)


def make_proposals(pool_count, pool_size, seeds, role, cap=16):
    result = np.empty((pool_count, len(seeds), cap), dtype=np.int64)
    for pool in range(pool_count):
        for seed_index, seed in enumerate(seeds):
            rng = random.Random(f"{ANALYSIS_ID}|{role}|{pool}|{seed}")
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


def random_regret(labels, pools, proposals, cap=16):
    prefix = proposals[:, :, :cap]
    proposed_labels = np.take_along_axis(labels[pools][:, None, :], prefix, axis=2)
    return labels[pools].max(axis=1)[:, None] - proposed_labels[:, :, 0], np.zeros(prefix.shape[:2], dtype=np.int64)


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


def target_scaler(labels):
    values = np.asarray(labels, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or values.std(ddof=0) <= 0:
        raise ValueError("invalid train labels")
    return {"mean": float(values.mean()), "scale": float(values.std(ddof=0)), "ddof": 0, "fit_role": "budget_train_only"}


def dynamic_objective(model, codes, labels, pools, mean, scale, lambda_):
    valid = pools >= 0
    rows = pools[valid]
    outputs = model(codes_to_one_hot(codes[rows]))
    standardized = (labels[rows] - mean) / scale
    width = pools.shape[1]
    errors = torch.zeros(pools.shape, device=outputs.device, dtype=outputs.dtype)
    errors[valid] = outputs - standardized
    full_labels = torch.zeros(pools.shape, device=outputs.device, dtype=outputs.dtype)
    full_labels[valid] = standardized
    full_outputs = torch.zeros(pools.shape, device=outputs.device, dtype=outputs.dtype)
    full_outputs[valid] = outputs
    mse = torch.sum((errors * errors).sum(dim=1) / valid.sum(dim=1)) / len(pools)
    if lambda_ == 0:
        return mse, mse.detach(), torch.zeros((), device=outputs.device, dtype=outputs.dtype)
    complete = valid.all(dim=1)
    tail = torch.argsort(full_outputs.detach(), dim=1, descending=True, stable=True)[:, :16]
    tail_scores = torch.gather(full_outputs, 1, tail)
    tail_labels = torch.gather(full_labels, 1, tail)
    left, right = torch.triu_indices(16, 16, offset=1, device=outputs.device)
    differences = tail_scores[:, left] - tail_scores[:, right]
    signs = torch.sign(tail_labels[:, left] - tail_labels[:, right])
    nontied = signs != 0
    logistic = torch.nn.functional.softplus(-signs * differences) * nontied
    per_pool = logistic.sum(dim=1) / nontied.sum(dim=1).clamp(min=1)
    pair = torch.sum(torch.where(complete & (nontied.sum(dim=1) > 0), per_pool, torch.zeros_like(per_pool))) / complete.sum().clamp(min=1)
    return mse + lambda_ * pair, mse.detach(), pair.detach()


def evaluate_pools(model, codes, labels, pools, mean, scale, lambda_, batch_pools=16):
    model.eval()
    mse_total = 0.0
    pair_total = 0.0
    complete_count = int(np.sum((pools >= 0).all(axis=1).cpu().numpy()))
    with torch.no_grad():
        for start in range(0, len(pools), batch_pools):
            end = min(len(pools), start + batch_pools)
            _, mse, pair = dynamic_objective(model, codes, labels, pools[start:end], mean, scale, lambda_)
            mse_total += float(mse.item()) * (end - start) / len(pools)
            batch_complete = int((pools[start:end] >= 0).all(dim=1).sum().item())
            pair_total += float(pair.item()) * batch_complete / max(1, complete_count)
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


def spearman(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    def ranks(values):
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2 + 1
            start = end
        return result
    x, y = ranks(left), ranks(right)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])
