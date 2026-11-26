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

ANALYSIS_ID = "tcell_listnet_adaptation_v2"
V1_ANALYSIS_ID = "tcell_target_adaptation_v1"


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
        return self.parent[value]

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


def component_identity(sequences, members, analysis_id=ANALYSIS_ID):
    values = sorted(sequences[index].decode("ascii") if isinstance(sequences[index], (bytes, np.bytes_)) else str(sequences[index]) for index in members)
    identity = hashlib.sha256("".join(values).encode("ascii")).hexdigest()
    order_key = hashlib.sha256(f"{analysis_id}|component|{identity}".encode()).hexdigest()
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


def component_pools(components, component_keys, pool_count=16, pool_size=64):
    remaining = list(range(len(components)))
    pools = []
    for _ in range(pool_count):
        local_components = [components[index] for index in remaining]
        local_keys = [component_keys[index] for index in remaining]
        chosen_local = allocate_exact(local_components, local_keys, pool_size)
        chosen = {remaining[index] for index in chosen_local}
        row = [member for component in sorted(chosen, key=lambda index: component_keys[index]) for member in components[component]]
        if len(row) != pool_size:
            raise ValueError("component pool size mismatch")
        pools.append(row)
        remaining = [index for index in remaining if index not in chosen]
    return np.asarray(pools, dtype=np.int64)


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


def run_name(budget, learning_rate, temperature, mse_weight):
    return f"listnet_{budget}_lr_{learning_rate:g}_tau_{temperature:g}_mse_{mse_weight:g}"


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


def make_pools(global_indices, sequences, role, budget, pool_size=64, analysis_id=V1_ANALYSIS_ID):
    keyed = sorted((hashlib.sha256(f"{analysis_id}|pools|{role}|{budget}|{sequences[index].decode('ascii')}".encode()).digest(), int(index)) for index in global_indices)
    rows = []
    for start in range(0, len(keyed), pool_size):
        row = [index for _, index in keyed[start:start + pool_size]]
        row.extend([-1] * (pool_size - len(row)))
        rows.append(row)
    if not rows:
        raise ValueError("empty pool role")
    return np.asarray(rows, dtype=np.int64)


def make_proposals(pool_count, pool_size, seeds, role, cap=16, analysis_id=ANALYSIS_ID):
    result = np.empty((pool_count, len(seeds), cap), dtype=np.int64)
    for pool in range(pool_count):
        for seed_index, seed in enumerate(seeds):
            rng = random.Random(f"{analysis_id}|{role}|{pool}|{seed}")
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


def primary_success(comparisons, isolation_pass=True, convergence_pass=True):
    comparison_success = {}
    for name, row in comparisons.items():
        comparison_success[name] = bool(row["effect"]["ci_high"] < 0 and -row["effect"]["effect"] >= row["minimum_required_absolute_reduction"])
    return {"comparison_pass": comparison_success, "success": bool(all(comparison_success.values()) and isolation_pass and convergence_pass)}


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
    return {"mean": float(values.mean()), "scale": float(values.std(ddof=0)), "ddof": 0, "fit_role": "v1_train_50000_only"}


def listnet_objective(model, codes, labels, pools, mean, scale, mse_weight, temperature):
    if temperature <= 0 or mse_weight < 0:
        raise ValueError("temperature must be positive and MSE weight nonnegative")
    valid = pools >= 0
    rows = pools[valid]
    outputs = model(codes_to_one_hot(codes[rows]))
    standardized = (labels[rows] - mean) / scale
    errors = torch.zeros(pools.shape, device=outputs.device, dtype=outputs.dtype)
    errors[valid] = outputs - standardized
    mse = torch.sum((errors * errors).sum(dim=1) / valid.sum(dim=1)) / len(pools)
    complete = valid.all(dim=1)
    if not complete.any():
        zero = torch.zeros((), device=outputs.device, dtype=outputs.dtype)
        return mse_weight * mse, mse.detach(), zero
    full_labels = torch.zeros(pools.shape, device=outputs.device, dtype=outputs.dtype)
    full_labels[valid] = standardized
    full_outputs = torch.zeros(pools.shape, device=outputs.device, dtype=outputs.dtype)
    full_outputs[valid] = outputs
    complete_outputs = full_outputs[complete] / temperature
    complete_labels = full_labels[complete] / temperature
    target_probabilities = torch.softmax(complete_labels, dim=1)
    listnet_per_pool = -(target_probabilities * torch.log_softmax(complete_outputs, dim=1)).sum(dim=1)
    listnet = listnet_per_pool.mean()
    total = listnet + mse_weight * mse
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite ListNet objective")
    return total, mse.detach(), listnet.detach()


def evaluate_pools(model, codes, labels, pools, mean, scale, mse_weight, temperature, batch_pools=16):
    model.eval()
    mse_total = 0.0
    listnet_total = 0.0
    complete_count = int(np.sum((pools >= 0).all(axis=1).cpu().numpy()))
    with torch.no_grad():
        for start in range(0, len(pools), batch_pools):
            end = min(len(pools), start + batch_pools)
            _, mse, listnet = listnet_objective(model, codes, labels, pools[start:end], mean, scale, mse_weight, temperature)
            mse_total += float(mse.item()) * (end - start) / len(pools)
            batch_complete = int((pools[start:end] >= 0).all(dim=1).sum().item())
            listnet_total += float(listnet.item()) * batch_complete / max(1, complete_count)
    total = listnet_total + mse_weight * mse_total
    if not np.isfinite([mse_total, listnet_total, total]).all():
        raise FloatingPointError("nonfinite evaluation")
    return {"loss": total, "mse": mse_total, "listnet": listnet_total}


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
