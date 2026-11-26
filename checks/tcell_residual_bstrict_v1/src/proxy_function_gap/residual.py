from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .mechanism import sequence_features
from .trainer.checkpoint import atomic_json_write
from .trainer.cuda import configure_cuda
from .trainer.data import sha256_file

BASES = np.full(256, 255, dtype=np.uint8)
for index, base in enumerate(b"ACGT"):
    BASES[base] = index


def digest(path):
    return sha256_file(Path(path))


def save_json(path, value):
    atomic_json_write(value, Path(path))


def load_json(path):
    return json.loads(Path(path).read_text())


def encode_sequences(sequences):
    values = np.asarray(sequences, dtype="S50")
    if values.ndim != 1:
        raise ValueError("one sequence vector required")
    bases = BASES[values.view(np.uint8).reshape(len(values), 50)]
    if np.any(bases == 255):
        raise ValueError("non-ACGT sequence")
    return bases


def one_hot(sequences):
    bases = encode_sequences(sequences)
    return (bases[..., None] == np.arange(4, dtype=np.uint8)).transpose(0, 2, 1).astype(np.float32)


def packed_halves(sequences):
    bases = encode_sequences(sequences).astype(np.uint64)
    packed = np.zeros((len(bases), 2), dtype=np.uint64)
    for position in range(25):
        packed[:, 0] = (packed[:, 0] << np.uint64(2)) | bases[:, position]
        packed[:, 1] = (packed[:, 1] << np.uint64(2)) | bases[:, position + 25]
    return packed


def decode_sequence(value):
    return value.decode("ascii") if isinstance(value, (bytes, np.bytes_)) else str(value)


def simple_features(sequences, p):
    bases = encode_sequences(sequences)
    gc = np.mean((bases == 1) | (bases == 2), axis=1)
    aug = np.zeros(len(bases), dtype=bool)
    run = np.zeros(len(bases), dtype=np.int16)
    longest = np.zeros(len(bases), dtype=np.int16)
    for position in range(50):
        run = np.where(bases[:, position] == 3, run + 1, 0)
        longest = np.maximum(longest, run)
        if position <= 47:
            aug |= (bases[:, position] == 0) & (bases[:, position + 1] == 3) & (bases[:, position + 2] == 2)
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack((p, p * p, gc, aug.astype(np.float64), longest.astype(np.float64), np.ones(len(p))))


def sequence_feature_check():
    rng = random.Random(17)
    for _ in range(100):
        sequence = "".join(rng.choice("ACGT") for _ in range(50))
        expected = sequence_features(sequence)
        actual = simple_features(np.asarray([sequence], dtype="S50"), np.asarray([2.0]))[0]
        assert actual[2] == expected["gc"]
        assert bool(actual[3]) == expected["uAUG"]
        assert actual[4] == expected["longest_poly_u"]
        assert actual[5] == 1.0


def hamming1(left, right):
    return len(left) == len(right) and sum(a != b for a, b in zip(left, right)) == 1


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


def frozen_members(config):
    import json
    value = load_json(config)
    return value


def cnn_predict(sequences, inference_config, batch_size=4096, memory_fraction=0.25):
    from .trainer.checkpoint import load_checkpoint, trainer_code_identity
    from .trainer.model import FreshSampleCNN, model_definition_hash, model_hash

    config = load_json(inference_config)
    device, dtype, gpu = configure_cuda(0, memory_fraction, "float32", 20261126)
    if str(device) != "cuda:0":
        raise RuntimeError("all CNN inference must use cuda:0")
    member_outputs = []
    member_reports = []
    sample_scales = []
    for member in config["members"]:
        if digest(member["checkpoint"]) != member["sha256"]:
            raise ValueError("checkpoint hash mismatch")
        checkpoint = load_checkpoint(member["checkpoint"], device)
        model = FreshSampleCNN()
        if checkpoint["model_definition_sha256"] != model_definition_hash(model):
            raise ValueError("model definition mismatch")
        model.load_state_dict(checkpoint["model_state"])
        model.to(device=device, dtype=dtype)
        model.eval()
        if model_hash(model) != member["model_hash"]:
            raise ValueError("frozen model hash mismatch")
        scaler = checkpoint["scaler"]
        if scaler["fit_role"] != "train" or scaler["ddof"] != 0:
            raise ValueError("train-only population scaler required")
        sample_scales.append(float(scaler["scale"]))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                values = one_hot(sequences[start:start + batch_size])
                inputs = torch.from_numpy(values).to(device=device, dtype=dtype)
                if not inputs.is_cuda or inputs.device.index != 0:
                    raise RuntimeError("CPU inference fallback forbidden")
                pred = model(inputs)
                outputs.append(pred.detach().cpu().numpy().astype(np.float64) * float(scaler["scale"]) + float(scaler["mean"]))
        torch.cuda.synchronize(device)
        member_outputs.append(np.concatenate(outputs))
        member_reports.append({"seed": member["seed"], "checkpoint_sha256": member["sha256"], "model_hash": member["model_hash"], "gpu": gpu, "rows": len(sequences), "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)})
        del model
        torch.cuda.empty_cache()
    if not sample_scales or max(sample_scales) != min(sample_scales):
        raise ValueError("member Sample train scales differ")
    predictions = np.stack(member_outputs)
    return predictions.mean(axis=0), predictions, {"members": member_reports, "sample_train_sd": sample_scales[0], "cpu_fallback": False, "actual_cuda": True}


def make_auxiliary_pools(sequences, context, split, seeds, cap=16, pool_size=64):
    keyed = sorted((hashlib.sha256(f"{context}|{split}|{decode_sequence(seq)}".encode()).digest(), index) for index, seq in enumerate(sequences))
    count = len(keyed) // pool_size
    if count < 1:
        raise ValueError("insufficient complete auxiliary pools")
    pools = np.asarray([index for _, index in keyed[:count * pool_size]], dtype=np.int64).reshape(count, pool_size)
    proposals = np.empty((count, len(seeds), cap), dtype=np.int64)
    for pool_index in range(count):
        for seed_index, seed in enumerate(seeds):
            rng = random.Random(f"{context}|{split}|{pool_index}|{seed}")
            proposals[pool_index, seed_index] = [min(pool_size - 1, int(rng.random() * pool_size)) for _ in range(cap)]
    return pools, proposals


def cap16_regret(scores, labels, pools, proposals):
    pool_scores = scores[pools]
    pool_labels = labels[pools]
    proposed_scores = np.take_along_axis(pool_scores[:, None, :], proposals, axis=2)
    proposed_labels = np.take_along_axis(pool_labels[:, None, :], proposals, axis=2)
    selected = np.argmax(proposed_scores, axis=2)
    selected_labels = np.take_along_axis(proposed_labels, selected[..., None], axis=2)[..., 0]
    regrets = pool_labels.max(axis=1)[:, None] - selected_labels
    return {"mean_regret": float(regrets.mean()), "pool_seed_regret": regrets}


def stable_auc_order(scores):
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores)))


def fit_ridge_cuda(x, y, lambda_, condition_max=1e12):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < x.shape[1] or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("nonfinite or underspecified ridge inputs")
    device, dtype, gpu = configure_cuda(0, 0.25, "float64", 20261126)
    tx = torch.from_numpy(x).to(device=device, dtype=dtype)
    ty = torch.from_numpy(y).to(device=device, dtype=dtype)
    penalty = torch.ones(x.shape[1], device=device, dtype=dtype)
    penalty[-1] = 0.0
    matrix = tx.T @ tx / len(x) + float(lambda_) * torch.diag(penalty)
    rhs = tx.T @ ty / len(x)
    condition = torch.linalg.cond(matrix)
    coefficients = torch.linalg.solve(matrix, rhs)
    fitted = tx @ coefficients
    residual = ty - fitted
    torch.cuda.synchronize(device)
    values = coefficients.detach().cpu().numpy()
    report = {
        "lambda": float(lambda_), "condition_number": float(condition.item()),
        "coefficients": values.tolist(), "train_rows": len(x), "features": x.shape[1],
        "finite": bool(np.all(np.isfinite(values)) and np.isfinite(condition.item()) and torch.isfinite(residual).all().item()),
        "converged": True, "solver": "closed_form_torch_linalg_solve", "device": str(device), "dtype": str(dtype),
        "train_standardized_mse": float(torch.mean(residual * residual).item()), "gpu": gpu, "cpu_fallback": False,
    }
    if report["condition_number"] > condition_max or not report["finite"]:
        raise FloatingPointError(f"ridge numerical gate failed: {report}")
    return values, report


def proposal_indices(pool_size, seed, cap):
    rng = random.Random(seed)
    return [min(pool_size - 1, int(rng.random() * pool_size)) for _ in range(cap)]


def cluster_bootstrap(values, seed, replicates=10000):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("invalid bootstrap cluster values")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = values[indices].mean(axis=1)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    p = min(1.0, 2.0 * min((1 + int(np.sum(draws <= 0))) / (replicates + 1), (1 + int(np.sum(draws >= 0))) / (replicates + 1)))
    return {"effect": float(values.mean()), "ci_low": float(lower), "ci_high": float(upper), "p_two_sided": float(p), "clusters": len(values), "replicates": replicates, "seed": seed}
