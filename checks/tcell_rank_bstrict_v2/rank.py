from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


DIRECTIONS = (
    ("n50_hek293t_r1", "defined_hepg2_r1"),
    ("defined_hepg2_r1", "n50_hek293t_r1"),
)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_context(root):
    root = Path(root)
    sequences = np.load(root / "sequences.npy", allow_pickle=False)
    y = np.load(root / "labels.npy", allow_pickle=False).astype(np.float64)
    p = np.load(root / "cnn_predictions.npy", allow_pickle=False).astype(np.float64)
    split = np.load(root / "split.npy", allow_pickle=False)
    if not (len(sequences) == len(y) == len(p) == len(split)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("invalid auxiliary arrays")
    return sequences, y, p, split


def simple_features(sequences, p):
    values = np.asarray(sequences, dtype="S50")
    bases = values.view(np.uint8).reshape(len(values), 50)
    if np.any((bases != ord("A")) & (bases != ord("C")) & (bases != ord("G")) & (bases != ord("T"))):
        raise ValueError("non-ACGT sequence")
    gc = ((bases == ord("C")) | (bases == ord("G"))).mean(axis=1, dtype=np.float64)
    uaug = np.zeros(len(bases), dtype=np.float64)
    longest = np.zeros(len(bases), dtype=np.float64)
    run = np.zeros(len(bases), dtype=np.float64)
    for position in range(50):
        run = np.where(bases[:, position] == ord("T"), run + 1, 0)
        longest = np.maximum(longest, run)
        if position <= 47:
            uaug = np.maximum(uaug, ((bases[:, position] == ord("A")) & (bases[:, position + 1] == ord("T")) & (bases[:, position + 2] == ord("G"))).astype(np.float64))
    p = np.asarray(p, dtype=np.float64)
    return np.column_stack((p, p * p, gc, uaug, longest, np.ones(len(p), dtype=np.float64)))


def make_pools(sequences, context, split, seeds, cap=16, pool_size=64):
    keyed = sorted((hashlib.sha256(f"{context}|{split}|{sequence.decode('ascii')}".encode()).digest(), index) for index, sequence in enumerate(sequences))
    count = len(keyed) // pool_size
    if count < 1:
        raise ValueError("insufficient complete pools")
    pools = np.asarray([index for _, index in keyed[:count * pool_size]], dtype=np.int64).reshape(count, pool_size)
    proposals = np.empty((count, len(seeds), cap), dtype=np.int64)
    for pool_index in range(count):
        for seed_index, seed in enumerate(seeds):
            rng = random.Random(f"{context}|{split}|{pool_index}|{seed}")
            proposals[pool_index, seed_index] = [min(pool_size - 1, int(rng.random() * pool_size)) for _ in range(cap)]
    return pools, proposals


def fixed_tail(p, pools, tail_k=16):
    order = np.argsort(-np.asarray(p)[pools], axis=1, kind="stable")[:, :tail_k]
    return np.take_along_axis(pools, order, axis=1)


def pair_arrays(y, p, z, tail):
    pair_i, pair_j = np.triu_indices(tail.shape[1], 1)
    left = tail[:, pair_i]
    right = tail[:, pair_j]
    dy = y[left] - y[right]
    valid = dy != 0
    labels = np.sign(dy[valid]).astype(np.float64)
    left = left[valid]
    right = right[valid]
    pool_ids = np.broadcast_to(np.arange(len(tail))[:, None], (len(tail), len(pair_i)))[valid]
    counts = np.bincount(pool_ids, minlength=len(tail))
    if np.any(counts == 0):
        raise ValueError("a training tail has no nontied MRL pairs")
    weights = (1.0 / len(tail) / counts[pool_ids]).astype(np.float64)
    p_diff = (p[left] - p[right]).astype(np.float64)
    z_diff = (z[left] - z[right]).astype(np.float64)
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise FloatingPointError("pair weights do not sum to one")
    return {"left": left, "right": right, "labels": labels, "weights": weights, "p_diff": p_diff, "z_diff": z_diff, "pool_ids": pool_ids}


def cap_regret(scores, labels, pools, proposals, cap=16):
    if proposals.shape[2] < cap:
        raise ValueError("proposal cap too small")
    prefix = proposals[:, :, :cap]
    pool_scores = scores[pools]
    pool_labels = labels[pools]
    proposed_scores = np.take_along_axis(pool_scores[:, None, :], prefix, axis=2)
    proposed_labels = np.take_along_axis(pool_labels[:, None, :], prefix, axis=2)
    selected = np.argmax(proposed_scores, axis=2)
    selected_labels = np.take_along_axis(proposed_labels, selected[..., None], axis=2)[..., 0]
    regrets = pool_labels.max(axis=1)[:, None] - selected_labels
    return regrets, selected


def pairwise_loss_numpy(pairs, beta, sample_sd, strength=1.0, lambda_=0.0):
    margin = pairs["labels"] * (pairs["p_diff"] + sample_sd * strength * (pairs["z_diff"] @ beta))
    return float(np.sum(pairs["weights"] * np.logaddexp(0.0, -margin)) + lambda_ * np.dot(beta, beta))


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    return device, {"name": properties.name, "capability": list(properties.major for properties in [properties]) + [properties.minor], "total_memory": properties.total_memory, "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": "torch.float64", "cpu_fallback": False}


def fit_pairwise_newton(pairs, lambda_, config):
    device, gpu = configure_cuda()
    solver = config["solver"]
    sample_sd = float(config["sample_train_sd"])
    labels = torch.from_numpy(pairs["labels"]).to(device=device, dtype=torch.float64)
    weights = torch.from_numpy(pairs["weights"]).to(device=device, dtype=torch.float64)
    p_diff = torch.from_numpy(pairs["p_diff"]).to(device=device, dtype=torch.float64)
    z_diff = torch.from_numpy(pairs["z_diff"]).to(device=device, dtype=torch.float64)
    beta = torch.zeros(z_diff.shape[1], device=device, dtype=torch.float64)
    penalty = 2.0 * float(lambda_) * torch.eye(z_diff.shape[1], device=device, dtype=torch.float64)
    initial_condition = None
    condition = None
    iterations = 0
    armijo_steps = 0
    previous_loss = None
    converged = False

    def values(candidate):
        correction = z_diff @ candidate
        signed = p_diff + sample_sd * correction
        margin = labels * signed
        loss = torch.sum(weights * torch.nn.functional.softplus(-margin)) + float(lambda_) * torch.dot(candidate, candidate)
        probability = torch.sigmoid(-margin)
        gradient = z_diff.T @ (weights * (-labels * probability) * sample_sd) + 2.0 * float(lambda_) * candidate
        curvature = weights * probability * (1.0 - probability) * sample_sd * sample_sd
        hessian = z_diff.T @ (z_diff * curvature[:, None]) + penalty
        return loss, gradient, hessian

    def loss_only(candidate):
        margin = labels * (p_diff + sample_sd * (z_diff @ candidate))
        return torch.sum(weights * torch.nn.functional.softplus(-margin)) + float(lambda_) * torch.dot(candidate, candidate)

    while not converged:
        loss, gradient, hessian = values(beta)
        if not (torch.isfinite(loss) and torch.isfinite(gradient).all() and torch.isfinite(hessian).all()):
            raise FloatingPointError("nonfinite pairwise optimization state")
        gradient_inf = float(torch.linalg.vector_norm(gradient, ord=float("inf")).item())
        loss_value = float(loss.item())
        condition = float(torch.linalg.cond(hessian).item())
        if initial_condition is None:
            initial_condition = condition
        if not np.isfinite(condition) or condition > solver["condition_number_max"]:
            raise FloatingPointError(f"Hessian condition gate failed: {condition}")
        if gradient_inf <= solver["gradient_inf_tolerance"]:
            converged = True
            break
        if previous_loss is not None and abs(previous_loss - loss_value) <= solver["relative_loss_tolerance"] * max(1.0, abs(previous_loss)) and gradient_inf <= solver["loss_stopping_gradient_inf_max"]:
            converged = True
            break
        direction = torch.linalg.solve(hessian, -gradient)
        print(json.dumps({"lambda": float(lambda_), "iteration": iterations, "loss": loss_value, "gradient_inf": gradient_inf, "condition": condition}), flush=True)
        directional = torch.dot(gradient, direction)
        if not torch.isfinite(direction).all() or float(directional.item()) >= 0:
            raise FloatingPointError("Newton direction is not finite descent")
        step = 1.0
        while True:
            candidate = beta + step * direction
            candidate_loss = loss_only(candidate)
            armijo_steps += 1
            if torch.isfinite(candidate_loss) and float(candidate_loss.item()) < loss_value and float(candidate_loss.item()) <= loss_value + 1e-4 * float(step) * float(directional.item()):
                break
            step = 0.5 * step
            if float(step) == 0.0:
                raise FloatingPointError("Armijo step underflow before convergence")
        previous_loss = loss_value
        beta = candidate
        iterations += 1
        print(json.dumps({"lambda": float(lambda_), "accepted_iteration": iterations, "step": float(step), "armijo_evaluations_total": armijo_steps}), flush=True)
    final_loss, final_gradient, final_hessian = values(beta)
    final_gradient_inf = float(torch.linalg.vector_norm(final_gradient, ord=float("inf")).item())
    condition = float(torch.linalg.cond(final_hessian).item())
    coefficients = beta.detach().cpu().numpy()
    report = {
        "lambda": float(lambda_), "converged": converged, "iterations": iterations,
        "armijo_evaluations": armijo_steps, "initial_hessian_condition": initial_condition,
        "final_hessian_condition": condition, "gradient_inf": final_gradient_inf,
        "loss": float(final_loss.item()), "coefficients_nonintercept": coefficients.tolist(),
        "train_pairs": len(pairs["labels"]), "device": str(device), "dtype": "torch.float64",
        "finite": bool(np.all(np.isfinite(coefficients)) and np.isfinite(condition) and np.isfinite(final_loss.item())),
        "gpu": gpu, "scientific_iteration_cap": None,
    }
    if not converged or not report["finite"] or condition > solver["condition_number_max"]:
        raise FloatingPointError(f"optimization gate failed: {report}")
    return coefficients, report


def cluster_bootstrap(pool_seed_regret, seed, replicates=10000):
    values = np.asarray(pool_seed_regret, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("pool by seed regret matrix required")
    clusters = values.mean(axis=1)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    draws = clusters[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"effect": float(clusters.mean()), "ci_low": float(low), "ci_high": float(high), "clusters": len(clusters), "replicates": replicates, "seed": seed}
