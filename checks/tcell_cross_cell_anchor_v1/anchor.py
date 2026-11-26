from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np


ANALYSIS_ID = "tcell_cross_cell_anchor_v1"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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


def valid_sequence(value):
    return len(value) == 50 and not set(value) - set("ACGT")


def eligible_hash_order(sequences):
    return sorted(sequences, key=lambda value: hashlib.sha256(f"{ANALYSIS_ID}|{value}".encode()).digest())


def isolate_h1(sequences, required):
    retained = []
    index = HalfIndex()
    for sequence in eligible_hash_order(sequences):
        exact, h1 = index.match(sequence)
        if exact or h1:
            continue
        retained.append(sequence)
        index.add(sequence)
        if len(retained) == required:
            return retained
    raise ValueError(f"insufficient H1-isolated eligible sequences: {len(retained)} < {required}")


def make_proposals(pool_count, pool_size, seeds, cap=16):
    proposals = np.empty((pool_count, len(seeds), cap), dtype=np.int64)
    for pool_index in range(pool_count):
        for seed_index, seed in enumerate(seeds):
            rng = random.Random(f"{ANALYSIS_ID}|{pool_index}|{seed}")
            proposals[pool_index, seed_index] = [min(pool_size - 1, int(rng.random() * pool_size)) for _ in range(cap)]
    return proposals


def cap_regret(scores, labels, pools, proposals, cap):
    prefix = proposals[:, :, :cap]
    pool_scores = np.asarray(scores)[pools]
    pool_labels = np.asarray(labels)[pools]
    proposed_scores = np.take_along_axis(pool_scores[:, None, :], prefix, axis=2)
    proposed_labels = np.take_along_axis(pool_labels[:, None, :], prefix, axis=2)
    selected = np.argmax(proposed_scores, axis=2)
    selected_labels = np.take_along_axis(proposed_labels, selected[..., None], axis=2)[..., 0]
    regret = pool_labels.max(axis=1)[:, None] - selected_labels
    return regret, selected


def random_regret(labels, pools, proposals, cap):
    prefix = proposals[:, :, :cap]
    selected = np.zeros(prefix.shape[:2], dtype=np.int64)
    pool_labels = np.asarray(labels)[pools]
    selected_labels = np.take_along_axis(pool_labels[:, None, :], prefix, axis=2)[:, :, 0]
    return pool_labels.max(axis=1)[:, None] - selected_labels, selected


def best_average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("finite vector with at least two items required")
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 1.0 - ((start + end - 1) / 2.0) / (len(values) - 1)
        start = end
    return ranks


def fusion_scores(cnn, hepg2, pools):
    fused = np.empty(len(cnn), dtype=np.float64)
    cnn_ranks = np.empty(len(cnn), dtype=np.float64)
    hepg2_ranks = np.empty(len(cnn), dtype=np.float64)
    for pool_index, pool in enumerate(pools):
        cnn_ranks[pool] = best_average_ranks(np.asarray(cnn)[pool])
        hepg2_ranks[pool] = best_average_ranks(np.asarray(hepg2)[pool])
        fused[pool] = 0.5 * cnn_ranks[pool] + 0.5 * hepg2_ranks[pool]
    return fused, cnn_ranks, hepg2_ranks


def bootstrap_clusters(cluster_values, seed, replicates=10000):
    values = np.asarray(cluster_values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("finite bootstrap clusters required")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = values[indices].mean(axis=1)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    p_value = min(1.0, 2.0 * min(
        (1 + int(np.sum(draws <= 0))) / (replicates + 1),
        (1 + int(np.sum(draws >= 0))) / (replicates + 1),
    ))
    return {
        "effect": float(values.mean()), "ci_low": float(lower), "ci_high": float(upper),
        "p_two_sided": float(p_value), "clusters": len(values),
        "replicates": replicates, "seed": seed,
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_rank = best_average_ranks(left)
    right_rank = best_average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def oracle_decomposition(labels, pools, proposals, selected, cap):
    prefix = proposals[:, :, :cap]
    pool_labels = np.asarray(labels)[pools]
    proposed_labels = np.take_along_axis(pool_labels[:, None, :], prefix, axis=2)
    selected_labels = np.take_along_axis(proposed_labels, selected[..., None], axis=2)[..., 0]
    proposal_best = proposed_labels.max(axis=2)
    pool_best = pool_labels.max(axis=1)[:, None]
    opportunity = proposal_best - selected_labels
    headroom = pool_best - proposal_best
    if not np.allclose(opportunity + headroom, pool_best - selected_labels, rtol=0, atol=1e-12):
        raise FloatingPointError("oracle decomposition mismatch")
    return opportunity, headroom
