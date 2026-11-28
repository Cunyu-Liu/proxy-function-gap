from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cap16_regret(scores, labels, pools, proposals):
    pool_scores = scores[pools]
    pool_labels = labels[pools]
    proposed_scores = np.take_along_axis(pool_scores[:, None, :], proposals, axis=2)
    proposed_labels = np.take_along_axis(pool_labels[:, None, :], proposals, axis=2)
    selected = np.argmax(proposed_scores, axis=2)
    selected_labels = np.take_along_axis(proposed_labels, selected[..., None], axis=2)[..., 0]
    regrets = pool_labels.max(axis=1)[:, None] - selected_labels
    return {"mean_regret": float(regrets.mean()), "pool_seed_regret": regrets}


def member_std(members) -> np.ndarray:
    return np.asarray(members, dtype=np.float64).std(axis=1, ddof=1)


def upts_score(p, std, kappa: float) -> np.ndarray:
    return np.asarray(p, dtype=np.float64) - float(kappa) * np.asarray(std, dtype=np.float64)


def _ranks(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def top16_by_score(pool_scores: np.ndarray) -> np.ndarray:
    return np.lexsort((np.arange(len(pool_scores)), -np.asarray(pool_scores, dtype=np.float64)))[:16]