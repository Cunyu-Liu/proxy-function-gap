import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

from .pool import Candidate, CandidatePool


class Aggregation(str, Enum):
    PRODUCT = "product"
    MIN = "min"
    MEAN = "mean"


@dataclass(frozen=True)
class Budget:
    max_proposals: Optional[int] = None
    max_proxy_item_forwards: Optional[int] = None
    max_verifier_item_forwards: Optional[int] = None
    max_qhat_item_forwards: Optional[int] = None

    def __post_init__(self) -> None:
        for limit in (
            self.max_proposals,
            self.max_proxy_item_forwards,
            self.max_verifier_item_forwards,
            self.max_qhat_item_forwards,
        ):
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
                raise ValueError("budget limits must be positive integers or None")


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    _LIMITS = {
        "proposals": "max_proposals",
        "proxy_item_forwards": "max_proxy_item_forwards",
        "verifier_item_forwards": "max_verifier_item_forwards",
        "qhat_item_forwards": "max_qhat_item_forwards",
    }
    _FIELDS = (
        "proposals",
        "proxy_item_forwards",
        "verifier_item_forwards",
        "qhat_item_forwards",
        "proxy_scorer_batches",
        "verifier_scorer_batches",
        "proxy_cache_hits",
        "proxy_cache_misses",
        "verifier_cache_hits",
        "verifier_cache_misses",
        "accepted_proposals",
        "rejected_proposals",
        "fallbacks",
    )

    def __init__(self, limits: Budget) -> None:
        self.limits = limits
        self._counts = {field: 0 for field in self._FIELDS}

    def charge(self, field: str, amount: int = 1) -> None:
        if field not in self._counts:
            raise KeyError(field)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise ValueError("charge amount must be a positive integer")
        if field in self._LIMITS:
            limit = getattr(self.limits, self._LIMITS[field])
            if limit is not None and self._counts[field] + amount > limit:
                raise BudgetExceeded(f"{field} exceeds limit {limit}")
        self._counts[field] += amount

    def snapshot(self) -> Dict[str, int]:
        return dict(self._counts)

    def delta(self, start: Dict[str, int]) -> Dict[str, int]:
        return {field: self._counts[field] - start[field] for field in self._FIELDS}


class ScoreCache:
    def __init__(self) -> None:
        self._values = {}
        self._missing = object()

    def get(self, kind: str, sequence_id: str):
        return self._values.get((kind, sequence_id), self._missing)

    def put(self, kind: str, sequence_id: str, value: float) -> None:
        self._values[(kind, sequence_id)] = value

    def contains(self, kind: str, sequence_id: str) -> bool:
        return (kind, sequence_id) in self._values


@dataclass(frozen=True)
class BatchScorer:
    scorer: Callable[[Sequence[Candidate]], Sequence[float]]


ItemScorer = Callable[[Candidate], float]
Scorer = Union[ItemScorer, BatchScorer]
ScorerSpec = Union[Scorer, Sequence[Scorer]]
PoolScorer = Callable[[], float]


def _members(spec: ScorerSpec) -> Tuple[Scorer, ...]:
    if isinstance(spec, BatchScorer) or callable(spec):
        members = (spec,)
    else:
        members = tuple(spec)
    if not members or any(not (isinstance(member, BatchScorer) or callable(member)) for member in members):
        raise ValueError("scorer ensembles must be non-empty and callable")
    return members


def _validate_score(value: float, bounded: bool) -> float:
    value = float(value)
    if not math.isfinite(value) or (bounded and not 0 <= value <= 1):
        domain = "[0, 1]" if bounded else "finite"
        raise ValueError(f"scorer returned {value}; expected {domain}")
    return value


def _score_uncached(
    pool: CandidatePool,
    indices: Sequence[int],
    members: Tuple[Scorer, ...],
    budget: CallBudget,
    kind: str,
) -> Tuple[float, ...]:
    candidates = [pool.candidate(index) for index in indices]
    member_values = []
    for member in members:
        budget.charge(f"{kind}_item_forwards", len(candidates))
        if isinstance(member, BatchScorer):
            budget.charge(f"{kind}_scorer_batches")
            values = list(member.scorer(candidates))
            if len(values) != len(candidates):
                raise ValueError("batch scorer output length must match input length")
        else:
            budget.charge(f"{kind}_scorer_batches", len(candidates))
            values = [member(candidate) for candidate in candidates]
        member_values.append([
            _validate_score(value, kind == "verifier") for value in values
        ])
    return tuple(
        math.fsum(member_values[member][position] for member in range(len(members))) / len(members)
        for position in range(len(candidates))
    )


def score_candidates(
    pool: CandidatePool,
    indices: Sequence[int],
    spec: ScorerSpec,
    budget: CallBudget,
    kind: str,
    cache: Optional[ScoreCache] = None,
) -> Tuple[float, ...]:
    if kind not in ("proxy", "verifier"):
        raise ValueError("score kind must be proxy or verifier")
    members = _members(spec)
    if not indices:
        return ()
    if cache is None:
        return _score_uncached(pool, indices, members, budget, kind)

    output: Dict[int, float] = {}
    missing = []
    missing_seen = set()
    for index in indices:
        sequence_id = pool.sequence_ids[index]
        if cache.contains(kind, sequence_id):
            budget.charge(f"{kind}_cache_hits")
            output[index] = cache.get(kind, sequence_id)
        elif index in missing_seen:
            budget.charge(f"{kind}_cache_hits")
        else:
            missing_seen.add(index)
            budget.charge(f"{kind}_cache_misses")
            missing.append(index)
    if missing:
        values = _score_uncached(pool, missing, members, budget, kind)
        for index, value in zip(missing, values):
            cache.put(kind, pool.sequence_ids[index], value)
    for index in indices:
        if index not in output:
            output[index] = cache.get(kind, pool.sequence_ids[index])
    return tuple(output[index] for index in indices)


def score_qhat(scorer: PoolScorer, budget: CallBudget) -> float:
    budget.charge("qhat_item_forwards")
    value = float(scorer())
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("qhat scorer must return a probability in [0, 1]")
    return value


def guidance_weight(
    verifier: float,
    qhat: Optional[float],
    alpha: float,
    aggregation: Aggregation = Aggregation.PRODUCT,
) -> float:
    if not math.isfinite(verifier) or not 0 <= verifier <= 1:
        raise ValueError("verifier must be a probability in [0, 1]")
    if qhat is not None and (not math.isfinite(qhat) or not 0 <= qhat <= 1):
        raise ValueError("qhat must be a probability in [0, 1]")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    aggregation = Aggregation(aggregation)
    if alpha == 0:
        return 1.0
    if qhat is None:
        return verifier ** alpha
    if aggregation == Aggregation.PRODUCT:
        combined = verifier * qhat
    elif aggregation == Aggregation.MIN:
        combined = min(verifier, qhat)
    else:
        combined = (verifier + qhat) / 2.0
    return combined ** alpha
