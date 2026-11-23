import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .pool import Candidate, CandidatePool


class Strategy(str, Enum):
    RANDOM = "random"
    BON = "bon"
    WSC = "wsc"
    GREEDY = "greedy"
    V_ONLY = "v-only"
    JOINT = "joint"


@dataclass(frozen=True)
class Budget:
    max_proposals: int
    max_proxy_calls: Optional[int] = None
    max_verifier_calls: Optional[int] = None
    max_qhat_calls: Optional[int] = None

    def __post_init__(self) -> None:
        limits = (
            self.max_proposals,
            self.max_proxy_calls,
            self.max_verifier_calls,
            self.max_qhat_calls,
        )
        for limit in limits:
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
                raise ValueError("budget limits must be positive integers or None")


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    _FIELDS = {
        "proposals": "max_proposals",
        "proxy_calls": "max_proxy_calls",
        "verifier_calls": "max_verifier_calls",
        "qhat_calls": "max_qhat_calls",
    }

    def __init__(self, limits: Budget) -> None:
        self.limits = limits
        self._counts = {
            field: 0
            for field in (
                "proposals",
                "proxy_calls",
                "verifier_calls",
                "qhat_calls",
                "accepted_proposals",
                "fallbacks",
            )
        }

    def charge(self, field: str) -> None:
        if field not in self._counts:
            raise KeyError(field)
        if field in self._FIELDS:
            limit = getattr(self.limits, self._FIELDS[field])
            if limit is not None and self._counts[field] >= limit:
                raise BudgetExceeded(f"{field} exceeds limit {limit}")
        self._counts[field] += 1

    def snapshot(self) -> Dict[str, int]:
        return dict(self._counts)


@dataclass(frozen=True)
class SelectionResult:
    strategy: str
    selected_index: int
    selected_id: str
    calls: Dict[str, int]
    proposed_indices: Tuple[int, ...]
    proposed_weights: Tuple[float, ...]
    accepted: bool
    used_fallback: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "selected_index": self.selected_index,
            "selected_id": self.selected_id,
            "calls": dict(self.calls),
            "proposed_indices": list(self.proposed_indices),
            "proposed_weights": list(self.proposed_weights),
            "accepted": self.accepted,
            "used_fallback": self.used_fallback,
        }


Scorer = Callable[[Candidate], float]
PoolScorer = Callable[[], float]


def _draw_index(probabilities: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    last_positive = None
    for index, probability in enumerate(probabilities):
        if probability > 0:
            last_positive = index
        cumulative += probability
        if threshold < cumulative:
            return index
    if last_positive is None:
        raise ValueError("at least one proposal probability must be positive")
    return last_positive


def _propose(pool: CandidatePool, budget: CallBudget, rng: random.Random) -> int:
    budget.charge("proposals")
    index = _draw_index(pool.masked_proposal_probs(), rng)
    if not pool.feasible[index]:
        raise AssertionError("constraint mask allowed an infeasible proposal")
    return index


def _score(
    scorer: Scorer, pool: CandidatePool, index: int, budget: CallBudget, field: str, bounded: bool
) -> float:
    budget.charge(field)
    value = float(scorer(pool.candidate(index)))
    if not math.isfinite(value) or (bounded and not 0 <= value <= 1):
        domain = "[0, 1]" if bounded else "finite"
        raise ValueError(f"scorer returned {value}; expected {domain}")
    return value


def _qhat(scorer: PoolScorer, budget: CallBudget) -> float:
    budget.charge("qhat_calls")
    value = float(scorer())
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("qhat scorer must return a value in [0, 1]")
    return value


def _weighted_position(weights: Sequence[float], rng: random.Random) -> int:
    if not weights or any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("weights must be non-empty, finite and non-negative")
    total = math.fsum(weights)
    if total == 0:
        return rng.randrange(len(weights))
    threshold = rng.random() * total
    cumulative = 0.0
    last_positive = None
    for index, weight in enumerate(weights):
        if weight > 0:
            last_positive = index
        cumulative += weight
        if threshold < cumulative:
            return index
    return last_positive


def _result(
    pool: CandidatePool,
    strategy: Strategy,
    selected_index: int,
    budget: CallBudget,
    proposed_indices: Sequence[int],
    proposed_weights: Sequence[float],
    accepted: bool,
    used_fallback: bool,
) -> SelectionResult:
    return SelectionResult(
        strategy=strategy.value,
        selected_index=selected_index,
        selected_id=pool.sequence_ids[selected_index],
        calls=budget.snapshot(),
        proposed_indices=tuple(proposed_indices),
        proposed_weights=tuple(proposed_weights),
        accepted=accepted,
        used_fallback=used_fallback,
    )


def select(
    pool: CandidatePool,
    strategy: Strategy,
    budget_limits: Budget,
    rng: random.Random,
    alpha: float = 1.0,
    proxy: Optional[Scorer] = None,
    verifier: Optional[Scorer] = None,
    qhat: Optional[PoolScorer] = None,
) -> SelectionResult:
    strategy = Strategy(strategy)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    required = {
        Strategy.BON: (proxy, "proxy"),
        Strategy.WSC: (verifier, "verifier"),
        Strategy.GREEDY: (verifier, "verifier"),
        Strategy.V_ONLY: (verifier, "verifier"),
        Strategy.JOINT: (verifier, "verifier"),
    }
    scorer, name = required.get(strategy, (object(), ""))
    if scorer is None:
        raise ValueError(f"{strategy.value} requires {name}")
    if strategy in (Strategy.GREEDY, Strategy.JOINT) and qhat is None:
        raise ValueError(f"{strategy.value} requires qhat")

    budget = CallBudget(budget_limits)
    proposed_indices: List[int] = []
    proposed_weights: List[float] = []

    if strategy == Strategy.RANDOM:
        index = _propose(pool, budget, rng)
        return _result(pool, strategy, index, budget, [index], [], False, False)

    if strategy == Strategy.BON:
        best_index = -1
        best_score = -math.inf
        for _ in range(budget_limits.max_proposals):
            index = _propose(pool, budget, rng)
            score = _score(proxy, pool, index, budget, "proxy_calls", False)
            proposed_indices.append(index)
            proposed_weights.append(score)
            if score > best_score:
                best_index = index
                best_score = score
        return _result(
            pool, strategy, best_index, budget, proposed_indices, proposed_weights, False, False
        )

    if strategy == Strategy.GREEDY:
        q_value = _qhat(qhat, budget)
        best_index = -1
        best_weight = -math.inf
        for _ in range(budget_limits.max_proposals):
            index = _propose(pool, budget, rng)
            v_value = _score(verifier, pool, index, budget, "verifier_calls", True)
            weight = (v_value * q_value) ** alpha
            proposed_indices.append(index)
            proposed_weights.append(weight)
            if weight > best_weight:
                best_index = index
                best_weight = weight
        return _result(
            pool, strategy, best_index, budget, proposed_indices, proposed_weights, False, False
        )

    if strategy == Strategy.WSC:
        for _ in range(budget_limits.max_proposals):
            index = _propose(pool, budget, rng)
            v_value = _score(verifier, pool, index, budget, "verifier_calls", True)
            proposed_indices.append(index)
            proposed_weights.append(v_value ** alpha)
        position = _weighted_position(proposed_weights, rng)
        return _result(
            pool,
            strategy,
            proposed_indices[position],
            budget,
            proposed_indices,
            proposed_weights,
            False,
            False,
        )

    q_value = _qhat(qhat, budget) if strategy == Strategy.JOINT else 1.0
    for _ in range(budget_limits.max_proposals):
        index = _propose(pool, budget, rng)
        v_value = _score(verifier, pool, index, budget, "verifier_calls", True)
        weight = v_value ** alpha if strategy == Strategy.V_ONLY else (v_value * q_value) ** alpha
        proposed_indices.append(index)
        proposed_weights.append(weight)
        if rng.random() < weight:
            budget.charge("accepted_proposals")
            return _result(
                pool, strategy, index, budget, proposed_indices, proposed_weights, True, False
            )

    fallback_weights = [weight / (1.0 - weight) for weight in proposed_weights]
    position = _weighted_position(fallback_weights, rng)
    budget.charge("fallbacks")
    return _result(
        pool,
        strategy,
        proposed_indices[position],
        budget,
        proposed_indices,
        proposed_weights,
        False,
        True,
    )
