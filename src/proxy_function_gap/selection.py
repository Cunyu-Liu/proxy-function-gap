import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from .budget import (
    Aggregation,
    Budget,
    CallBudget,
    PoolScorer,
    ScoreCache,
    ScorerSpec,
    guidance_weight,
    score_candidates,
    score_qhat,
)
from .pool import CandidatePool, ZeroNormalizerError


class Strategy(str, Enum):
    RANDOM = "random"
    BON = "bon"
    WSC = "wsc"
    GREEDY = "greedy"
    FP_TILT = "fp-tilt"
    FP_REJ = "fp-rej"


class ZeroWeightPolicy(str, Enum):
    ABSTAIN = "abstain"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class SelectionResult:
    strategy: str
    selected_indices: Tuple[int, ...]
    selected_ids: Tuple[str, ...]
    selected_scores: Tuple[float, ...]
    calls: Dict[str, int]
    proposed_indices: Tuple[int, ...]
    proposed_weights: Tuple[float, ...]
    accepted: bool
    used_fallback: bool
    abstain_reason: Optional[str]

    @property
    def selected_index(self) -> Optional[int]:
        return self.selected_indices[0] if self.selected_indices else None

    @property
    def selected_id(self) -> Optional[str]:
        return self.selected_ids[0] if self.selected_ids else None

    @property
    def abstain(self) -> bool:
        return not self.selected_indices

    @property
    def acceptance_rate(self) -> float:
        return self.calls["accepted_proposals"] / self.calls["proposals"] if self.calls["proposals"] else 0.0

    @property
    def fallback_rate(self) -> float:
        return float(self.calls["fallbacks"])

    @property
    def coverage(self) -> float:
        return 0.0 if self.abstain else 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "selected_indices": list(self.selected_indices),
            "selected_ids": list(self.selected_ids),
            "selected_scores": list(self.selected_scores),
            "calls": dict(self.calls),
            "proposed_indices": list(self.proposed_indices),
            "proposed_weights": list(self.proposed_weights),
            "accepted": self.accepted,
            "used_fallback": self.used_fallback,
            "abstain": self.abstain,
            "abstain_reason": self.abstain_reason,
            "acceptance_rate": self.acceptance_rate,
            "fallback_rate": self.fallback_rate,
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class MultiTrajectoryResult:
    strategy: str
    selected_indices: Tuple[int, ...]
    selected_ids: Tuple[str, ...]
    selected_scores: Tuple[float, ...]
    calls: Dict[str, int]
    trajectory_results: Tuple[SelectionResult, ...]
    abstain_reason: Optional[str]

    @property
    def selected_index(self) -> Optional[int]:
        return self.selected_indices[0] if self.selected_indices else None

    @property
    def selected_id(self) -> Optional[str]:
        return self.selected_ids[0] if self.selected_ids else None

    @property
    def abstain(self) -> bool:
        return not self.selected_indices

    @property
    def acceptance_rate(self) -> float:
        return self.calls["accepted_proposals"] / self.calls["proposals"] if self.calls["proposals"] else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.calls["fallbacks"] / len(self.trajectory_results)

    @property
    def coverage(self) -> float:
        return sum(not result.abstain for result in self.trajectory_results) / len(self.trajectory_results)

    def to_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "selected_indices": list(self.selected_indices),
            "selected_ids": list(self.selected_ids),
            "selected_scores": list(self.selected_scores),
            "calls": dict(self.calls),
            "abstain": self.abstain,
            "abstain_reason": self.abstain_reason,
            "acceptance_rate": self.acceptance_rate,
            "fallback_rate": self.fallback_rate,
            "coverage": self.coverage,
            "trajectories": [result.to_dict() for result in self.trajectory_results],
        }


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
        raise ValueError("at least one probability must be positive")
    return last_positive


def _weighted_position(weights: Sequence[float], rng: random.Random) -> int:
    if not weights or any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("weights must be non-empty, finite and non-negative")
    total = math.fsum(weights)
    if total == 0:
        raise ZeroNormalizerError("weighted choice has zero normalizer")
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


def _propose(pool: CandidatePool, budget: CallBudget, rng: random.Random) -> int:
    budget.charge("proposals")
    index = _draw_index(pool.masked_proposal_probs(), rng)
    if not pool.feasible[index]:
        raise AssertionError("constraint mask allowed an infeasible proposal")
    return index


def _rank_identities(
    indices: Sequence[int], scores: Sequence[float], top_k: int, aggregate: str
) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    first_position = {}
    identity_scores = {}
    for position, (index, score) in enumerate(zip(indices, scores)):
        if index not in identity_scores:
            first_position[index] = position
            identity_scores[index] = score
        elif aggregate == "sum":
            identity_scores[index] += score
        else:
            identity_scores[index] = max(identity_scores[index], score)
    ordered = sorted(identity_scores, key=lambda index: (-identity_scores[index], first_position[index]))[:top_k]
    return tuple(ordered), tuple(identity_scores[index] for index in ordered)


def _result(
    pool: CandidatePool,
    strategy: Strategy,
    selected_indices: Sequence[int],
    selected_scores: Sequence[float],
    budget: CallBudget,
    start: Dict[str, int],
    proposed_indices: Sequence[int],
    proposed_weights: Sequence[float],
    accepted: bool = False,
    used_fallback: bool = False,
    abstain_reason: Optional[str] = None,
) -> SelectionResult:
    indices = tuple(selected_indices)
    if len(set(indices)) != len(indices):
        raise AssertionError("top-k contains duplicate sequence identities")
    return SelectionResult(
        strategy=strategy.value,
        selected_indices=indices,
        selected_ids=tuple(pool.sequence_ids[index] for index in indices),
        selected_scores=tuple(selected_scores),
        calls=budget.delta(start),
        proposed_indices=tuple(proposed_indices),
        proposed_weights=tuple(proposed_weights),
        accepted=accepted,
        used_fallback=used_fallback,
        abstain_reason=abstain_reason,
    )


def _zero_weight_result(
    pool: CandidatePool,
    strategy: Strategy,
    eligible_indices: Sequence[int],
    top_k: int,
    policy: ZeroWeightPolicy,
    budget: CallBudget,
    start: Dict[str, int],
    proposed_indices: Sequence[int],
    proposed_weights: Sequence[float],
    rng: random.Random,
    fallback_already_counted: bool = False,
) -> SelectionResult:
    if policy == ZeroWeightPolicy.ABSTAIN:
        return _result(
            pool,
            strategy,
            (),
            (),
            budget,
            start,
            proposed_indices,
            proposed_weights,
            abstain_reason="zero_guidance_weight",
        )
    unique = list(dict.fromkeys(eligible_indices))
    if not unique:
        raise ValueError("uniform fallback requires at least one eligible identity")
    rng.shuffle(unique)
    selected = unique[:top_k]
    if not fallback_already_counted:
        budget.charge("fallbacks")
    return _result(
        pool,
        strategy,
        selected,
        [0.0] * len(selected),
        budget,
        start,
        proposed_indices,
        proposed_weights,
        used_fallback=True,
    )


def _random_without_replacement(
    pool: CandidatePool,
    top_k: int,
    proposal_cap: int,
    budget: CallBudget,
    start: Dict[str, int],
    rng: random.Random,
) -> SelectionResult:
    eligible = [index for index, p in enumerate(pool.masked_proposal_probs()) if p > 0]
    if top_k > len(eligible) or proposal_cap < top_k:
        raise ValueError("random top-k requires enough proposal cap and positive-mass identities")
    remaining = list(eligible)
    selected = []
    for _ in range(top_k):
        probabilities = [pool.masked_proposal_probs()[index] for index in remaining]
        total = math.fsum(probabilities)
        position = _draw_index([p / total for p in probabilities], rng)
        budget.charge("proposals")
        selected.append(remaining.pop(position))
    return _result(pool, Strategy.RANDOM, selected, (), budget, start, selected, ())


def _select_with_budget(
    pool: CandidatePool,
    strategy: Strategy,
    proposal_cap: int,
    budget: CallBudget,
    rng: random.Random,
    alpha: float,
    aggregation: Aggregation,
    zero_weight_policy: ZeroWeightPolicy,
    top_k: int,
    proxy: Optional[ScorerSpec],
    verifier: Optional[ScorerSpec],
    qhat: Optional[PoolScorer],
    abstain_below_q: Optional[float],
    cache: Optional[ScoreCache],
) -> SelectionResult:
    start = budget.snapshot()
    if isinstance(proposal_cap, bool) or not isinstance(proposal_cap, int) or proposal_cap <= 0:
        raise ValueError("proposal_cap must be a positive integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if abstain_below_q is not None and not 0 <= abstain_below_q <= 1:
        raise ValueError("abstain_below_q must be in [0, 1]")
    strategy = Strategy(strategy)
    aggregation = Aggregation(aggregation)
    zero_weight_policy = ZeroWeightPolicy(zero_weight_policy)

    q_value = None
    q_for_gate = abstain_below_q is not None
    q_for_rejection = strategy == Strategy.FP_REJ and qhat is not None and alpha > 0
    if q_for_gate or q_for_rejection:
        if qhat is None:
            raise ValueError("qhat is required for coverage gating or joint FP-REJ")
        q_value = score_qhat(qhat, budget)
        if q_for_gate and q_value < abstain_below_q:
            return _result(
                pool, strategy, (), (), budget, start, (), (), abstain_reason="qhat_below_threshold"
            )

    if strategy == Strategy.RANDOM:
        return _random_without_replacement(pool, top_k, proposal_cap, budget, start, rng)

    if strategy == Strategy.FP_TILT:
        base = pool.masked_proposal_probs()
        eligible = [index for index, probability in enumerate(base) if probability > 0]
        if top_k > len(eligible):
            raise ValueError("FP-TILT top-k exceeds positive-mass identities")
        if alpha == 0:
            distribution = base
        else:
            if verifier is None:
                raise ValueError("FP-TILT requires verifier")
            scores = score_candidates(pool, eligible, verifier, budget, "verifier", cache)
            full_scores = [0.0] * len(pool)
            for index, score in zip(eligible, scores):
                full_scores[index] = score
            try:
                distribution = pool.exact_tilted_distribution(full_scores, alpha)
            except ZeroNormalizerError:
                return _zero_weight_result(
                    pool, strategy, eligible, top_k, zero_weight_policy, budget, start, (), (), rng
                )
        ranked = sorted(eligible, key=lambda index: (-distribution[index], index))[:top_k]
        return _result(
            pool, strategy, ranked, [distribution[index] for index in ranked], budget, start, (), ()
        )

    if strategy == Strategy.FP_REJ:
        if top_k != 1:
            raise ValueError("single FP-REJ is top-1; use select_trajectories for top-k identity voting")
        if alpha > 0 and verifier is None:
            raise ValueError("FP-REJ requires verifier when alpha > 0")
        proposed_indices = []
        proposed_weights = []
        for _ in range(proposal_cap):
            index = _propose(pool, budget, rng)
            if alpha == 0:
                weight = 1.0
            else:
                v_value = score_candidates(pool, [index], verifier, budget, "verifier", cache)[0]
                weight = guidance_weight(v_value, q_value, alpha, aggregation)
            proposed_indices.append(index)
            proposed_weights.append(weight)
            if rng.random() < weight:
                budget.charge("accepted_proposals")
                return _result(
                    pool,
                    strategy,
                    [index],
                    [weight],
                    budget,
                    start,
                    proposed_indices,
                    proposed_weights,
                    accepted=True,
                )
            budget.charge("rejected_proposals")
        odds = [weight / (1.0 - weight) for weight in proposed_weights]
        if math.fsum(odds) == 0:
            budget.charge("fallbacks")
            return _zero_weight_result(
                pool,
                strategy,
                proposed_indices,
                1,
                zero_weight_policy,
                budget,
                start,
                proposed_indices,
                proposed_weights,
                rng,
                fallback_already_counted=True,
            )
        position = _weighted_position(odds, rng)
        budget.charge("fallbacks")
        index = proposed_indices[position]
        return _result(
            pool,
            strategy,
            [index],
            [proposed_weights[position]],
            budget,
            start,
            proposed_indices,
            proposed_weights,
            used_fallback=True,
        )

    if strategy == Strategy.BON and proxy is None:
        raise ValueError("BoN requires proxy")
    if strategy == Strategy.GREEDY and verifier is None:
        raise ValueError("greedy requires verifier")
    if strategy == Strategy.WSC and alpha > 0 and verifier is None:
        raise ValueError("WSC requires verifier when alpha > 0")

    proposed_indices = [_propose(pool, budget, rng) for _ in range(proposal_cap)]
    if strategy == Strategy.BON:
        scores = score_candidates(pool, proposed_indices, proxy, budget, "proxy", cache)
        selected, selected_scores = _rank_identities(proposed_indices, scores, top_k, "max")
        return _result(
            pool, strategy, selected, selected_scores, budget, start, proposed_indices, scores
        )
    if strategy == Strategy.WSC:
        if alpha == 0:
            weights = [1.0] * len(proposed_indices)
        else:
            v_values = score_candidates(pool, proposed_indices, verifier, budget, "verifier", cache)
            weights = [guidance_weight(value, None, alpha) for value in v_values]
        if math.fsum(weights) == 0:
            return _zero_weight_result(
                pool,
                strategy,
                proposed_indices,
                top_k,
                zero_weight_policy,
                budget,
                start,
                proposed_indices,
                weights,
                rng,
            )
        selected, selected_scores = _rank_identities(proposed_indices, weights, top_k, "sum")
        return _result(
            pool, strategy, selected, selected_scores, budget, start, proposed_indices, weights
        )

    rewards = score_candidates(pool, proposed_indices, verifier, budget, "verifier", cache)
    if math.fsum(rewards) == 0:
        return _zero_weight_result(
            pool,
            strategy,
            proposed_indices,
            top_k,
            zero_weight_policy,
            budget,
            start,
            proposed_indices,
            rewards,
            rng,
        )
    selected, selected_scores = _rank_identities(proposed_indices, rewards, top_k, "max")
    return _result(pool, strategy, selected, selected_scores, budget, start, proposed_indices, rewards)


def select(
    pool: CandidatePool,
    strategy: Strategy,
    proposal_cap: int,
    rng: random.Random,
    budget_limits: Optional[Budget] = None,
    alpha: float = 1.0,
    aggregation: Aggregation = Aggregation.PRODUCT,
    zero_weight_policy: ZeroWeightPolicy = ZeroWeightPolicy.ABSTAIN,
    top_k: int = 1,
    proxy: Optional[ScorerSpec] = None,
    verifier: Optional[ScorerSpec] = None,
    qhat: Optional[PoolScorer] = None,
    abstain_below_q: Optional[float] = None,
    cache: Optional[ScoreCache] = None,
) -> SelectionResult:
    budget = CallBudget(budget_limits or Budget())
    return _select_with_budget(
        pool,
        Strategy(strategy),
        proposal_cap,
        budget,
        rng,
        alpha,
        Aggregation(aggregation),
        ZeroWeightPolicy(zero_weight_policy),
        top_k,
        proxy,
        verifier,
        qhat,
        abstain_below_q,
        cache,
    )


def select_trajectories(
    pool: CandidatePool,
    strategy: Strategy,
    proposal_cap: int,
    trajectories: int,
    top_k: int,
    rng: random.Random,
    budget_limits: Optional[Budget] = None,
    alpha: float = 1.0,
    aggregation: Aggregation = Aggregation.PRODUCT,
    zero_weight_policy: ZeroWeightPolicy = ZeroWeightPolicy.ABSTAIN,
    proxy: Optional[ScorerSpec] = None,
    verifier: Optional[ScorerSpec] = None,
    qhat: Optional[PoolScorer] = None,
    abstain_below_q: Optional[float] = None,
    cache: Optional[ScoreCache] = None,
) -> MultiTrajectoryResult:
    if isinstance(trajectories, bool) or not isinstance(trajectories, int) or trajectories <= 0:
        raise ValueError("trajectories must be a positive integer")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    strategy = Strategy(strategy)
    budget = CallBudget(budget_limits or Budget())
    results = []
    for _ in range(trajectories):
        results.append(
            _select_with_budget(
                pool,
                strategy,
                proposal_cap,
                budget,
                rng,
                alpha,
                Aggregation(aggregation),
                ZeroWeightPolicy(zero_weight_policy),
                1,
                proxy,
                verifier,
                qhat,
                abstain_below_q,
                cache,
            )
        )
    first_position = {}
    votes = {}
    for result in results:
        if result.abstain:
            continue
        index = result.selected_index
        if index not in votes:
            first_position[index] = len(first_position)
            votes[index] = 0
        votes[index] += 1
    ordered = sorted(votes, key=lambda index: (-votes[index], first_position[index]))[:top_k]
    return MultiTrajectoryResult(
        strategy=strategy.value,
        selected_indices=tuple(ordered),
        selected_ids=tuple(pool.sequence_ids[index] for index in ordered),
        selected_scores=tuple(float(votes[index]) for index in ordered),
        calls=budget.snapshot(),
        trajectory_results=tuple(results),
        abstain_reason="all_trajectories_abstained" if not ordered else None,
    )
