import hashlib
import json
import random
from dataclasses import dataclass

import numpy as np

from .constraints import UTRConstraints, utr_constraint_mask
from .pool import CandidatePool
from .pool_manifest import PoolData
from .selection import Budget, select
from .trainer.checkpoint import atomic_json_write


STRATEGIES = ("random", "bon", "lcb", "wsc", "greedy", "fp-rej", "fp-tilt")
ALPHAS = (0, 1, 5, 10, 20)
BUDGETS = (1, 2, 4, 8, 16)
PAIRED_SEEDS = (101, 202, 303, 404, 505)


class EnsembleScoreProvider:
    def __init__(self, pool: PoolData, member_scorers, verifier_values=None, v_model=None):
        if len(member_scorers) != 3 or any(not callable(scorer) for scorer in member_scorers):
            raise ValueError("three actual member scorers are required")
        self.pool = pool
        self.member_scorers = tuple(member_scorers)
        self.verifier_values = verifier_values
        self.v_model = v_model
        self.member_item_forwards = {seed: 0 for seed in (11, 23, 37)}
        self.proxy_member_item_forwards = 0
        self.verifier_member_item_forwards = 0
        self.v_model_item_forwards = 0

    def _features(self, candidate, kind):
        values = []
        for seed, scorer in zip((11, 23, 37), self.member_scorers):
            values.append(float(scorer(candidate)))
            self.member_item_forwards[seed] += 1
            if kind == "proxy":
                self.proxy_member_item_forwards += 1
            else:
                self.verifier_member_item_forwards += 1
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        value_range = float(np.max(values) - np.min(values))
        return mean, std, value_range

    def proxy_mean(self, candidate):
        return self._features(candidate, "proxy")[0]

    def lcb(self, candidate):
        mean, std, _ = self._features(candidate, "proxy")
        return mean, std

    def verifier(self, candidate):
        mean, std, value_range = self._features(candidate, "verifier")
        if self.v_model is not None:
            value = self.v_model(mean, std, value_range, self.pool.support[candidate.index], candidate.index)
            self.v_model_item_forwards += getattr(self.v_model, "last_item_forwards", 1)
            return value
        return self.verifier_values[candidate.index]

    def total_member_item_forwards(self):
        return sum(self.member_item_forwards.values())


def _lcb_select(pool, provider, proposal_cap, rng, lambda_):
    distribution = pool.masked_proposal_probs()
    best_index = None
    best_score = -float("inf")
    for _ in range(proposal_cap):
        threshold = rng.random()
        cumulative = 0.0
        for index, probability in enumerate(distribution):
            cumulative += probability
            if threshold < cumulative:
                break
        mean, std = provider.lcb(pool.candidate(index))
        score = mean - lambda_ * std
        if best_index is None or score > best_score:
            best_index = index
            best_score = score
    return best_index


@dataclass(frozen=True)
class SelectionRecord:
    pool_id: str
    strategy: str
    alpha: float
    proposal_cap: int
    seed: int
    selected_index: int
    selected_id: str
    regret: float
    proxy_item_forwards: int
    ensemble_member_item_forwards: int
    member_item_forwards: dict
    proposals: int
    abstain: bool
    verifier_item_forwards: int = 0
    v_model_item_forwards: int = 0
    noise_sigma: float = 0.0
    proxy_member_item_forwards: int = None
    verifier_member_item_forwards: int = None
    q_model_item_forwards: int = 0
    total_model_item_forwards: int = None

    def __post_init__(self):
        if self.proxy_member_item_forwards is None:
            object.__setattr__(self, "proxy_member_item_forwards", 3 * self.proxy_item_forwards)
        if self.verifier_member_item_forwards is None:
            object.__setattr__(self, "verifier_member_item_forwards", 3 * self.verifier_item_forwards)
        object.__setattr__(
            self,
            "total_model_item_forwards",
            self.ensemble_member_item_forwards + self.v_model_item_forwards + self.q_model_item_forwards,
        )

    def to_dict(self):
        value = {
            "pool_id": self.pool_id,
            "strategy": self.strategy,
            "alpha": self.alpha,
            "proposal_cap": self.proposal_cap,
            "seed": self.seed,
            "selected_index": self.selected_index,
            "selected_id": self.selected_id,
            "regret": self.regret,
            "proxy_item_forwards": self.proxy_item_forwards,
            "ensemble_member_item_forwards": self.ensemble_member_item_forwards,
            "member_item_forwards": dict(self.member_item_forwards),
            "proposals": self.proposals,
            "abstain": self.abstain,
            "verifier_item_forwards": self.verifier_item_forwards,
            "v_model_item_forwards": self.v_model_item_forwards,
            "proxy_member_item_forwards": self.proxy_member_item_forwards,
            "verifier_member_item_forwards": self.verifier_member_item_forwards,
            "q_model_item_forwards": self.q_model_item_forwards,
            "total_model_item_forwards": self.total_model_item_forwards,
        }
        if self.noise_sigma > 0:
            value["noise_sigma"] = self.noise_sigma
        return value


def evaluate_strategy(pool: PoolData, member_scorers, verifier_values, strategy, alpha, proposal_cap, seed, lambda_=1.0, v_model=None):
    feasible = utr_constraint_mask(tuple(value.replace("T", "U") for value in pool.utr50), UTRConstraints(min_length=50, max_length=50))
    candidate_pool = CandidatePool(pool.sequence_ids, pool.utr50, pool.proposal_probs, feasible=feasible)
    provider = EnsembleScoreProvider(pool, member_scorers, verifier_values, v_model)
    selected_index = None
    proxy_calls = 0
    proposals = 0
    verifier_calls = 0
    if strategy == "lcb":
        selected_index = _lcb_select(candidate_pool, provider, proposal_cap, random.Random(seed), lambda_)
        proposals = proposal_cap
        proxy_calls = proposal_cap
    else:
        result = select(
            candidate_pool,
            strategy,
            proposal_cap,
            random.Random(seed),
            budget_limits=Budget(),
            alpha=alpha,
            proxy=provider.proxy_mean,
            verifier=provider.verifier,
        )
        selected_index = result.selected_index
        proxy_calls = result.calls["proxy_item_forwards"]
        verifier_calls = result.calls["verifier_item_forwards"]
        proposals = result.calls["proposals"]
    regret = None if selected_index is None or pool.measured is None else float(np.max(pool.measured) - pool.measured[selected_index])
    return SelectionRecord(
        pool_id=pool.pool_id,
        strategy=strategy,
        alpha=alpha,
        proposal_cap=proposal_cap,
        seed=seed,
        selected_index=-1 if selected_index is None else selected_index,
        selected_id="ABSTAIN" if selected_index is None else pool.sequence_ids[selected_index],
        regret=regret,
        proxy_item_forwards=proxy_calls,
        ensemble_member_item_forwards=provider.total_member_item_forwards(),
        member_item_forwards=dict(provider.member_item_forwards),
        proposals=proposals,
        abstain=selected_index is None,
        verifier_item_forwards=verifier_calls,
        v_model_item_forwards=provider.v_model_item_forwards,
        proxy_member_item_forwards=provider.proxy_member_item_forwards,
        verifier_member_item_forwards=provider.verifier_member_item_forwards,
    )


def run_budget_grid(pool: PoolData, member_scorers, verifier_values, alphas=ALPHAS, budgets=BUDGETS, seeds=PAIRED_SEEDS, lambda_=1.0, v_model=None, actual_cost_budget=None):
    if actual_cost_budget is not None and (isinstance(actual_cost_budget, bool) or not isinstance(actual_cost_budget, int) or actual_cost_budget <= 0):
        raise ValueError("actual_cost_budget must be a positive integer or None")
    records = []
    actual_cost = 0
    for alpha in alphas:
        for proposal_cap in budgets:
            for seed in seeds:
                for strategy in STRATEGIES:
                    record = evaluate_strategy(pool, member_scorers, verifier_values, strategy, alpha, proposal_cap, seed, lambda_, v_model)
                    records.append(record)
                    actual_cost += record.total_model_item_forwards
                    if actual_cost_budget is not None and actual_cost >= actual_cost_budget:
                        return records
    return records


def selection_config_hash(records):
    payload = json.dumps([record.to_dict() for record in records], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_selection_records(records, path, metadata=None):
    config_hash = selection_config_hash(records)
    execution_metadata = dict(metadata or {})
    execution_metadata.setdefault("cost_ledger_schema_version", "2.0")
    execution_metadata.setdefault("pre_cost_ledger_results_require_rerun", True)
    atomic_json_write(
        {
            "schema_version": "1.0",
            "selection_config_sha256": config_hash,
            "q_used_in_candidate_ranking": False,
            "execution_metadata": execution_metadata,
            "records": [record.to_dict() for record in records],
        },
        path,
    )
    return config_hash
