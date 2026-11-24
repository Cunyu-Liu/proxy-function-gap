import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .bayesian_ridge import BayesianRidge
from .pool_manifest import load_pool_manifest
from .regret_acquisition import EvaluationVault, QueryLabelVault, UniqueQueryLedger, UnlabeledPool
from .trainer.checkpoint import atomic_json_write
from .trainer.cuda import configure_cuda
from .trainer.data import sha256_file


FEATURE_NAMES = (
    "proxy_mean",
    "proxy_std",
    "proxy_range",
    "support",
    "pred_seed_11",
    "pred_seed_23",
    "pred_seed_37",
)
STRATEGIES = ("zero_label", "random_full_budget", "full_q")


@dataclass(frozen=True)
class ReplayConfig:
    development_manifest: Path
    query_manifest: Path
    output_dir: Path
    query_budget: int = 64
    random_seed: int = 20261124
    top_k: int = 1
    noise_variance: float = 0.04
    prior_variance: float = 1.0
    min_repair_margin: float = 0.0
    gpu_index: int = 0
    memory_fraction: float = 0.05
    dtype: str = "float64"

    def __post_init__(self):
        if isinstance(self.query_budget, bool) or not isinstance(self.query_budget, int) or self.query_budget <= 0:
            raise ValueError("query_budget must be a positive integer")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k != 1:
            raise ValueError("the preregistered replay deployment requires top_k=1")
        if not np.isfinite(self.noise_variance) or self.noise_variance <= 0:
            raise ValueError("noise_variance must be finite and positive")
        if not np.isfinite(self.prior_variance) or self.prior_variance <= 0:
            raise ValueError("prior_variance must be finite and positive")
        if not np.isfinite(self.min_repair_margin) or self.min_repair_margin < 0:
            raise ValueError("min_repair_margin must be finite and non-negative")
        if not 0 < self.memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0, 1]")

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text())
        allowed = {
            "development_manifest", "query_manifest", "output_dir", "query_budget", "random_seed",
            "top_k", "noise_variance", "prior_variance", "min_repair_margin", "gpu_index",
            "memory_fraction", "dtype",
        }
        if set(value) - allowed:
            raise ValueError(f"unknown replay config fields: {sorted(set(value) - allowed)}")
        required = {"development_manifest", "query_manifest", "output_dir", "query_budget"}
        if required - set(value):
            raise ValueError(f"missing replay config fields: {sorted(required - set(value))}")
        return cls(
            development_manifest=Path(value["development_manifest"]),
            query_manifest=Path(value["query_manifest"]),
            output_dir=Path(value["output_dir"]),
            query_budget=value["query_budget"],
            random_seed=value.get("random_seed", 20261124),
            top_k=value.get("top_k", 1),
            noise_variance=value.get("noise_variance", 0.04),
            prior_variance=value.get("prior_variance", 1.0),
            min_repair_margin=value.get("min_repair_margin", 0.0),
            gpu_index=value.get("gpu_index", 0),
            memory_fraction=value.get("memory_fraction", 0.05),
            dtype=value.get("dtype", "float64"),
        )

    def to_dict(self):
        return {
            "development_manifest": str(self.development_manifest),
            "query_manifest": str(self.query_manifest),
            "output_dir": str(self.output_dir),
            "query_budget": self.query_budget,
            "random_seed": self.random_seed,
            "top_k": self.top_k,
            "noise_variance": self.noise_variance,
            "prior_variance": self.prior_variance,
            "min_repair_margin": self.min_repair_margin,
            "gpu_index": self.gpu_index,
            "memory_fraction": self.memory_fraction,
            "dtype": self.dtype,
        }

    def sha256(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FrozenFeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features):
        features = np.asarray(features, dtype=np.float64)
        mean = features.mean(axis=0)
        scale = features.std(axis=0, ddof=0)
        return cls(mean=mean, scale=np.where(scale > 0, scale, 1.0))

    def transform(self, features):
        return (np.asarray(features, dtype=np.float64) - self.mean) / self.scale


def frozen_features(pool):
    return np.column_stack((
        pool.proxy_mean,
        pool.proxy_std,
        pool.proxy_range,
        pool.support,
        pool.member_predictions[0],
        pool.member_predictions[1],
        pool.member_predictions[2],
    )).astype(np.float64)


def _spd_inverse_cuda(matrix):
    factor = torch.linalg.cholesky(matrix)
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return torch.cholesky_solve(identity, factor)


def fit_ridge_cuda(features, labels, config: ReplayConfig, seed_offset=0):
    device, dtype, gpu = configure_cuda(
        config.gpu_index, config.memory_fraction, config.dtype, config.random_seed + seed_offset
    )
    x = torch.as_tensor(np.array(features, dtype=np.float64, copy=True), device=device, dtype=dtype)
    y = torch.as_tensor(np.asarray(labels, dtype=np.float64), device=device, dtype=dtype)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError("CUDA ridge requires matching non-empty feature and label arrays")
    if not x.is_cuda or not y.is_cuda:
        raise RuntimeError("CPU ridge training fallback is forbidden")
    dimension = x.shape[1]
    prior_mean = torch.zeros(dimension, device=device, dtype=dtype)
    prior_covariance = torch.eye(dimension, device=device, dtype=dtype) * config.prior_variance
    prior_precision = _spd_inverse_cuda(prior_covariance)
    precision = prior_precision + x.T @ x / config.noise_variance
    covariance = _spd_inverse_cuda(precision)
    right_hand_side = prior_precision @ prior_mean + x.T @ y / config.noise_variance
    mean = covariance @ right_hand_side
    torch.cuda.synchronize(device)
    if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
        raise RuntimeError("CUDA ridge closed-form solve did not converge to finite values")
    model = BayesianRidge(
        np.zeros(dimension),
        np.eye(dimension) * config.prior_variance,
        config.noise_variance,
        mean.detach().cpu().numpy(),
        covariance.detach().cpu().numpy(),
        features,
        labels,
    )
    metadata = {
        "gpu": gpu,
        "training_device": str(x.device),
        "label_device": str(y.device),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "convergence": {
            "method": "closed_form_spd_cholesky_solve",
            "iterative_epochs": 0,
            "finite_posterior": True,
            "cpu_fallback": False,
        },
    }
    return model, metadata


def select_posterior_mean_cuda(model, features, config: ReplayConfig):
    device, dtype, gpu = configure_cuda(
        config.gpu_index, config.memory_fraction, config.dtype, config.random_seed
    )
    x = torch.as_tensor(np.array(features, dtype=np.float64, copy=True), device=device, dtype=dtype)
    weights = torch.as_tensor(model.mean, device=device, dtype=dtype)
    means = x @ weights
    if not means.is_cuda:
        raise RuntimeError("CPU posterior-mean inference fallback is forbidden")
    order = torch.argsort(-means, stable=True)[: config.top_k]
    torch.cuda.synchronize(device)
    if not torch.isfinite(means).all():
        raise RuntimeError("non-finite posterior-mean predictions")
    indices = tuple(int(index) for index in order.detach().cpu().tolist())
    metadata = {
        "gpu": gpu,
        "inference_device": str(means.device),
        "posterior_mean_argmax": True,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cpu_fallback": False,
    }
    return indices, metadata


class ReplayLearner:
    def __init__(self, development_features, development_labels, query_pool, query_vault, config):
        self.development_features = np.asarray(development_features, dtype=np.float64)
        self.development_labels = np.asarray(development_labels, dtype=np.float64)
        self.query_pool = query_pool
        self.query_vault = query_vault
        self.config = config

    def run(self, strategy):
        if strategy not in STRATEGIES:
            raise ValueError("strategy is outside the preregistered repair-headroom set")
        ledger = UniqueQueryLedger(max_labels=self.config.query_budget)
        if strategy == "zero_label":
            query_order = ()
        elif strategy == "random_full_budget":
            rng = np.random.default_rng(self.config.random_seed)
            query_order = tuple(self.query_pool.sequence_ids[index] for index in rng.permutation(len(self.query_pool)))
        else:
            query_order = self.query_pool.sequence_ids
        revealed = self.query_vault.reveal(query_order, ledger)
        if revealed:
            canonical_ids = tuple(item for item in self.query_pool.sequence_ids if item in revealed)
            indices = tuple(self.query_pool.index(item) for item in canonical_ids)
            features = np.vstack((self.development_features, self.query_pool.features[list(indices)]))
            labels = np.concatenate((self.development_labels, [revealed[item] for item in canonical_ids]))
        else:
            features = self.development_features
            labels = self.development_labels
        model, training = fit_ridge_cuda(features, labels, self.config, seed_offset=ledger.used)
        selected_indices, inference = select_posterior_mean_cuda(model, self.query_pool.features, self.config)
        selected_ids = tuple(self.query_pool.sequence_ids[index] for index in selected_indices)
        return {
            "strategy": strategy,
            "selected_ids": selected_ids,
            "selected_indices": selected_indices,
            "query_order": query_order,
            "unique_queries_used": ledger.used,
            "unique_query_budget": ledger.max_labels,
            "posterior_observation_count": model.observation_count,
            "training": training,
            "inference": inference,
            "evaluation_labels_read": False,
            "per_round_evaluation_scores_read": False,
        }


class ReplayEvaluator:
    def __init__(self, evaluation_vault):
        self.evaluation_vault = evaluation_vault

    def evaluate(self, learner_result):
        return {
            "strategy": learner_result["strategy"],
            "regret": self.evaluation_vault.regret(learner_result["selected_ids"]),
            "labels_returned_to_learner": False,
            "per_round_scores_returned_to_learner": False,
        }


def _require_roles(development, query):
    if development.pool_role != "dev" or query.pool_role != "calibration":
        raise PermissionError("replay is restricted to source dev/calibration roles")
    if development.synthetic or query.synthetic:
        raise PermissionError("repair-headroom replay requires real frozen pools")
    if development.pool_id == query.pool_id:
        raise PermissionError("development and query pools must be distinct")


def run_replay(config: ReplayConfig):
    development = load_pool_manifest(config.development_manifest)
    query = load_pool_manifest(config.query_manifest)
    _require_roles(development, query)
    if config.query_budget != len(query):
        raise ValueError("this preregistered prototype requires query_budget equal to the full query pool")
    scaler = FrozenFeatureScaler.fit(frozen_features(development))
    development_features = scaler.transform(frozen_features(development))
    query_pool = UnlabeledPool(query.row_ids, scaler.transform(frozen_features(query)))
    query_vault = QueryLabelVault(query_pool, query.measured)
    evaluation_vault = EvaluationVault(query_pool, query.measured)
    learner_results = []
    evaluator_results = []
    for strategy in STRATEGIES:
        learner_result = ReplayLearner(
            development_features, development.measured, query_pool, query_vault, config
        ).run(strategy)
        learner_results.append(learner_result)
        evaluator_results.append(ReplayEvaluator(evaluation_vault).evaluate(learner_result))
    regrets = {item["strategy"]: item["regret"] for item in evaluator_results}
    repair_headroom = regrets["zero_label"] - regrets["full_q"]
    passed = repair_headroom > config.min_repair_margin
    random_result = next(item for item in learner_results if item["strategy"] == "random_full_budget")
    full_result = next(item for item in learner_results if item["strategy"] == "full_q")
    same_information_set = (
        random_result["posterior_observation_count"] == full_result["posterior_observation_count"]
        and random_result["unique_queries_used"] == full_result["unique_queries_used"] == config.query_budget
        and random_result["selected_ids"] == full_result["selected_ids"]
        and regrets["random_full_budget"] == regrets["full_q"]
    )
    if not same_information_set:
        raise RuntimeError("random full-budget and full-Q information sets differ")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    common = {
        "schema_version": "1.0",
        "config_sha256": config.sha256(),
        "config": config.to_dict(),
        "development_manifest_sha256": development.manifest_sha256,
        "query_manifest_sha256": query.manifest_sha256,
        "development_pool_id": development.pool_id,
        "query_pool_id": query.pool_id,
        "development_role": "development",
        "query_role": "query",
        "source_roles": {"development": development.pool_role, "query": query.pool_role},
        "feature_names": list(FEATURE_NAMES),
        "feature_scaler": {"fit_role": "development", "mean": scaler.mean.tolist(), "scale": scaler.scale.tolist()},
        "outer_labels_read": False,
        "outer_metrics_computed": False,
        "acquisition_compared_to_baseline": False,
    }
    learner_payload = {
        **common,
        "results": learner_results,
        "evaluation_labels_returned": False,
        "per_round_evaluation_scores_returned": False,
    }
    atomic_json_write(learner_payload, config.output_dir / "learner_replay.json")
    evaluator_payload = {
        **common,
        "results": evaluator_results,
        "repair_headroom": repair_headroom,
        "gate": {
            "rule": "zero_label_regret - full_q_regret > min_repair_margin",
            "min_repair_margin": config.min_repair_margin,
            "pass": passed,
            "status": "HEADROOM_PRESENT" if passed else "STOP_NO_REPAIR_HEADROOM",
            "random_full_budget_matches_full_q": same_information_set,
        },
        "evaluation_labels_returned_to_learner": False,
        "per_round_scores_returned_to_learner": False,
    }
    atomic_json_write(evaluator_payload, config.output_dir / "evaluator_final.json")
    manifest = {
        "config_sha256": config.sha256(),
        "status": evaluator_payload["gate"]["status"],
        "repair_headroom": repair_headroom,
        "files": {
            name: sha256_file(config.output_dir / name)
            for name in ("learner_replay.json", "evaluator_final.json")
        },
        "acquisition_superiority_claimed": False,
    }
    atomic_json_write(manifest, config.output_dir / "manifest.json")
    return manifest
