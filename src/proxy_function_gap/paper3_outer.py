import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .paper3_headroom import FrozenScaler, fit_model, predict_selection, raw_representation
from .paper3_headroom_v3 import _hamming, diversity_scores_cuda, expected_improvement_cuda
from .paper3_headroom_v4 import HeadroomV4Config, STRATEGIES, regret_analytic_ladder_cuda
from .pool_manifest import combine_pools, load_pool_manifest
from .regret_acquisition import QueryLabelVault, UniqueQueryLedger, UnlabeledPool
from .trainer.checkpoint import atomic_json_write
from .trainer.cuda import configure_cuda


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class OuterProtocolConfig:
    schema_version: str
    protocol_frozen_before_dry_run_results: bool
    source_v4_config: Path
    source_v4_config_sha256: str
    source_v4_scientific_sha256: str
    mode: str
    dry_run_pool_id: str
    primary_outer_pool_ids: tuple
    secondary_outer_pool_ids: tuple
    representation: str
    primary_strategy: str
    comparator_strategy: str
    strategies: tuple
    budgets: tuple
    primary_budget: int
    batch_size: int
    split_seed: int
    split_rule: str
    near_duplicate_max_hamming: int
    within_batch_near_duplicate_max_hamming: int
    prior_variance: float
    noise_variance: float
    bootstrap_resamples: int
    bootstrap_seed: int
    bootstrap_confidence_level: float
    primary_success_rule: str
    output_dir: Path

    def __post_init__(self):
        if self.schema_version != "paper3-outer-protocol-v1" or not self.protocol_frozen_before_dry_run_results:
            raise ValueError("the frozen Paper3 outer v1 protocol is required")
        if self.mode != "calibration_dry_run" or not self.dry_run_pool_id.startswith("calibration_"):
            raise PermissionError("this work package permits only calibration dry runs")
        if len(self.primary_outer_pool_ids) != 16 or any(not item.startswith("outer_designed_") for item in self.primary_outer_pool_ids):
            raise ValueError("the 16 designed outer pool IDs are preregistered")
        if len(self.secondary_outer_pool_ids) != 3 or any(not item.startswith("outer_random_") for item in self.secondary_outer_pool_ids):
            raise ValueError("the three random outer pool IDs are preregistered")
        if self.representation != "kmer4" or self.primary_strategy != "expected_improvement" or self.comparator_strategy != "random":
            raise ValueError("the kmer4 EI-versus-random primary comparison is frozen")
        if tuple(self.strategies) != STRATEGIES or tuple(self.budgets) != (0, 4, 8, 16, 32):
            raise ValueError("six strategies and 0/4/8/16/32 budgets are frozen")
        if self.primary_budget != 16 or self.batch_size != 4:
            raise ValueError("the primary budget and batch size are frozen")
        if self.split_seed != 20261125 or self.split_rule != "sha256_group_order_balanced_32q_32e":
            raise ValueError("the unlabeled group split is frozen")
        if self.near_duplicate_max_hamming != 1 or self.within_batch_near_duplicate_max_hamming != 8:
            raise ValueError("Q/E and within-batch near-duplicate rules are frozen")
        if self.prior_variance != 0.1 or self.noise_variance != 0.16:
            raise ValueError("development-only V4 hyperparameters are frozen")
        if self.bootstrap_resamples != 10000 or self.bootstrap_seed != 20261125 or self.bootstrap_confidence_level != 0.95:
            raise ValueError("the pool-cluster bootstrap is frozen")
        if self.primary_success_rule != "upper_95_percentile_CI_of_pool_mean_EI_minus_random_regret_at_16_below_0":
            raise ValueError("the primary success rule is frozen")

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text())
        fields = set(cls.__dataclass_fields__)
        if set(value) != fields:
            raise ValueError(f"outer config schema mismatch: missing={sorted(fields-set(value))}, unknown={sorted(set(value)-fields)}")
        tuple_fields = ("primary_outer_pool_ids", "secondary_outer_pool_ids", "strategies", "budgets")
        converted = {key: tuple(value[key]) for key in tuple_fields}
        converted.update(source_v4_config=Path(value["source_v4_config"]), output_dir=Path(value["output_dir"]))
        return cls(**{**value, **converted})

    def to_dict(self):
        result = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            result[key] = str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        return result

    def sha256(self):
        payload = self.to_dict()
        payload.pop("output_dir")
        return digest(payload)

    def v4(self):
        source = HeadroomV4Config.from_json(self.source_v4_config)
        if source.sha256() != self.source_v4_config_sha256:
            raise ValueError("source V4 config hash mismatch")
        return source

    def runtime(self, output_dir):
        return self.v4().runtime(self.prior_variance, self.noise_variance, output_dir)


def split_group_indices(pool, seed=20261125):
    groups = {}
    for index, group in enumerate(pool.group_ids):
        groups.setdefault(group, []).append(index)
    ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{pool.pool_id}:{group}".encode()).hexdigest())
    query, evaluation = [], []
    for group in ordered:
        if len(query) <= len(evaluation):
            query.extend(groups[group])
        else:
            evaluation.extend(groups[group])
    query, evaluation = sorted(query), sorted(evaluation)
    if len(query) != 32 or len(evaluation) != 32:
        raise ValueError("the frozen split requires exactly 32 query and 32 evaluation rows")
    if set(query) & set(evaluation) or set(pool.group_ids[index] for index in query) & set(pool.group_ids[index] for index in evaluation):
        raise PermissionError("Q/E group isolation failed")
    for q in query:
        for e in evaluation:
            if _hamming(pool.utr50[q], pool.utr50[e]) <= 1:
                raise PermissionError("Q/E exact or Hamming-1 conflict")
    return tuple(query), tuple(evaluation)


def subset_pool(pool, indices, pool_id):
    from dataclasses import replace
    indices = tuple(indices)
    values = {
        "pool_id": pool_id, "manifest_path": None,
        "row_ids": tuple(pool.row_ids[index] for index in indices),
        "sequence_ids": tuple(pool.sequence_ids[index] for index in indices),
        "utr50": tuple(pool.utr50[index] for index in indices),
        "group_ids": tuple(pool.group_ids[index] for index in indices),
        "proxy_mean": pool.proxy_mean[list(indices)], "proxy_std": pool.proxy_std[list(indices)],
        "proxy_range": pool.proxy_range[list(indices)], "support": pool.support[list(indices)],
        "measured": None if pool.measured is None else pool.measured[list(indices)],
        "proposal_probs": pool.proposal_probs[list(indices)], "source_manifests": getattr(pool, "source_manifests", None),
    }
    if not hasattr(pool, "source_manifests"):
        values.pop("source_manifests")
    if not hasattr(pool, "manifest_path"):
        values.pop("manifest_path")
    if hasattr(pool, "member_predictions"):
        values["member_predictions"] = pool.member_predictions[:, list(indices)]
    return replace(pool, **values)


def require_development_isolation(development, query, evaluation):
    development_groups = set(development.group_ids)
    development_sequences = set(development.utr50)
    for pool in (query, evaluation):
        if development_groups & set(pool.group_ids) or development_sequences & set(pool.utr50):
            raise PermissionError("development and Q/E overlap")
        for sequence in pool.utr50:
            if any(_hamming(sequence, source) <= 1 for source in development.utr50):
                raise PermissionError("development and Q/E Hamming-1 conflict")


def learner_curve(development_features, development_labels, query_pool, query_vault,
                  evaluation_pool, proxy_mean, query_sequences, strategy, config, runtime, seed):
    if strategy not in STRATEGIES:
        raise ValueError("unknown acquisition strategy")
    if set(query_pool.sequence_ids) & set(evaluation_pool.sequence_ids):
        raise PermissionError("Learner Q/E identities overlap")
    ledger = UniqueQueryLedger(max_labels=config.budgets[-1])
    rng = np.random.default_rng(seed)
    random_order = list(rng.permutation(len(query_pool)))
    selected, records, acquisition_metadata = [], [], []
    revealed_labels = {}
    model = None
    for step in range(len(query_pool) + 1):
        if step in config.budgets:
            if selected:
                revealed_ids = tuple(query_pool.sequence_ids[index] for index in selected)
                x = np.vstack((development_features, query_pool.features[selected]))
                y = np.concatenate((development_labels, [revealed_labels[item] for item in revealed_ids]))
            else:
                x, y = development_features, development_labels
            model, training = fit_model(x, y, runtime)
            choice, inference = predict_selection(model, evaluation_pool.features, runtime)
            records.append({
                "budget": step, "selected_index": int(choice),
                "selected_id": evaluation_pool.sequence_ids[choice],
                "unique_queries_used": ledger.used,
                "training_device": training["training_device"],
                "inference_device": inference["inference_device"],
            })
        if step == len(query_pool):
            break
        batch_start = step // config.batch_size * config.batch_size
        available = [index for index in range(len(query_pool)) if index not in selected]
        exclusions = 0
        for chosen in selected[batch_start:]:
            before = len(available)
            available = [index for index in available if _hamming(query_sequences[chosen], query_sequences[index]) > config.within_batch_near_duplicate_max_hamming]
            exclusions += before - len(available)
        if not available:
            raise RuntimeError("batch constraint left no feasible query")
        if strategy == "random":
            choice = int(next(index for index in random_order if index in available))
            metadata = {"device": "not_applicable", "cpu_fallback": False}
        else:
            available_x = query_pool.features[available]
            if strategy == "max_variance":
                from .paper3_headroom import posterior_variance_cuda
                scores = posterior_variance_cuda(model, available_x, runtime)
                metadata = {"device": str(scores.device), "cpu_fallback": False}
            elif strategy == "diversity":
                labeled_x = np.vstack((development_features, query_pool.features[selected])) if selected else development_features
                scores = diversity_scores_cuda(available_x, labeled_x, runtime)
                metadata = {"device": str(scores.device), "cpu_fallback": False}
            elif strategy == "expected_improvement":
                scores = expected_improvement_cuda(model, available_x, query_pool.features, runtime)
                metadata = {"device": str(scores.device), "cpu_fallback": False, "integration": "analytic_normal"}
            elif strategy == "top_proxy":
                device, dtype, _ = configure_cuda(runtime.gpu_index, runtime.memory_fraction, runtime.dtype, seed)
                scores = torch.as_tensor(proxy_mean[available], device=device, dtype=dtype)
                metadata = {"device": str(scores.device), "cpu_fallback": False}
            else:
                scores, metadata = regret_analytic_ladder_cuda(model, available_x, evaluation_pool.features, config.v4(), runtime.noise_variance)
            best_position = int(torch.argmax(scores).item())
            choice = available[best_position]
            metadata.update(chosen_available_position=best_position, chosen_score=float(scores[best_position].item()))
        query_id = query_pool.sequence_ids[choice]
        revealed_labels.update(query_vault.reveal((query_id,), ledger))
        metadata.update({
            "step": step, "batch_index": step // config.batch_size,
            "chosen_index": int(choice), "chosen_query_id": query_id,
            "feasible_count": len(available), "within_batch_near_duplicates_excluded": exclusions,
            "unique_queries_used": ledger.used,
        })
        selected.append(choice)
        acquisition_metadata.append(metadata)
    return {
        "strategy": strategy, "seed": seed, "records": records,
        "query_order": [query_pool.sequence_ids[index] for index in selected],
        "acquisition_metadata": acquisition_metadata,
        "evaluation_labels_read": False, "evaluation_scores_returned_to_learner": False,
    }


def score_frozen_curve(curve, evaluation_ids, evaluation_labels):
    if curve["evaluation_labels_read"] or curve["evaluation_scores_returned_to_learner"]:
        raise PermissionError("Learner curve has been contaminated by evaluation")
    labels = dict(zip(evaluation_ids, evaluation_labels))
    if set(curve["query_order"]) & set(labels):
        raise PermissionError("query IDs entered the evaluation vault")
    oracle = max(evaluation_labels)
    records = []
    for record in curve["records"]:
        regret = float(oracle - labels[record["selected_id"]])
        records.append({**record, "regret": regret})
    return {"strategy": curve["strategy"], "seed": curve["seed"], "records": records,
            "labels_returned_to_learner": False, "per_round_scores_returned_to_learner": False}


def cluster_bootstrap_mean(values, resamples=10000, seed=20261125, confidence_level=0.95):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("cluster values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(means, [alpha, 1 - alpha])
    return {"cluster_count": len(values), "resamples": resamples, "seed": seed,
            "confidence_level": confidence_level, "mean": float(values.mean()),
            "ci_lower": float(lower), "ci_upper": float(upper), "success_upper_below_zero": bool(upper < 0)}


def run_calibration_dry_round(config, output_dir=None):
    output_dir = config.output_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = config.to_dict()
    frozen["output_dir"] = str(output_dir)
    atomic_json_write(frozen, output_dir / "frozen_config.json")
    source = config.v4()
    development_pools = [load_pool_manifest(source.pool_root / item / "pool_manifest.json") for item in source.development_pool_ids]
    calibration = load_pool_manifest(source.pool_root / config.dry_run_pool_id / "pool_manifest.json")
    if any(pool.pool_role != "dev" for pool in development_pools) or calibration.pool_role != "calibration":
        raise PermissionError("dry run is restricted to development/calibration")
    development = combine_pools(development_pools, "combined_development")
    query_indices, evaluation_indices = split_group_indices(calibration, config.split_seed)
    query = subset_pool(calibration, query_indices, calibration.pool_id + "_query")
    evaluation = subset_pool(calibration, evaluation_indices, calibration.pool_id + "_evaluation")
    require_development_isolation(development, query, evaluation)
    development_raw, _ = raw_representation(config.representation, development, source)
    query_raw, _ = raw_representation(config.representation, query, source)
    evaluation_raw, _ = raw_representation(config.representation, evaluation, source)
    scaler = FrozenScaler.fit(development_raw, development.measured)
    development_x = scaler.features(development_raw)
    development_y = scaler.labels(development.measured)
    query_pool = UnlabeledPool(query.row_ids, scaler.features(query_raw))
    evaluation_pool = UnlabeledPool(evaluation.row_ids, scaler.features(evaluation_raw))
    query_vault = QueryLabelVault(query_pool, scaler.labels(query.measured))
    runtime = config.runtime(output_dir)
    curves = []
    for random_seed in source.random_seeds:
        curves.append(learner_curve(development_x, development_y, query_pool, query_vault,
                                    evaluation_pool, query.proxy_mean, query.utr50,
                                    "random", config, runtime, random_seed))
    for offset, strategy in enumerate(STRATEGIES[1:], start=1):
        curves.append(learner_curve(development_x, development_y, query_pool, query_vault,
                                    evaluation_pool, query.proxy_mean, query.utr50,
                                    strategy, config, runtime, source.acquisition_seed + offset))
    common = {
        "schema_version": config.schema_version, "config_sha256": config.sha256(),
        "mode": "calibration_dry_run", "source_pool_id": calibration.pool_id,
        "source_manifest_sha256": calibration.manifest_sha256,
        "development_manifest_sha256": development.manifest_sha256,
        "query_ids": list(query.row_ids), "evaluation_ids": list(evaluation.row_ids),
        "query_sequences": list(query.utr50), "evaluation_sequences": list(evaluation.utr50),
        "feature_scaler": scaler.to_dict(), "outer_labels_read": False,
        "outer_metrics_computed": False, "scientific_conclusion": False,
    }
    atomic_json_write({**common, "curves": curves}, output_dir / "learner_frozen.json")
    scored = [score_frozen_curve(curve, evaluation.row_ids, evaluation.measured) for curve in curves]
    full_selections = {next(record["selected_id"] for record in curve["records"] if record["budget"] == 32) for curve in scored}
    if len(full_selections) != 1:
        raise RuntimeError("full-Q selections differ across strategies")
    atomic_json_write({**common, "curves": scored, "full_q_same_selection": True}, output_dir / "evaluator_dry_run.json")
    summary = {**common, "status": "CALIBRATION_DRY_RUN_COMPLETE", "curve_count": len(scored),
               "scientific_sha256": digest({"learner": curves, "evaluator": scored})}
    atomic_json_write(summary, output_dir / "summary.json")
    return summary
