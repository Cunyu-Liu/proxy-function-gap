import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .paper3_headroom import (
    REPRESENTATIONS,
    FrozenScaler,
    HeadroomConfig,
    fit_model,
    gate_case,
    posterior_variance_cuda,
    predict_selection,
    raw_representation,
)
from .pool_manifest import combine_pools, load_pool_manifest
from .trainer.checkpoint import atomic_json_write
from .trainer.cuda import configure_cuda


STRATEGIES = ("random", "max_variance", "diversity", "expected_improvement", "top_proxy", "regret_quadrature")


@dataclass(frozen=True)
class HeadroomV3Config:
    schema_version: str
    protocol_frozen_before_v3_results: bool
    source_v2_config_sha256: str
    source_v2_scientific_sha256: str
    pool_root: Path
    development_pool_ids: tuple
    query_pool_ids: tuple
    representations: tuple
    curve_cohort: tuple
    strategies: tuple
    output_dir: Path
    budgets: tuple
    batch_size: int
    near_duplicate_max_hamming: int
    random_seeds: tuple
    acquisition_seed: int
    posterior_sample_ladder: tuple
    label_quadrature_ladder: tuple
    convergence_score_atol: float
    inner_cv_folds: int
    prior_variance_grid: tuple
    noise_variance_grid: tuple
    min_repair_margin: float
    gpu_index: int
    memory_fraction: float
    dtype: str
    cnn_batch_size: int
    cnn_members: tuple

    def __post_init__(self):
        if self.schema_version != "paper3-dev-query-headroom-v3" or self.protocol_frozen_before_v3_results is not True:
            raise ValueError("a frozen Paper3 V3 protocol is required")
        if tuple(self.representations) != REPRESENTATIONS or tuple(self.strategies) != STRATEGIES:
            raise ValueError("all frozen representations and strategies are required in fixed order")
        if any(not item.startswith("dev_") for item in self.development_pool_ids):
            raise PermissionError("development IDs must have dev roles")
        if any(not item.startswith("calibration_") for item in self.query_pool_ids):
            raise PermissionError("query IDs must have calibration roles")
        if any("outer" in item.lower() for item in (*self.development_pool_ids, *self.query_pool_ids)):
            raise PermissionError("outer paths and IDs are forbidden")
        if len(self.curve_cohort) != 6 or len({(item["representation"], item["query_pool_id"]) for item in self.curve_cohort}) != 6:
            raise ValueError("the fixed V2 curve cohort must contain six unique cases")
        if any(item["representation"] not in self.representations or item["query_pool_id"] not in self.query_pool_ids for item in self.curve_cohort):
            raise ValueError("curve cohort is outside the frozen case set")
        if tuple(self.budgets) != (0, 4, 8, 16, 32, 64) or self.batch_size != 4:
            raise ValueError("V3 budgets and batch size are frozen")
        if self.near_duplicate_max_hamming != 8:
            raise ValueError("the within-batch Hamming rule is frozen")
        if len(self.random_seeds) != 3 or len(set(self.random_seeds)) != 3:
            raise ValueError("exactly three unique random seeds are required")
        if tuple(self.posterior_sample_ladder) != (32, 64, 128) or tuple(self.label_quadrature_ladder) != (8, 16, 32):
            raise ValueError("the S/K doubling ladder is frozen")
        if self.inner_cv_folds != 4 or not self.prior_variance_grid or not self.noise_variance_grid:
            raise ValueError("the development-only inner CV grid is incomplete")
        if not np.isfinite(self.convergence_score_atol) or self.convergence_score_atol <= 0:
            raise ValueError("convergence tolerance must be finite and positive")
        if len(self.cnn_members) != 3:
            raise ValueError("three frozen CNN members are required")

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text())
        fields = set(cls.__dataclass_fields__)
        if set(value) != fields:
            raise ValueError(f"config fields differ from frozen schema: missing={sorted(fields-set(value))}, unknown={sorted(set(value)-fields)}")
        tuple_fields = (
            "development_pool_ids", "query_pool_ids", "representations", "curve_cohort", "strategies",
            "budgets", "random_seeds", "posterior_sample_ladder", "label_quadrature_ladder",
            "prior_variance_grid", "noise_variance_grid", "cnn_members",
        )
        converted = {key: tuple(value[key]) for key in tuple_fields}
        converted.update(pool_root=Path(value["pool_root"]), output_dir=Path(value["output_dir"]))
        return cls(**{**value, **converted})

    def to_dict(self):
        result = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value
        return result

    def sha256(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def runtime(self, prior_variance, noise_variance, output_dir=None):
        return HeadroomConfig(
            schema_version="paper3-dev-query-headroom-v2",
            protocol_frozen_before_pool_results=True,
            pool_root=self.pool_root,
            development_pool_ids=self.development_pool_ids,
            query_pool_ids=self.query_pool_ids,
            representations=self.representations,
            output_dir=self.output_dir if output_dir is None else Path(output_dir),
            budgets=self.budgets,
            random_seeds=self.random_seeds,
            acquisition_seed=self.acquisition_seed,
            posterior_samples=self.posterior_sample_ladder[-1],
            label_samples=self.label_quadrature_ladder[-1],
            noise_variance=float(noise_variance),
            prior_variance=float(prior_variance),
            min_repair_margin=self.min_repair_margin,
            gpu_index=self.gpu_index,
            memory_fraction=self.memory_fraction,
            dtype=self.dtype,
            cnn_batch_size=self.cnn_batch_size,
            cnn_members=self.cnn_members,
        )


def posterior_mean_cuda(model, features, config):
    device, dtype, _ = configure_cuda(config.gpu_index, config.memory_fraction, config.dtype, config.acquisition_seed)
    x = torch.as_tensor(features, device=device, dtype=dtype)
    mean = torch.as_tensor(model.mean, device=device, dtype=dtype)
    prediction = x @ mean
    torch.cuda.synchronize(device)
    if not prediction.is_cuda or not torch.isfinite(prediction).all():
        raise RuntimeError("invalid CUDA posterior-mean prediction")
    return prediction


def select_inner_hyperparameters(raw_features, measured, folds, config):
    raw_features = np.asarray(raw_features, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    folds = np.asarray(folds, dtype=np.int64)
    records = []
    for prior in config.prior_variance_grid:
        for noise in config.noise_variance_grid:
            fold_records = []
            for fold in range(config.inner_cv_folds):
                train = folds != fold
                valid = ~train
                if not train.any() or not valid.any():
                    raise ValueError("inner development folds must be non-empty")
                scaler = FrozenScaler.fit(raw_features[train], measured[train])
                runtime = config.runtime(prior, noise)
                model, training = fit_model(scaler.features(raw_features[train]), scaler.labels(measured[train]), runtime)
                prediction = posterior_mean_cuda(model, scaler.features(raw_features[valid]), runtime)
                prediction_original = prediction.detach().cpu().numpy() * scaler.target_scale + scaler.target_mean
                mse = float(np.mean((prediction_original - measured[valid]) ** 2))
                fold_records.append({
                    "fold": fold, "train_rows": int(train.sum()), "validation_rows": int(valid.sum()),
                    "measured_mrl_mse": mse, "training_device": training["training_device"],
                    "inference_device": str(prediction.device), "target_scaler_fit": "inner_training_development_only",
                })
            mean_mse = float(np.mean([item["measured_mrl_mse"] for item in fold_records]))
            records.append({
                "prior_variance": float(prior), "noise_variance": float(noise),
                "folds": fold_records, "mean_fold_measured_mrl_mse": mean_mse,
            })
    def selection_key(item):
        return (
            item["mean_fold_measured_mrl_mse"], abs(np.log(item["prior_variance"] / 1.0)),
            abs(np.log(item["noise_variance"] / 0.04)), item["prior_variance"], item["noise_variance"],
        )
    selected = min(records, key=selection_key)
    return {
        "selection_role": "inner_development_blocked_cv_only",
        "fold_assignment": "development_pool_numeric_id_mod_4",
        "metric": "equal_weight_mean_fold_measured_MRL_MSE",
        "tie_break": "closest_to_v2_prior_then_v2_noise_then_smaller_prior_then_smaller_noise",
        "query_labels_used": False, "outer_labels_read": False,
        "selected": {key: selected[key] for key in ("prior_variance", "noise_variance", "mean_fold_measured_mrl_mse")},
        "grid": records,
    }


def expected_improvement_cuda(model, features, evaluation_features, config):
    device, dtype, _ = configure_cuda(config.gpu_index, config.memory_fraction, config.dtype, config.acquisition_seed)
    x = torch.as_tensor(features, device=device, dtype=dtype)
    evaluation = torch.as_tensor(evaluation_features, device=device, dtype=dtype)
    mean = torch.as_tensor(model.mean, device=device, dtype=dtype)
    covariance = torch.as_tensor(model.covariance, device=device, dtype=dtype)
    mu = x @ mean
    incumbent = torch.max(evaluation @ mean)
    variance = torch.einsum("ij,jk,ik->i", x, covariance, x).clamp_min(0)
    sigma = torch.sqrt(variance)
    safe_sigma = sigma.clamp_min(torch.finfo(dtype).tiny)
    z = (mu - incumbent) / safe_sigma
    normal_cdf = 0.5 * (1 + torch.erf(z / np.sqrt(2.0)))
    normal_pdf = torch.exp(-0.5 * z.square()) / np.sqrt(2.0 * np.pi)
    improvement = sigma * (z * normal_cdf + normal_pdf)
    improvement = torch.where(sigma > 0, improvement, (mu - incumbent).clamp_min(0))
    torch.cuda.synchronize(device)
    if not improvement.is_cuda or not torch.isfinite(improvement).all():
        raise RuntimeError("invalid analytic CUDA expected improvement")
    return improvement


def diversity_scores_cuda(available_features, labeled_features, config):
    device, dtype, _ = configure_cuda(config.gpu_index, config.memory_fraction, config.dtype, config.acquisition_seed)
    available = torch.as_tensor(available_features, device=device, dtype=dtype)
    labeled = torch.as_tensor(labeled_features, device=device, dtype=dtype)
    distances = torch.cdist(available, labeled, p=2)
    scores = distances.min(dim=1).values
    torch.cuda.synchronize(device)
    if not scores.is_cuda or not torch.isfinite(scores).all():
        raise RuntimeError("invalid CUDA diversity scores")
    return scores


def _normal_quadrature(count, device, dtype):
    nodes, weights = np.polynomial.hermite.hermgauss(count)
    nodes = nodes * np.sqrt(2.0)
    weights = weights / np.sqrt(np.pi)
    return (
        torch.as_tensor(nodes, device=device, dtype=dtype),
        torch.as_tensor(weights, device=device, dtype=dtype),
    )


def regret_quadrature_ladder_cuda(model, available_features, evaluation_features, config, seed, noise_variance):
    device, dtype, gpu = configure_cuda(config.gpu_index, config.memory_fraction, config.dtype, seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    max_samples = config.posterior_sample_ladder[-1]
    max_labels = config.label_quadrature_ladder[-1]
    dimension = len(model.mean)
    z_half = torch.randn((max_samples // 2, dimension), device=device, dtype=dtype, generator=generator)
    z_all = torch.stack((z_half, -z_half), dim=1).reshape(max_samples, dimension)
    epsilon_half = torch.randn((max_samples // 2, max_labels), device=device, dtype=dtype, generator=generator)
    epsilon_all = torch.stack((epsilon_half, -epsilon_half), dim=1).reshape(max_samples, max_labels)
    mean = torch.as_tensor(model.mean, device=device, dtype=dtype)
    covariance = torch.as_tensor(model.covariance, device=device, dtype=dtype)
    query = torch.as_tensor(available_features, device=device, dtype=dtype)
    evaluation = torch.as_tensor(evaluation_features, device=device, dtype=dtype)
    factor = torch.linalg.cholesky(covariance)
    evaluation_factor = evaluation @ factor
    query_factor = query @ factor
    evaluation_mean = evaluation @ mean
    query_mean = query @ mean
    v = query @ covariance
    predictive_variance = (torch.sum(v * query, dim=1) + float(noise_variance)).clamp_min(torch.finfo(dtype).tiny)
    v_evaluation = v @ evaluation.T
    levels = []
    previous_scores = None
    previous_choice = None
    for samples, label_count in zip(config.posterior_sample_ladder, config.label_quadrature_ladder):
        z = z_all[:samples]
        epsilon = epsilon_all[:samples, :label_count]
        nodes, weights = _normal_quadrature(label_count, device, dtype)
        base = z @ evaluation_factor.T
        projected = z @ query_factor.T
        conditional_mean = (
            evaluation_mean[None, :]
            + v_evaluation[:, None, :] * (nodes / torch.sqrt(predictive_variance)[:, None])[:, :, None]
        )
        residual = (
            projected.T[:, None, :]
            + np.sqrt(float(noise_variance)) * epsilon.T[None, :, :]
        ) / predictive_variance[:, None, None]
        outcomes = (
            conditional_mean[:, :, None, :] + base[None, None, :, :]
            - residual[:, :, :, None] * v_evaluation[:, None, None, :]
        )
        updated_choice = torch.argmax(conditional_mean, dim=2)
        selected_outcomes = torch.gather(
            outcomes, 3, updated_choice[:, :, None, None].expand(-1, -1, samples, 1)
        ).squeeze(3)
        per_sample_regret = outcomes.max(dim=3).values - selected_outcomes
        expected_regret = (per_sample_regret.mean(dim=2) * weights[None, :]).sum(dim=1)
        scores = -expected_regret
        choice = int(torch.argmax(scores).item())
        delta = None if previous_scores is None else float(torch.max(torch.abs(scores - previous_scores)).item())
        levels.append({
            "posterior_samples": samples, "label_quadrature_nodes": label_count,
            "chosen_available_position": choice, "scores": scores.detach().cpu().tolist(),
            "max_absolute_score_delta_from_previous": delta,
        })
        previous_scores = scores
        previous_choice = choice
    torch.cuda.synchronize(device)
    converged = (
        levels[-2]["chosen_available_position"] == levels[-1]["chosen_available_position"]
        and levels[-1]["max_absolute_score_delta_from_previous"] <= config.convergence_score_atol
    )
    metadata = {
        "device": str(device), "gpu": gpu, "cpu_fallback": False,
        "common_random_numbers": True, "antithetic_posterior_samples": True,
        "label_integration": "deterministic_gauss_hermite",
        "joint_gaussian_conditional_outcomes": True,
        "convergence_rule": "last_two_choices_equal_and_max_score_delta<=0.01",
        "converged": bool(converged), "levels": levels,
    }
    return previous_scores, metadata


def _hamming(left, right):
    if len(left) != len(right):
        raise ValueError("near-duplicate sequences must have equal length")
    return sum(a != b for a, b in zip(left, right))


def curve_case_v3(development_features, development_labels, query_features, query_labels, proxy_mean, sequences, strategy, config, runtime, seed):
    if strategy not in STRATEGIES:
        raise ValueError("unknown V3 acquisition strategy")
    rng = np.random.default_rng(seed)
    random_order = list(rng.permutation(len(query_features)))
    selected = []
    records = []
    acquisition_metadata = []
    model = None
    for step in range(len(query_features) + 1):
        if step in config.budgets:
            if selected:
                x = np.vstack((development_features, query_features[selected]))
                y = np.concatenate((development_labels, query_labels[selected]))
            else:
                x, y = development_features, development_labels
            model, training = fit_model(x, y, runtime)
            choice, inference = predict_selection(model, query_features, runtime)
            regret = float(np.max(query_labels) - query_labels[choice])
            records.append({
                "budget": step, "selected_index": choice, "regret_scaled": regret,
                "unique_queries_used": len(selected), "training_device": training["training_device"],
                "inference_device": inference["inference_device"],
            })
        if step == len(query_features):
            break
        batch_start = (step // config.batch_size) * config.batch_size
        available = [index for index in range(len(query_features)) if index not in selected]
        batch_exclusions = 0
        for chosen in selected[batch_start:]:
            before = len(available)
            available = [index for index in available if _hamming(sequences[chosen], sequences[index]) > config.near_duplicate_max_hamming]
            batch_exclusions += before - len(available)
        if not available:
            raise RuntimeError("frozen batch constraint left no feasible query")
        if strategy == "random":
            choice = int(next(index for index in random_order if index in available))
            metadata = {"device": "not_applicable", "cpu_fallback": False}
        else:
            available_x = query_features[available]
            if strategy == "max_variance":
                scores = posterior_variance_cuda(model, available_x, runtime)
                metadata = {"device": str(scores.device), "cpu_fallback": False}
            elif strategy == "diversity":
                labeled_x = np.vstack((development_features, query_features[selected])) if selected else development_features
                scores = diversity_scores_cuda(available_x, labeled_x, runtime)
                metadata = {"device": str(scores.device), "cpu_fallback": False}
            elif strategy == "expected_improvement":
                scores = expected_improvement_cuda(model, available_x, query_features, runtime)
                metadata = {"device": str(scores.device), "cpu_fallback": False, "integration": "analytic_normal"}
            elif strategy == "top_proxy":
                device, dtype, _ = configure_cuda(config.gpu_index, config.memory_fraction, config.dtype, seed)
                scores = torch.as_tensor(proxy_mean[available], device=device, dtype=dtype)
                metadata = {"device": str(scores.device), "cpu_fallback": False}
            else:
                scores, metadata = regret_quadrature_ladder_cuda(model, available_x, query_features, config, seed + step, runtime.noise_variance)
            best_position = int(torch.argmax(scores).item())
            choice = available[best_position]
            metadata["chosen_available_position"] = best_position
            metadata["chosen_score"] = float(scores[best_position].item())
        metadata.update({
            "step": step, "batch_index": step // config.batch_size,
            "chosen_index": choice, "feasible_count": len(available),
            "within_batch_near_duplicates_excluded": batch_exclusions,
            "near_duplicate_rule": "within_batch_hamming<=8",
        })
        acquisition_metadata.append(metadata)
        selected.append(choice)
    return {
        "strategy": strategy, "seed": seed, "records": records, "query_order": selected,
        "acquisition_metadata": acquisition_metadata, "evaluation_scores_returned_to_learner": False,
        "batch_size": config.batch_size, "near_duplicate_max_hamming": config.near_duplicate_max_hamming,
    }


def run_study_v3(config: HeadroomV3Config, output_dir=None):
    output_dir = config.output_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = config.to_dict()
    frozen["output_dir"] = str(output_dir)
    atomic_json_write(frozen, output_dir / "frozen_config.json")
    development_pools = [load_pool_manifest(config.pool_root / item / "pool_manifest.json") for item in config.development_pool_ids]
    query_pools = [load_pool_manifest(config.pool_root / item / "pool_manifest.json") for item in config.query_pool_ids]
    if any(pool.pool_role != "dev" or pool.synthetic for pool in development_pools) or any(pool.pool_role != "calibration" or pool.synthetic for pool in query_pools):
        raise PermissionError("only real dev/calibration pools are allowed")
    development = combine_pools(development_pools, "combined_development")
    folds = np.concatenate([
        np.full(len(pool), int(pool.pool_id.rsplit("_", 1)[1]) % config.inner_cv_folds, dtype=np.int64)
        for pool in development_pools
    ])
    cohort = {(item["representation"], item["query_pool_id"]) for item in config.curve_cohort}
    cases = []
    selections = {}
    for representation in config.representations:
        development_raw, development_metadata = raw_representation(representation, development, config)
        selection = select_inner_hyperparameters(development_raw, development.measured, folds, config)
        selections[representation] = selection
        atomic_json_write(selection, output_dir / representation / "inner_development_selection.json")
        selected_prior = selection["selected"]["prior_variance"]
        selected_noise = selection["selected"]["noise_variance"]
        runtime = config.runtime(selected_prior, selected_noise, output_dir)
        scaler = FrozenScaler.fit(development_raw, development.measured)
        development_x = scaler.features(development_raw)
        development_y = scaler.labels(development.measured)
        for query in query_pools:
            query_raw, query_metadata = raw_representation(representation, query, config)
            query_x = scaler.features(query_raw)
            query_y = scaler.labels(query.measured)
            result = gate_case(development_x, development_y, query_x, query_y, runtime)
            result.update({
                "representation": representation, "query_pool_id": query.pool_id,
                "development_pool_ids": list(config.development_pool_ids),
                "development_manifest_sha256": development.manifest_sha256,
                "query_manifest_sha256": query.manifest_sha256,
                "development_rows": len(development), "query_rows": len(query),
                "query_sequences": list(query.utr50),
                "feature_scaler": scaler.to_dict(),
                "selected_prior_variance": selected_prior, "selected_noise_variance": selected_noise,
                "inner_development_selection_sha256": hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "development_representation_metadata": development_metadata,
                "query_representation_metadata": query_metadata,
                "zero_label_selected_id": query.row_ids[result["zero_label"]["selected_index"]],
                "full_q_selected_id": query.row_ids[result["full_q"]["selected_index"]],
                "oracle_selected_id": query.row_ids[result["oracle"]["selected_index"]],
                "zero_label_regret": result["zero_label"]["regret_in_scaled_units"] * scaler.target_scale,
                "full_q_regret": result["full_q"]["regret_in_scaled_units"] * scaler.target_scale,
                "repair_headroom": result["repair_headroom_scaled"] * scaler.target_scale,
                "fixed_v2_curve_cohort_member": (representation, query.pool_id) in cohort,
                "outer_labels_read": False, "outer_metrics_computed": False,
            })
            case_dir = output_dir / representation / query.pool_id
            atomic_json_write(result, case_dir / "gate.json")
            if (representation, query.pool_id) in cohort:
                curves = []
                for random_seed in config.random_seeds:
                    curves.append(curve_case_v3(development_x, development_y, query_x, query_y, query.proxy_mean, query.utr50, "random", config, runtime, random_seed))
                for offset, strategy in enumerate(STRATEGIES[1:], start=1):
                    curves.append(curve_case_v3(development_x, development_y, query_x, query_y, query.proxy_mean, query.utr50, strategy, config, runtime, config.acquisition_seed + offset))
                for curve in curves:
                    for record in curve["records"]:
                        record["regret"] = record["regret_scaled"] * scaler.target_scale
                result["curves"] = curves
                result["curve_status"] = "RUN_FIXED_V2_COHORT"
                atomic_json_write(curves, case_dir / "curves.json")
            else:
                result["curves"] = []
                result["curve_status"] = "NOT_IN_FIXED_V2_CURVE_COHORT"
            result["stop_reason"] = None if result["fixed_v2_curve_cohort_member"] else "not one of the six V2-preregistered curve cases"
            atomic_json_write(result, case_dir / "result.json")
            cases.append({
                "representation": representation, "query_pool_id": query.pool_id,
                "zero_label_regret": result["zero_label_regret"], "full_q_regret": result["full_q_regret"],
                "repair_headroom": result["repair_headroom"], "gate_pass": result["gate"]["pass"],
                "curve_status": result["curve_status"], "stop_reason": result["stop_reason"],
                "zero_label_selected_id": result["zero_label_selected_id"], "full_q_selected_id": result["full_q_selected_id"],
                "fixed_v2_curve_cohort_member": result["fixed_v2_curve_cohort_member"],
                "selected_prior_variance": selected_prior, "selected_noise_variance": selected_noise,
            })
            atomic_json_write({"config_sha256": config.sha256(), "completed_cases": len(cases), "total_cases": 48, "cases": cases}, output_dir / "summary.json")
    summary = {
        "schema_version": config.schema_version, "config_sha256": config.sha256(),
        "status": "COMPLETED_ALL_48_GATES_AND_FIXED_SIX_CURVE_CASES",
        "development_rows": len(development), "query_pool_count": len(query_pools),
        "case_count": len(cases), "headroom_present_count": sum(item["gate_pass"] for item in cases),
        "headroom_absent_count": sum(not item["gate_pass"] for item in cases),
        "fixed_curve_case_count": len(cohort), "cases": cases,
        "inner_development_selections": selections,
        "outer_labels_read": False, "outer_metrics_computed": False,
        "acquisition_superiority_claimed": False,
    }
    atomic_json_write(summary, output_dir / "summary.json")
    return summary
