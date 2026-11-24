import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .bayesian_ridge import BayesianRidge
from .paper3_replay import ReplayConfig, fit_ridge_cuda, select_posterior_mean_cuda
from .pool_manifest import combine_pools, load_pool_manifest
from .trainer.checkpoint import atomic_json_write, load_checkpoint
from .trainer.cuda import configure_cuda
from .trainer.data import encode_50nt, sha256_file
from .trainer.model import FreshSampleCNN, model_definition_hash, model_hash, state_dict_hash


REPRESENTATIONS = ("summary4", "kmer4", "cnn64")


@dataclass(frozen=True)
class HeadroomConfig:
    schema_version: str
    protocol_frozen_before_pool_results: bool
    pool_root: Path
    development_pool_ids: tuple
    query_pool_ids: tuple
    representations: tuple
    output_dir: Path
    budgets: tuple
    random_seeds: tuple
    acquisition_seed: int
    posterior_samples: int
    label_samples: int
    noise_variance: float
    prior_variance: float
    min_repair_margin: float
    gpu_index: int
    memory_fraction: float
    dtype: str
    cnn_batch_size: int
    cnn_members: tuple

    def __post_init__(self):
        if self.schema_version != "paper3-dev-query-headroom-v2" or self.protocol_frozen_before_pool_results is not True:
            raise ValueError("a frozen Paper3 development/query v2 protocol is required")
        if not self.development_pool_ids or not self.query_pool_ids:
            raise ValueError("development and query pool sets must be non-empty")
        if any(not item.startswith("dev_") for item in self.development_pool_ids):
            raise PermissionError("development IDs must have dev roles")
        if any(not item.startswith("calibration_") for item in self.query_pool_ids):
            raise PermissionError("query IDs must have calibration roles")
        if len(set(self.development_pool_ids)) != len(self.development_pool_ids) or len(set(self.query_pool_ids)) != len(self.query_pool_ids):
            raise ValueError("pool IDs must be unique")
        if tuple(self.representations) != REPRESENTATIONS:
            raise ValueError("all preregistered representations must be present in fixed order")
        if tuple(sorted(set(self.budgets))) != tuple(self.budgets) or self.budgets[0] != 0 or self.budgets[-1] <= 0:
            raise ValueError("budgets must be unique, sorted and span zero through a positive full Q")
        if len(self.random_seeds) != 3 or len(set(self.random_seeds)) != 3:
            raise ValueError("exactly three unique random seeds are preregistered")
        if self.posterior_samples <= 0 or self.label_samples <= 0 or self.cnn_batch_size <= 0:
            raise ValueError("sample and batch sizes must be positive")
        if not np.isfinite(self.noise_variance) or self.noise_variance <= 0 or not np.isfinite(self.prior_variance) or self.prior_variance <= 0:
            raise ValueError("Bayesian variances must be finite and positive")
        if not np.isfinite(self.min_repair_margin) or self.min_repair_margin < 0:
            raise ValueError("min_repair_margin must be finite and non-negative")
        if len(self.cnn_members) != 3:
            raise ValueError("three frozen CNN members are required")
        for member in self.cnn_members:
            if len(member["sha256"]) != 64 or len(member["model_hash"]) != 64:
                raise ValueError("CNN checkpoint and model hashes are required")
        for value in (*self.development_pool_ids, *self.query_pool_ids):
            if "outer" in value.lower():
                raise PermissionError("outer paths and IDs are forbidden")

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text())
        fields = set(cls.__dataclass_fields__)
        if set(value) != fields:
            raise ValueError(f"config fields differ from frozen schema: missing={sorted(fields-set(value))}, unknown={sorted(set(value)-fields)}")
        return cls(
            **{
                **value,
                "pool_root": Path(value["pool_root"]),
                "development_pool_ids": tuple(value["development_pool_ids"]),
                "query_pool_ids": tuple(value["query_pool_ids"]),
                "representations": tuple(value["representations"]),
                "output_dir": Path(value["output_dir"]),
                "budgets": tuple(value["budgets"]),
                "random_seeds": tuple(value["random_seeds"]),
                "cnn_members": tuple(value["cnn_members"]),
            }
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "protocol_frozen_before_pool_results": self.protocol_frozen_before_pool_results,
            "pool_root": str(self.pool_root),
            "development_pool_ids": list(self.development_pool_ids),
            "query_pool_ids": list(self.query_pool_ids),
            "representations": list(self.representations),
            "output_dir": str(self.output_dir),
            "budgets": list(self.budgets),
            "random_seeds": list(self.random_seeds),
            "acquisition_seed": self.acquisition_seed,
            "posterior_samples": self.posterior_samples,
            "label_samples": self.label_samples,
            "noise_variance": self.noise_variance,
            "prior_variance": self.prior_variance,
            "min_repair_margin": self.min_repair_margin,
            "gpu_index": self.gpu_index,
            "memory_fraction": self.memory_fraction,
            "dtype": self.dtype,
            "cnn_batch_size": self.cnn_batch_size,
            "cnn_members": list(self.cnn_members),
        }

    def sha256(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def replay_config(self):
        return ReplayConfig(
            Path("development"), Path("query"), self.output_dir, 64,
            random_seed=self.acquisition_seed, noise_variance=self.noise_variance,
            prior_variance=self.prior_variance, min_repair_margin=self.min_repair_margin,
            gpu_index=self.gpu_index, memory_fraction=self.memory_fraction, dtype=self.dtype,
        )


@dataclass(frozen=True)
class FrozenScaler:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    keep: np.ndarray
    target_mean: float
    target_scale: float

    @classmethod
    def fit(cls, features, labels):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        mean = features.mean(axis=0)
        raw_scale = features.std(axis=0)
        keep = raw_scale > 1e-12
        if not keep.any():
            raise ValueError("representation has no development variation")
        target_scale = float(labels.std())
        return cls(mean, np.where(keep, raw_scale, 1.0), keep, float(labels.mean()), target_scale if target_scale > 0 else 1.0)

    def features(self, values):
        return ((np.asarray(values, dtype=np.float64) - self.feature_mean) / self.feature_scale)[:, self.keep]

    def labels(self, values):
        return (np.asarray(values, dtype=np.float64) - self.target_mean) / self.target_scale

    def to_dict(self):
        return {
            "fit_role": "combined_development_only",
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "retained_columns": self.keep.tolist(),
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
        }


def summary4(pool):
    return np.column_stack((pool.proxy_mean, pool.proxy_std, pool.proxy_range, pool.support)).astype(np.float64)


def kmer4(sequences, k=4):
    if k != 4:
        raise ValueError("the frozen representation uses k=4")
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    result = np.zeros((len(sequences), 256), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        if len(sequence) < k or set(sequence) - set(mapping):
            raise ValueError("k-mer input must contain only A/C/G/T")
        for start in range(len(sequence) - k + 1):
            index = 0
            for base in sequence[start:start + k]:
                index = index * 4 + mapping[base]
            result[row, index] += 1.0
        result[row] /= len(sequence) - k + 1
    return result


def cnn64(pool, config):
    inputs = encode_50nt(pool.utr50)
    hidden_members = []
    prediction_members = []
    metadata = []
    for member in config.cnn_members:
        checkpoint_path = Path(member["checkpoint"])
        if sha256_file(checkpoint_path) != member["sha256"]:
            raise ValueError(f"CNN checkpoint hash mismatch for seed {member['seed']}")
        device, dtype, gpu = configure_cuda(config.gpu_index, config.memory_fraction, "float32", member["seed"])
        checkpoint = load_checkpoint(checkpoint_path, device)
        model = FreshSampleCNN()
        if checkpoint["model_definition_sha256"] != model_definition_hash(model):
            raise ValueError("CNN model definition mismatch")
        model.load_state_dict(checkpoint["model_state"])
        model.to(device=device, dtype=dtype)
        model.eval()
        if model_hash(model) != member["model_hash"] or state_dict_hash(model.state_dict()) != checkpoint["model_state_sha256"]:
            raise ValueError("frozen CNN model hash mismatch")
        hidden_batches = []
        prediction_batches = []
        with torch.no_grad():
            for start in range(0, len(inputs), config.cnn_batch_size):
                values = torch.from_numpy(inputs[start:start + config.cnn_batch_size]).to(device=device, dtype=dtype)
                if not values.is_cuda:
                    raise RuntimeError("CPU CNN inference is forbidden")
                convolution = model.features(values).mean(dim=2)
                hidden = model.regressor[1](model.regressor[0](convolution))
                hidden = model.regressor[2](hidden)
                prediction = model.regressor[3](hidden).squeeze(1)
                if not hidden.is_cuda or not prediction.is_cuda:
                    raise RuntimeError("CNN representation inference left CUDA")
                hidden_batches.append(hidden.detach().cpu().numpy())
                prediction_batches.append(prediction.detach().cpu().numpy())
        torch.cuda.synchronize(device)
        hidden_members.append(np.concatenate(hidden_batches).astype(np.float64))
        scaled_prediction = np.concatenate(prediction_batches).astype(np.float64)
        prediction_members.append(scaled_prediction * checkpoint["scaler"]["scale"] + checkpoint["scaler"]["mean"])
        metadata.append({"seed": member["seed"], "checkpoint_sha256": member["sha256"], "model_hash": member["model_hash"], "gpu": gpu, "cpu_fallback": False})
        del model
        torch.cuda.empty_cache()
    expected = np.asarray(pool.member_predictions)
    actual = np.asarray(prediction_members)
    max_prediction_error = float(np.max(np.abs(actual - expected)))
    if max_prediction_error > 2e-5:
        raise ValueError(f"CNN representation predictions do not reproduce frozen member predictions: {max_prediction_error}")
    hidden = np.asarray(hidden_members)
    return np.concatenate((hidden.mean(axis=0), hidden.std(axis=0, ddof=1)), axis=1), {
        "members": metadata,
        "hidden_definition": "member penultimate 32-ReLU activations; ensemble mean and sample standard deviation",
        "max_member_prediction_absolute_error": max_prediction_error,
        "forward_rows": len(pool),
        "device": "cuda:0",
        "cpu_fallback": False,
    }


def raw_representation(name, pool, config):
    if name == "summary4":
        return summary4(pool), {"definition": "proxy mean/std/range and train-only support", "device": "not_applicable"}
    if name == "kmer4":
        return kmer4(pool.utr50), {"definition": "normalized 4-mer frequencies", "device": "not_applicable"}
    if name == "cnn64":
        return cnn64(pool, config)
    raise ValueError(f"unknown representation {name}")


def fit_model(features, labels, config):
    model, metadata = fit_ridge_cuda(features, labels, config.replay_config())
    return model, metadata


def predict_selection(model, features, config):
    indices, metadata = select_posterior_mean_cuda(model, features, config.replay_config())
    return int(indices[0]), metadata


def rank_and_scale(values):
    singular = np.linalg.svd(values, compute_uv=False)
    return {
        "rows": int(values.shape[0]), "columns": int(values.shape[1]),
        "rank": int(np.linalg.matrix_rank(values)),
        "singular_values": singular.tolist(),
        "condition_nonzero": float(singular[0] / singular[np.linalg.matrix_rank(values) - 1]),
    }


def posterior_change(zero_model, full_model, query_features):
    zero = query_features @ zero_model.mean
    full = query_features @ full_model.mean
    delta = full - zero
    zero_rank = np.argsort(np.argsort(zero, kind="stable"), kind="stable")
    full_rank = np.argsort(np.argsort(full, kind="stable"), kind="stable")
    return {
        "prediction_delta_mean_absolute": float(np.mean(np.abs(delta))),
        "prediction_delta_max_absolute": float(np.max(np.abs(delta))),
        "prediction_delta_l2": float(np.linalg.norm(delta)),
        "zero_prediction_range": [float(zero.min()), float(zero.max())],
        "full_prediction_range": [float(full.min()), float(full.max())],
        "rank_correlation": float(np.corrcoef(zero_rank, full_rank)[0, 1]),
        "zero_weights": zero_model.mean.tolist(),
        "full_weights": full_model.mean.tolist(),
    }


def gate_case(development_features, development_labels, query_features, query_labels, config):
    zero_model, zero_training = fit_model(development_features, development_labels, config)
    full_x = np.vstack((development_features, query_features))
    full_y = np.concatenate((development_labels, query_labels))
    full_model, full_training = fit_model(full_x, full_y, config)
    brute = BayesianRidge.fit(full_x, full_y, prior_covariance=np.eye(full_x.shape[1]) * config.prior_variance, noise_variance=config.noise_variance)
    weight_error = float(np.max(np.abs(brute.mean - full_model.mean)))
    covariance_error = float(np.max(np.abs(brute.covariance - full_model.covariance)))
    if weight_error > 1e-8 or covariance_error > 1e-8:
        raise RuntimeError("CUDA and brute-force full-Q refits differ")
    zero_index, zero_inference = predict_selection(zero_model, query_features, config)
    full_index, full_inference = predict_selection(full_model, query_features, config)
    oracle_index = int(np.argmax(query_labels))
    zero_regret = float(query_labels[oracle_index] - query_labels[zero_index])
    full_regret = float(query_labels[oracle_index] - query_labels[full_index])
    headroom = zero_regret - full_regret
    passed = headroom > config.min_repair_margin
    return {
        "zero_label": {"selected_index": zero_index, "regret_in_scaled_units": zero_regret, "training": zero_training, "inference": zero_inference},
        "full_q": {"selected_index": full_index, "regret_in_scaled_units": full_regret, "training": full_training, "inference": full_inference},
        "oracle": {"selected_index": oracle_index, "zero_baseline_oracle_gap_scaled": zero_regret},
        "repair_headroom_scaled": headroom,
        "gate": {"rule": "zero_label_regret - full_q_regret > min_repair_margin", "min_repair_margin": config.min_repair_margin, "pass": passed, "status": "HEADROOM_PRESENT" if passed else "STOP_NO_REPAIR_HEADROOM"},
        "brute_force_refit": {"weight_max_absolute_error": weight_error, "covariance_max_absolute_error": covariance_error, "argmax_matches": int(np.argmax(query_features @ brute.mean)) == full_index},
        "posterior_change": posterior_change(zero_model, full_model, query_features),
        "development_diagnostics": rank_and_scale(development_features),
        "query_diagnostics": rank_and_scale(query_features),
        "development_label_diagnostics": {"mean": float(development_labels.mean()), "std": float(development_labels.std()), "min": float(development_labels.min()), "max": float(development_labels.max())},
        "query_label_diagnostics": {"mean": float(query_labels.mean()), "std": float(query_labels.std()), "min": float(query_labels.min()), "max": float(query_labels.max())},
        "outer_labels_read": False,
        "outer_metrics_computed": False,
    }


def posterior_variance_cuda(model, features, config):
    device, dtype, _ = configure_cuda(config.gpu_index, config.memory_fraction, config.dtype, config.acquisition_seed)
    x = torch.as_tensor(features, device=device, dtype=dtype)
    covariance = torch.as_tensor(model.covariance, device=device, dtype=dtype)
    variance = torch.einsum("ij,jk,ik->i", x, covariance, x)
    if not variance.is_cuda or not torch.isfinite(variance).all():
        raise RuntimeError("invalid CUDA posterior variance")
    return variance


def regret_scores_cuda(model, available_features, evaluation_features, config, seed):
    device, dtype, gpu = configure_cuda(config.gpu_index, config.memory_fraction, config.dtype, seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    mean = torch.as_tensor(model.mean, device=device, dtype=dtype)
    covariance = torch.as_tensor(model.covariance, device=device, dtype=dtype)
    query = torch.as_tensor(available_features, device=device, dtype=dtype)
    evaluation = torch.as_tensor(evaluation_features, device=device, dtype=dtype)
    noise = float(config.noise_variance)
    samples = int(config.posterior_samples)
    label_samples = int(config.label_samples)
    v = query @ covariance
    latent_variance = torch.sum(v * query, dim=1).clamp_min(0)
    predictive_variance = latent_variance + noise
    query_mean = query @ mean
    fantasy = query_mean[:, None] + torch.sqrt(predictive_variance)[:, None] * torch.randn((len(query), label_samples), device=device, dtype=dtype, generator=generator)
    updated_mean = mean[None, None, :] + v[:, None, :] * ((fantasy - query_mean[:, None]) / predictive_variance[:, None])[:, :, None]
    updated_choice = torch.argmax(updated_mean @ evaluation.T, dim=2)
    z = torch.randn((samples, model.mean.size), device=device, dtype=dtype, generator=generator)
    base_outcomes = z @ evaluation.T
    xz = z @ query.T
    v_evaluation = v @ evaluation.T
    epsilon = torch.randn((label_samples, samples), device=device, dtype=dtype, generator=generator)
    coefficient = xz.T[:, None, :] / predictive_variance[:, None, None] - (np.sqrt(noise) / predictive_variance)[:, None, None] * epsilon[None, :, :]
    outcomes = updated_mean @ evaluation.T
    outcomes = outcomes[:, :, None, :] + base_outcomes[None, None, :, :] - coefficient[:, :, :, None] * v_evaluation[:, None, None, :]
    selected_outcomes = torch.gather(outcomes, 3, updated_choice[:, :, None, None].expand(-1, -1, samples, 1)).squeeze(3)
    expected_regret = (outcomes.max(dim=3).values - selected_outcomes).mean(dim=(1, 2))
    scores = -expected_regret
    torch.cuda.synchronize(device)
    if not scores.is_cuda or not torch.isfinite(scores).all():
        raise RuntimeError("non-finite CUDA regret acquisition")
    return scores, {"device": str(scores.device), "cpu_fallback": False, "gpu": gpu, "posterior_samples": samples, "label_samples": label_samples}


def curve_case(development_features, development_labels, query_features, query_labels, strategy, config, seed):
    rng = np.random.default_rng(seed)
    random_order = list(rng.permutation(len(query_features))) if strategy == "random" else None
    selected = []
    available = list(range(len(query_features)))
    records = []
    acquisition_metadata = []
    for step in range(len(query_features) + 1):
        if step in config.budgets:
            if selected:
                x = np.vstack((development_features, query_features[selected]))
                y = np.concatenate((development_labels, query_labels[selected]))
            else:
                x, y = development_features, development_labels
            model, training = fit_model(x, y, config)
            choice, inference = predict_selection(model, query_features, config)
            regret = float(np.max(query_labels) - query_labels[choice])
            records.append({"budget": step, "selected_index": choice, "regret_scaled": regret, "unique_queries_used": len(selected), "training_device": training["training_device"], "inference_device": inference["inference_device"]})
        if step == len(query_features):
            break
        if strategy == "random":
            choice = int(random_order[step])
        else:
            if strategy == "max_variance":
                scores = posterior_variance_cuda(model, query_features[available], config)
                metadata = {"device": str(scores.device), "cpu_fallback": False}
            elif strategy == "regret":
                scores, metadata = regret_scores_cuda(model, query_features[available], query_features, config, seed + step)
            else:
                raise ValueError("strategy must be random, regret or max_variance")
            best_position = int(torch.argmax(scores).item())
            choice = available[best_position]
            acquisition_metadata.append({"step": step, "chosen_index": choice, **metadata})
        if choice not in available:
            raise RuntimeError("acquisition selected an already revealed query")
        available.remove(choice)
        selected.append(choice)
    return {"strategy": strategy, "seed": seed, "records": records, "query_order": selected, "acquisition_metadata": acquisition_metadata, "evaluation_scores_returned_to_learner": False}


def run_study(config: HeadroomConfig):
    config.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json_write(config.to_dict(), config.output_dir / "frozen_config.json")
    development_pools = [load_pool_manifest(config.pool_root / item / "pool_manifest.json") for item in config.development_pool_ids]
    query_pools = [load_pool_manifest(config.pool_root / item / "pool_manifest.json") for item in config.query_pool_ids]
    if any(pool.pool_role != "dev" or pool.synthetic for pool in development_pools) or any(pool.pool_role != "calibration" or pool.synthetic for pool in query_pools):
        raise PermissionError("only real dev/calibration pools are allowed")
    if any(len(pool) != config.budgets[-1] for pool in query_pools):
        raise ValueError("the final budget must equal every query pool size")
    development = combine_pools(development_pools, "combined_development")
    cases = []
    for representation in config.representations:
        development_raw, development_metadata = raw_representation(representation, development, config)
        scaler = FrozenScaler.fit(development_raw, development.measured)
        development_x = scaler.features(development_raw)
        development_y = scaler.labels(development.measured)
        for query in query_pools:
            query_raw, query_metadata = raw_representation(representation, query, config)
            query_x = scaler.features(query_raw)
            query_y = scaler.labels(query.measured)
            result = gate_case(development_x, development_y, query_x, query_y, config)
            result.update({
                "representation": representation, "query_pool_id": query.pool_id,
                "development_pool_ids": list(config.development_pool_ids),
                "development_manifest_sha256": development.manifest_sha256,
                "query_manifest_sha256": query.manifest_sha256,
                "development_rows": len(development), "query_rows": len(query),
                "feature_scaler": scaler.to_dict(),
                "development_representation_metadata": development_metadata,
                "query_representation_metadata": query_metadata,
                "zero_label_selected_id": query.row_ids[result["zero_label"]["selected_index"]],
                "full_q_selected_id": query.row_ids[result["full_q"]["selected_index"]],
                "oracle_selected_id": query.row_ids[result["oracle"]["selected_index"]],
                "zero_label_regret": result["zero_label"]["regret_in_scaled_units"] * scaler.target_scale,
                "full_q_regret": result["full_q"]["regret_in_scaled_units"] * scaler.target_scale,
                "repair_headroom": result["repair_headroom_scaled"] * scaler.target_scale,
            })
            case_dir = config.output_dir / representation / query.pool_id
            atomic_json_write(result, case_dir / "gate.json")
            if result["gate"]["pass"]:
                curves = []
                for random_seed in config.random_seeds:
                    curves.append(curve_case(development_x, development_y, query_x, query_y, "random", config, random_seed))
                for offset, strategy in enumerate(("max_variance", "regret"), start=1):
                    curves.append(curve_case(development_x, development_y, query_x, query_y, strategy, config, config.acquisition_seed + offset))
                for curve in curves:
                    for record in curve["records"]:
                        record["regret"] = record["regret_scaled"] * scaler.target_scale
                result["curves"] = curves
                result["curve_status"] = "RUN_HEADROOM_PRESENT"
                atomic_json_write(curves, case_dir / "curves.json")
            else:
                result["curves"] = []
                result["curve_status"] = "STOP_NO_REPAIR_HEADROOM"
            result["stop_reason"] = None if result["gate"]["pass"] else "zero-minus-full-Q did not exceed the frozen positive margin"
            atomic_json_write(result, case_dir / "result.json")
            cases.append({
                "representation": representation, "query_pool_id": query.pool_id,
                "zero_label_regret": result["zero_label_regret"], "full_q_regret": result["full_q_regret"],
                "repair_headroom": result["repair_headroom"], "gate_pass": result["gate"]["pass"],
                "curve_status": result["curve_status"], "stop_reason": result["stop_reason"],
                "zero_label_selected_id": result["zero_label_selected_id"], "full_q_selected_id": result["full_q_selected_id"],
            })
            atomic_json_write({"config_sha256": config.sha256(), "completed_cases": len(cases), "total_cases": len(config.representations) * len(query_pools), "cases": cases}, config.output_dir / "summary.json")
    summary = {
        "schema_version": config.schema_version, "config_sha256": config.sha256(),
        "status": "COMPLETED_ALL_PREREGISTERED_CASES", "development_rows": len(development),
        "query_pool_count": len(query_pools), "case_count": len(cases),
        "headroom_present_count": sum(item["gate_pass"] for item in cases),
        "headroom_absent_count": sum(not item["gate_pass"] for item in cases),
        "cases": cases, "outer_labels_read": False, "outer_metrics_computed": False,
        "acquisition_superiority_claimed": False,
    }
    atomic_json_write(summary, config.output_dir / "summary.json")
    return summary
