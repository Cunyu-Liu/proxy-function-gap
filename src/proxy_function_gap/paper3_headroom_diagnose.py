import argparse
from pathlib import Path

import numpy as np

from .bayesian_ridge import BayesianRidge
from .paper3_replay import FrozenFeatureScaler, ReplayConfig, fit_ridge_cuda, frozen_features
from .pool_manifest import load_pool_manifest
from .trainer.checkpoint import atomic_json_write


def fit(features, labels, config):
    return fit_ridge_cuda(features, labels, config)[0]


def regret(pool, index):
    return float(np.max(pool.measured) - pool.measured[index])


def run(config_path, output_path):
    config = ReplayConfig.from_json(config_path)
    development = load_pool_manifest(config.development_manifest)
    query = load_pool_manifest(config.query_manifest)
    scaler = FrozenFeatureScaler.fit(frozen_features(development))
    development_x = scaler.transform(frozen_features(development))
    query_x = scaler.transform(frozen_features(query))
    full_x = np.vstack((development_x, query_x))
    full_y = np.concatenate((development.measured, query.measured))
    zero = fit(development_x, development.measured, config)
    full = fit(full_x, full_y, config)
    brute = BayesianRidge.fit(full_x, full_y, prior_covariance=np.eye(full_x.shape[1]) * config.prior_variance, noise_variance=config.noise_variance)
    zero_prediction = query_x @ zero.mean
    full_prediction = query_x @ full.mean
    zero_index = int(np.argmax(zero_prediction))
    full_index = int(np.argmax(full_prediction))
    oracle_index = int(np.argmax(query.measured))
    singular = np.linalg.svd(development_x, compute_uv=False)
    prior_sensitivity = []
    for prior_variance in (0.01, 0.1, 1.0, 100.0, 1000000.0):
        local = ReplayConfig(
            config.development_manifest, config.query_manifest, config.output_dir,
            config.query_budget, prior_variance=prior_variance,
            noise_variance=config.noise_variance, gpu_index=config.gpu_index,
            memory_fraction=config.memory_fraction, dtype=config.dtype,
        )
        local_zero = fit(development_x, development.measured, local)
        local_full = fit(full_x, full_y, local)
        zero_choice = int(np.argmax(query_x @ local_zero.mean))
        full_choice = int(np.argmax(query_x @ local_full.mean))
        prior_sensitivity.append({
            "prior_variance": prior_variance, "zero_index": zero_choice, "full_index": full_choice,
            "zero_regret": regret(query, zero_choice), "full_q_regret": regret(query, full_choice),
            "headroom": regret(query, zero_choice) - regret(query, full_choice),
            "prediction_delta_max_absolute": float(np.max(np.abs(query_x @ (local_full.mean - local_zero.mean)))),
        })
    centered = ReplayConfig(
        config.development_manifest, config.query_manifest, config.output_dir,
        config.query_budget, prior_variance=config.prior_variance,
        noise_variance=config.noise_variance, gpu_index=config.gpu_index,
        memory_fraction=config.memory_fraction, dtype=config.dtype,
    )
    target_mean = float(development.measured.mean())
    centered_zero = fit(development_x, development.measured - target_mean, centered)
    centered_full = fit(full_x, full_y - target_mean, centered)
    centered_zero_index = int(np.argmax(query_x @ centered_zero.mean))
    centered_full_index = int(np.argmax(query_x @ centered_full.mean))
    result = {
        "schema_version": "paper3-original-headroom-diagnostic-v1",
        "development_pool_id": development.pool_id, "query_pool_id": query.pool_id,
        "development_rows": len(development), "query_rows": len(query),
        "development_labels": {"mean": float(development.measured.mean()), "std": float(development.measured.std()), "min": float(development.measured.min()), "max": float(development.measured.max())},
        "query_labels": {"mean": float(query.measured.mean()), "std": float(query.measured.std()), "min": float(query.measured.min()), "max": float(query.measured.max())},
        "development_feature_rank": int(np.linalg.matrix_rank(development_x)),
        "development_feature_columns": development_x.shape[1],
        "development_singular_values": singular.tolist(),
        "proxy_mean_minus_member_average_max_absolute": float(np.max(np.abs(development.proxy_mean - development.member_predictions.mean(axis=0)))),
        "zero": {"index": zero_index, "id": query.row_ids[zero_index], "regret": regret(query, zero_index)},
        "full_q": {"index": full_index, "id": query.row_ids[full_index], "regret": regret(query, full_index)},
        "oracle": {"index": oracle_index, "id": query.row_ids[oracle_index], "value": float(query.measured[oracle_index])},
        "posterior_change": {
            "prediction_delta_mean_absolute": float(np.mean(np.abs(full_prediction - zero_prediction))),
            "prediction_delta_max_absolute": float(np.max(np.abs(full_prediction - zero_prediction))),
            "prediction_delta_l2": float(np.linalg.norm(full_prediction - zero_prediction)),
            "zero_prediction_range": [float(zero_prediction.min()), float(zero_prediction.max())],
            "full_prediction_range": [float(full_prediction.min()), float(full_prediction.max())],
            "rank_correlation": float(np.corrcoef(np.argsort(np.argsort(zero_prediction)), np.argsort(np.argsort(full_prediction)))[0, 1]),
        },
        "brute_force_refit": {
            "weight_max_absolute_error": float(np.max(np.abs(brute.mean - full.mean))),
            "covariance_max_absolute_error": float(np.max(np.abs(brute.covariance - full.covariance))),
            "argmax_matches": int(np.argmax(query_x @ brute.mean)) == full_index,
        },
        "prior_sensitivity": prior_sensitivity,
        "target_centering": {
            "development_target_mean": target_mean, "zero_index": centered_zero_index, "full_index": centered_full_index,
            "zero_regret": regret(query, centered_zero_index), "full_q_regret": regret(query, centered_full_index),
            "headroom": regret(query, centered_zero_index) - regret(query, centered_full_index),
        },
        "interpretation_scope": "diagnostics on the already reported development/query pool; prior sensitivity is not a new selected protocol",
        "outer_labels_read": False, "outer_metrics_computed": False,
    }
    atomic_json_write(result, Path(output_path))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose the original Paper3 zero-headroom result")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    import json
    print(json.dumps(run(args.config, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
