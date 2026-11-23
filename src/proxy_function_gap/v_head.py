import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pool_manifest import PoolData
from .small_model import SmallModelConfig, apply_platt, fit_oof_classifier, fit_platt, log_loss, predict_with_folds
from .trainer.checkpoint import atomic_json_write, atomic_torch_save
from .trainer.data import sha256_file


FEATURE_SETS = (
    ("proxy_mean",),
    ("proxy_mean", "proxy_std", "proxy_range", "support"),
)


@dataclass(frozen=True)
class VHeadConfig:
    deltas: tuple = (0.0, 0.25, 0.5)
    feature_sets: tuple = FEATURE_SETS
    model_types: tuple = ("logistic", "mlp")
    learning_rate: float = 0.05
    min_delta: float = 0.0001
    patience: int = 3
    fold_count: int = 3
    memory_fraction: float = 0.05

    def sha256(self):
        value = {
            "deltas": list(self.deltas),
            "feature_sets": [list(item) for item in self.feature_sets],
            "model_types": list(self.model_types),
            "learning_rate": self.learning_rate,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "fold_count": self.fold_count,
            "memory_fraction": self.memory_fraction,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_matrix(pool: PoolData, names):
    values = {
        "proxy_mean": pool.proxy_mean,
        "proxy_std": pool.proxy_std,
        "proxy_range": pool.proxy_range,
        "support": pool.support,
    }
    return np.column_stack([values[name] for name in names]).astype(np.float64)


def fit_v_head(dev_pool: PoolData, calibration_pool: PoolData, config: VHeadConfig):
    candidates = []
    selected = None
    for delta in config.deltas:
        for feature_names in config.feature_sets:
            for model_type in config.model_types:
                small = SmallModelConfig(
                    model_type=model_type,
                    feature_names=feature_names,
                    learning_rate=config.learning_rate,
                    min_delta=config.min_delta,
                    patience=config.patience,
                    fold_count=config.fold_count,
                    memory_fraction=config.memory_fraction,
                )
                x_dev = feature_matrix(dev_pool, feature_names)
                y_dev = (dev_pool.proxy_mean > dev_pool.measured + delta).astype(np.float64)
                oof, folds, assignments = fit_oof_classifier(x_dev, y_dev, dev_pool.group_ids, small)
                score = log_loss(y_dev, oof)
                record = {
                    "delta": delta,
                    "feature_names": list(feature_names),
                    "model_type": model_type,
                    "oof_log_loss": score,
                }
                candidates.append(record)
                if selected is None or score < selected[0] - 1e-12:
                    selected = (score, delta, feature_names, model_type, small, oof, folds, assignments, y_dev)
    _, delta, feature_names, model_type, small, oof, folds, assignments, y_dev = selected
    calibration_x = feature_matrix(calibration_pool, feature_names)
    calibration_targets = (calibration_pool.proxy_mean > calibration_pool.measured + delta).astype(np.float64)
    calibration_raw = predict_with_folds(folds, calibration_x, small)
    platt = fit_platt(calibration_raw, calibration_targets, calibration_pool.group_ids, small)
    calibrated = apply_platt(platt, calibration_raw, small)
    return {
        "config": config,
        "small_config": small,
        "dev_pool": dev_pool,
        "calibration_pool": calibration_pool,
        "candidates": candidates,
        "selected": {"delta": delta, "feature_names": feature_names, "model_type": model_type, "oof_log_loss": selected[0]},
        "oof": oof,
        "oof_targets": y_dev,
        "folds": folds,
        "assignments": assignments,
        "calibration_raw": calibration_raw,
        "calibration_targets": calibration_targets,
        "calibrated": calibrated,
        "platt": platt,
    }


def save_v_head_artifacts(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = result["config"]
    dev_targets = result["oof_targets"]
    oof_path = output / "oof_predictions.npz"
    np.savez(
        oof_path,
        raw_risk=result["oof"],
        raw_v=1 - result["oof"],
        target=dev_targets,
        row_id=np.asarray(result["dev_pool"].row_ids),
        group_id=np.asarray(result["dev_pool"].group_ids),
        fold=np.asarray([result["assignments"][group] for group in result["dev_pool"].group_ids]),
    )
    calibration_path = output / "calibration_predictions.npz"
    np.savez(
        calibration_path,
        row_id=np.asarray(result["calibration_pool"].row_ids),
        group_id=np.asarray(result["calibration_pool"].group_ids),
        raw_risk=result["calibration_raw"],
        calibrated_risk=result["calibrated"],
        calibrated_v=1 - result["calibrated"],
        target=result["calibration_targets"],
    )
    folds_path = output / "fold_models.pt"
    calibrator_path = output / "platt.pt"
    atomic_torch_save(result["folds"], folds_path)
    atomic_torch_save(result["platt"], calibrator_path)
    manifest = {
        "schema_version": "1.0",
        "config_sha256": config.sha256(),
        "input_pools": [
            {"path": str(path), "sha256": sha256}
            for path, sha256 in result["dev_pool"].source_manifests + result["calibration_pool"].source_manifests
        ],
        "selected": result["selected"],
        "candidate_scores": result["candidates"],
        "oof_log_loss": result["selected"]["oof_log_loss"],
        "fold_models": [
            {"fold": artifact["fold"], "best_epoch": artifact["best_epoch"], "model_hash": artifact["model_hash"], "history": artifact["history"]}
            for artifact in result["folds"]
        ],
        "calibrator_model_hash": result["platt"]["model_hash"],
        "calibrator_loss_type": result["platt"]["loss_type"],
        "fold_loss_type": result["folds"][0]["loss_type"],
        "files": {
            path.name: sha256_file(path)
            for path in (oof_path, calibration_path, folds_path, calibrator_path)
        },
        "dev_grouped_oof_only_for_selection": True,
        "calibration_only_for_final_probability_calibration": True,
        "V_definition": "V=1-r, r=P(proxy_mean-measured>delta)",
        "outer_labels_read": False,
        "outer_metrics_computed": False,
    }
    atomic_json_write(manifest, output / "v_manifest.json")
    return manifest
