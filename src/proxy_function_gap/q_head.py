import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pool_manifest import PoolData
from .selection_eval import SelectionRecord
from .small_model import SmallModelConfig, apply_platt, fit_oof_classifier, fit_platt, predict_with_folds
from .trainer.checkpoint import atomic_json_write, atomic_torch_save
from .trainer.data import sha256_file


@dataclass(frozen=True)
class QHeadConfig:
    delta_regret: float = 0.1
    thresholds: tuple = (0.1, 0.25, 0.5, 0.75, 0.9)
    min_coverage: float = 0.5
    learning_rate: float = 0.05
    min_delta: float = 0.0001
    patience: int = 3
    fold_count: int = 3
    memory_fraction: float = 0.05

    def sha256(self):
        value = {
            "delta_regret": self.delta_regret,
            "thresholds": list(self.thresholds),
            "min_coverage": self.min_coverage,
            "learning_rate": self.learning_rate,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "fold_count": self.fold_count,
            "memory_fraction": self.memory_fraction,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pool_features(pool: PoolData, record: SelectionRecord):
    return np.asarray([
        np.max(pool.proxy_mean),
        np.mean(pool.proxy_std),
        np.max(pool.proxy_range),
        np.mean(pool.support),
        np.log1p(len(pool)),
        np.log1p(record.ensemble_member_item_forwards),
    ], dtype=np.float64)


def _examples(examples, selection_config_sha256):
    features = []
    targets = []
    regrets = []
    groups = []
    for pool, record in examples:
        if record.pool_id != pool.pool_id or record.abstain:
            raise ValueError("Q examples require a frozen non-abstain selection")
        if not selection_config_sha256:
            raise PermissionError("selector must be frozen before Q training")
        features.append(pool_features(pool, record))
        regrets.append(record.regret)
        groups.append(pool.pool_id)
    return np.asarray(features), np.asarray(regrets), np.asarray(groups)


def fit_q_head(dev_examples, calibration_examples, config: QHeadConfig, selection_config_sha256):
    small = SmallModelConfig(
        model_type="logistic",
        feature_names=("max_proxy", "mean_std", "max_range", "mean_support", "log_pool_size", "log_member_forwards"),
        learning_rate=config.learning_rate,
        min_delta=config.min_delta,
        patience=config.patience,
        fold_count=config.fold_count,
        memory_fraction=config.memory_fraction,
    )
    x_dev, dev_regret, dev_groups = _examples(dev_examples, selection_config_sha256)
    y_dev = (dev_regret > config.delta_regret).astype(np.float64)
    oof, folds, assignments = fit_oof_classifier(x_dev, y_dev, dev_groups, small)
    threshold_rows = []
    selected = None
    for threshold in config.thresholds:
        abstain = oof >= threshold
        coverage = float(1 - abstain.mean())
        if coverage < config.min_coverage:
            continue
        mean_regret = float(dev_regret[~abstain].mean()) if (~abstain).any() else 0.0
        row = {"threshold": threshold, "coverage": coverage, "mean_regret": mean_regret}
        threshold_rows.append(row)
        if selected is None or mean_regret < selected[0] - 1e-12 or (mean_regret == selected[0] and coverage > selected[1]):
            selected = (mean_regret, coverage, threshold)
    if selected is None:
        raise ValueError("no Q threshold satisfies minimum coverage")
    x_cal, cal_regret, cal_groups = _examples(calibration_examples, selection_config_sha256)
    y_cal = (cal_regret > config.delta_regret).astype(np.float64)
    raw_cal = predict_with_folds(folds, x_cal, small)
    platt = fit_platt(raw_cal, y_cal, cal_groups, small)
    calibrated = apply_platt(platt, raw_cal, small)
    return {
        "config": config,
        "small_config": small,
        "selection_config_sha256": selection_config_sha256,
        "folds": folds,
        "assignments": assignments,
        "oof": oof,
        "oof_targets": y_dev,
        "oof_regret": dev_regret,
        "dev_groups": dev_groups,
        "threshold_rows": threshold_rows,
        "selected_threshold": selected[2],
        "calibration_raw": raw_cal,
        "calibration_targets": y_cal,
        "calibration_regret": cal_regret,
        "calibration_groups": cal_groups,
        "calibrated": calibrated,
        "platt": platt,
        "input_pools": tuple(dict.fromkeys(source for pool, _ in dev_examples + calibration_examples for source in pool.source_manifests)),
    }


def save_q_head_artifacts(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    oof_path = output / "q_oof_predictions.npz"
    np.savez(oof_path, raw_q=result["oof"], target=result["oof_targets"], regret=result["oof_regret"], pool_id=np.asarray(result["dev_groups"]))
    calibration_path = output / "q_calibration_predictions.npz"
    np.savez(calibration_path, raw_q=result["calibration_raw"], calibrated_q=result["calibrated"], target=result["calibration_targets"], regret=result["calibration_regret"], pool_id=np.asarray(result["calibration_groups"]))
    folds_path = output / "q_fold_models.pt"
    platt_path = output / "q_platt.pt"
    atomic_torch_save(result["folds"], folds_path)
    atomic_torch_save(result["platt"], platt_path)
    manifest = {
        "schema_version": "1.0",
        "config_sha256": result["config"].sha256(),
        "selection_config_sha256": result["selection_config_sha256"],
        "input_pools": [{"path": str(path), "sha256": sha256} for path, sha256 in result["input_pools"]],
        "selection_sources": result.get("selection_sources", []),
        "selector_frozen_before_q": True,
        "q_usage": "abstain/risk-coverage only; never candidate ranking",
        "delta_regret": result["config"].delta_regret,
        "selected_threshold": result["selected_threshold"],
        "threshold_rows": result["threshold_rows"],
        "fold_models": [
            {"fold": artifact["fold"], "best_epoch": artifact["best_epoch"], "model_hash": artifact["model_hash"], "history": artifact["history"]}
            for artifact in result["folds"]
        ],
        "calibrator_model_hash": result["platt"]["model_hash"],
        "calibrator_loss_type": result["platt"]["loss_type"],
        "files": {path.name: sha256_file(path) for path in (oof_path, calibration_path, folds_path, platt_path)},
        "outer_labels_read": False,
        "outer_metrics_computed": False,
        "E1_enabled": False,
    }
    atomic_json_write(manifest, output / "q_manifest.json")
    return manifest


def q_abstain(q_probability, threshold):
    return bool(q_probability >= threshold)
