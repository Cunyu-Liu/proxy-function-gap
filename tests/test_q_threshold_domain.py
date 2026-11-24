import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import verify_q_head
from proxy_function_gap.outer_learner import ProductionQScorer
from proxy_function_gap.q_head import Q_THRESHOLD_DOMAIN, QHeadConfig, fit_q_head, save_q_head_artifacts


class Pool(SimpleNamespace):
    def __len__(self):
        return len(self.proxy_mean)


def q_examples(regrets, prefix):
    examples = []
    for index, regret in enumerate(regrets):
        pool_id = f"{prefix}-{index}"
        pool = Pool(
            pool_id=pool_id,
            proxy_mean=np.asarray([0.2, 0.8]),
            proxy_std=np.asarray([0.1, 0.1]),
            proxy_range=np.asarray([0.2, 0.2]),
            support=np.asarray([0.5, 0.5]),
            source_manifests=(),
        )
        record = SimpleNamespace(
            pool_id=pool_id,
            seed=101,
            regret=float(regret),
            abstain=False,
            ensemble_member_item_forwards=6,
        )
        examples.append((pool, record))
    return examples


def test_q_threshold_is_selected_on_platt_calibrated_oof(monkeypatch, tmp_path):
    raw_oof = np.asarray([0.2, 0.4, 0.6, 0.8])
    folds = [{"fold": 0, "best_epoch": 1, "model_hash": "a" * 64, "history": []}]
    platt = {"model_hash": "b" * 64, "loss_type": "log_loss"}
    monkeypatch.setattr("proxy_function_gap.q_head.fit_oof_classifier", lambda *args: (raw_oof, folds, {}))
    monkeypatch.setattr("proxy_function_gap.q_head.predict_with_folds", lambda *args: np.asarray([0.3, 0.7]))
    monkeypatch.setattr("proxy_function_gap.q_head.fit_platt", lambda *args: platt)
    monkeypatch.setattr("proxy_function_gap.q_head.apply_platt", lambda artifact, raw, config: 1 - np.asarray(raw))

    result = fit_q_head(
        q_examples([0.0, 0.0, 1.0, 1.0], "dev"),
        q_examples([0.0, 1.0], "calibration"),
        QHeadConfig(thresholds=(0.5, 0.75), min_coverage=0.5),
        "f" * 64,
    )

    assert np.allclose(result["calibrated_oof"], [0.8, 0.6, 0.4, 0.2])
    assert result["selected_threshold"] == 0.75
    assert result["threshold_rows"] == [
        {"threshold": 0.5, "coverage": 0.5, "mean_regret": 1.0},
        {"threshold": 0.75, "coverage": 0.75, "mean_regret": 2 / 3},
    ]
    manifest = save_q_head_artifacts(result, tmp_path / "q")
    assert manifest["threshold_domain"] == Q_THRESHOLD_DOMAIN
    assert manifest["threshold_grid"] == [0.5, 0.75]
    assert manifest["min_coverage"] == 0.5
    with np.load(tmp_path / "q" / "q_oof_predictions.npz", allow_pickle=False) as saved:
        assert np.allclose(saved["raw_q"], raw_oof)
        assert np.allclose(saved["calibrated_q"], result["calibrated_oof"])


@pytest.mark.parametrize("domain", [None, "raw_oof_probability"])
def test_production_q_scorer_rejects_missing_or_raw_threshold_domain(tmp_path, domain):
    manifest = {"calibrator_loss_type": "log_loss", "E1_enabled": False, "files": {}}
    if domain is not None:
        manifest["threshold_domain"] = domain
    (tmp_path / "q_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(PermissionError, match="threshold_domain.*unsafe"):
        ProductionQScorer(tmp_path)


def test_verify_q_head_rejects_legacy_manifest_before_reading_pools(monkeypatch, tmp_path):
    (tmp_path / "q_manifest.json").write_text(json.dumps({"files": {}}))
    monkeypatch.setattr(sys, "argv", [
        "verify_q_head.py",
        "--artifacts", str(tmp_path),
        "--pool-root", str(tmp_path / "missing-pools"),
        "--selection-dir", str(tmp_path / "missing-selections"),
        "--count", "48",
        "--round", "1",
    ])

    with pytest.raises(PermissionError, match="threshold_domain.*unsafe"):
        verify_q_head.main()


def test_production_q_scorer_compares_calibrated_probability_in_declared_domain(monkeypatch):
    scorer = ProductionQScorer.__new__(ProductionQScorer)
    scorer.manifest = {"threshold_domain": Q_THRESHOLD_DOMAIN}
    scorer.device = torch.device("cpu")
    scorer.dtype = torch.float32
    scorer.folds = [("fold-low", None), ("fold-high", None)]
    scorer.platt = ("platt", None)
    scorer.threshold = 0.75
    platt_inputs = []

    def predict(model, scaler, values, device, dtype):
        if model == "fold-low":
            return np.asarray([0.2])
        if model == "fold-high":
            return np.asarray([0.4])
        platt_inputs.append(values.copy())
        return np.asarray([0.8])

    monkeypatch.setattr("proxy_function_gap.outer_learner.predict_model", predict)
    pool = Pool(
        pool_id="outer-0",
        proxy_mean=np.asarray([0.2, 0.8]),
        proxy_std=np.asarray([0.1, 0.1]),
        proxy_range=np.asarray([0.2, 0.2]),
        support=np.asarray([0.5, 0.5]),
    )
    decision = scorer.predict(pool, SimpleNamespace(seed=101, ensemble_member_item_forwards=6))

    assert np.allclose(platt_inputs, [[[0.3]]])
    assert decision["raw_q_probability"] == pytest.approx(0.3)
    assert decision["q_probability"] == 0.8
    assert decision["threshold"] == 0.75
    assert decision["threshold_domain"] == Q_THRESHOLD_DOMAIN
    assert decision["abstain"] is True
