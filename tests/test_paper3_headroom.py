import json

import numpy as np
import pytest
import torch

from proxy_function_gap.paper3_headroom import (
    HeadroomConfig,
    FrozenScaler,
    curve_case,
    gate_case,
    kmer4,
    regret_scores_cuda,
    summary4,
)


def config(tmp_path, budgets=(0, 2), posterior_samples=4, label_samples=3):
    members = tuple({"seed": seed, "checkpoint": "unused", "sha256": "0" * 64, "model_hash": "1" * 64} for seed in (11, 23, 37))
    return HeadroomConfig(
        schema_version="paper3-dev-query-headroom-v2",
        protocol_frozen_before_pool_results=True,
        pool_root=tmp_path / "pools",
        development_pool_ids=("dev_000",),
        query_pool_ids=("calibration_000",),
        representations=("summary4", "kmer4", "cnn64"),
        output_dir=tmp_path / "output",
        budgets=budgets,
        random_seeds=(11, 23, 37),
        acquisition_seed=47,
        posterior_samples=posterior_samples,
        label_samples=label_samples,
        noise_variance=0.04,
        prior_variance=1.0,
        min_repair_margin=0.0,
        gpu_index=0,
        memory_fraction=0.05,
        dtype="float64",
        cnn_batch_size=32,
        cnn_members=members,
    )


def test_kmer4_counts_and_normalization():
    actual = kmer4(("AAAA",))
    assert actual.shape == (1, 256)
    assert actual[0, 0] == pytest.approx(1.0)
    assert actual.sum() == pytest.approx(1.0)
    actual = kmer4(("AAAAC",))
    assert actual[0, 0] == pytest.approx(0.5)
    assert actual[0, 1] == pytest.approx(0.5)


def test_frozen_scaler_uses_only_development_statistics():
    development = np.array([[0.0, 1.0], [2.0, 1.0]])
    scaler = FrozenScaler.fit(development, np.array([2.0, 4.0]))
    query = scaler.features(np.array([[4.0, 100.0]]))
    assert query.tolist() == [[3.0]]
    assert scaler.labels(np.array([3.0])).tolist() == [0.0]
    assert scaler.to_dict()["fit_role"] == "combined_development_only"


def test_gate_detects_headroom_and_brute_refit(tmp_path):
    cfg = config(tmp_path)
    development_x = np.eye(2)
    development_y = np.zeros(2)
    query_x = np.eye(2)
    query_y = np.array([0.0, 1.0])
    result = gate_case(development_x, development_y, query_x, query_y, cfg)
    assert result["gate"]["pass"] is True
    assert result["repair_headroom_scaled"] == pytest.approx(1.0)
    assert result["brute_force_refit"]["weight_max_absolute_error"] < 1e-10
    assert result["posterior_change"]["prediction_delta_max_absolute"] > 0
    assert result["zero_label"]["training"]["training_device"] == "cuda:0"


def test_gate_stops_without_relaxing_margin(tmp_path):
    cfg = config(tmp_path)
    result = gate_case(np.eye(2), np.zeros(2), np.eye(2), np.array([1.0, 0.0]), cfg)
    assert result["gate"]["pass"] is False
    assert result["gate"]["status"] == "STOP_NO_REPAIR_HEADROOM"
    assert result["repair_headroom_scaled"] == pytest.approx(0.0)


def test_regret_acquisition_is_cuda_and_deterministic(tmp_path):
    cfg = config(tmp_path)
    development_x = np.eye(2)
    development_y = np.zeros(2)
    query_x = np.eye(2)
    query_y = np.array([0.0, 1.0])
    result = gate_case(development_x, development_y, query_x, query_y, cfg)
    from proxy_function_gap.paper3_replay import fit_ridge_cuda
    model, _ = fit_ridge_cuda(development_x, development_y, cfg.replay_config())
    first, metadata = regret_scores_cuda(model, query_x, query_x, cfg, 91)
    second, _ = regret_scores_cuda(model, query_x, query_x, cfg, 91)
    assert torch.equal(first, second)
    assert first.is_cuda
    assert metadata["cpu_fallback"] is False


def test_curves_respect_unique_budget_and_never_return_scores(tmp_path):
    cfg = config(tmp_path)
    for strategy in ("random", "max_variance", "regret"):
        result = curve_case(np.eye(2), np.zeros(2), np.eye(2), np.array([0.0, 1.0]), strategy, cfg, 19)
        assert [item["budget"] for item in result["records"]] == [0, 2]
        assert result["records"][-1]["unique_queries_used"] == 2
        assert len(set(result["query_order"])) == 2
        assert result["evaluation_scores_returned_to_learner"] is False
        assert all(item["training_device"] == "cuda:0" for item in result["records"])


def test_config_rejects_outer_and_missing_preregistered_representation(tmp_path):
    cfg = config(tmp_path)
    value = cfg.to_dict()
    value["query_pool_ids"] = ["outer_000"]
    path = tmp_path / "outer.json"
    path.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="calibration roles"):
        HeadroomConfig.from_json(path)
    value = cfg.to_dict()
    value["representations"] = ["kmer4", "cnn64", "summary4"]
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fixed order"):
        HeadroomConfig.from_json(path)
