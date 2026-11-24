import json
from pathlib import Path

import numpy as np
import pytest
import torch

from proxy_function_gap.paper3_headroom import fit_model
from proxy_function_gap.paper3_headroom_v3 import (
    HeadroomV3Config,
    curve_case_v3,
    expected_improvement_cuda,
    regret_quadrature_ladder_cuda,
    select_inner_hyperparameters,
)


def config(tmp_path):
    value = json.loads(Path("configs/paper3_headroom_v3.json").read_text())
    value["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return HeadroomV3Config.from_json(path)


def test_config_rejects_outer_and_changed_cohort(tmp_path):
    cfg = config(tmp_path)
    value = cfg.to_dict()
    value["query_pool_ids"] = ["outer_000" if item == "calibration_000" else item for item in value["query_pool_ids"]]
    path = tmp_path / "outer.json"
    path.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="calibration roles"):
        HeadroomV3Config.from_json(path)
    value = cfg.to_dict()
    value["curve_cohort"] = value["curve_cohort"][:-1]
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="six unique"):
        HeadroomV3Config.from_json(path)


def test_analytic_expected_improvement_matches_scalar_formula(tmp_path):
    cfg = config(tmp_path).runtime(1.0, 0.04)
    model, _ = fit_model(np.eye(2), np.zeros(2), cfg)
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    actual = expected_improvement_cuda(model, features, features, cfg)
    sigma = np.sqrt(1.0 / (1.0 + 1.0 / cfg.noise_variance))
    expected = sigma / np.sqrt(2 * np.pi)
    assert actual.is_cuda
    assert actual.cpu().numpy() == pytest.approx(np.repeat(expected, 2), rel=1e-6)


def test_regret_quadrature_is_cuda_deterministic_and_has_full_ladder(tmp_path):
    protocol = config(tmp_path)
    cfg = protocol.runtime(1.0, 0.04)
    model, _ = fit_model(np.eye(2), np.zeros(2), cfg)
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    first, metadata = regret_quadrature_ladder_cuda(model, features, features, protocol, 91, cfg.noise_variance)
    second, _ = regret_quadrature_ladder_cuda(model, features, features, protocol, 91, cfg.noise_variance)
    assert torch.equal(first, second)
    assert first.is_cuda and torch.all(first <= 0)
    assert [(item["posterior_samples"], item["label_quadrature_nodes"]) for item in metadata["levels"]] == [(32, 8), (64, 16), (128, 32)]
    assert all(len(item["scores"]) == 3 for item in metadata["levels"])
    assert metadata["common_random_numbers"] and metadata["antithetic_posterior_samples"]


def test_inner_selection_uses_blocked_development_folds_and_cuda(tmp_path):
    protocol = config(tmp_path)
    rng = np.random.default_rng(4)
    features = rng.normal(size=(16, 2))
    labels = features[:, 0] * 2 + rng.normal(scale=0.1, size=16)
    folds = np.arange(16) % 4
    result = select_inner_hyperparameters(features, labels, folds, protocol)
    assert len(result["grid"]) == 9
    assert all(len(item["folds"]) == 4 for item in result["grid"])
    assert all(fold["training_device"] == "cuda:0" for item in result["grid"] for fold in item["folds"])
    assert result["query_labels_used"] is False and result["outer_labels_read"] is False


def test_random_curve_applies_same_within_batch_constraint(tmp_path):
    protocol = config(tmp_path)
    runtime = protocol.runtime(1.0, 0.04)
    rng = np.random.default_rng(5)
    query_x = rng.normal(size=(64, 2))
    query_y = query_x[:, 0] - query_x[:, 1]
    sequences = []
    for group in range(16):
        bits = f"{group:04b}"
        sequence = "".join(("A" if bit == "1" else "C") * 10 for bit in bits) + "C" * 10
        sequences.extend([sequence] * 4)
    result = curve_case_v3(
        np.eye(2), np.zeros(2), query_x, query_y, np.zeros(64),
        sequences, "random", protocol, runtime, 19,
    )
    assert [item["budget"] for item in result["records"]] == [0, 4, 8, 16, 32, 64]
    assert len(set(result["query_order"])) == 64
    assert sum(item["within_batch_near_duplicates_excluded"] for item in result["acquisition_metadata"]) > 0
    for item in result["acquisition_metadata"]:
        start = item["step"] // 4 * 4
        for previous in result["query_order"][start:item["step"]]:
            distance = sum(a != b for a, b in zip(sequences[previous], sequences[item["chosen_index"]]))
            assert distance > protocol.near_duplicate_max_hamming
    assert result["records"][-1]["training_device"] == "cuda:0"
