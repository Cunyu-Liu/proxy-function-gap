import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from proxy_function_gap.paper3_headroom import fit_model
from proxy_function_gap.paper3_headroom_v4 import (
    HeadroomV4Config,
    _normal_quadrature,
    curve_case_v4,
    independent_expected_max_quadrature_cuda,
    regret_analytic_ladder_cuda,
)
from proxy_function_gap.paper3_headroom_v4_numerics import (
    exhaustive_scores,
    nested_independent_mc_cuda,
    tiny_model_and_features,
    two_candidate_closed_form_scores,
)
from proxy_function_gap.trainer.cuda import configure_cuda


def config(tmp_path):
    value = json.loads(Path("configs/paper3_headroom_v4.json").read_text())
    value["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return HeadroomV4Config.from_json(path)


def test_config_rejects_outer_changed_cohort_and_changed_ladder(tmp_path):
    cfg = config(tmp_path)
    value = cfg.to_dict()
    value["query_pool_ids"] = ["outer_000" if item == "calibration_000" else item for item in value["query_pool_ids"]]
    path = tmp_path / "outer.json"
    path.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="calibration roles"):
        HeadroomV4Config.from_json(path)
    value = cfg.to_dict()
    value["curve_cohort"] = value["curve_cohort"][:-1]
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="six V2 curve cases"):
        HeadroomV4Config.from_json(path)
    value = cfg.to_dict()
    value["value_quadrature_ladder"] = [16, 32, 32]
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="quadrature ladder"):
        HeadroomV4Config.from_json(path)


def test_independent_expected_max_matches_two_normal_closed_form(tmp_path):
    cfg = config(tmp_path)
    device, dtype, _ = configure_cuda(cfg.gpu_index, cfg.memory_fraction, cfg.dtype, 19)
    mean = torch.tensor([[[0.3, -0.2], [1.0, 0.7]]], device=device, dtype=dtype)
    variance = torch.tensor([[[0.4, 0.2], [0.1, 0.3]]], device=device, dtype=dtype)
    nodes, weights = _normal_quadrature(64, device, dtype)
    actual = independent_expected_max_quadrature_cuda(mean, variance, nodes, weights).cpu().numpy()
    expected = []
    for row_mean, row_variance in zip(mean[0].cpu().numpy(), variance[0].cpu().numpy()):
        delta = row_mean[0] - row_mean[1]
        sigma = math.sqrt(sum(row_variance))
        z = delta / sigma
        cdf = 0.5 * math.erfc(-z / math.sqrt(2))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        expected.append(row_mean[0] * cdf + row_mean[1] * (1 - cdf) + sigma * pdf)
    assert actual[0] == pytest.approx(expected, abs=1e-10)


def test_regret_analytic_is_cuda_deterministic_no_mc_and_full_ladder(tmp_path):
    protocol = config(tmp_path)
    model, features, _ = tiny_model_and_features(protocol)
    first, metadata = regret_analytic_ladder_cuda(model, features, features, protocol, 0.04)
    second, _ = regret_analytic_ladder_cuda(model, features, features, protocol, 0.04)
    assert torch.equal(first, second)
    assert first.is_cuda and torch.all(first <= 0)
    assert [(item["label_quadrature_nodes"], item["value_quadrature_nodes"]) for item in metadata["levels"]] == [(64, 32), (64, 64), (64, 128)]
    assert all(len(item["scores"]) == 3 for item in metadata["levels"])
    assert metadata["posterior_samples"] == 0 and metadata["label_monte_carlo_samples"] == 0
    assert metadata["posterior_dependence"] == "diagonal_independent_fantasy_posterior_approximation"
    assert not metadata["joint_gaussian_multivariate_max_claimed"]


def test_regret_analytic_matches_exhaustive_and_two_candidate_closed_form(tmp_path):
    protocol = config(tmp_path)
    model, features, _ = tiny_model_and_features(protocol)
    scores, _ = regret_analytic_ladder_cuda(model, features, features, protocol, 0.04)
    exhaustive, _ = exhaustive_scores(model, features, 0.04)
    assert scores.cpu().numpy() == pytest.approx(exhaustive, abs=1e-8)
    scores_two, _ = regret_analytic_ladder_cuda(model, features, features[:2], protocol, 0.04)
    closed_form = two_candidate_closed_form_scores(model, features, 0.04)
    assert scores_two.cpu().numpy() == pytest.approx(closed_form, abs=1e-8)


def test_regret_analytic_matches_reduced_nested_cuda_mc(tmp_path):
    protocol = config(tmp_path)
    model, features, _ = tiny_model_and_features(protocol)
    analytic, _ = regret_analytic_ladder_cuda(model, features, features, protocol, 0.04)
    reduced = SimpleNamespace(
        gpu_index=protocol.gpu_index, memory_fraction=protocol.memory_fraction,
        dtype=protocol.dtype, validation_seed=protocol.validation_seed,
        validation_label_mc_samples=256, validation_posterior_mc_samples=8192,
        runtime=protocol.runtime,
    )
    mc, standard_errors, metadata = nested_independent_mc_cuda(model, features, reduced)
    errors = torch.abs(analytic - mc).cpu().numpy()
    tolerance = np.maximum(0.05, 5 * standard_errors.cpu().numpy())
    assert np.all(errors <= tolerance)
    assert mc.is_cuda and metadata["posterior_mc_samples"] == 8192


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
    result = curve_case_v4(
        np.eye(2), np.zeros(2), query_x, query_y, np.zeros(64),
        sequences, "random", protocol, runtime, 19,
    )
    assert [item["budget"] for item in result["records"]] == [0, 4, 8, 16, 32, 64]
    json.dumps(result)
    assert len(set(result["query_order"])) == 64
    assert sum(item["within_batch_near_duplicates_excluded"] for item in result["acquisition_metadata"]) > 0
    for item in result["acquisition_metadata"]:
        start = item["step"] // 4 * 4
        for previous in result["query_order"][start:item["step"]]:
            distance = sum(a != b for a, b in zip(sequences[previous], sequences[item["chosen_index"]]))
            assert distance > protocol.near_duplicate_max_hamming
    assert result["records"][-1]["training_device"] == "cuda:0"
