import hashlib
from pathlib import Path

import numpy as np
import torch

from select_models import ordered
from adaptation import (
    ANALYSIS_ID, DualPlateau, FreshSampleCNN, allocate_exact, cap_regret,
    cluster_bootstrap, component_pools, encode_codes, h1_components,
    holm_adjust, listnet_objective, make_proposals, primary_success,
    random_regret, run_name, state_dict_hash,
)


class VectorModel(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.asarray(values, dtype=torch.float32))

    def forward(self, inputs):
        return self.values[:inputs.shape[0]]


def test_h1_components_keep_exact_h1_together_not_h2():
    sequences = np.asarray(["A" * 50, "A" * 49 + "C", "A" * 48 + "CC", "G" * 50], dtype="S50")
    components, edges = h1_components(sequences)
    assert sorted(sorted(row) for row in components) == [[0, 1, 2], [3]] and edges == 2


def test_component_pool_packing_exact_and_disjoint():
    components = [[0, 1], [2], [3, 4], [5], [6], [7, 8], [9]]
    keys = [str(index) for index in range(len(components))]
    pools = component_pools(components, keys, pool_count=2, pool_size=3)
    assert pools.shape == (2, 3)
    membership = [{next(index for index, members in enumerate(components) if value in members) for value in row} for row in pools]
    assert not membership[0] & membership[1]
    assert sorted(pools.ravel().tolist()) == list(range(6))


def test_listnet_matches_manual_full_group_cross_entropy():
    codes = torch.zeros((64, 50), dtype=torch.uint8)
    labels = torch.arange(64, dtype=torch.float32)
    pools = torch.arange(64).reshape(1, 64)
    model = VectorModel(np.linspace(3, -2, 64))
    loss, mse, listnet = listnet_objective(model, codes, labels, pools, 0.0, 1.0, 0.25, 2.0)
    target = torch.softmax(labels / 2.0, dim=0)
    expected = -(target * torch.log_softmax(model.values / 2.0, dim=0)).sum()
    expected_mse = torch.mean((model.values - labels) ** 2)
    assert torch.allclose(listnet, expected, atol=1e-6)
    assert torch.allclose(mse, expected_mse, atol=1e-5)
    assert torch.allclose(loss, expected + 0.25 * expected_mse, atol=1e-5)


def test_listnet_supervises_all_64_not_prediction_tail():
    codes = torch.zeros((64, 50), dtype=torch.uint8)
    labels = torch.arange(64, dtype=torch.float32)
    pools = torch.arange(64).reshape(1, 64)
    model = VectorModel(np.linspace(3, -2, 64))
    loss, _, _ = listnet_objective(model, codes, labels, pools, 0.0, 1.0, 0.0, 1.0)
    loss.backward()
    assert model.values.grad.shape == (64,)
    assert torch.all(model.values.grad.abs() > 0)


def test_incomplete_pool_excluded_from_listnet_but_not_mse():
    codes = torch.zeros((4, 50), dtype=torch.uint8)
    labels = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pools = torch.tensor([[0, 1, -1, -1]])
    model = VectorModel([1.0, 2.0, 3.0, 4.0])
    loss, mse, listnet = listnet_objective(model, codes, labels, pools, 0.0, 1.0, 2.0, 1.0)
    assert float(listnet) == 0 and torch.allclose(loss, 2 * mse)


def test_plateau_natural_convergence_no_max_epoch():
    plateau = DualPlateau(0.0, 2)
    assert not plateau.update(1, 2.0, 2.0)["stop"]
    assert not plateau.update(2, 2.0, 2.0)["stop"]
    assert plateau.update(3, 2.0, 2.0)["stop"] and plateau.best_epoch == 1


def test_shared_prefix_random_and_regret():
    scores = np.asarray([0.0, 1.0, 2.0, 3.0])
    labels = np.asarray([0.0, 4.0, 1.0, 2.0])
    pools = np.asarray([[0, 1, 2, 3]])
    proposals = np.asarray([[[0, 1, 2, 3]]])
    regret, selected = cap_regret(scores, labels, pools, proposals, cap=2)
    random, _ = random_regret(labels, pools, proposals, cap=2)
    assert regret.tolist() == [[0.0]] and selected.tolist() == [[1]] and random.tolist() == [[4.0]]


def test_bootstrap_and_ten_comparison_cap_holm():
    result = cluster_bootstrap(np.asarray([[1.0, 1.0], [3.0, 3.0]]), 7, 100)
    adjusted = holm_adjust([0.01] * 10)
    assert result["effect"] == 2 and result["clusters"] == 2
    assert len(adjusted) == 10 and np.allclose(adjusted, 0.1)


def test_dev_selection_uses_mean_replicate_regret_and_frozen_ties():
    rows = [
        {"name": "higher", "dev_cap16_regret_mean_replicate": 2.0, "dev_combined_loss_mean": 0.0, "learning_rate": 0.001, "temperature": 1.0, "mse_weight": 1.0},
        {"name": "selected", "dev_cap16_regret_mean_replicate": 1.0, "dev_combined_loss_mean": 0.0, "learning_rate": 0.001, "temperature": 1.0, "mse_weight": 1.0},
    ]
    assert ordered(rows)[0]["name"] == "selected"


def test_primary_requires_both_baselines_and_minimum_effect():
    good = {"effect": {"effect": -0.2, "ci_high": -0.01}, "minimum_required_absolute_reduction": 0.1}
    weak = {"effect": {"effect": -0.05, "ci_high": -0.01}, "minimum_required_absolute_reduction": 0.1}
    assert primary_success({"original_cnn": good, "v1_main": good})["success"]
    assert not primary_success({"original_cnn": good, "v1_main": weak})["success"]


def test_model_capacity_and_run_identity():
    shapes = [tuple(value.shape) for value in FreshSampleCNN().state_dict().values()]
    assert shapes == [(32, 4, 5), (32,), (64, 32, 5), (64,), (32, 64), (32,), (1, 32), (1,)]
    assert run_name(50000, 1e-5, 0.5, 10.0) == "listnet_50000_lr_1e-05_tau_0.5_mse_10"
    state_hash = state_dict_hash(FreshSampleCNN().state_dict())
    assert hashlib.sha256(b"FreshSampleCNN" + state_hash.encode()).hexdigest() != state_hash


def test_prepare_and_prediction_freeze_cannot_read_test_labels():
    prepare = Path(__file__).resolve().parents[1].joinpath("prepare.py").read_text()
    freeze = Path(__file__).resolve().parents[1].joinpath("freeze_predictions.py").read_text()
    assert 'row["rl"]' not in prepare and "gzip.open" not in freeze and 'row["rl"]' not in freeze


def test_listnet_objective_real_cuda():
    if not torch.cuda.is_available():
        return
    codes = torch.zeros((64, 50), dtype=torch.uint8).cuda()
    labels = torch.arange(64, dtype=torch.float32).cuda()
    model = VectorModel(np.linspace(3, -2, 64)).cuda()
    loss, _, _ = listnet_objective(model, codes, labels, torch.arange(64).cuda().reshape(1, 64), 0.0, 1.0, 0.1, 0.5)
    loss.backward()
    assert loss.is_cuda and torch.isfinite(loss) and model.values.grad.is_cuda
