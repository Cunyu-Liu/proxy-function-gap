import hashlib

import numpy as np
import torch

from select_models import ordered
from adaptation import (
    DualPlateau, FreshSampleCNN, HalfIndex, allocate_exact, cap_regret,
    cluster_bootstrap, component_identity, dynamic_objective, encode_codes,
    h1_components, holm_adjust, make_proposals, random_regret, run_name,
    state_dict_hash,
)


def test_h1_components_keep_exact_h1_together_not_h2():
    sequences = np.asarray(["A" * 50, "A" * 49 + "C", "A" * 48 + "CC", "G" * 50], dtype="S50")
    components, edges = h1_components(sequences)
    membership = sorted(sorted(row) for row in components)
    assert membership == [[0, 1, 2], [3]] and edges == 2


def test_component_allocation_exact_and_skips_oversize():
    components = [[0, 1], [2], [3], [4, 5], [6]]
    keys = [str(index) for index in range(len(components))]
    chosen = allocate_exact(components, keys, 3)
    assert sum(len(components[index]) for index in chosen) == 3


def test_dynamic_tail_loss_gradient_and_mse_only():
    sequences = ["A" * index + "C" * (50 - index) for index in range(16)]
    codes = torch.from_numpy(encode_codes(sequences))
    labels = torch.arange(16, dtype=torch.float32)
    pools = torch.arange(16).reshape(1, 16)
    model = FreshSampleCNN()
    loss, mse, pair = dynamic_objective(model, codes, labels, pools, 0.0, 1.0, 0.1)
    assert torch.isfinite(loss) and torch.isfinite(mse) and torch.isfinite(pair)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    model.zero_grad()
    loss, _, pair = dynamic_objective(model, codes, labels, pools, 0.0, 1.0, 0.0)
    assert float(pair) == 0 and torch.isfinite(loss)


def test_incomplete_pool_mse_and_pair_complete_only():
    codes = torch.from_numpy(encode_codes(["A" * 50, "C" * 50]))
    labels = torch.tensor([1.0, 2.0])
    pools = torch.tensor([[0, 1, -1, -1] + [-1] * 12])
    model = FreshSampleCNN()
    loss, mse, pair = dynamic_objective(model, codes, labels, pools, 0.0, 1.0, 1.0)
    assert torch.isfinite(loss) and float(pair) == 0


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
    random, random_selected = random_regret(labels, pools, proposals, cap=2)
    assert regret.tolist() == [[0.0]] and selected.tolist() == [[1]]
    assert random.tolist() == [[4.0]] and random_selected.tolist() == [[0]]
    cap1, _ = cap_regret(scores, labels, pools, proposals, cap=1)
    random1, _ = random_regret(labels, pools, proposals, cap=1)
    assert np.array_equal(cap1, random1)


def test_bootstrap_and_all_cap_holm_monotonicity():
    result = cluster_bootstrap(np.asarray([[1.0, 1.0], [3.0, 3.0]]), 7, 100)
    assert result["effect"] == 2 and result["clusters"] == 2
    adjusted = holm_adjust([0.01, 0.02, 1.0, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    assert len(adjusted) == 10 and np.all(adjusted >= np.asarray([0.01, 0.02, 1.0, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]))


def test_model_capacity_and_run_identity():
    shapes = [tuple(value.shape) for value in FreshSampleCNN().state_dict().values()]
    assert shapes == [(32, 4, 5), (32,), (64, 32, 5), (64,), (32, 64), (32,), (1, 32), (1,)]
    assert run_name("nested", 1000, 1e-4, 0.1) == "main_1000_lr_0.0001_lambda_0.1"


def test_dev_selection_uses_worst_repeat_not_only_mean():
    rows = [
        {"name": "mean_better_single_worse", "dev_cap16_regret_worst_repeat": 2.0, "dev_cap16_regret": 1.0, "dev_combined_loss_mean": 0.0, "learning_rate": 0.001, "lambda": 1.0},
        {"name": "both_repeats_good", "dev_cap16_regret_worst_repeat": 1.5, "dev_cap16_regret": 1.4, "dev_combined_loss_mean": 0.0, "learning_rate": 0.001, "lambda": 1.0},
    ]
    assert ordered(rows)[0]["name"] == "both_repeats_good"


def test_original_model_hash_wrapper():
    model = FreshSampleCNN()
    state_hash = state_dict_hash(model.state_dict())
    assert hashlib.sha256(b"FreshSampleCNN" + state_hash.encode()).hexdigest() != state_hash


def test_dynamic_objective_real_cuda():
    if not torch.cuda.is_available():
        return
    codes = torch.from_numpy(encode_codes(["A" * index + "C" * (50 - index) for index in range(16)])).cuda()
    labels = torch.arange(16, dtype=torch.float32).cuda()
    model = FreshSampleCNN().cuda()
    loss, _, _ = dynamic_objective(model, codes, labels, torch.arange(16).cuda().reshape(1, 16), 0.0, 1.0, 0.1)
    loss.backward()
    assert loss.is_cuda and torch.isfinite(loss)
    assert all(parameter.grad is None or parameter.grad.is_cuda for parameter in model.parameters())
