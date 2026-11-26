import json

import numpy as np
import torch

from backbone import DualPlateau, FreshSampleCNN, HalfIndex, cap_regret, cluster_bootstrap, context_scalers, dynamic_objective, encode_codes, make_pools


def test_half_index_exact_h1_and_not_h2():
    index = HalfIndex()
    index.add("A" * 50)
    assert index.match("A" * 50) == (True, False)
    assert index.match("A" * 49 + "C") == (False, True)
    assert index.match("A" * 24 + "C" + "A" * 25) == (False, True)
    assert index.match("A" * 48 + "CC") == (False, False)


def test_component_pooling_and_padding():
    sequences = ["A" * 50, "C" * 50, "G" * 50]
    pools = make_pools(sequences, "synthetic", "train", pool_size=2)
    assert pools.shape == (2, 2)
    assert np.all(pools[0] >= 0) and np.all(pools[1, 1:] == -1)


def test_context_scaler_is_train_only_and_dynamic_loss_gradients():
    codes = torch.from_numpy(encode_codes(["A" * index + "C" * (50 - index) for index in range(16)]))
    labels = torch.arange(16, dtype=torch.float32)
    pools = torch.arange(16).reshape(1, 16)
    pool_contexts = np.asarray([0])
    scalers = context_scalers(np.asarray([1.0, 3.0]), np.asarray([0, 0]))
    scalers = {0: scalers[0]}
    model = FreshSampleCNN()
    loss, mse, pair = dynamic_objective(model, codes, labels, pools, pool_contexts, scalers, 0.1, [1])
    assert torch.isfinite(loss) and torch.isfinite(mse) and torch.isfinite(pair)
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0 for parameter in model.parameters())


def test_mse_only_has_zero_pair_and_no_max_epoch_plateau():
    model = FreshSampleCNN()
    codes = torch.from_numpy(encode_codes(["A" * 50, "C" * 50]))
    labels = torch.tensor([1.0, 2.0])
    loss, _, pair = dynamic_objective(model, codes, labels, torch.tensor([[0, 1]]), np.asarray([0]), {0: {"mean": 0.0, "scale": 1.0}}, 0.0, [1])
    assert float(pair) == 0.0 and torch.isfinite(loss)
    plateau = DualPlateau(0.0, 2)
    assert not plateau.update(1, 2.0, 2.0)["stop"]
    assert not plateau.update(2, 2.0, 2.0)["stop"]
    assert plateau.update(3, 2.0, 2.0)["stop"]
    assert plateau.best_epoch == 1


def test_cap_regret_shared_prefix_and_cluster_bootstrap():
    scores = np.asarray([0.0, 1.0, 2.0, 3.0])
    labels = np.asarray([0.0, 4.0, 1.0, 2.0])
    pools = np.asarray([[0, 1, 2, 3]])
    proposals = np.asarray([[[0, 1, 2, 3]]])
    regret, selected = cap_regret(scores, labels, pools, proposals, cap=2)
    assert regret.tolist() == [[0.0]] and selected.tolist() == [[1]]
    result = cluster_bootstrap(np.asarray([[1.0, 1.0], [3.0, 3.0]]), 7, 100)
    assert result["effect"] == 2.0 and result["clusters"] == 2


def test_model_capacity_is_unchanged():
    model = FreshSampleCNN()
    shapes = [tuple(value.shape) for value in model.state_dict().values()]
    assert shapes == [(32, 4, 5), (32,), (64, 32, 5), (64,), (32, 64), (32,), (1, 32), (1,)]


def test_dynamic_objective_real_cuda():
    if not torch.cuda.is_available():
        return
    model = FreshSampleCNN().cuda()
    codes = torch.from_numpy(encode_codes(["A" * index + "C" * (50 - index) for index in range(16)])).cuda()
    labels = torch.arange(16, dtype=torch.float32).cuda()
    loss, _, _ = dynamic_objective(model, codes, labels, torch.arange(16).cuda().reshape(1, 16), torch.tensor([0]).cuda(), {0: {"mean": 0.0, "scale": 1.0}}, 0.1, [1])
    loss.backward()
    assert loss.is_cuda and torch.isfinite(loss)
    assert all(parameter.grad is None or parameter.grad.is_cuda for parameter in model.parameters())
    model.zero_grad()
    incomplete = torch.arange(16).cuda().reshape(1, 16).clone()
    incomplete[0, -1] = -1
    partial_loss, _, _ = dynamic_objective(model, codes, labels, incomplete, torch.tensor([0]).cuda(), {0: {"mean": 0.0, "scale": 1.0}}, 0.1, [1])
    partial_loss.backward()
    assert torch.isfinite(partial_loss)
