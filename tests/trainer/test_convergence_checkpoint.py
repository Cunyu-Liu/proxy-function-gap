import json

import pytest
import torch

from proxy_function_gap.trainer.checkpoint import atomic_json_write, atomic_torch_save, load_checkpoint
from proxy_function_gap.trainer.convergence import DualPlateau


def test_plateau_requires_both_train_and_dev_to_stop_significantly_improving():
    plateau = DualPlateau(min_delta=0.1, patience=2)
    assert not plateau.update(1, 1.0, 1.0)["stop"]
    train_only = plateau.update(2, 0.5, 1.0)
    assert train_only["train_significant_improvement"]
    assert plateau.plateau_epochs == 0
    dev_only = plateau.update(3, 0.5, 0.5)
    assert dev_only["dev_significant_improvement"]
    assert plateau.plateau_epochs == 0
    assert not plateau.update(4, 0.5, 0.5)["stop"]
    assert plateau.update(5, 0.5, 0.5)["stop"]
    assert plateau.best_epoch == 3


def test_small_improvement_updates_best_but_counts_as_plateau():
    plateau = DualPlateau(min_delta=0.1, patience=1)
    plateau.update(1, 1.0, 1.0)
    decision = plateau.update(2, 0.95, 0.95)
    assert decision["dev_best"]
    assert decision["stop"]
    assert plateau.best_dev == 0.95
    assert plateau.best_epoch == 2


def test_nonfinite_loss_is_rejected():
    with pytest.raises(ValueError):
        DualPlateau(0.1, 1).update(1, float("nan"), 1.0)


def test_atomic_checkpoint_roundtrip_and_json(tmp_path):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"weight": torch.tensor([1.0]), "epoch": 3}, path)
    loaded = load_checkpoint(path, torch.device("cpu"))
    assert loaded["epoch"] == 3
    assert loaded["weight"].item() == 1.0
    atomic_json_write({"pass": True}, tmp_path / "summary.json")
    assert json.loads((tmp_path / "summary.json").read_text()) == {"pass": True}
    assert not list(tmp_path.glob(".*.pt.*"))
    assert not list(tmp_path.glob(".*.json.*"))


def test_atomic_save_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"epoch": 1}, path)
    def fail(*_):
        raise RuntimeError("injected")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(RuntimeError, match="injected"):
        atomic_torch_save({"epoch": 2}, path)
    assert load_checkpoint(path, torch.device("cpu"))["epoch"] == 1
    assert not list(tmp_path.glob(".checkpoint.pt.*"))
