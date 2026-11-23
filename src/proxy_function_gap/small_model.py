import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .trainer.convergence import DualPlateau
from .trainer.cuda import configure_cuda
from .trainer.model import model_hash


@dataclass(frozen=True)
class SmallModelConfig:
    model_type: str = "logistic"
    loss_type: str = "squared_hinge"
    feature_names: tuple = ("proxy_mean", "proxy_std", "proxy_range", "support")
    learning_rate: float = 0.05
    min_delta: float = 0.0001
    patience: int = 3
    fold_count: int = 3
    memory_fraction: float = 0.05
    dtype: str = "float32"

    def __post_init__(self):
        if self.model_type not in ("logistic", "mlp") or self.loss_type not in ("squared_hinge", "log_loss") or not self.feature_names:
            raise ValueError("invalid small-model declaration")
        if self.learning_rate <= 0 or self.min_delta < 0 or self.patience <= 0 or self.fold_count < 2:
            raise ValueError("invalid convergence or fold configuration")


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values):
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        return cls(mean=mean, scale=np.where(scale > 0, scale, 1.0))

    def transform(self, values):
        return (values - self.mean) / self.scale

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "fit_role": "train_fold"}


def build_classifier(model_type, feature_count):
    if model_type == "logistic":
        return nn.Linear(feature_count, 1)
    if model_type == "mlp":
        return nn.Sequential(nn.Linear(feature_count, 4), nn.ReLU(), nn.Linear(4, 1))
    raise ValueError(model_type)


def _loss(model, values, targets, device, dtype, loss_type="squared_hinge"):
    values = torch.from_numpy(values).to(device=device, dtype=dtype)
    targets = torch.from_numpy(targets).to(device=device, dtype=dtype)
    logits = model(values).squeeze(1)
    if loss_type == "log_loss":
        return nn.functional.binary_cross_entropy_with_logits(logits, targets), logits
    signed_targets = targets * 2 - 1
    return torch.mean(torch.relu(1 - signed_targets * logits) ** 2), logits


def _eval_loss(model, values, targets, device, dtype, loss_type="squared_hinge"):
    with torch.no_grad():
        loss, _ = _loss(model, values, targets, device, dtype, loss_type)
        return float(loss.item())


def predict_model(model, scaler, values, device, dtype):
    model.eval()
    transformed = scaler.transform(values).astype(np.float32)
    with torch.no_grad():
        _, logits = _loss(model, transformed, np.zeros(len(values), dtype=np.float32), device, dtype)
        return torch.sigmoid(logits).detach().cpu().numpy()


def train_classifier(x_train, y_train, x_dev, y_dev, config: SmallModelConfig, seed):
    device, dtype, gpu = configure_cuda(0, config.memory_fraction, config.dtype, seed)
    scaler = FeatureScaler.fit(x_train)
    train_values = scaler.transform(x_train).astype(np.float32)
    dev_values = scaler.transform(x_dev).astype(np.float32)
    model = build_classifier(config.model_type, x_train.shape[1])
    model.to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    plateau = DualPlateau(config.min_delta, config.patience)
    best_state = copy.deepcopy(model.state_dict())
    history = []
    epoch = 0
    while True:
        epoch += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _loss(model, train_values, y_train.astype(np.float32), device, dtype, config.loss_type)
        loss.backward()
        optimizer.step()
        online = float(loss.item())
        model.eval()
        train_loss = _eval_loss(model, train_values, y_train.astype(np.float32), device, dtype, config.loss_type)
        dev_loss = _eval_loss(model, dev_values, y_dev.astype(np.float32), device, dtype, config.loss_type)
        decision = plateau.update(epoch, train_loss, dev_loss)
        if decision["dev_best"]:
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "epoch": epoch,
            "train_online_loss": online,
            "train_eval_loss": train_loss,
            "dev_eval_loss": dev_loss,
            "best_epoch": plateau.best_epoch,
            "plateau_epochs": plateau.plateau_epochs,
            "dev_absolute_best": decision["dev_best"],
        })
        if epoch % 100 == 0:
            print(f"small_model_epoch model={config.model_type} seed={seed} epoch={epoch} train={train_loss:.8f} dev={dev_loss:.8f} plateau={plateau.plateau_epochs}", flush=True)
        if decision["stop"]:
            break
    model.load_state_dict(best_state)
    artifact = {
        "model_type": config.model_type,
        "loss_type": config.loss_type,
        "feature_names": list(config.feature_names),
        "model_state": {name: value.detach().cpu().clone() for name, value in best_state.items()},
        "scaler": scaler,
        "history": history,
        "model_hash": model_hash(model),
        "best_epoch": plateau.best_epoch,
        "gpu": gpu,
    }
    return model, artifact


def fold_assignments(groups, fold_count):
    unique = sorted(set(groups))
    if len(unique) < fold_count:
        raise ValueError("not enough groups for grouped OOF")
    return {group: index % fold_count for index, group in enumerate(unique)}


def fit_oof_classifier(values, targets, groups, config: SmallModelConfig):
    groups = np.asarray(groups)
    assignments = fold_assignments(groups, config.fold_count)
    folds = []
    oof = np.zeros(len(values), dtype=np.float64)
    for fold in range(config.fold_count):
        dev_mask = np.asarray([assignments[group] == fold for group in groups])
        train_mask = ~dev_mask
        if not train_mask.any() or not dev_mask.any():
            raise ValueError("invalid grouped fold")
        fold_config = SmallModelConfig(
            model_type=config.model_type,
            loss_type=config.loss_type,
            feature_names=config.feature_names,
            learning_rate=config.learning_rate,
            min_delta=config.min_delta,
            patience=config.patience,
            fold_count=config.fold_count,
            memory_fraction=config.memory_fraction,
            dtype=config.dtype,
        )
        model, artifact = train_classifier(values[train_mask], targets[train_mask], values[dev_mask], targets[dev_mask], fold_config, 1000 + fold)
        artifact["fold"] = fold
        folds.append(artifact)
        device = torch.device("cuda:0")
        oof[dev_mask] = predict_model(model, artifact["scaler"], values[dev_mask], device, torch.float32)
    return oof, folds, assignments


def predict_with_folds(folds, values, config: SmallModelConfig):
    device, dtype, _ = configure_cuda(0, config.memory_fraction, config.dtype, 0)
    predictions = []
    for artifact in folds:
        model = build_classifier(artifact["model_type"], values.shape[1])
        model.load_state_dict(artifact["model_state"])
        model.to(device=device, dtype=dtype)
        model.eval()
        predictions.append(predict_model(model, artifact["scaler"], values, device, dtype))
    return np.mean(predictions, axis=0)


def fit_platt(raw_risk, targets, groups, config: SmallModelConfig):
    groups = np.asarray(groups)
    assignments = fold_assignments(groups, config.fold_count)
    dev_mask = np.asarray([assignments[group] == 0 for group in groups])
    train_mask = ~dev_mask
    platt_config = SmallModelConfig(
        model_type="logistic",
        loss_type="log_loss",
        feature_names=("raw_risk",),
        learning_rate=config.learning_rate,
        min_delta=config.min_delta,
        patience=config.patience,
        fold_count=config.fold_count,
        memory_fraction=config.memory_fraction,
        dtype=config.dtype,
    )
    _, artifact = train_classifier(raw_risk[train_mask].reshape(-1, 1), targets[train_mask], raw_risk[dev_mask].reshape(-1, 1), targets[dev_mask], platt_config, 2024)
    return artifact


def apply_platt(artifact, raw_risk, config: SmallModelConfig):
    device, dtype, _ = configure_cuda(0, config.memory_fraction, config.dtype, 0)
    model = build_classifier("logistic", 1)
    model.load_state_dict(artifact["model_state"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return predict_model(model, artifact["scaler"], raw_risk.reshape(-1, 1), device, dtype)


def log_loss(targets, probabilities):
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-np.mean(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities)))
