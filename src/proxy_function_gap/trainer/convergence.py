import math
from dataclasses import dataclass


@dataclass
class DualPlateau:
    min_delta: float
    patience: int
    best_train: float = math.inf
    best_dev: float = math.inf
    best_epoch: int = 0
    plateau_epochs: int = 0

    def __post_init__(self):
        if self.min_delta < 0 or self.patience <= 0:
            raise ValueError("invalid plateau configuration")

    def update(self, epoch: int, train_loss: float, dev_loss: float):
        if not math.isfinite(train_loss) or not math.isfinite(dev_loss):
            raise ValueError("non-finite loss")
        train_significant = train_loss < self.best_train - self.min_delta
        dev_significant = dev_loss < self.best_dev - self.min_delta
        dev_best = dev_loss < self.best_dev
        if train_loss < self.best_train:
            self.best_train = train_loss
        if dev_best:
            self.best_dev = dev_loss
            self.best_epoch = epoch
        if train_significant or dev_significant:
            self.plateau_epochs = 0
        else:
            self.plateau_epochs += 1
        return {
            "stop": self.plateau_epochs >= self.patience,
            "train_significant_improvement": train_significant,
            "dev_significant_improvement": dev_significant,
            "dev_best": dev_best,
            "plateau_epochs": self.plateau_epochs,
            "best_epoch": self.best_epoch,
        }

    def state_dict(self):
        return {
            "min_delta": self.min_delta,
            "patience": self.patience,
            "best_train": self.best_train,
            "best_dev": self.best_dev,
            "best_epoch": self.best_epoch,
            "plateau_epochs": self.plateau_epochs,
        }

    def load_state_dict(self, state):
        self.best_train = float(state["best_train"])
        self.best_dev = float(state["best_dev"])
        self.best_epoch = int(state["best_epoch"])
        self.plateau_epochs = int(state["plateau_epochs"])
