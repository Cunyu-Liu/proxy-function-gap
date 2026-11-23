import hashlib

import torch
from torch import nn


class FreshSampleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, inputs):
        return self.regressor(self.features(inputs)).squeeze(1)


def model_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    digest.update(model.__class__.__name__.encode())
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
