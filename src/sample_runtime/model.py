import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from .provenance import OFFICIAL_COMMIT, RETRAINED_WEIGHTS, ORIGINAL_WEIGHTS, WEIGHT_SHA256

NUCLEOTIDES = {"A": 0, "C": 1, "G": 2, "T": 3}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_cuda(requested="cuda:0"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU inference is forbidden")
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError(f"Requested device must be CUDA: {requested}")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(0.10, device=device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return device


def encode_utrs(utrs):
    result = np.zeros((len(utrs), 54, 4), dtype=np.float32)
    for row, utr in enumerate(utrs):
        if len(utr) != 50 or set(utr) - set(NUCLEOTIDES):
            raise ValueError(f"Expected exactly 50 A/C/G/T bases: {utr!r}")
        for col, nucleotide in enumerate(utr + "ATGG"):
            result[row, col, NUCLEOTIDES[nucleotide]] = 1.0
    return result


class SampleEvolutionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 40, kernel_size=8, padding=0, bias=True)
        self.conv2 = nn.Conv1d(40, 40, kernel_size=8, padding=0, bias=True)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(1600, 40, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.dense2 = nn.Linear(40, 1, bias=True)

    def forward(self, inputs):
        x = torch.relu(self.conv1(inputs))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1).contiguous()
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = torch.relu(x)
        return self.dense2(x).squeeze(1)


def _find_tensor(group, layer, kind):
    suffixes = (f"/{kind}:0", "_W:0" if kind == "kernel" else "_b:0")
    found = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.startswith(layer + "/") and any(s in name for s in suffixes):
            found.append(obj[...])
    group.visititems(visit)
    if len(found) != 1:
        raise ValueError(f"Expected one {layer}/{kind}, found {len(found)}")
    return found[0]


def _torch_conv(weight):
    if weight.ndim == 4:
        if weight.shape[1] != 1:
            raise ValueError(f"Unexpected Keras convolution shape: {weight.shape}")
        weight = weight[:, 0, :, :]
    return torch.from_numpy(np.ascontiguousarray(weight.transpose(2, 1, 0)))


def _torch_dense(weight):
    return torch.from_numpy(np.ascontiguousarray(weight.T))


def load_evolution_model(path, expected_sha256, device):
    path = Path(path)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Weight hash mismatch for {path}: {actual_sha256}")
    model = SampleEvolutionModel()
    with h5py.File(path, "r") as handle:
        config = json.loads(handle.attrs["model_config"])
        first_layer = config["config"][0]["config"]
        if first_layer["batch_input_shape"] != [None, 54, 4]:
            raise ValueError(f"Unexpected input shape: {first_layer[batch_input_shape]}")
        weights = handle["model_weights"]
        with torch.no_grad():
            for name in ("convolution1d_1", "convolution1d_2", "dense_1", "dense_2"):
                kernel = _find_tensor(weights, name, "kernel")
                bias = _find_tensor(weights, name, "bias")
                if name.startswith("convolution"):
                    target = model.conv1 if name.endswith("_1") else model.conv2
                    target.weight.copy_(_torch_conv(kernel))
                else:
                    target = model.dense1 if name.endswith("_1") else model.dense2
                    target.weight.copy_(_torch_dense(kernel))
                target.bias.copy_(torch.from_numpy(bias))
        keras_version = str(handle.attrs["keras_version"])
    model.eval()
    model.cuda(device=device)
    for parameter in model.parameters():
        if not parameter.is_cuda or parameter.dtype != torch.float32:
            raise RuntimeError("Model parameters must be float32 CUDA tensors")
        parameter.requires_grad_(False)
    return model, {"path": str(path), "sha256": actual_sha256, "keras_version": keras_version, "official_commit": OFFICIAL_COMMIT}


def load_both_models(device):
    original, original_info = load_evolution_model(ORIGINAL_WEIGHTS, WEIGHT_SHA256["original"], device)
    retrained, retrained_info = load_evolution_model(RETRAINED_WEIGHTS, WEIGHT_SHA256["retrained"], device)
    return {"original": original, "retrained": retrained}, {"original": original_info, "retrained": retrained_info}


def predict_both(models, utrs, device, batch_size=128):
    encoded = encode_utrs(utrs)
    outputs = {"original": [], "retrained": []}
    with torch.inference_mode():
        for start in range(0, len(utrs), batch_size):
            inputs = torch.from_numpy(encoded[start:start + batch_size]).cuda(device=device)
            inputs = inputs.permute(0, 2, 1).contiguous()
            if not inputs.is_cuda or inputs.dtype != torch.float32:
                raise RuntimeError("Inputs must be float32 CUDA tensors")
            for name, model in models.items():
                values = model(inputs)
                if not values.is_cuda or values.device != inputs.device:
                    raise RuntimeError("Model output is not on the requested CUDA device")
                outputs[name].append(values.detach().cpu().numpy())
            torch.cuda.synchronize(device)
    return {name: np.concatenate(values).astype(np.float64) for name, values in outputs.items()}
