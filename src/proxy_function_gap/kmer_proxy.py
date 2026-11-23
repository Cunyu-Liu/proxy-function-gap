import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .mechanism import sequence_features
from .trainer.checkpoint import atomic_json_write, atomic_torch_save
from .trainer.convergence import DualPlateau
from .trainer.cuda import configure_cuda
from .trainer.data import sha256_file
from .trainer.model import model_hash


BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


def kmer_features(sequence, k=4):
    values = np.zeros(4 ** k + 3, dtype=np.float32)
    for index in range(len(sequence) - k + 1):
        code = 0
        for base in sequence[index:index + k]:
            code = code * 4 + BASES[base]
        values[code] += 1
    values[:4 ** k] /= len(sequence) - k + 1
    simple = sequence_features(sequence)
    values[-3:] = [simple["gc"], simple["uAUG"], simple["longest_poly_u"] / 50]
    return values


class KmerRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(259, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, values):
        return self.layers(values).squeeze(1)


def build_feature_pack(input_npz, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    with np.load(input_npz, allow_pickle=False) as source:
        for role in ("train", "dev"):
            sequences = source["utr50"].tolist()
            values = np.asarray([kmer_features(sequence) for sequence in sequences], dtype=np.float32)
            path = output / f"{role}.npz"
            np.savez(path, x=values, y=source["measured_rl"], row_id=source["row_id"], sequence_id=source["sequence_id"], utr50=source["utr50"])
            files[role] = {"path": str(path), "sha256": sha256_file(path), "rows": len(values)}
    manifest = {"source_npz": str(input_npz), "source_sha256": sha256_file(input_npz), "features": "4-mer frequencies plus GC/uAUG/longest-poly-U", "feature_count": 259, "files": files, "outer_labels_read": False, "synthetic": False}
    atomic_json_write(manifest, output / "feature_manifest.json")
    return manifest


def _eval(model, values, targets, device, dtype):
    losses = []
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            x = torch.from_numpy(values[start:start + 4096]).to(device=device, dtype=dtype)
            y = torch.from_numpy(targets[start:start + 4096]).to(device=device, dtype=dtype)
            losses.append(torch.mean((model(x) - y) ** 2).item() * len(x))
    return float(sum(losses) / len(values))


def train_kmer(feature_manifest, seed, output_dir, memory_fraction=0.08, learning_rate=0.001, min_delta=0.0001, patience=5):
    manifest_path = Path(feature_manifest)
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["files"].values():
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError("k-mer feature hash mismatch")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device, dtype, gpu = configure_cuda(0, memory_fraction, "float32", seed)
    with np.load(manifest["files"]["train"]["path"], allow_pickle=False) as pack:
        x_train = pack["x"]
        y_raw = pack["y"]
    with np.load(manifest["files"]["dev"]["path"], allow_pickle=False) as pack:
        x_dev = pack["x"]
        y_dev_raw = pack["y"]
    mean = float(y_raw.mean())
    scale = float(y_raw.std())
    if scale == 0:
        scale = 1.0
    y_train = ((y_raw - mean) / scale).astype(np.float32)
    y_dev = ((y_dev_raw - mean) / scale).astype(np.float32)
    model = KmerRegressor().to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    plateau = DualPlateau(min_delta, patience)
    best_state = copy.deepcopy(model.state_dict())
    history = []
    epoch = 0
    while True:
        epoch += 1
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(x_train), generator=generator).numpy()
        online = 0.0
        for start in range(0, len(order), 2048):
            indices = order[start:start + 2048]
            values = torch.from_numpy(x_train[indices]).to(device=device, dtype=dtype)
            targets = torch.from_numpy(y_train[indices]).to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(values) - targets) ** 2)
            loss.backward()
            optimizer.step()
            online += loss.item() * len(indices)
        online /= len(order)
        model.eval()
        train_loss = _eval(model, x_train, y_train, device, dtype)
        dev_loss = _eval(model, x_dev, y_dev, device, dtype)
        decision = plateau.update(epoch, train_loss, dev_loss)
        if decision["dev_best"]:
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_online_loss": online, "train_eval_loss": train_loss, "dev_eval_loss": dev_loss, "best_epoch": plateau.best_epoch, "plateau_epochs": plateau.plateau_epochs})
        if epoch % 10 == 0:
            print(f"kmer_seed={seed} epoch={epoch} train={train_loss:.6f} dev={dev_loss:.6f} plateau={plateau.plateau_epochs}", flush=True)
        if decision["stop"]:
            break
    model.load_state_dict(best_state)
    artifact = {"seed": seed, "epoch": epoch, "best_epoch": plateau.best_epoch, "model_state": best_state, "model_hash": model_hash(model), "scaler": {"mean": mean, "scale": scale, "fit_role": "train"}, "feature_manifest_sha256": sha256_file(manifest_path), "history": history, "gpu": gpu, "converged_dev_plateau": True, "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    atomic_torch_save(artifact, output / "best.pt")
    atomic_json_write({"seed": seed, "epoch": epoch, "best_epoch": plateau.best_epoch, "model_hash": artifact["model_hash"], "feature_manifest_sha256": artifact["feature_manifest_sha256"], "converged_dev_plateau": True, "outer_labels_read": False}, output / "training_manifest.json")
    return artifact


def predict_kmer(checkpoint_path, sequences, memory_fraction=0.05):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    device, dtype, gpu = configure_cuda(0, memory_fraction, "float32", checkpoint["seed"])
    model = KmerRegressor()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device=device, dtype=dtype)
    model.eval()
    if model_hash(model) != checkpoint["model_hash"]:
        raise ValueError("k-mer model hash mismatch")
    values = np.asarray([kmer_features(sequence) for sequence in sequences], dtype=np.float32)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            x = torch.from_numpy(values[start:start + 4096]).to(device=device, dtype=dtype)
            if not x.is_cuda:
                raise RuntimeError("CPU inference fallback is forbidden")
            outputs.append(model(x).cpu().numpy())
    normalized = np.concatenate(outputs)
    result = normalized * checkpoint["scaler"]["scale"] + checkpoint["scaler"]["mean"]
    torch.cuda.synchronize(device)
    return result, gpu


def score_kmer_pool(pool, checkpoints, output):
    from .mechanism import sequence_features
    predictions = []
    gpu_reports = []
    for checkpoint in checkpoints:
        values, gpu = predict_kmer(checkpoint, pool.utr50)
        predictions.append(values)
        gpu_reports.append(gpu)
    values = np.asarray(predictions)
    rows = [{"sequence_id": sequence_id, "utr50": pool.utr50[index], "kmer_mean": float(values[:, index].mean()), "kmer_std": float(values[:, index].std(ddof=1)), "kmer_range": float(values[:, index].max() - values[:, index].min()), **sequence_features(pool.utr50[index])} for index, sequence_id in enumerate(pool.sequence_ids)]
    atomic_json_write({"metadata": {"pool_id": pool.pool_id, "actual_cuda_inference": True, "gpu": gpu_reports, "kmer_item_forwards": len(checkpoints) * len(pool), "outer_labels_read": False}, "records": rows}, output)
    return {"pool_id": pool.pool_id, "records": len(rows), "outer_labels_read": False}


def evaluate_cross_architecture(pool_dir, pool_id, cnn_scores_path, kmer_scores_path, output):
    from .mechanism import _labels
    cnn = {row["sequence_id"]: row for row in json.loads(Path(cnn_scores_path).read_text())["records"]}
    kmer = {row["sequence_id"]: row for row in json.loads(Path(kmer_scores_path).read_text())["records"]}
    labels = _labels(pool_dir)
    rows = []
    for sequence_id, fresh in cnn.items():
        alternative = kmer[sequence_id]
        rows.append({"sequence_id": sequence_id, "fresh_error": fresh["proxy_mean"] - labels[sequence_id], "kmer_error": alternative["kmer_mean"] - labels[sequence_id], "architecture_gap": fresh["proxy_mean"] - alternative["kmer_mean"], "fresh_spread": fresh["proxy_std"], "kmer_spread": alternative["kmer_std"]})
    fresh_error = np.asarray([abs(row["fresh_error"]) for row in rows])
    kmer_error = np.asarray([abs(row["kmer_error"]) for row in rows])
    gaps = np.asarray([abs(row["architecture_gap"]) for row in rows])
    cut = float(np.quantile(gaps, 0.25))
    high_cut = float(np.quantile(np.maximum(fresh_error, kmer_error), 0.75))
    low = gaps <= cut
    high = np.maximum(fresh_error, kmer_error) >= high_cut
    result = {
        "pool_id": pool_id,
        "candidate_count": len(rows),
        "fresh_mean_abs_error": float(fresh_error.mean()),
        "kmer_mean_abs_error": float(kmer_error.mean()),
        "mean_architecture_gap": float(gaps.mean()),
        "architecture_error_correlation": float(np.corrcoef(fresh_error, kmer_error)[0, 1]),
        "low_gap_cut": cut,
        "low_gap_high_error_count": int((low & high).sum()),
        "low_gap_mean_max_error": float(np.maximum(fresh_error, kmer_error)[low].mean()),
        "other_gap_mean_max_error": float(np.maximum(fresh_error, kmer_error)[~low].mean()),
        "labels_returned_to_learner": False,
    }
    atomic_json_write(result, output)
    return result
