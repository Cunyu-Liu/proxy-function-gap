import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from scipy import sparse
from sklearn.linear_model import Ridge
from torch import nn

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def encode(sequences):
    result = np.zeros((len(sequences), len(sequences[0]), len(AMINO_ACIDS)), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        for j, aa in enumerate(sequence):
            result[i, j, AA_TO_INDEX[aa]] = 1.0
    return result


def metrics(y, prediction):
    return {"mae": float(np.mean(np.abs(y - prediction))), "spearman": float(spearmanr(y, prediction).statistic)}


class SmallCNN(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(20, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * channels, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        return self.head(self.dropout(x)).squeeze(1)


def predict(model, x, batch_size=1024):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunks.append(model(x[start:start + batch_size]).float().cpu().numpy())
    return np.concatenate(chunks)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = Path(config["paths"]["artifacts"])
    assert not (root / "predictions" / "PREDICTIONS_FROZEN.json").exists(), "Refusing to retrain after test predictions were frozen"
    rows = read_csv(root / "learner" / "low_order_labeled.csv")
    assert all(r["role"] in {"train", "dev"} and int(r["order"]) <= 2 for r in rows)
    train = [r for r in rows if r["role"] == "train"]
    dev = [r for r in rows if r["role"] == "dev"]
    test = read_csv(root / "learner" / "high_order_sequences.csv")
    assert all(int(r["order"]) >= 5 for r in test)
    assert "DMS_score" not in test[0]
    x_train = torch.from_numpy(encode([r["mutated_sequence"] for r in train]))
    x_dev = torch.from_numpy(encode([r["mutated_sequence"] for r in dev]))
    y_train_raw = np.asarray([float(r["DMS_score"]) for r in train], dtype=np.float64)
    y_dev_raw = np.asarray([float(r["DMS_score"]) for r in dev], dtype=np.float64)
    label_mean = float(y_train_raw.mean())
    label_std = float(y_train_raw.std())
    assert label_std > 0
    y_train = torch.from_numpy(((y_train_raw - label_mean) / label_std).astype(np.float32))
    assert torch.cuda.is_available(), "Real CUDA is required"
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    grid = config["models"]["cnn_grid"]
    combinations = list(itertools.product(grid["channels"], grid["kernel_size"], grid["learning_rate"], grid["weight_decay"], grid["dropout"]))
    all_runs = []
    for channels, kernel_size, learning_rate, weight_decay, dropout in combinations:
        name = f"cnn_c{channels}_k{kernel_size}_lr{learning_rate}_wd{weight_decay}_do{dropout}"
        for seed in config["models"]["seeds"]:
            started = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = SmallCNN(channels, kernel_size, dropout).to(device)
            assert next(model.parameters()).is_cuda
            x_train_device = x_train.to(device)
            x_dev_device = x_dev.to(device)
            y_train_device = y_train.to(device)
            torch.cuda.reset_peak_memory_stats()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            run_dir = root / "training" / name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            history_path = run_dir / "history.jsonl"
            best_dev = math.inf
            best_epoch = 0
            stale = 0
            with open(history_path, "w") as history:
                for epoch in range(1, config["models"]["max_epochs"] + 1):
                    model.train()
                    permutation = torch.randperm(len(train), generator=generator)
                    losses = []
                    for start in range(0, len(train), config["models"]["batch_size"]):
                        indices = permutation[start:start + config["models"]["batch_size"]].to(device)
                        optimizer.zero_grad(set_to_none=True)
                        output = model(x_train_device[indices])
                        loss = nn.functional.mse_loss(output, y_train_device[indices])
                        assert torch.isfinite(loss)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        losses.append(float(loss.detach().cpu()))
                    dev_prediction = predict(model, x_dev_device) * label_std + label_mean
                    dev_mae = float(np.mean(np.abs(y_dev_raw - dev_prediction)))
                    record = {"epoch": epoch, "train_mse_standardized": float(np.mean(losses)), "dev_mae": dev_mae, "cuda_memory_allocated": int(torch.cuda.memory_allocated()), "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated())}
                    history.write(json.dumps(record, sort_keys=True) + "\n")
                    history.flush()
                    if dev_mae < best_dev - 1e-6:
                        best_dev = dev_mae
                        best_epoch = epoch
                        stale = 0
                        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "channels": channels, "kernel_size": kernel_size, "dropout": dropout, "label_mean": label_mean, "label_std": label_std, "epoch": epoch, "seed": seed}, run_dir / "best.pt")
                    else:
                        stale += 1
                    if stale >= config["models"]["early_stopping_patience"]:
                        break
            checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            dev_prediction = predict(model, x_dev_device) * label_std + label_mean
            run_metrics = metrics(y_dev_raw, dev_prediction)
            summary = {"model": name, "seed": seed, "channels": channels, "kernel_size": kernel_size, "best_epoch": best_epoch, "epochs_run": epoch, "dev": run_metrics, "label_mean": label_mean, "label_std": label_std, "cuda_device": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda, "torch_version": torch.__version__, "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()), "elapsed_seconds": time.time() - started}
            assert summary["cuda_max_memory_allocated"] > 0
            atomic_json(run_dir / "summary.json", summary)
            np.save(run_dir / "dev_predictions.npy", dev_prediction.astype(np.float64))
            all_runs.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
            del model, x_train_device, x_dev_device, y_train_device
            torch.cuda.empty_cache()
    grouped = {}
    for run in all_runs:
        grouped.setdefault(run["model"], []).append(run)
    assert all(len(runs) == len(config["models"]["seeds"]) for runs in grouped.values())
    ranking = []
    for name, runs in grouped.items():
        ranking.append({"model": name, "mean_dev_mae": float(np.mean([r["dev"]["mae"] for r in runs])), "mean_dev_spearman": float(np.mean([r["dev"]["spearman"] for r in runs])), "channels": runs[0]["channels"], "kernel_size": runs[0]["kernel_size"]})
    ranking.sort(key=lambda r: (round(r["mean_dev_mae"], 12), -r["mean_dev_spearman"], r["channels"], r["kernel_size"]))
    selected = ranking[0]
    selection = {"selection_rule": config["models"]["selection"], "selected_model": selected["model"], "ranking": ranking, "runs": all_runs, "outer_test_used": False}
    atomic_json(root / "training" / "dev_selection.json", selection)
    x_test = torch.from_numpy(encode([r["mutated_sequence"] for r in test])).to(device)
    test_predictions = []
    checkpoint_hashes = {}
    for seed in config["models"]["seeds"]:
        checkpoint_path = root / "training" / selected["model"] / f"seed_{seed}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = SmallCNN(checkpoint["channels"], checkpoint["kernel_size"], checkpoint["dropout"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        test_predictions.append(predict(model, x_test) * checkpoint["label_std"] + checkpoint["label_mean"])
        checkpoint_hashes[str(checkpoint_path.relative_to(root))] = sha256_file(checkpoint_path)
        del model
    cnn_prediction = np.mean(test_predictions, axis=0)
    x_train_sparse_rows, x_train_sparse_cols = [], []
    all_rows = train + dev + test
    sparse_rows, sparse_cols = [], []
    for i, r in enumerate(all_rows):
        for j, aa in enumerate(r["mutated_sequence"]):
            sparse_rows.append(i)
            sparse_cols.append(j * 20 + AA_TO_INDEX[aa])
    design = sparse.csr_matrix((np.ones(len(sparse_rows), dtype=np.float64), (sparse_rows, sparse_cols)), shape=(len(all_rows), 238 * 20))
    design_train = design[:len(train)]
    design_dev = design[len(train):len(train) + len(dev)]
    design_test = design[len(train) + len(dev):]
    additive_results = []
    for alpha in config["models"]["additive_ridge_alpha_grid"]:
        ridge = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
        ridge.fit(design_train, y_train_raw)
        dev_prediction = ridge.predict(design_dev)
        additive_results.append({"alpha": alpha, **metrics(y_dev_raw, dev_prediction)})
    additive_results.sort(key=lambda r: (round(r["mae"], 12), -r["spearman"], r["alpha"]))
    best_alpha = additive_results[0]["alpha"]
    ridge = Ridge(alpha=best_alpha, fit_intercept=True, solver="lsqr")
    ridge.fit(design_train, y_train_raw)
    additive_prediction = ridge.predict(design_test)
    atomic_json(root / "training" / "additive_dev_selection.json", {"ranking": additive_results, "selected_alpha": best_alpha, "outer_test_used": False})
    predictions_dir = root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    variant_ids = np.asarray([r["variant_id"] for r in test])
    pool_ids = np.asarray([r["pool_id"] for r in test])
    orders = np.asarray([int(r["order"]) for r in test])
    np.savez(predictions_dir / "cnn_seed_predictions.npz", variant_id=variant_ids, pool_id=pool_ids, order=orders, prediction=np.asarray(test_predictions), seed=np.asarray(config["models"]["seeds"]))
    np.savez(predictions_dir / "cnn_ensemble.npz", variant_id=variant_ids, pool_id=pool_ids, order=orders, prediction=cnn_prediction)
    np.savez(predictions_dir / "additive.npz", variant_id=variant_ids, pool_id=pool_ids, order=orders, prediction=additive_prediction)
    code_root = Path(config["paths"]["code"])
    code_hashes = {str(p.relative_to(code_root)): sha256_file(p) for p in sorted(code_root.rglob("*.py"))}
    manifest = {"status": "FROZEN_BEFORE_TEST_LABEL_EVALUATION", "config_sha256": sha256_file(config_path), "code_sha256": code_hashes, "selected_cnn": selected["model"], "selected_additive_alpha": best_alpha, "checkpoint_sha256": checkpoint_hashes, "prediction_files": {p.name: sha256_file(p) for p in sorted(predictions_dir.glob("*.npz"))}, "test_row_count": len(test), "test_label_columns_in_learner_view": [], "cuda_training_runs": len(all_runs), "all_cuda_runs_had_allocation": all(r["cuda_max_memory_allocated"] > 0 for r in all_runs)}
    atomic_json(predictions_dir / "PREDICTIONS_FROZEN.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
