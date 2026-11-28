#!/usr/bin/env python3
"""Validation-selected frozen-RNA-FM baselines for RNAcompete Ray2013."""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from log_run import RunLogger, sha256_file
from rnacompete_ray2013_tasklog import TaskLogger

VERSION = "rnacompete-ray2013-v1.0-id90-strict3train"
ARCHES = ("linear", "mlp")
TRACKS = ("sequence", "rbp_ood")
WEIGHT_DECAYS = (1e-4, 1e-2, 1.0)
K_VALUES = (1, 5, 20)
SEEDS = (0, 1, 2)


def ids_sha256(ids):
    return hashlib.sha256(("\n".join(list(ids)) + "\n").encode()).hexdigest()


def atomic_json(path, value):
    with open(path + ".tmp", "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(path + ".tmp", path)


def atomic_npz(path, **arrays):
    with open(path + ".tmp", "wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(path + ".tmp", path)


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    x = torch.tensor([1.25, -2.0], device=device, requires_grad=True)
    y = torch.tensor([2.0, 3.0], device=device)
    observed = (x * y).detach().cpu().numpy()
    if not np.array_equal(observed, np.array([2.5, -6.0], dtype=np.float32)):
        raise RuntimeError(f"CUDA numeric probe failed: {observed}")
    (x * y).sum().backward()
    if not torch.equal(x.grad, y) or not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA backward probe failed")
    torch.cuda.synchronize(device)
    info = {"device": "cuda:0", "name": torch.cuda.get_device_name(device),
            "numeric_probe": observed.tolist(), "backward_probe": "passed",
            "torch": str(torch.__version__)}
    print("DEVICE_PROBE " + json.dumps(info), flush=True)
    del x, y
    torch.cuda.empty_cache()
    return device, info


def split_roles(test_fold):
    validation = (test_fold + 1) % 5
    train = [fold for fold in range(5) if fold not in (test_fold, validation)]
    if len(train) != 3:
        raise ValueError("strict3train violation")
    return {"train": train, "validation": [validation], "test": [test_fold]}


def regression_metrics(y, prediction):
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(y)
    if not np.isfinite(prediction[finite]).all():
        raise ValueError("Non-finite prediction at observed label")
    if not finite.any():
        return {"spearman": np.nan, "pearson": np.nan, "mse": np.nan, "n": 0}
    y = y[finite]
    prediction = prediction[finite]
    mse = float(np.mean((y - prediction) ** 2))
    if len(y) < 2 or np.unique(y).size < 2 or np.unique(prediction).size < 2:
        return {"spearman": np.nan, "pearson": np.nan, "mse": mse, "n": len(y)}
    yc = y - y.mean()
    pc = prediction - prediction.mean()
    pearson = float(np.dot(yc, pc) / np.sqrt(np.dot(yc, yc) * np.dot(pc, pc)))
    yr = rankdata(y)
    pr = rankdata(prediction)
    yrc = yr - yr.mean()
    prc = pr - pr.mean()
    spearman = float(np.dot(yrc, prc) / np.sqrt(np.dot(yrc, yrc) * np.dot(prc, prc)))
    return {"spearman": spearman, "pearson": pearson, "mse": mse, "n": len(y)}


def macro_metrics(labels, prediction):
    labels = np.asarray(labels)
    prediction = np.asarray(prediction)
    if labels.ndim == 1:
        labels = labels[:, None]
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if prediction.shape[1] == 1 and labels.shape[1] > 1:
        prediction = np.tile(prediction, (1, labels.shape[1]))
    if prediction.shape != labels.shape:
        raise ValueError(f"Prediction shape {prediction.shape} != labels {labels.shape}")
    rows = [regression_metrics(labels[:, col], prediction[:, col]) for col in range(labels.shape[1])]
    valid = [row for row in rows if np.isfinite(row["spearman"])]
    if not valid:
        return {"macro_spearman": np.nan, "macro_pearson": np.nan,
                "macro_mse": float(np.mean([row["mse"] for row in rows])), "n_valid_spearman": 0}
    return {"macro_spearman": float(np.mean([row["spearman"] for row in valid])),
            "macro_pearson": float(np.mean([row["pearson"] for row in valid])),
            "macro_mse": float(np.mean([row["mse"] for row in rows])),
            "n_valid_spearman": len(valid)}


def better_candidate(candidate, incumbent):
    if incumbent is None:
        return True
    left = (candidate["macro_spearman"], -candidate["macro_mse"], -candidate["regularization"])
    right = (incumbent["macro_spearman"], -incumbent["macro_mse"], -incumbent["regularization"])
    return left > right


class BenchmarkData:
    def __init__(self, frozen_root, embedding_dir, label_cache, encoder="rnafm", emb_dim=640):
        self.frozen_root = frozen_root
        self.label_cache = label_cache
        self.encoder = encoder
        self.emb_dim = emb_dim
        manifest_path = os.path.join(frozen_root, "split_manifest.json")
        with open(manifest_path) as handle:
            self.manifest = json.load(handle)
        if self.manifest["version"] != VERSION or self.manifest["status"] != "accepted":
            raise ValueError("Frozen benchmark is not accepted")
        self.sequences = pd.read_csv(os.path.join(frozen_root, "sequence_folds.tsv.gz"), sep="\t")
        self.experiments = pd.read_csv(os.path.join(frozen_root, "experiments.tsv"), sep="\t")
        self.experiments = self.experiments.sort_values("Motif_ID").reset_index(drop=True)
        if not self.experiments.sequence_track_included.all() or len(self.experiments) != 244:
            raise ValueError("Expected all 244 frozen sequence-track experiments")
        cache = np.load(label_cache)
        self.labels = cache["labels"]
        if cache["sequence_ids"].tolist() != self.sequences.sequence_id.tolist():
            raise ValueError("Label cache sequence IDs differ from frozen order")
        if cache["experiment_ids"].tolist() != self.experiments.Motif_ID.tolist():
            raise ValueError("Label cache experiment IDs differ from frozen order")
        if self.labels.shape != (len(self.sequences), len(self.experiments)):
            raise ValueError("Invalid label cache shape")
        if not np.isfinite(self.labels).any() or np.isinf(self.labels).any():
            raise ValueError("Invalid labels")
        with open(os.path.join(embedding_dir, f"{encoder}.ids")) as handle:
            embedding_ids = [line.strip() for line in handle]
        if embedding_ids != self.sequences.sequence_id.tolist():
            raise ValueError("Embedding IDs differ from frozen order")
        self.embeddings = np.load(os.path.join(embedding_dir, f"{encoder}.mean_masked.fp16.npy"), mmap_mode="r")
        if self.embeddings.shape != (len(self.sequences), emb_dim):
            raise ValueError("Invalid embedding shape")
        for start in range(0, len(self.sequences), 4096):
            if not np.isfinite(np.asarray(self.embeddings[start:start + 4096])).all():
                raise ValueError("Non-finite shared embeddings")

    def context(self, track, fold, include_test=False):
        roles = split_roles(fold)
        if track == "sequence":
            row_sets = {role: np.flatnonzero(self.sequences.fold.isin(folds).to_numpy())
                        for role, folds in roles.items()}
            task_sets = {role: np.arange(len(self.experiments)) for role in roles}
        elif track == "rbp_ood":
            row_sets = {role: np.arange(len(self.sequences)) for role in roles}
            task_sets = {role: np.flatnonzero(self.experiments.rbp_fold.isin(folds).to_numpy())
                         for role, folds in roles.items()}
        else:
            raise ValueError(track)
        if not include_test:
            row_sets.pop("test")
            task_sets.pop("test")
        for role in row_sets:
            if not len(row_sets[role]) or not len(task_sets[role]):
                raise ValueError(f"Empty {track}/{fold}/{role}")
        return {"track": track, "fold": fold, "roles": roles,
                "rows": row_sets, "tasks": task_sets}

    def arrays(self, context, role):
        rows = context["rows"][role]
        tasks = context["tasks"][role]
        x = np.asarray(self.embeddings[rows], dtype=np.float32)
        y = np.asarray(self.labels[np.ix_(rows, tasks)], dtype=np.float32)
        return x, y


def prepare_labels(args):
    frozen = args.frozen_root
    sequences = pd.read_csv(os.path.join(frozen, "sequence_folds.tsv.gz"), sep="\t",
                            usecols=["sequence_id"])
    experiments = pd.read_csv(os.path.join(frozen, "experiments.tsv"), sep="\t").sort_values("Motif_ID")
    probes = pd.read_csv(os.path.join(frozen, "probes.tsv.gz"), sep="\t",
                         usecols=["probe_id", "sequence_id"])
    raw = pd.read_csv(os.path.join(args.input_root, "norm_data.txt.gz"), sep="\t", low_memory=False)
    motif_ids = experiments.Motif_ID.tolist()
    if raw.columns[:3].tolist() != ["Probe_Set", "RNA_Seq", "Probe_ID"] or set(raw.columns[3:]) != set(motif_ids):
        raise ValueError("Normalized input schema differs from frozen experiments")
    raw = raw.set_index("Probe_ID").loc[probes.probe_id]
    sequence_index = pd.Index(sequences.sequence_id)
    probe_rows = sequence_index.get_indexer(probes.sequence_id)
    if (probe_rows < 0).any():
        raise ValueError("Probe sequence missing from frozen sequence table")
    labels = np.full((len(sequences), len(experiments)), np.nan, dtype=np.float32)
    order = np.argsort(probe_rows, kind="stable")
    sorted_rows = probe_rows[order]
    for col, motif in enumerate(motif_ids):
        grouped = pd.DataFrame({"row": sorted_rows,
                                "value": raw[motif].to_numpy(dtype=np.float64)[order]}).groupby(
            "row", sort=False).value.median()
        labels[grouped.index.to_numpy(), col] = grouped.to_numpy()
        if (col + 1) % 25 == 0 or col + 1 == len(motif_ids):
            print(f"LABEL_CACHE {col + 1}/{len(motif_ids)}", flush=True)
    counts = np.isfinite(labels).sum(axis=0)
    expected = experiments.n_finite_unique.to_numpy()
    if not np.array_equal(counts, expected):
        raise ValueError("Label cache finite counts differ from frozen experiment audit")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "labels.unique_rna.npz")
    atomic_npz(path, labels=labels, sequence_ids=sequences.sequence_id.to_numpy(dtype=str),
               experiment_ids=experiments.Motif_ID.to_numpy(dtype=str))
    result = {"status": "accepted", "kind": "label-cache-not-model-result", "version": VERSION,
              "shape": list(labels.shape), "finite": int(np.isfinite(labels).sum()),
              "nan_unobserved": int(np.isnan(labels).sum()),
              "frozen_manifest_sha256": sha256_file(os.path.join(frozen, "split_manifest.json")),
              "labels_sha256": sha256_file(path), "aggregation": "median finite probes; NaN unobserved"}
    atomic_json(os.path.join(args.out, "label_manifest.json"), result)
    print("LABEL_CACHE_OK " + json.dumps(result), flush=True)


def make_model(arch, input_dim, output_dim):
    import torch
    if arch == "linear":
        return torch.nn.Linear(input_dim, output_dim)
    if arch == "mlp":
        return torch.nn.Sequential(torch.nn.Linear(input_dim, 256), torch.nn.GELU(),
                                   torch.nn.Dropout(0.1), torch.nn.Linear(256, output_dim))
    raise ValueError(arch)


def label_placeholders(labels):
    observed = np.isfinite(labels)
    return observed, np.where(observed, labels, 0).astype(np.float32)


def masked_mse(prediction, target, observed):
    if prediction.shape != target.shape:
        prediction = prediction.expand_as(target)
    count = observed.sum()
    if count.item() == 0:
        raise ValueError("Masked MSE received an all-NaN batch")
    return (((prediction - target) ** 2) * observed).sum() / count


def predict_neural(model, x, batch=2048):
    import torch
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            outputs.append(model(x[start:start + batch]).float())
    return torch.cat(outputs).cpu().numpy()


def aligned_prediction(prediction, n_tasks):
    prediction = np.asarray(prediction)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if prediction.shape[1] == 1 and n_tasks > 1:
        prediction = np.tile(prediction, (1, n_tasks))
    if prediction.ndim != 2 or prediction.shape[1] != n_tasks:
        raise ValueError("Prediction cannot be aligned to experiment IDs")
    return prediction


def save_validation_predictions(path, data, context, prediction):
    rows = context["rows"]["validation"]
    tasks = context["tasks"]["validation"]
    prediction = aligned_prediction(prediction, len(tasks))
    atomic_npz(path, sequence_ids=data.sequences.sequence_id.to_numpy(dtype=str)[rows],
               experiment_ids=data.experiments.Motif_ID.to_numpy(dtype=str)[tasks], prediction=prediction)


def train_core(data, track, arch, fold, seed, weight_decay, out, args, device_info, selection_path=None):
    import torch
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, ".train_complete")):
        raise FileExistsError(f"Training output already complete: {out}")
    context = data.context(track, fold, include_test=False)
    xtr, ytr = data.arrays(context, "train")
    xva, _ = data.arrays(context, "validation")
    feature_mean = xtr.mean(axis=0, keepdims=True)
    feature_std = xtr.std(axis=0, keepdims=True).clip(min=1e-5)
    xtr = (xtr - feature_mean) / feature_std
    xva = (xva - feature_mean) / feature_std
    if track == "sequence":
        observed = np.isfinite(ytr)
        counts = observed.sum(axis=0)
        if (counts < 2).any():
            raise ValueError("Fewer than two finite train labels in a task")
        sums = np.where(observed, ytr, 0).sum(axis=0)
        label_mean = sums / counts
        label_var = (np.where(observed, (ytr - label_mean) ** 2, 0).sum(axis=0) / counts)
        label_std = np.sqrt(label_var).clip(min=1e-5)
        ytr = (ytr - label_mean) / label_std
    else:
        observed = np.isfinite(ytr)
        label_mean = np.array([np.nanmean(ytr)], dtype=np.float32)
        label_std = np.array([max(float(np.nanstd(ytr)), 1e-5)], dtype=np.float32)
        ytr = (ytr - label_mean) / label_std
    observed, ytr = label_placeholders(ytr)
    seed_value = int.from_bytes(hashlib.sha256(
        f"RNAcompete|{track}|{arch}|{fold}|{seed}".encode()).digest()[:4], "little")
    torch.manual_seed(seed_value)
    np.random.seed(seed_value % (2**32))
    device = torch.device("cuda:0")
    xtr_t = torch.from_numpy(xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    mask_t = torch.from_numpy(observed.astype(np.float32)).to(device)
    xva_t = torch.from_numpy(xva).to(device)
    output_dim = len(context["tasks"]["train"]) if track == "sequence" else 1
    model = make_model(arch, data.embeddings.shape[1], output_dim).to(device)
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Training head is not on CUDA")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_value)
    history = []
    best = None
    best_state = None
    best_epoch = 0
    started = time.time()
    for epoch in range(1, 1001):
        model.train()
        order = torch.randperm(len(xtr_t), device=device, generator=generator)
        loss_sum = 0.0
        for start in range(0, len(order), 1024):
            indices = order[start:start + 1024]
            prediction = model(xtr_t[indices])
            loss = masked_mse(prediction, ytr_t[indices], mask_t[indices])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        model.eval()
        val_prediction = predict_neural(model, xva_t)
        val_raw = val_prediction * label_std + label_mean
        _, yva = data.arrays(context, "validation")
        val_metrics = macro_metrics(yva, val_raw)
        row = {"epoch": epoch, "train_loss": loss_sum / len(order), **val_metrics}
        history.append(row)
        if np.isfinite(val_metrics["macro_spearman"]) and (
                best is None or val_metrics["macro_spearman"] > best["macro_spearman"] + 1e-4):
            best = val_metrics
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_prediction = val_raw.copy()
        if epoch % 10 == 0 or epoch == 1:
            print(f"TRAIN {track}/{arch}/f{fold}/s{seed}/wd{weight_decay} epoch={epoch} "
                  f"best={best_epoch} val={val_metrics['macro_spearman']}", flush=True)
        if best_state is not None and epoch - best_epoch >= 20:
            break
    if best_state is None or epoch >= 1000:
        raise RuntimeError("Training failed to early-stop before the 1000-epoch failsafe")
    model.load_state_dict(best_state)
    checkpoint = {"model_state": best_state, "arch": arch, "track": track, "fold": fold,
                  "seed": seed, "weight_decay": weight_decay, "feature_mean": feature_mean,
                  "feature_std": feature_std, "label_mean": label_mean, "label_std": label_std,
                  "best_epoch": best_epoch, "stopped_epoch": epoch, "patience": 20,
                  "min_delta": 1e-4, "seed_value": seed_value, "cuda": device_info,
                  "context_roles": context["roles"], "output_dim": output_dim}
    torch.save(checkpoint, os.path.join(out, "best.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out, "validation_history.tsv"), sep="\t", index=False)
    save_validation_predictions(os.path.join(out, "validation_predictions.npz"), data, context, best_prediction)
    metrics = {"best_epoch": best_epoch, "stopped_epoch": epoch, "early_stopped": True,
               "patience": 20, "min_delta": 1e-4, "epochs_failsafe": 1000,
               **best, "n_train_rows": len(context["rows"]["train"]),
               "n_train_tasks": len(context["tasks"]["train"]),
               "train_row_id_sha256": ids_sha256(data.sequences.sequence_id.to_numpy()[context["rows"]["train"]]),
               "train_task_id_sha256": ids_sha256(data.experiments.Motif_ID.to_numpy()[context["tasks"]["train"]]),
               "cuda": device_info, "elapsed_seconds": time.time() - started}
    atomic_json(os.path.join(out, "train_metrics.json"), metrics)
    config = {"version": VERSION, "track": track, "arch": arch, "fold": fold, "seed": seed,
              "weight_decay": weight_decay, "lr": 1e-3, "batch": 1024, "loss": "masked_mse",
              "patience": 20, "min_delta": 1e-4, "max_epochs": 1000,
              "selection_path": selection_path, "test_read_during_training": False,
              "frozen_manifest_sha256": sha256_file(os.path.join(data.frozen_root, "split_manifest.json")),
              "label_cache_sha256": sha256_file(data.label_cache),
              "code_sha256": sha256_file(os.path.abspath(__file__))}
    atomic_json(os.path.join(out, "config.json"), config)
    with open(os.path.join(out, ".train_complete") + ".tmp", "w") as handle:
        handle.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    os.replace(os.path.join(out, ".train_complete") + ".tmp", os.path.join(out, ".train_complete"))
    del model, optimizer, xtr_t, ytr_t, mask_t, xva_t
    torch.cuda.empty_cache()
    return {"out": out, "weight_decay": weight_decay, "regularization": weight_decay, **metrics}


def select_neural(data, track, arch, args, device_info):
    if args.fold != 0 or args.seed != 0:
        raise ValueError("Hyperparameter selection is frozen to fold0/seed0")
    candidates = []
    for weight_decay in WEIGHT_DECAYS:
        out = os.path.join(args.out, "selection_runs", f"{track}_{arch}_wd{weight_decay:g}")
        candidates.append(train_core(data, track, arch, 0, 0, weight_decay, out, args, device_info))
    best = None
    for candidate in candidates:
        if better_candidate(candidate, best):
            best = candidate
    selection = {"status": "frozen", "version": VERSION, "track": track, "arch": arch,
                 "selection_fold": 0, "selection_seed": 0, "selection_metric": "validation_macro_spearman",
                 "tie_breakers": ["lower_validation_macro_mse", "lower_weight_decay"],
                 "weight_decay": best["weight_decay"], "candidate_weight_decays": list(WEIGHT_DECAYS),
                 "candidate_metrics": [{key: value for key, value in row.items() if key not in ("cuda", "out")}
                                       for row in candidates],
                 "test_used": False, "contract_sha256": sha256_file(args.contract),
                 "code_sha256": sha256_file(os.path.abspath(__file__))}
    path = os.path.join(args.out, "selections", f"{track}_{arch}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    atomic_json(path, selection)
    print("SELECTION_FROZEN " + json.dumps({"path": path, "weight_decay": best["weight_decay"],
          "validation_macro_spearman": best["macro_spearman"]}), flush=True)


def train_selected(data, track, arch, fold, seed, args, device_info):
    selection_path = os.path.join(args.out, "selections", f"{track}_{arch}.json")
    with open(selection_path) as handle:
        selection = json.load(handle)
    if selection["status"] != "frozen" or selection["track"] != track or selection["arch"] != arch:
        raise ValueError("Invalid frozen selection")
    out = os.path.join(args.out, "runs", f"{track}_{arch}_f{fold}_s{seed}")
    result = train_core(data, track, arch, fold, seed, float(selection["weight_decay"]), out, args,
                        device_info, selection_path=selection_path)
    config_path = os.path.join(out, "config.json")
    with open(config_path) as handle:
        config = json.load(handle)
    config["selection_sha256"] = sha256_file(selection_path)
    atomic_json(config_path, config)
    print("TRAIN_SELECTED_OK " + json.dumps({"out": out, "best_epoch": result["best_epoch"]}), flush=True)


def load_neural_checkpoint(run, track, arch, fold):
    import torch
    if not os.path.isfile(os.path.join(run, ".train_complete")):
        raise ValueError("Cannot evaluate incomplete training run")
    checkpoint = torch.load(os.path.join(run, "best.pt"), map_location="cpu", weights_only=False)
    if (checkpoint["track"], checkpoint["arch"], checkpoint["fold"]) != (track, arch, fold):
        raise ValueError("Checkpoint identity mismatch")
    with open(os.path.join(run, "config.json")) as handle:
        config = json.load(handle)
    return checkpoint, config


def metrics_table(data, context, split, prediction):
    rows = context["rows"][split]
    tasks = context["tasks"][split]
    labels = data.labels[np.ix_(rows, tasks)]
    prediction = np.asarray(prediction)
    if prediction.ndim == 1:
        prediction = np.tile(prediction[:, None], (1, len(tasks)))
    if prediction.shape != labels.shape or not np.isfinite(prediction[np.isfinite(labels)]).all():
        raise ValueError(f"Invalid prediction matrix {prediction.shape} versus {labels.shape}")
    records = []
    probe_sets = data.sequences.probe_set.to_numpy()[rows]
    for local_task, task_index in enumerate(tasks):
        experiment = data.experiments.iloc[task_index]
        for probe_set in ("All", "SetA", "SetB"):
            selected = np.ones(len(rows), dtype=bool) if probe_set == "All" else probe_sets == probe_set
            row = regression_metrics(labels[selected, local_task], prediction[selected, local_task])
            records.append({"Motif_ID": experiment.Motif_ID, "species_gene": experiment.species_gene,
                            "frequency_tier": experiment.frequency_tier, "split": split,
                            "probe_set": probe_set, **row})
    return pd.DataFrame(records)


def save_test_products(data, context, prediction, out, config):
    rows = context["rows"]["test"]
    tasks = context["tasks"]["test"]
    prediction = aligned_prediction(np.asarray(prediction, dtype=np.float32), len(tasks))
    atomic_npz(os.path.join(out, "test_predictions.npz"),
               sequence_ids=data.sequences.sequence_id.to_numpy(dtype=str)[rows],
               experiment_ids=data.experiments.Motif_ID.to_numpy(dtype=str)[tasks], prediction=prediction)
    table = metrics_table(data, context, "test", prediction)
    table.to_csv(os.path.join(out, "test_metrics.tsv"), sep="\t", index=False)
    atomic_json(os.path.join(out, "eval_config.json"),
                {**config, "test_used_for_selection": False, "evaluation_only": True,
                 "test_row_id_sha256": ids_sha256(data.sequences.sequence_id.to_numpy()[rows]),
                 "test_task_id_sha256": ids_sha256(data.experiments.Motif_ID.to_numpy()[tasks])})
    with open(os.path.join(out, ".eval_complete") + ".tmp", "w") as handle:
        handle.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    os.replace(os.path.join(out, ".eval_complete") + ".tmp", os.path.join(out, ".eval_complete"))
    return table


def evaluate_neural(data, track, arch, fold, seed, args, device_info):
    import torch
    run = os.path.join(args.out, "runs", f"{track}_{arch}_f{fold}_s{seed}")
    if os.path.exists(os.path.join(run, ".eval_complete")):
        raise FileExistsError(f"Evaluation already complete: {run}")
    checkpoint, config = load_neural_checkpoint(run, track, arch, fold)
    if checkpoint["seed"] != seed:
        raise ValueError("Checkpoint seed mismatch")
    context = data.context(track, fold, include_test=True)
    rows = context["rows"]["test"]
    x = np.asarray(data.embeddings[rows], dtype=np.float32)
    x = (x - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    device = torch.device("cuda:0")
    model = make_model(arch, data.embeddings.shape[1], checkpoint["output_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    x_t = torch.from_numpy(x).to(device)
    prediction = predict_neural(model, x_t)
    prediction = prediction * checkpoint["label_std"] + checkpoint["label_mean"]
    table = save_test_products(data, context, prediction, run, config)
    del model, x_t
    torch.cuda.empty_cache()
    print("EVALUATE_OK " + json.dumps({"run": run, "rows": len(table),
          "macro_test_spearman": float(table[table.probe_set == "All"].spearman.mean())}), flush=True)


def knn_consensus_labels(data, context):
    _, labels = data.arrays(context, "train")
    if context["track"] == "rbp_ood":
        observed = np.isfinite(labels)
        counts = observed.sum(axis=1)
        if (counts == 0).any():
            raise ValueError("Train sequence has no finite RBP label")
        labels = (np.where(observed, labels, 0).sum(axis=1) / counts)[:, None]
    return labels


def knn_predict(data, context, split, k_values, device):
    import torch
    train_rows = context["rows"]["train"]
    eval_rows = context["rows"][split]
    xtr = np.asarray(data.embeddings[train_rows], dtype=np.float32)
    xev = np.asarray(data.embeddings[eval_rows], dtype=np.float32)
    xtr /= np.linalg.norm(xtr, axis=1, keepdims=True).clip(min=1e-12)
    xev /= np.linalg.norm(xev, axis=1, keepdims=True).clip(min=1e-12)
    labels = knn_consensus_labels(data, context)
    xtr_t = torch.from_numpy(xtr).to(device)
    labels_t = torch.from_numpy(np.where(np.isfinite(labels), labels, 0)).to(device)
    mask_t = torch.from_numpy(np.isfinite(labels).astype(np.float32)).to(device)
    max_k = min(256, len(xtr))
    predictions = {k: [] for k in k_values}
    with torch.no_grad():
        for start in range(0, len(xev), 64):
            batch = torch.from_numpy(xev[start:start + 64]).to(device)
            similarity = batch @ xtr_t.T
            values, indices = similarity.topk(max_k, dim=1)
            neighbor_labels = labels_t[indices]
            neighbor_mask = mask_t[indices]
            finite_rank = neighbor_mask.cumsum(dim=1)
            for k in k_values:
                masks = ((finite_rank > 0) & (finite_rank <= k)).float()
                weights = values.clamp(min=0) ** 2
                weighted_mask = weights.unsqueeze(-1) * masks
                denominator = weighted_mask.sum(dim=1, keepdim=True)
                weighted_mask = torch.where(denominator == 0, masks, weighted_mask)
                denominator = weighted_mask.sum(dim=1, keepdim=True)
                prediction = (weighted_mask * neighbor_labels).sum(dim=1) / denominator.squeeze(1)
                predictions[k].append(prediction.float().cpu().numpy())
    del xtr_t, labels_t, mask_t
    torch.cuda.empty_cache()
    return {k: np.concatenate(chunks) for k, chunks in predictions.items()}


def select_knn(data, track, args, device_info):
    context = data.context(track, 0, include_test=False)
    predictions = knn_predict(data, context, "validation", K_VALUES, torch_device())
    _, labels = data.arrays(context, "validation")
    candidates = []
    observed = np.isfinite(labels)
    for k, prediction in predictions.items():
        aligned = aligned_prediction(prediction, labels.shape[1])
        if not np.isfinite(aligned[observed]).all():
            candidates.append({"k": k, "regularization": k, "macro_spearman": -1e300,
                               "macro_pearson": None, "macro_mse": 1e300,
                               "n_valid_spearman": 0, "valid_candidate": False})
            continue
        row = {"k": k, "regularization": k, "valid_candidate": True, **macro_metrics(labels, prediction)}
        candidates.append(row)
    best = None
    for candidate in candidates:
        if better_candidate(candidate, best):
            best = candidate
    if best is None or not best["valid_candidate"]:
        raise ValueError(f"No finite-at-observed-labels kNN validation candidate: {track}")
    out = os.path.join(args.out, "knn", f"{track}_f0")
    os.makedirs(out, exist_ok=True)
    save_validation_predictions(os.path.join(out, "validation_predictions.npz"), data, context, predictions[best["k"]])
    selection = {"status": "frozen", "version": VERSION, "track": track, "arch": "knn",
                 "selection_fold": 0, "selection_metric": "validation_macro_spearman",
                 "tie_breakers": ["lower_validation_macro_mse", "lower_k"], "k": best["k"],
                 "candidate_k": list(K_VALUES), "candidate_metrics": candidates, "test_used": False,
                 "contract_sha256": sha256_file(args.contract),
                 "code_sha256": sha256_file(os.path.abspath(__file__))}
    path = os.path.join(args.out, "selections", f"{track}_knn.json")
    atomic_json(path, selection)
    atomic_json(os.path.join(out, "config.json"),
                {"track": track, "arch": "knn", "fold": 0, "k": best["k"],
                 "selection_sha256": sha256_file(path), "cuda": device_info})
    print("KNN_SELECTION_FROZEN " + json.dumps({"track": track, "k": best["k"]}), flush=True)


def torch_device():
    import torch
    return torch.device("cuda:0")


def evaluate_knn(data, track, fold, args, device_info):
    selection_path = os.path.join(args.out, "selections", f"{track}_knn.json")
    with open(selection_path) as handle:
        selection = json.load(handle)
    if selection["status"] != "frozen" or selection["track"] != track:
        raise ValueError("Invalid kNN selection")
    out = os.path.join(args.out, "knn", f"{track}_f{fold}")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, ".eval_complete")):
        raise FileExistsError(f"Evaluation already complete: {out}")
    context = data.context(track, fold, include_test=True)
    prediction = knn_predict(data, context, "test", [int(selection["k"])], torch_device())[int(selection["k"])]
    config = {"version": VERSION, "track": track, "arch": "knn", "fold": fold, "k": selection["k"],
              "selection_sha256": sha256_file(selection_path), "cuda": device_info,
              "code_sha256": sha256_file(os.path.abspath(__file__))}
    table = save_test_products(data, context, prediction, out, config)
    atomic_json(os.path.join(out, "config.json"), config)
    print("KNN_EVALUATE_OK " + json.dumps({"run": out, "rows": len(table)}), flush=True)


def audit_selection_boundaries(root):
    selection_dir = os.path.join(root, "selections")
    expected = [f"{track}_{arch}.json" for track in TRACKS for arch in (*ARCHES, "knn")]
    for name in expected:
        path = os.path.join(selection_dir, name)
        with open(path) as handle:
            selection = json.load(handle)
        if selection["status"] != "frozen" or selection["test_used"] is not False:
            raise ValueError(f"Invalid test-free selection: {path}")
        if selection["selection_fold"] != 0:
            raise ValueError(f"Selection is not restricted to fold0: {path}")
    selection_runs = os.path.join(root, "selection_runs")
    for directory, _, files in os.walk(selection_runs):
        if any(name.startswith("test_") or name == ".eval_complete" for name in files):
            raise ValueError(f"Test artifact found in selection tree: {directory}")
        if "config.json" in files:
            with open(os.path.join(directory, "config.json")) as handle:
                config = json.load(handle)
            if config.get("test_read_during_training") is not False:
                raise ValueError(f"Selection training read test: {directory}")
    return {"n_selections": len(expected), "test_used_for_selection": False,
            "selection_tree_test_artifacts": 0}


def replay_neural_checkpoint(data, run, track, arch, fold, context, saved_prediction, eval_config):
    import torch
    checkpoint, _ = load_neural_checkpoint(run, track, arch, fold)
    if checkpoint["stopped_epoch"] - checkpoint["best_epoch"] != 20:
        raise ValueError(f"Checkpoint patience mismatch: {run}")
    history = pd.read_csv(os.path.join(run, "validation_history.tsv"), sep="\t")
    if len(history) != checkpoint["stopped_epoch"]:
        raise ValueError(f"Checkpoint history length mismatch: {run}")
    selection_path = os.path.join(os.path.dirname(os.path.dirname(run)), "selections", f"{track}_{arch}.json")
    if eval_config.get("selection_sha256") != sha256_file(selection_path):
        raise ValueError(f"Frozen selection SHA mismatch: {run}")
    rows = context["rows"]["test"]
    x = np.asarray(data.embeddings[rows], dtype=np.float32)
    x = (x - checkpoint["feature_mean"]) / checkpoint["feature_std"]
    device = torch.device("cuda:0")
    model = make_model(arch, data.embeddings.shape[1], checkpoint["output_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prediction = predict_neural(model, torch.from_numpy(x).to(device))
    prediction = prediction * checkpoint["label_std"] + checkpoint["label_mean"]
    prediction = aligned_prediction(prediction, len(context["tasks"]["test"]))
    if not np.allclose(prediction, saved_prediction, rtol=1e-6, atol=1e-6, equal_nan=True):
        raise ValueError(f"Checkpoint test prediction replay mismatch: {run}")
    del model
    torch.cuda.empty_cache()
    return {"checkpoint_replay_passed": True, "best_epoch": checkpoint["best_epoch"],
            "stopped_epoch": checkpoint["stopped_epoch"], "selection_sha256": eval_config["selection_sha256"]}


def replay_run(data, track, arch, fold, seed, root):
    if arch == "knn":
        run = os.path.join(root, "knn", f"{track}_f{fold}")
    else:
        run = os.path.join(root, "runs", f"{track}_{arch}_f{fold}_s{seed}")
    if not os.path.isfile(os.path.join(run, ".eval_complete")):
        raise ValueError(f"Missing accepted evaluation: {run}")
    saved = pd.read_csv(os.path.join(run, "test_metrics.tsv"), sep="\t")
    cache = np.load(os.path.join(run, "test_predictions.npz"))
    context = data.context(track, fold, include_test=True)
    rows = context["rows"]["test"]
    tasks = context["tasks"]["test"]
    if cache["sequence_ids"].tolist() != data.sequences.sequence_id.to_numpy()[rows].tolist():
        raise ValueError(f"Test sequence ID replay mismatch: {run}")
    if cache["experiment_ids"].tolist() != data.experiments.Motif_ID.to_numpy()[tasks].tolist():
        raise ValueError(f"Test experiment ID replay mismatch: {run}")
    prediction = cache["prediction"]
    observed_labels = data.labels[np.ix_(rows, tasks)]
    if not np.isfinite(prediction[np.isfinite(observed_labels)]).all():
        raise ValueError(f"Non-finite test prediction at observed label: {run}")
    replay = metrics_table(data, context, "test", prediction)
    keys = ["Motif_ID", "species_gene", "frequency_tier", "split", "probe_set"]
    left = saved.sort_values(keys).reset_index(drop=True)
    right = replay.sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]) or len(left) != len(right):
        raise ValueError(f"Metric schema replay mismatch: {run}")
    for column in ("spearman", "pearson", "mse"):
        if not np.allclose(left[column], right[column], rtol=1e-6, atol=1e-8, equal_nan=True):
            raise ValueError(f"Metric value replay mismatch {column}: {run}")
    if not np.array_equal(left.n, right.n):
        raise ValueError(f"Metric support replay mismatch: {run}")
    with open(os.path.join(run, "eval_config.json")) as handle:
        config = json.load(handle)
    if config["test_used_for_selection"] is not False or config["evaluation_only"] is not True:
        raise ValueError(f"Test selection boundary missing: {run}")
    result = {"run": run, "n_metric_rows": len(saved), "replay_passed": True,
              "test_rows": len(rows), "test_tasks": len(tasks)}
    if arch != "knn":
        result.update(replay_neural_checkpoint(data, run, track, arch, fold, context, prediction, config))
    return result


def verify_gate(data, root, fold0_only=True):
    selection_audit = audit_selection_boundaries(root)
    records = []
    folds = [0] if fold0_only else list(range(5))
    seeds = SEEDS if fold0_only else SEEDS
    for fold in folds:
        for track in TRACKS:
            for arch in ARCHES:
                for seed in seeds:
                    records.append(replay_run(data, track, arch, fold, seed, root))
            records.append(replay_run(data, track, "knn", fold, None, root))
    result = {"status": "passed", "scope": "fold0_three_seed" if fold0_only else "five_fold_three_seed",
              "selection_audit": selection_audit,
              "runs": records, "n_evaluations": len(records), "model_performance_claim": False,
              "expansion_allowed": fold0_only, "version": VERSION}
    path = os.path.join(root, "fold0_gate.json" if fold0_only else "final_gate.json")
    atomic_json(path, result)
    print("RNAcompete_GATE_OK " + json.dumps({"path": path, "n_evaluations": len(records)}), flush=True)


def summarize(data, root):
    gate_path = os.path.join(root, "final_gate.json")
    if not os.path.isfile(gate_path):
        raise ValueError("Full final gate is required before summary")
    frames = []
    for fold in range(5):
        for track in TRACKS:
            for arch in (*ARCHES, "knn"):
                seeds = [None] if arch == "knn" else SEEDS
                for seed in seeds:
                    run = (os.path.join(root, "knn", f"{track}_f{fold}") if arch == "knn" else
                           os.path.join(root, "runs", f"{track}_{arch}_f{fold}_s{seed}"))
                    frame = pd.read_csv(os.path.join(run, "test_metrics.tsv"), sep="\t")
                    frame["track"] = track
                    frame["arch"] = arch
                    frame["fold"] = fold
                    frame["seed"] = seed
                    frames.append(frame)
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(os.path.join(root, "all_experiment_metrics.tsv.gz"), sep="\t", index=False,
                       compression="gzip")
    sequence = all_metrics[all_metrics.track == "sequence"]
    tiers = sequence.frequency_tier.copy()
    overall = sequence.copy()
    overall["frequency_tier"] = "Overall"
    sequence = pd.concat([sequence, overall], ignore_index=True)
    summary = sequence.groupby(["arch", "seed", "fold", "frequency_tier", "probe_set"]).agg(
        spearman=("spearman", "mean"), pearson=("pearson", "mean"), mse=("mse", "mean"),
        n_valid_spearman=("spearman", "count"), n_experiments=("Motif_ID", "nunique")).reset_index()
    summary.to_csv(os.path.join(root, "sequence_summary.tsv"), sep="\t", index=False)
    rbp = all_metrics[all_metrics.track == "rbp_ood"]
    rbp.groupby(["arch", "seed", "fold", "probe_set"]).agg(
        spearman=("spearman", "mean"), pearson=("pearson", "mean"), mse=("mse", "mean"),
        n_valid_spearman=("spearman", "count"), n_experiments=("Motif_ID", "nunique")).reset_index().to_csv(
            os.path.join(root, "rbp_ood_experiment_summary.tsv"), sep="\t", index=False)
    rbp.groupby(["arch", "seed", "fold", "probe_set", "species_gene"]).agg(
        spearman=("spearman", "mean"), pearson=("pearson", "mean"), mse=("mse", "mean")).reset_index().groupby(
            ["arch", "seed", "fold", "probe_set"]).agg(
                spearman=("spearman", "mean"), pearson=("pearson", "mean"), mse=("mse", "mean"),
                n_rbp_groups=("species_gene", "nunique")).reset_index().to_csv(
                    os.path.join(root, "rbp_ood_group_summary.tsv"), sep="\t", index=False)
    print("RNACOMPETE_SUMMARY_OK", flush=True)


def data_from_args(args):
    return BenchmarkData(args.frozen_root, args.embedding_dir, args.label_cache,
                         encoder=getattr(args, "encoder", "rnafm"),
                         emb_dim=getattr(args, "emb_dim", 640))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare-labels", "select", "train", "evaluate", "knn-select",
                                            "knn-evaluate", "verify-fold0", "verify-final", "summarize"])
    parser.add_argument("--frozen-root", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/" + VERSION)
    parser.add_argument("--input-root", default="/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/ray2013_nature12311")
    parser.add_argument("--embedding-dir")
    parser.add_argument("--encoder", choices=["rnafm", "rinalmo"], default="rnafm")
    parser.add_argument("--emb-dim", type=int, default=640)
    parser.add_argument("--label-cache")
    parser.add_argument("--out", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--track", choices=TRACKS)
    parser.add_argument("--arch", choices=ARCHES)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", required=True)
    parser.add_argument("--task-log")
    args = parser.parse_args()
    args.prog = os.path.abspath(__file__)
    if args.task_log is None:
        args.task_log = os.path.join(args.out, "task_runs.jsonl")
    if not args.out.startswith("/teams/"):
        raise ValueError("Large model artifacts must be under /teams")
    os.makedirs(args.out, exist_ok=True)
    if args.command == "prepare-labels":
        prepare_labels(args)
        return
    if args.label_cache is None:
        args.label_cache = os.path.join(args.out, "labels.unique_rna.npz")
    if args.embedding_dir is None:
        args.embedding_dir = os.path.join(args.out, "embeddings")
    if args.fold not in range(5) or args.seed not in SEEDS:
        raise ValueError("Invalid fold or seed")
    if (args.encoder == "rinalmo") != (args.emb_dim == 1280):
        raise ValueError("encoder/emb-dim mismatch: rnafm=640, rinalmo=1280")
    if args.command in ("select", "train", "evaluate", "knn-select", "knn-evaluate") and not args.track:
        raise ValueError("--track is required")
    if args.command in ("select", "train", "evaluate") and not args.arch:
        raise ValueError("--arch is required")
    logger = RunLogger(args.log, stage="RNAcompete", kind="model-" + args.command)
    logger.start(config={key: value for key, value in vars(args).items()},
                 seed=args.seed, data_version=VERSION,
                 inputs=[os.path.join(args.frozen_root, "split_manifest.json")],
                 code_version=sha256_file(os.path.abspath(__file__)))
    task_logger = TaskLogger(args.task_log, args)
    task_logger.start(cuda_required=args.command != "summarize")
    device_info = {"required": args.command != "summarize",
                   "device": "cuda:0" if args.command != "summarize" else None,
                   "probe": "not_completed" if args.command != "summarize" else "not_required"}
    try:
        if args.command in ("verify-fold0", "verify-final", "summarize"):
            if args.command != "summarize":
                _, device_info = require_cuda()
                device_info.update({"required": True, "probe": "passed"})
            data = data_from_args(args)
            if args.command == "verify-fold0":
                verify_gate(data, args.out, fold0_only=True)
            elif args.command == "verify-final":
                verify_gate(data, args.out, fold0_only=False)
            else:
                summarize(data, args.out)
        else:
            _, device_info = require_cuda()
            device_info.update({"required": True, "probe": "passed"})
            data = data_from_args(args)
            if args.command == "select":
                select_neural(data, args.track, args.arch, args, device_info)
            elif args.command == "train":
                train_selected(data, args.track, args.arch, args.fold, args.seed, args, device_info)
            elif args.command == "evaluate":
                evaluate_neural(data, args.track, args.arch, args.fold, args.seed, args, device_info)
            elif args.command == "knn-select":
                select_knn(data, args.track, args, device_info)
            else:
                evaluate_knn(data, args.track, args.fold, args, device_info)
        task_logger.finish("completed", device_info)
        logger.finish(metrics={"status": "passed", "command": args.command}, outputs=[args.out])
    except Exception as exc:
        task_logger.finish("failed", device_info, exc)
        logger.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
