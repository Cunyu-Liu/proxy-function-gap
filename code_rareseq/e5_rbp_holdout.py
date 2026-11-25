#!/usr/bin/env python3
"""E5 整 RBP 留出基准：validation 早停的共享 frozen 头与拒识基线。"""
import argparse
import gc
import hashlib
import json
import os
import resource

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from e1_frozen_baselines import TaskLoader, stable_seed
from log_run import RunLogger, sha256_file


METRICS = ["auroc_ood", "fpr95", "aurc", "ece", "brier"]
METRIC_VERSION = 2
TRAIN_FOLD_POLICY = "assigned-folds-only-v1"
COMPATIBLE_TRAIN_CODE_HASHES = {
    "bbe4b60734d26c98ab040ad7491f1a75a7126f492e62632fb25f5256569cd909",
    "cb180b4299633c485c4bb5a2d3801b520c2d9a88114b7b2b492011dbc326f9c6",
    "192b77cadda6d566e45254f99489aa55c6a59fd9d6f70c7fa4b9cfe3a0fedee8",
}


def atomic_json(value, path):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def atomic_csv(frame, path):
    temporary = path + ".tmp"
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def hash_ids(ids):
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E5 RBP holdout; CPU fallback is forbidden")
    if not torch.cuda.device_count():
        raise RuntimeError("CUDA reports zero devices")
    torch.cuda.init()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    probe = torch.arange(256, dtype=torch.float32, device=device).reshape(16, 16)
    actual = (probe @ probe.T).cpu().numpy()
    expected = np.arange(256, dtype=np.float32).reshape(16, 16)
    expected = expected @ expected.T
    if actual.shape != (16, 16) or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-4, atol=1e-2):
        raise RuntimeError("CUDA numerical validation failed")
    info = {"device": "cuda:0", "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch_version": str(torch.__version__), "numerical_probe": "passed"}
    print("CUDA_PROBE", json.dumps(info), flush=True)
    return device, info


def log_host_memory(phase, metadata=None):
    values = {"phase": phase, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, value = line.split(":", 1)
                    values[key] = value.strip()
    except FileNotFoundError:
        pass
    if metadata:
        values.update(metadata)
    print("HOST_MEMORY", json.dumps(values), flush=True)


def binary_metrics(y, score):
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def fpr95(y_ood, score):
    threshold = np.quantile(score[y_ood == 1], 0.05)
    return float((score[y_ood == 0] >= threshold).mean())


def ood_tpr_at_known_fpr5(y_ood, score):
    threshold = np.quantile(score[y_ood == 0], 0.95)
    return float((score[y_ood == 1] >= threshold).mean())


def aurc(y, probability, unknown):
    error = (probability >= 0.5).astype(int) != y
    order = np.argsort(unknown, kind="mergesort")
    coverage = np.arange(1, len(y) + 1) / len(y)
    risk = np.cumsum(error[order]) / np.arange(1, len(y) + 1)
    return float(np.trapezoid(risk, coverage))


def ece(y, probability, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probability >= lo) & (probability < hi if hi < 1 else probability <= hi)
        if mask.any():
            value += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(value)


def assigned_train_mask(loader, ids, labels):
    folds = np.asarray([loader.fold_by_wid.get(window_id, -1) for window_id in ids])
    if ((labels == 1) & (folds < 0) & ~np.isin(folds, [-1])).any():
        raise ValueError("Training positives have invalid negative fold assignments")
    return ~((labels == 1) & ~np.isin(folds, [0, 1, 2, 3, 4]))


def load_pair(loader, base_tasks, heldout, track, fold):
    selected = [t for t in base_tasks if t["track"] == track and t["fold"] == fold]
    if not selected:
        raise ValueError(f"No tasks for {track}/fold{fold}")
    splits = {name: [[], []] for name in ("train", "validation", "test")}
    weights, val_tiers, target_x, target_y, target_ids, known_test_ids = [], [], [], [], [], []
    roster = []
    for task in selected:
        arrays = loader.arrays(task)
        for ids, values, labels in arrays:
            if len(ids) != len(labels) or len(ids) != values.shape[0]:
                raise ValueError(f"Invalid row alignment: {task['experiment']}/{track}")
            if not np.isfinite(values).all() or not np.isin(labels, [0, 1]).all():
                raise ValueError(f"Non-finite embeddings or invalid labels: {task['experiment']}/{track}")
        task_record = {"experiment": task["experiment"], "rbp": task["rbp"], "tier": task["tier"],
                       "fold": fold, "track": track, "role": "heldout_test" if task["rbp"] == heldout else "known"}
        for split_index, split in enumerate(("train", "validation", "test")):
            ids, values, labels = arrays[split_index]
            removed = 0
            if split == "train" and task["rbp"] != heldout:
                keep = assigned_train_mask(loader, np.asarray(ids), labels)
                removed = int((~keep).sum())
                ids = list(np.asarray(ids)[keep])
                values, labels = values[keep], labels[keep]
            task_record[split + "_n"] = len(labels)
            task_record[split + "_ids_sha256"] = hash_ids(ids)
            task_record[split + "_positives"] = int(labels.sum())
            if split == "train":
                task_record["train_unassigned_positives_removed"] = removed
            if task["rbp"] == heldout:
                if split == "test":
                    target_ids.append(ids)
                    target_x.append(values)
                    target_y.append(labels)
                continue
            splits[split][0].append(values)
            splits[split][1].append(labels)
            if split == "train":
                weights.append(np.full(len(labels), 1 / len(labels), dtype=np.float32))
            if split == "validation":
                val_tiers.extend([task["tier"]] * len(labels))
            if split == "test":
                known_test_ids.append(ids)
        roster.append(task_record)
    if not target_x:
        raise ValueError(f"Heldout RBP has no test task: {heldout}/{track}")
    data = {"x": np.concatenate(splits["train"][0]), "y": np.concatenate(splits["train"][1]),
            "w": np.concatenate(weights), "vx": np.concatenate(splits["validation"][0]),
            "vy": np.concatenate(splits["validation"][1]), "vt": np.asarray(val_tiers),
            "tx": np.concatenate(splits["test"][0]), "ty": np.concatenate(splits["test"][1]),
            "target_x": np.concatenate(target_x), "target_y": np.concatenate(target_y),
            "known_ids": np.concatenate(known_test_ids), "target_ids": np.concatenate(target_ids)}
    manifest = {"heldout": heldout, "track": track, "fold": fold, "tasks": roster,
                "train_fold_policy": TRAIN_FOLD_POLICY,
                "train_unassigned_positives_removed": sum(
                    task.get("train_unassigned_positives_removed", 0) for task in roster),
                "heldout_test_ids_sha256": hash_ids(np.concatenate(target_ids)),
                "train_alignment_sha256": hashlib.sha256(json.dumps(
                    [{k: v for k, v in row.items() if k.startswith("train_") or k in ("experiment", "track")}
                     for row in roster if row["role"] == "known"], sort_keys=True).encode()).hexdigest(),
                "validation_alignment_sha256": hashlib.sha256(json.dumps(
                    [{k: v for k, v in row.items() if k.startswith("validation_") or k in ("experiment", "track")}
                     for row in roster if row["role"] == "known"], sort_keys=True).encode()).hexdigest(),
                "test_used_for_selection": False}
    if (data["vt"] == "tail").sum() == 0:
        raise ValueError(f"Empty tail validation: {heldout}/{track}")
    return data, manifest


def validation_scores(model, values, batch, device):
    import torch
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch):
            batch_x = torch.from_numpy(values[start:start + batch]).to(device)
            outputs.append(model(batch_x).squeeze(1).float().cpu().numpy())
    return np.concatenate(outputs)


def save_checkpoint(path, model_state, mean, std, args, code_hash, gpu, manifest, selection):
    import torch
    value = {"format_version": 1, "state_dict": model_state,
             "normalization_mean": torch.from_numpy(mean), "normalization_std": torch.from_numpy(std),
             "input_dim": int(mean.shape[1]), "hidden": args.hidden, "dropout": args.dropout,
             "config": vars(args).copy(), "code_sha256": code_hash, "cuda": gpu,
             "data_manifest": manifest, "selection": selection, "test_used_for_selection": False}
    temporary = path + ".tmp"
    torch.save(value, temporary)
    os.replace(temporary, path)


def train_shared(data, args, seed, device, gpu, code_hash, manifest, history_path, checkpoint_path, metadata):
    import torch
    torch.manual_seed(seed)
    x, val_x = data["x"], data["vx"]
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True).clip(min=1e-5)
    x -= mean
    x /= std
    val_x -= mean
    val_x /= std
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(data["y"].astype(np.float32)).to(device)
    wt = torch.from_numpy(data["w"].astype(np.float32)).to(device)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], args.hidden), torch.nn.GELU(),
                                torch.nn.Dropout(args.dropout), torch.nn.Linear(args.hidden, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    history = []
    best_state, best_auprc, best_epoch, stale = None, -np.inf, 0, 0
    tail = data["vt"] == "tail"
    stop_reason = "max_epochs_failsafe"
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        order = torch.randperm(len(xt), device=device, generator=generator)
        total = 0.0
        for start in range(0, len(order), args.batch):
            index = order[start:start + args.batch]
            logits = model(xt[index]).squeeze(1)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yt[index], reduction="none") * wt[index]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(index)
        score = validation_scores(model, val_x, args.batch, device)
        if not np.isfinite(score).all():
            raise FloatingPointError(f"Non-finite validation logits at epoch {epoch}")
        auroc, auprc = binary_metrics(data["vy"][tail], score[tail])
        improved = auprc > best_auprc + args.min_delta
        if improved:
            best_auprc, best_epoch, stale = auprc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        row = {**metadata, "epoch": epoch, "val_tail_auroc": auroc, "val_tail_auprc": auprc,
               "train_loss": total / len(order), "stale_epochs": stale, "is_best": improved}
        history.append(row)
        atomic_csv(pd.DataFrame(history), history_path)
        print("EPOCH", json.dumps(row), flush=True)
        if improved:
            save_checkpoint(checkpoint_path, best_state, mean, std, args, code_hash, gpu, manifest,
                            {"best_epoch": best_epoch, "best_val_tail_auprc": best_auprc})
        if stale >= args.patience:
            stop_reason = "validation_early_stop"
            break
    if best_state is None:
        raise RuntimeError("No valid checkpoint")
    selection = {"best_epoch": best_epoch, "best_val_tail_auprc": best_auprc,
                 "epochs_run": len(history), "stop_reason": stop_reason,
                 "selection_metric": "validation tail pooled AUPRC", "patience": args.patience,
                 "min_delta": args.min_delta, "max_epochs_failsafe": args.max_epochs}
    save_checkpoint(checkpoint_path, best_state, mean, std, args, code_hash, gpu, manifest, selection)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    check_score = validation_scores(model, val_x, args.batch, device)
    _, check_auprc = binary_metrics(data["vy"][tail], check_score[tail])
    if not np.isclose(check_auprc, best_auprc, rtol=0, atol=1e-8):
        raise RuntimeError(f"Checkpoint validation mismatch: {check_auprc} != {best_auprc}")
    print("CHECKPOINT_VALIDATED", json.dumps({**metadata, "val_tail_auprc": check_auprc}), flush=True)
    return model, mean, std, pd.DataFrame(history), selection


def predict_logits(model, values, mean, std, args, device):
    import torch
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), args.batch):
            normalized = (values[start:start + args.batch] - mean) / std
            tensor = torch.from_numpy(normalized).to(device)
            outputs.append(model(tensor).squeeze(1).float().cpu().numpy())
    return np.concatenate(outputs)


def restore_checkpoint_for_eval(checkpoint_path, data, args, device):
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = torch.nn.Sequential(torch.nn.Linear(checkpoint["input_dim"], checkpoint["hidden"]),
                                torch.nn.GELU(), torch.nn.Dropout(checkpoint["dropout"]),
                                torch.nn.Linear(checkpoint["hidden"], 1))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    mean = checkpoint["normalization_mean"].numpy()
    std = checkpoint["normalization_std"].numpy()
    data["vx"] -= mean
    data["vx"] /= std
    score = validation_scores(model, data["vx"], args.batch, device)
    tail = data["vt"] == "tail"
    _, auprc = binary_metrics(data["vy"][tail], score[tail])
    expected = checkpoint["selection"]["best_val_tail_auprc"]
    if not np.isclose(auprc, expected, rtol=0, atol=1e-8):
        raise RuntimeError(f"Resume checkpoint validation mismatch: {auprc} != {expected}")
    return model, mean, std, checkpoint["selection"], checkpoint


def rows_from_window_scores(score_path, metadata, selection, counts):
    with np.load(score_path, allow_pickle=False) as saved:
        required = {"window_id", "occurrence", "class_label", "y_ood", "logit", "probability",
                    "msp", "energy", "deep_knn", "max_cosine", "metric_version"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"Missing score arrays: {sorted(missing)}")
        if int(saved["metric_version"]) != METRIC_VERSION:
            raise ValueError("OOD metric version mismatch")
        y_ood = saved["y_ood"].astype(np.int8)
        y_class = saved["class_label"].astype(np.int8)
        probability = saved["probability"].astype(np.float64)
        scores = {method: saved[method].astype(np.float64) for method in ("msp", "energy", "deep_knn")}
        arrays = [probability, saved["logit"], saved["max_cosine"], *scores.values()]
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("Non-finite per-window scores")
        if not np.isin(y_ood, [0, 1]).all() or not np.isin(y_class, [0, 1]).all():
            raise ValueError("Invalid per-window labels")
        n = len(y_ood)
        if any(len(value) != n for value in (saved["window_id"], saved["occurrence"], saved["logit"], *scores.values())):
            raise ValueError("Per-window score row misalignment")
        if not np.array_equal(scores["energy"], -np.abs(saved["logit"].astype(np.float64))):
            raise ValueError("Energy direction/version mismatch")
        rows = []
        for method, score in scores.items():
            rows.append({**metadata, "method": method, "metric_version": METRIC_VERSION,
                         "auroc_ood": float(roc_auc_score(y_ood, score)), "fpr95": fpr95(y_ood, score),
                         "ood_tpr_at_known_fpr5": ood_tpr_at_known_fpr5(y_ood, score),
                         "aurc": aurc(y_class, probability, score), "ece": ece(y_class, probability),
                         "brier": float(np.mean((probability - y_class) ** 2)),
                         **selection, **counts})
    return rows


def evaluate_and_save(model, mean, std, data, knn_x, args, device, metadata, selection,
                      score_path, pair_path, manifest_path, manifest):
    import torch
    known_logit = predict_logits(model, data["tx"], mean, std, args, device)
    unknown_logit = predict_logits(model, data["target_x"], mean, std, args, device)
    all_logit = np.concatenate([known_logit, unknown_logit])
    probability = 1 / (1 + np.exp(-all_logit))
    y_ood = np.concatenate([np.zeros(len(known_logit), dtype=np.int8),
                            np.ones(len(unknown_logit), dtype=np.int8)])
    y_class = np.concatenate([data["ty"], data["target_y"]]).astype(np.int8)
    window_id = np.concatenate([data["known_ids"], data["target_ids"]]).astype(str)
    occurrence = pd.Series(window_id).groupby(window_id).cumcount().to_numpy(dtype=np.int32)
    a = torch.nn.functional.normalize(torch.from_numpy(knn_x).to(device), dim=1)
    similarities = []
    with torch.no_grad():
        for values in (data["tx"], data["target_x"]):
            for start in range(0, len(values), 1024):
                b = torch.nn.functional.normalize(
                    torch.from_numpy(values[start:start + 1024]).to(device), dim=1)
                similarities.append((b @ a.T).max(dim=1).values.cpu().numpy())
    similarity = np.concatenate(similarities)
    scores = {"msp": -np.maximum(probability, 1 - probability),
              "energy": -np.abs(all_logit), "deep_knn": -similarity}
    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    temporary = score_path + ".tmp"
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, window_id=window_id, occurrence=occurrence,
                            sample_role=np.where(y_ood == 0, "known", "heldout"),
                            class_label=y_class, y_ood=y_ood, logit=all_logit.astype(np.float32),
                            probability=probability, max_cosine=similarity.astype(np.float32),
                            msp=scores["msp"], energy=scores["energy"], deep_knn=scores["deep_knn"],
                            metric_version=np.int64(METRIC_VERSION),
                            energy_definition=np.array("-abs(logit); larger=more OOD"),
                            aurc_definition=np.array("ascending unknown; stable mergesort; occurrence order for ties"))
    os.replace(temporary, score_path)
    counts = {"n_train": len(data["y"]), "n_validation": len(data["vy"]),
              "n_known": len(data["ty"]), "n_unknown": len(data["target_y"])}
    rows = rows_from_window_scores(score_path, metadata, selection, counts)
    atomic_csv(pd.DataFrame(rows), pair_path)
    manifest.update({"selection": selection, "metric_version": METRIC_VERSION,
                     "energy_definition": "-abs(logit)", "aurc_sort": "ascending_mergesort",
                     "fpr95_definition": "known FPR at 95% OOD TPR",
                     "score_rows": len(y_ood), "test_used_for_selection": False,
                     "artifacts": {"pair_sha256": sha256_file(pair_path),
                                   "history_sha256": sha256_file(manifest["history_path"]),
                                   "checkpoint_sha256": sha256_file(manifest["checkpoint_path"]),
                                   "scores_sha256": sha256_file(score_path)}})
    manifest.pop("history_path", None)
    manifest.pop("checkpoint_path", None)
    atomic_json(manifest, manifest_path)
    return rows


def aggregate_pairs(out, expected):
    frames = []
    for heldout, track in expected:
        path = os.path.join(out, "pairs", f"{heldout}_{track}.tsv")
        manifest_path = os.path.join(out, "manifests", f"{heldout}_{track}.json")
        if not os.path.exists(path) or not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        if manifest.get("metric_version") == METRIC_VERSION and manifest.get("train_fold_policy") == TRAIN_FOLD_POLICY:
            frames.append(pd.read_csv(path, sep="\t"))
    if not frames:
        return
    results = pd.concat(frames, ignore_index=True)
    atomic_csv(results, os.path.join(out, "holdout_results.tsv"))
    summary = results.groupby(["track", "method"])[METRICS].mean().round(8).reset_index()
    atomic_csv(summary, os.path.join(out, "holdout_summary.tsv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--positives-audit", required=True)
    ap.add_argument("--split-clstr", required=True)
    ap.add_argument("--exps-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdouts", required=True)
    ap.add_argument("--model", default="rnafm")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--tracks", default="neg1c,neg2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--knn-cap", type=int, default=50000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    if args.patience < 1 or args.max_epochs < args.patience or args.min_delta < 0:
        raise ValueError("Invalid convergence settings")
    holdouts = args.holdouts.split(",")
    tracks = args.tracks.split(",")
    if len(set(holdouts)) != len(holdouts) or len(set(tracks)) != len(tracks):
        raise ValueError("Duplicate holdouts or tracks")
    code_hash = sha256_file(os.path.abspath(__file__))
    if os.path.exists(args.out) and os.listdir(args.out) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.out}")
    if args.resume:
        previous_path = os.path.join(args.out, "config.json")
        if not os.path.exists(previous_path):
            raise FileNotFoundError("Resume requires config.json")
        with open(previous_path) as handle:
            previous = json.load(handle)
        if previous["code_sha256"] != code_hash and previous["code_sha256"] not in COMPATIBLE_TRAIN_CODE_HASHES:
            raise ValueError("Resume code hash mismatch")
        ignored = {"log", "resume"}
        if {k: v for k, v in previous["args"].items() if k not in ignored} != {k: v for k, v in vars(args).items() if k not in ignored}:
            raise ValueError("Resume configuration mismatch")
    for subdirectory in ("pairs", "history", "checkpoints", "manifests", "scores"):
        os.makedirs(os.path.join(args.out, subdirectory), exist_ok=True)
    run_kind = f"rbp-holdout-convergence-s{args.seed}-{args.tracks.replace(',', '-')}"
    lg = RunLogger(args.log, stage="E5", kind=run_kind)
    config = vars(args).copy()
    config.update({"code_sha256": code_hash, "selection_metric": "validation tail pooled AUPRC",
                   "test_used_for_selection": False, "metric_version": METRIC_VERSION,
                   "energy_definition": "-abs(logit); larger=more OOD",
                   "aurc_definition": "ascending unknown; stable mergesort; occurrence order for ties",
                   "fpr95_definition": "known FPR at 95% OOD TPR; known=0, OOD=1, larger score=more OOD",
                   "train_fold_policy": TRAIN_FOLD_POLICY,
                   "evidence_level": "pilot fold0 test evaluation; descriptive until complete multiseed audit"})
    run_id = lg.start(config=config, seed=args.seed, data_version=args.data_version,
                      inputs=[args.split_clstr, args.exps_file, args.positives_audit], code_version=code_hash)
    try:
        import torch
        device, gpu = require_cuda()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        input_hashes = {name: sha256_file(getattr(args, name)) for name in
                        ("split_clstr", "exps_file", "positives_audit")}
        run_config = {"run_id": run_id, "args": vars(args).copy(), "code_sha256": code_hash,
                      "input_sha256": input_hashes, "cuda": gpu, "test_used_for_selection": False,
                      "train_fold_policy": TRAIN_FOLD_POLICY}
        atomic_json(run_config, os.path.join(args.out, "config.json"))
        if args.resume:
            atomic_json(run_config, os.path.join(args.out, f"resume_{run_id}.json"))
        loader = TaskLoader(args)
        base_tasks = list(loader.tasks())
        expected = [(heldout, track) for heldout in holdouts for track in tracks]
        for heldout, track in expected:
            prefix = f"{heldout}_{track}"
            pair_path = os.path.join(args.out, "pairs", prefix + ".tsv")
            history_path = os.path.join(args.out, "history", prefix + ".tsv")
            checkpoint_path = os.path.join(args.out, "checkpoints", prefix + ".pt")
            manifest_path = os.path.join(args.out, "manifests", prefix + ".json")
            score_path = os.path.join(args.out, "scores", prefix + ".npz")
            artifact_paths = {"pair_sha256": pair_path, "history_sha256": history_path,
                              "checkpoint_sha256": checkpoint_path, "scores_sha256": score_path}
            present = [os.path.exists(path) for path in
                       (pair_path, history_path, checkpoint_path, manifest_path, score_path)]
            if all(present):
                with open(manifest_path) as handle:
                    previous_manifest = json.load(handle)
                hashes_valid = all(previous_manifest.get("artifacts", {}).get(key) == sha256_file(path)
                                   for key, path in artifact_paths.items())
                if (previous_manifest.get("metric_version") == METRIC_VERSION and
                        previous_manifest.get("train_fold_policy") == TRAIN_FOLD_POLICY and hashes_valid):
                    print("RESUME_SKIP_HASH_VERIFIED", heldout, track, flush=True)
                    continue
            if any(present):
                quarantine = os.path.join(args.out, "quarantine", f"{prefix}_{run_id}")
                os.makedirs(quarantine, exist_ok=False)
                for path in (pair_path, history_path, checkpoint_path, manifest_path, score_path):
                    if os.path.exists(path):
                        os.replace(path, os.path.join(quarantine, os.path.basename(path)))
                print("QUARANTINED_INCOMPLETE_OR_INVALID", heldout, track, quarantine, flush=True)
            log_host_memory("before_load", {"holdout": heldout, "track": track})
            data, current_manifest = load_pair(loader, base_tasks, heldout, track, int(args.folds))
            log_host_memory("after_load", {"holdout": heldout, "track": track})
            seed = stable_seed(args.seed, {"experiment": heldout, "fold": int(args.folds), "track": track})
            metadata = {"run_id": run_id, "holdout": heldout, "track": track,
                        "seed": args.seed, "stable_seed": seed}
            rng = np.random.default_rng(seed)
            cap = min(args.knn_cap, len(data["x"]))
            knn_x = data["x"][rng.choice(len(data["x"]), cap, replace=False)].copy()
            repair = all(os.path.exists(path) for path in (history_path, checkpoint_path, manifest_path))
            if repair:
                with open(manifest_path) as handle:
                    manifest = json.load(handle)
                if manifest.get("train_fold_policy") != TRAIN_FOLD_POLICY:
                    print("RETRAIN_FOLD_POLICY", heldout, track, flush=True)
                    repair = False
            if repair:
                for key in ("train_alignment_sha256", "validation_alignment_sha256", "heldout_test_ids_sha256"):
                    if manifest[key] != current_manifest[key]:
                        raise ValueError(f"Resume data alignment changed: {heldout}/{track}/{key}")
                manifest["training_run_id"] = manifest.get("training_run_id", manifest.get("run_id"))
                model, mean, std, selection, checkpoint = restore_checkpoint_for_eval(checkpoint_path, data, args, device)
                print("METRIC_REPAIR_FROM_CHECKPOINT", heldout, track, manifest["training_run_id"], flush=True)
            else:
                manifest = current_manifest
                model, mean, std, history, selection = train_shared(
                    data, args, seed, device, gpu, code_hash, manifest, history_path, checkpoint_path, metadata)
            manifest.update({"run_id": run_id, "seed": args.seed, "stable_seed": seed,
                             "n_train": len(data["y"]), "n_validation": len(data["vy"]),
                             "n_known_test": len(data["ty"]), "n_heldout_test": len(data["target_y"]),
                             "history_path": history_path, "checkpoint_path": checkpoint_path})
            log_host_memory("before_evaluation", {"holdout": heldout, "track": track})
            rows = evaluate_and_save(model, mean, std, data, knn_x, args, device, metadata, selection,
                                     score_path, pair_path, manifest_path, manifest)
            log_host_memory("after_evaluation", {"holdout": heldout, "track": track})
            aggregate_pairs(args.out, expected)
            print("HOLDOUT_DONE", heldout, track, json.dumps(selection),
                  json.dumps([{k: row[k] for k in ("method", "auroc_ood", "fpr95", "aurc")} for row in rows]), flush=True)
            del data, model, knn_x
            gc.collect()
            torch.cuda.empty_cache()
        aggregate_pairs(args.out, expected)
        result = {"n_holdouts": len(holdouts), "n_pairs": len(expected), "cuda": gpu,
                  "test_used_for_selection": False}
        lg.finish(metrics=result, outputs=[args.out])
        print("E5_RUN_DONE", run_id, json.dumps(result), flush=True)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
