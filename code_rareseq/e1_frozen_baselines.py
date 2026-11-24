#!/usr/bin/env python3
"""RARE-Seq E1 — 严格 train/val/test 的冻结嵌入现代头与簇去重 OverfitNN。

val 用于按尾部 RBP 平均 AUPRC 选择 epoch 或邻居数；test 仅在选择后汇总。
现代头采用两遍训练，第二遍重训至选定 epoch，避免按测试集早停。
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_run import RunLogger, sha256_file  # noqa: E402


def metrics(y, score):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def read_exps(path):
    return [line.strip() for line in open(path) if line.strip()]


def rbp_of(experiment):
    return experiment.rsplit("_", 1)[0]


def load_tiers(path):
    audit = pd.read_csv(path, sep="\t")
    per_rbp = audit.groupby("rbp")["n_sites"].sum()
    q1, q2 = per_rbp.quantile([1 / 3, 2 / 3])
    return {r: "tail" if v <= q1 else ("head" if v > q2 else "mid") for r, v in per_rbp.items()}


def parse_clstr(path):
    mapping = {}
    cid = None
    with open(path) as handle:
        for line in handle:
            if line.startswith(">Cluster"):
                cid = int(line.split()[1])
            elif line.strip():
                mapping[line.split(">")[1].split("...")[0]] = cid
    return mapping


class TaskLoader:
    def __init__(self, args):
        self.args = args
        self.fold_map = pd.read_csv(
            os.path.join(args.neg_dir, "window_fold_map.tsv"), sep="\t",
            usecols=["window_id", "fold"])
        self.fold_by_wid = dict(zip(self.fold_map.window_id, self.fold_map.fold))
        self.fold_map = None
        self.tiers = load_tiers(args.positives_audit)

    def tasks(self):
        for name in read_exps(self.args.exps_file):
            pos = pd.read_csv(os.path.join(self.args.windows_dir, f"{name}.windows.tsv"), sep="\t")
            neg = pd.read_csv(os.path.join(self.args.neg_dir, f"{name}.negatives.tsv"), sep="\t")
            pos_fold = pos.window_id.map(self.fold_by_wid)
            if pos_fold.isna().any():
                raise ValueError(f"{name}: positive windows missing frozen fold assignment")
            for fold in [int(x) for x in self.args.folds.split(",")]:
                val_fold = (fold + 1) % 5
                for track in self.args.tracks.split(","):
                    ntr = neg[(neg.track == track) & (~neg.fold.isin([fold, val_fold]))]
                    nva = neg[(neg.track == track) & (neg.fold == val_fold)]
                    nte = neg[(neg.track == track) & (neg.fold == fold)]
                    ptr = pos[~pos_fold.isin([fold, val_fold])]
                    pva = pos[pos_fold == val_fold]
                    pte = pos[pos_fold == fold]
                    splits = []
                    for p, n in ((ptr, ntr), (pva, nva), (pte, nte)):
                        splits.append((p.window_id.tolist() + n.window_id.tolist(),
                                       np.array([1] * len(p) + [0] * len(n), dtype=np.int64)))
                    if any(len(s[0]) == 0 for s in splits):
                        raise ValueError(f"{name}/fold{fold}/{track}: empty train/val/test split")
                    yield {"experiment": name, "rbp": rbp_of(name), "tier": self.tiers[rbp_of(name)],
                           "fold": fold, "track": track, "splits": splits}

    def embeddings(self, name, ids):
        idx = {}
        arrays = {}
        for kind in ("pos", "neg"):
            stem = os.path.join(self.args.emb_dir, f"{name}.{kind}.{self.args.model}")
            arrays[kind] = np.load(stem + ".npy", mmap_mode="r")
            with open(stem + ".ids") as handle:
                for i, line in enumerate(handle):
                    idx[line.strip()] = (kind, i)
        missing = [wid for wid in ids if wid not in idx]
        if missing:
            raise KeyError(f"{name}: {len(missing)} windows missing {self.args.model} embeddings")
        dim = arrays["pos"].shape[1]
        out = np.empty((len(ids), dim), dtype=np.float32)
        for kind in ("pos", "neg"):
            positions = [i for i, wid in enumerate(ids) if idx[wid][0] == kind]
            if positions:
                source = [idx[ids[i]][1] for i in positions]
                out[positions] = np.asarray(arrays[kind][source], dtype=np.float32)
        return out

    def arrays(self, task):
        result = []
        for ids, y in task["splits"]:
            result.append((ids, self.embeddings(task["experiment"], ids), y))
        return result


def stable_seed(base, task):
    key = f"{base}|{task['experiment']}|{task['fold']}|{task['track']}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def predict_torch(model, x, batch):
    import torch
    outputs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outputs.append(model(x[i:i + batch]).squeeze(1).float().cpu().numpy())
    return np.concatenate(outputs)


def train_modern(task, arrays, args, epochs, pass_id, device):
    import torch
    import torch.nn.functional as F
    seed = stable_seed(args.seed, task)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    (train_ids, xtr, ytr), (_, xva, yva), (test_ids, xte, yte) = arrays
    mean = xtr.mean(axis=0, keepdims=True)
    std = xtr.std(axis=0, keepdims=True).clip(min=1e-5)
    xtr = (xtr - mean) / std
    xva = (xva - mean) / std
    xte = (xte - mean) / std
    xtr_t = torch.from_numpy(xtr).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    xva_t = torch.from_numpy(xva).to(device)
    xte_t = torch.from_numpy(xte).to(device)
    hidden = args.hidden
    model = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(args.dropout),
        torch.nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    val_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(xtr_t), device=device, generator=generator)
        total_loss = 0.0
        for i in range(0, len(order), args.batch):
            ii = order[i:i + args.batch]
            logits = model(xtr_t[ii]).squeeze(1)
            target = ytr_t[ii]
            if args.loss == "bce":
                target = target * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
                loss = F.binary_cross_entropy_with_logits(logits, target)
            elif args.loss == "nnpu":
                positive = target > 0.5
                loss_positive_as1 = F.binary_cross_entropy_with_logits(
                    logits[positive], torch.ones_like(logits[positive]))
                loss_positive_as0 = F.binary_cross_entropy_with_logits(
                    logits[positive], torch.zeros_like(logits[positive]))
                loss_unlabeled_as0 = F.binary_cross_entropy_with_logits(
                    logits[~positive], torch.zeros_like(logits[~positive]))
                pi = args.pi
                risk = pi * loss_positive_as1 + (loss_unlabeled_as0 - pi * loss_positive_as0) / (1 - pi)
                loss = torch.clamp(risk, min=0.0)
            else:
                ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                prob = torch.sigmoid(logits)
                p_t = prob * target + (1 - prob) * (1 - target)
                loss = (((1 - p_t) ** args.focal_gamma) * ce).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach()) * len(ii)
        model.eval()
        val_score = predict_torch(model, xva_t, args.batch)
        val_auroc, val_auprc = metrics(yva, val_score)
        val_rows.append({"epoch": epoch, "val_auroc": val_auroc, "val_auprc": val_auprc,
                         "train_loss": total_loss / len(order)})
    result = {"val_history": val_rows}
    if pass_id == 2:
        model.eval()
        result["val_score"] = predict_torch(model, xva_t, args.batch)
        result["test_score"] = predict_torch(model, xte_t, args.batch)
        result["val_ids"] = arrays[1][0]
        result["test_ids"] = test_ids
        result["val_y"] = yva
        result["test_y"] = yte
    del model, opt, xtr_t, ytr_t, xva_t, xte_t
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def prediction_rows(task, split, ids, y, score):
    row = {"experiment": task["experiment"], "rbp": task["rbp"], "tier": task["tier"],
           "fold": task["fold"], "track": task["track"], "split": split,
           "window_id": ids, "label": y, "score": score}
    for key in ("source_experiment", "target_experiment", "eval_protocol"):
        if key in task:
            row[key] = task[key]
    return pd.DataFrame(row)


def run_modern(loader, args, lg):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = list(loader.tasks())
    histories = []
    train_examples = 0
    for i, task in enumerate(tasks, 1):
        arrays = loader.arrays(task)
        train_examples += len(arrays[0][2])
        result = train_modern(task, arrays, args, args.epochs, 1, device)
        for row in result["val_history"]:
            histories.append({**{k: task[k] for k in ("experiment", "rbp", "tier", "fold", "track")}, **row})
        print(f"PASS1 {i}/{len(tasks)} {task['experiment']} f{task['fold']} {task['track']}", flush=True)
    history = pd.DataFrame(histories)
    history.to_csv(os.path.join(args.out, "validation_history.tsv"), sep="\t", index=False)
    tail = history[history.tier == "tail"]
    selected = tail.groupby(["track", "epoch"]).val_auprc.mean().groupby("track").idxmax()
    best_epoch = {track: int(epoch) for track, epoch in selected}
    if not best_epoch:
        raise ValueError("No tail-RBP validation tasks; refusing overall-AUPRC early stopping")
    print("SELECTED_EPOCH", best_epoch, flush=True)
    rows, predictions, val_predictions = [], [], []
    for i, task in enumerate(tasks, 1):
        arrays = loader.arrays(task)
        result = train_modern(task, arrays, args, best_epoch[task["track"]], 2, device)
        for split in ("val", "test"):
            y = result[f"{split}_y"]
            score = result[f"{split}_score"]
            auroc, auprc = metrics(y, score)
            rows.append({**{k: task[k] for k in ("experiment", "rbp", "tier", "fold", "track")},
                         "split": split, "best_epoch": best_epoch[task["track"]],
                         "n_train": len(arrays[0][2]), "n_eval": len(y), "auroc": auroc, "auprc": auprc})
            if split == "test":
                predictions.append(prediction_rows(task, split, result[f"{split}_ids"], y, score))
            else:
                val_predictions.append(prediction_rows(task, split, result[f"{split}_ids"], y, score))
        print(f"PASS2 {i}/{len(tasks)} {task['experiment']} f{task['fold']} {task['track']}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(args.out, "modern_results.tsv"), sep="\t", index=False)
    summary = results.groupby(["track", "split", "tier"])[["auroc", "auprc"]].mean().round(6)
    summary.to_csv(os.path.join(args.out, "modern_summary.tsv"), sep="\t")
    pd.concat(predictions, ignore_index=True).to_csv(
        os.path.join(args.out, "modern_test_predictions.tsv.gz"), sep="\t", index=False, compression="gzip")
    pd.concat(val_predictions, ignore_index=True).to_csv(
        os.path.join(args.out, "modern_val_predictions.tsv.gz"), sep="\t", index=False, compression="gzip")
    config = vars(args).copy()
    config.update({"selected_epoch_by_track": best_epoch,
                   "selection": "mean validation AUPRC over tail RBPs; test excluded",
                   "logit_adjustment": "analytic no-op: balanced binary pi=0.5 gives tau*log(pi/(1-pi))=0"})
    with open(os.path.join(args.out, "modern_config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    test = results[results.split == "test"]
    return {"n_tasks": len(tasks), "best_epoch": best_epoch,
            "train_examples_per_pass": train_examples,
            "test_macro": {track: {"auroc": float(g.auroc.mean()), "auprc": float(g.auprc.mean())}
                           for track, g in test.groupby("track")}}


def knn_scores(xtr, ytr, xev, ks, device, batch=2048):
    import torch
    a = torch.nn.functional.normalize(torch.from_numpy(xtr).to(device), dim=1)
    b = torch.nn.functional.normalize(torch.from_numpy(xev).to(device), dim=1)
    max_k = min(max(ks), len(a))
    out = {k: [] for k in ks}
    with torch.no_grad():
        for i in range(0, len(b), batch):
            sim = b[i:i + batch] @ a.T
            top = sim.topk(max_k, dim=1).indices
            labels = torch.from_numpy(ytr).to(device)[top].float()
            for k in ks:
                out[k].append(labels[:, :min(k, max_k)].mean(1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def run_knn(loader, args, lg):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cluster = parse_clstr(args.split_clstr)
    ks = [int(x) for x in args.ks.split(",")]
    rows, cached = [], {}
    for i, task in enumerate(loader.tasks(), 1):
        arrays = loader.arrays(task)
        (train_ids, xtr, ytr), (_, xva, yva), (_, xte, yte) = arrays
        seen = set()
        keep = []
        for j, (wid, y) in enumerate(zip(train_ids, ytr)):
            if y == 1:
                cid = cluster.get(wid, f"missing:{wid}")
                if cid in seen:
                    continue
                seen.add(cid)
            keep.append(j)
        keep = np.array(keep, dtype=np.int64)
        xtr, ytr = xtr[keep], ytr[keep]
        val_scores = knn_scores(xtr, ytr, xva, ks, device)
        test_scores = knn_scores(xtr, ytr, xte, ks, device)
        cached[(task["experiment"], task["fold"], task["track"])] = (arrays, val_scores, test_scores)
        for k in ks:
            va = metrics(yva, val_scores[k])
            rows.append({**{key: task[key] for key in ("experiment", "rbp", "tier", "fold", "track")},
                         "k": k, "split": "val", "n_train": len(ytr), "n_eval": len(yva),
                         "auroc": va[0], "auprc": va[1]})
        print(f"KNN {i} {task['experiment']} f{task['fold']} {task['track']} train={len(ytr)}", flush=True)
    val_results = pd.DataFrame(rows)
    tail = val_results[val_results.tier == "tail"]
    selected = tail.groupby(["track", "k"]).auprc.mean().groupby("track").idxmax()
    best_k = {track: int(k) for track, k in selected}
    print("SELECTED_K", best_k, flush=True)
    final_rows, predictions, val_predictions = [], [], []
    for task in loader.tasks():
        key = (task["experiment"], task["fold"], task["track"])
        arrays, val_scores, test_scores = cached[key]
        k = best_k[task["track"]]
        for split, scores in (("val", val_scores), ("test", test_scores)):
            ids, _, y = arrays[1 if split == "val" else 2]
            score = scores[k]
            auroc, auprc = metrics(y, score)
            final_rows.append({**{key2: task[key2] for key2 in ("experiment", "rbp", "tier", "fold", "track")},
                               "k": k, "split": split, "n_eval": len(y), "auroc": auroc, "auprc": auprc})
            if split == "test":
                predictions.append(prediction_rows(task, split, ids, y, score))
            else:
                val_predictions.append(prediction_rows(task, split, ids, y, score))
    val_results.to_csv(os.path.join(args.out, "knn_validation.tsv"), sep="\t", index=False)
    results = pd.DataFrame(final_rows)
    results.to_csv(os.path.join(args.out, "knn_results.tsv"), sep="\t", index=False)
    results.groupby(["track", "split", "tier"])[["auroc", "auprc"]].mean().round(6).to_csv(
        os.path.join(args.out, "knn_summary.tsv"), sep="\t")
    pd.concat(predictions, ignore_index=True).to_csv(
        os.path.join(args.out, "knn_test_predictions.tsv.gz"), sep="\t", index=False, compression="gzip")
    pd.concat(val_predictions, ignore_index=True).to_csv(
        os.path.join(args.out, "knn_val_predictions.tsv.gz"), sep="\t", index=False, compression="gzip")
    with open(os.path.join(args.out, "knn_config.json"), "w") as handle:
        json.dump({"selected_k_by_track": best_k, "selection": "tail-RBP mean validation AUPRC",
                   "dedup": "one training positive per frozen id90 cluster; negatives unchanged",
                   "ks": ks}, handle, indent=2)
    test = results[results.split == "test"]
    return {"n_tasks": len(results) // 2, "best_k": best_k,
            "test_macro": {track: {"auroc": float(g.auroc.mean()), "auprc": float(g.auprc.mean())}
                           for track, g in test.groupby("track")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["modern", "knn"])
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--positives-audit", required=True)
    ap.add_argument("--split-clstr", required=True)
    ap.add_argument("--exps-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="rnafm")
    ap.add_argument("--folds", default="0")
    ap.add_argument("--tracks", default="neg1c,neg2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["bce", "focal", "nnpu"], default="bce")
    ap.add_argument("--pi", type=float, default=0.1, help="nnPU positive-class prior")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    ap.add_argument("--stage", default="E1", choices=["E1", "E2", "E4"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage=args.stage, kind=f"frozen-{args.command}")
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=args.seed, data_version=args.data_version,
             inputs=[args.split_clstr, args.exps_file], code_version=config["code_sha256"])
    try:
        loader = TaskLoader(args)
        result = run_modern(loader, args, lg) if args.command == "modern" else run_knn(loader, args, lg)
        lg.finish(metrics=result, outputs=[args.out])
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise
    print("E1_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
