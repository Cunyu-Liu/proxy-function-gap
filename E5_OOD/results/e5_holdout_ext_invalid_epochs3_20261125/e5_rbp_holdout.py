#!/usr/bin/env python3
"""E5 整 RBP 留出基准：共享 frozen 头 + MSP/energy/deep-kNN 拒识。"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from e1_frozen_baselines import TaskLoader, stable_seed
from log_run import RunLogger, sha256_file


def binary_metrics(y, score):
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def ood_metrics(y, score):
    return float(roc_auc_score(y, score))


def fpr95(y, score):
    threshold = np.quantile(score[y == 0], 0.95)
    return float((score[y == 1] >= threshold).mean())


def aurc(y, probability, unknown):
    error = (probability >= 0.5).astype(int) != y
    order = np.argsort(-unknown)
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


def train_shared(x, y, weights, val_x, val_y, val_tiers, args, seed, device):
    import torch
    torch.manual_seed(seed)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True).clip(min=1e-5)
    x = (x - mean) / std
    val_x = (val_x - mean) / std
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    wt = torch.from_numpy(weights.astype(np.float32)).to(device)
    vt = torch.from_numpy(val_x).to(device)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], args.hidden), torch.nn.GELU(),
                                torch.nn.Dropout(args.dropout), torch.nn.Linear(args.hidden, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device=device); generator.manual_seed(seed)
    history = []
    best_state = None
    best_auprc = -np.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(xt), device=device, generator=generator)
        total = 0.0
        for i in range(0, len(order), args.batch):
            ii = order[i:i + args.batch]
            logits = model(xt[ii]).squeeze(1)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yt[ii], reduction="none") * wt[ii]).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()) * len(ii)
        model.eval()
        with torch.no_grad():
            score = torch.cat([model(vt[i:i + args.batch]).squeeze(1) for i in range(0, len(vt), args.batch)]).cpu().numpy()
        tail = val_tiers == "tail"
        auroc, auprc = binary_metrics(val_y[tail], score[tail])
        history.append({"epoch": epoch, "val_tail_auroc": auroc, "val_tail_auprc": auprc,
                        "train_loss": total / len(order)})
        if auprc > best_auprc:
            best_auprc = auprc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print("EPOCH", epoch, history[-1], flush=True)
    best = max(history, key=lambda r: r["val_tail_auprc"])
    model.load_state_dict(best_state)
    return model, mean, std, best


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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--knn-cap", type=int, default=50000)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E5", kind="rbp-holdout")
    config = vars(args).copy(); config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=args.seed, data_version=args.data_version,
             inputs=[args.exps_file], code_version=config["code_sha256"])
    try:
        import torch
        device = "cuda"
        loader = TaskLoader(args)
        base_tasks = list(loader.tasks())
        rows = []
        for heldout in args.holdouts.split(","):
            for track in args.tracks.split(","):
                selected = [t for t in base_tasks if t["track"] == track and t["fold"] == int(args.folds)]
                train_x, train_y, train_w, val_x, val_y, val_tiers = [], [], [], [], [], []
                test_x, test_y, test_tiers = [], [], []
                target_x, target_y = None, None
                for task in selected:
                    arrays = loader.arrays(task)
                    if task["rbp"] == heldout:
                        target_x, target_y = arrays[2][1], arrays[2][2]
                        continue
                    for split_index, (xlist, ylist) in enumerate(((train_x, train_y), (val_x, val_y), (test_x, test_y))):
                        xlist.append(arrays[split_index][1]); ylist.append(arrays[split_index][2])
                    train_w.append(np.full(len(arrays[0][2]), 1 / len(arrays[0][2]), dtype=np.float32))
                    val_tiers.extend([task["tier"]] * len(arrays[1][2]))
                    test_tiers.extend([task["tier"]] * len(arrays[2][2]))
                x = np.concatenate(train_x); y = np.concatenate(train_y); w = np.concatenate(train_w)
                vx = np.concatenate(val_x); vy = np.concatenate(val_y); vt = np.array(val_tiers)
                tx = np.concatenate(test_x); ty = np.concatenate(test_y)
                seed = stable_seed(args.seed, {"experiment": heldout, "fold": int(args.folds), "track": track})
                model, mean, std, best = train_shared(x, y, w, vx, vy, vt, args, seed, device)
                with torch.no_grad():
                    def logits(values):
                        values = (values - mean) / std
                        tensor = torch.from_numpy(values).to(device)
                        return torch.cat([model(tensor[i:i + args.batch]).squeeze(1) for i in range(0, len(tensor), args.batch)]).cpu().numpy()
                    known_logit = logits(tx); unknown_logit = logits(target_x)
                all_logit = np.concatenate([known_logit, unknown_logit])
                probability = 1 / (1 + np.exp(-all_logit))
                y_ood = np.concatenate([np.zeros(len(known_logit)), np.ones(len(unknown_logit))])
                y_class = np.concatenate([ty, target_y])
                rng = np.random.default_rng(seed)
                cap = min(args.knn_cap, len(x))
                knn_x = x[rng.choice(len(x), cap, replace=False)]
                a = torch.nn.functional.normalize(torch.from_numpy(knn_x).to(device), dim=1)
                b = torch.nn.functional.normalize(torch.from_numpy(np.concatenate([tx, target_x])).to(device), dim=1)
                similarities = []
                for i in range(0, len(b), 1024):
                    similarities.append((b[i:i + 1024] @ a.T).max(dim=1).values.cpu().numpy())
                similarity = np.concatenate(similarities)
                scores = {"msp": -np.maximum(probability, 1 - probability),
                          "energy": np.abs(all_logit), "deep_knn": -similarity}
                for method, score in scores.items():
                    rows.append({"holdout": heldout, "track": track, "method": method,
                                 "auroc_ood": ood_metrics(y_ood, score), "fpr95": fpr95(y_ood, score),
                                 "aurc": aurc(y_class, probability, score),
                                 "ece": ece(y_class, probability), "brier": float(np.mean((probability - y_class) ** 2)),
                                 "best_epoch": best["epoch"], "n_train": len(y), "n_known": len(ty), "n_unknown": len(target_y)})
                pd.DataFrame(rows).to_csv(os.path.join(args.out, "holdout_results.tsv"), sep="\t", index=False)
                print("HOLDOUT_DONE", heldout, track, rows[-3:], flush=True)
        results = pd.DataFrame(rows)
        results.to_csv(os.path.join(args.out, "holdout_results.tsv"), sep="\t", index=False)
        results.groupby(["track", "method"])[["auroc_ood", "fpr95", "aurc", "ece", "brier"]].mean().round(6).to_csv(
            os.path.join(args.out, "holdout_summary.tsv"), sep="\t")
        lg.finish(metrics={"n_rows": len(results), "n_holdouts": len(args.holdouts.split(","))}, outputs=[args.out])
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
