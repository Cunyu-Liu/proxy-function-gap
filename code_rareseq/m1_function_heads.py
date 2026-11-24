#!/usr/bin/env python3
"""M1: shared, independent, multi-head and function-conditioned residual probes."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_frozen_baselines import TaskLoader, metrics, prediction_rows, stable_seed  # noqa: E402
from log_run import RunLogger, sha256_file  # noqa: E402


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for M1; CPU fallback is forbidden")
    if not torch.cuda.device_count():
        raise RuntimeError("CUDA reports zero devices")
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    probe = torch.randn(256, 64, device="cuda", dtype=torch.float32)
    result = (probe @ probe.T).float().cpu().numpy()
    if result.shape != (256, 256) or not np.isfinite(result).all():
        raise RuntimeError("CUDA numerical validation failed")
    return name, capability


def build_model(arch, dim, tasks, hidden, dropout, rank, shrinkage):
    import torch
    if arch == "independent":
        class Independent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = torch.nn.ModuleList([
                    torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                        torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))
                    for _ in tasks])
            def forward(self, x, task_index):
                return self.heads[task_index](x).squeeze(1)
        return Independent()
    if arch == "shared":
        return torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                   torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))
    if arch in {"multitask", "residual"}:
        class Conditional(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(),
                                                 torch.nn.Dropout(dropout))
                self.output = torch.nn.Linear(hidden, len(tasks) if arch == "multitask" else 1)
                if arch == "residual":
                    self.U = torch.nn.Linear(dim, rank, bias=False)
                    self.coefficients = torch.nn.Parameter(torch.zeros(len(tasks), rank))
                    torch.nn.init.normal_(self.coefficients, std=0.01)
            def forward(self, x, task_index):
                base = self.trunk(x)
                if arch == "multitask":
                    return self.output(base)[:, task_index]
                value = self.output(base).squeeze(1)
                return value + self.U(x) @ self.coefficients[task_index]
        return Conditional()
    raise ValueError(f"Unknown architecture: {arch}")


def forward(model, arch, x, task_index):
    return model(x).squeeze(1) if arch == "shared" else model(x, task_index)


def load_track(loader, args, track, device):
    import torch
    tasks = [t for t in loader.tasks() if t["track"] == track]
    if not tasks:
        raise ValueError(f"No tasks for {track}")
    names = [t["experiment"] for t in tasks]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate experiment identity")
    identity_key = "rbp" if args.arch == "residual" else "experiment"
    identities = list(dict.fromkeys(t[identity_key] for t in tasks))
    identity_index = {name: index for index, name in enumerate(identities)}
    loaded = []
    global_rng = np.random.default_rng(stable_seed(args.seed, {"experiment": "m1", "fold": args.fold, "track": track}))
    permutation = global_rng.permutation(len(identities)) if args.permute_identity else np.arange(len(identities))
    for task_index, task in enumerate(tasks):
        arrays = loader.arrays(task)
        split_tensors = []
        for ids, x, y in arrays:
            if len(ids) != len(y) or len(ids) != x.shape[0]:
                raise ValueError(f"Invalid IDs or row counts: {task['experiment']}/{track}")
            if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
                raise ValueError(f"Non-finite embeddings or invalid labels: {task['experiment']}/{track}")
            split_tensors.append((ids, torch.from_numpy(x).to(device),
                                  torch.from_numpy(y.astype(np.float32)).to(device)))
        loaded.append({"task": task, "arrays": split_tensors,
                       "model_index": int(permutation[identity_index[task[identity_key]]])})
    x_train = torch.cat([v["arrays"][0][1] for v in loaded])
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True).clamp_min(1e-5)
    for value in loaded:
        value["arrays"] = [(ids, (x - mean) / std, y) for ids, x, y in value["arrays"]]
    return loaded, {"mean": mean.cpu(), "std": std.cpu(), "identity_permutation": permutation.tolist()}


def evaluate(model, arch, loaded, batch, split_index, collect=False):
    import torch
    model.eval()
    metric_rows, predictions, losses = [], [], []
    with torch.no_grad():
        for value in loaded:
            ids, x, y = value["arrays"][split_index]
            scores, logits = [], []
            task_index = value["model_index"]
            for start in range(0, len(y), batch):
                logit = forward(model, arch, x[start:start + batch], task_index)
                logits.append(logit.float().cpu().numpy())
                scores.append(torch.sigmoid(logit).float().cpu().numpy())
            score, logit = np.concatenate(scores), np.concatenate(logits)
            losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(logit), torch.from_numpy(y.cpu().numpy())).item()))
            auroc, auprc = metrics(y.cpu().numpy(), score)
            task = value["task"]
            metric_rows.append({**{k: task[k] for k in ("experiment", "rbp", "tier", "fold", "track")},
                                "split": "val" if split_index == 1 else "test", "n_eval": len(y),
                                "auroc": auroc, "auprc": auprc, "bce": losses[-1]})
            if collect:
                frame = prediction_rows(task, "val" if split_index == 1 else "test", ids,
                                        y.cpu().numpy(), score)
                frame["logit"] = logit
                predictions.append(frame)
    return metric_rows, predictions, float(np.mean(losses))


def train_track(loaded, args, track, device):
    import torch
    identity_key = "rbp" if args.arch == "residual" else "experiment"
    tasks = list(dict.fromkeys(v["task"][identity_key] for v in loaded))
    seed = stable_seed(args.seed, {"experiment": "M1", "fold": args.fold, "track": track})
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    model = build_model(args.arch, loaded[0]["arrays"][0][1].shape[1], tasks,
                        args.hidden, args.dropout, args.rank, args.shrinkage).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generators = []
    for value in loaded:
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_seed(seed, value["task"]))
        generators.append(generator)
    best_score, best_epoch, stale = -math.inf, 0, 0
    history = []
    best_state = None
    start_time = time.time()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss, epoch_count = 0.0, 0
        order = torch.randperm(len(loaded), generator=torch.Generator(device=device).manual_seed(seed + epoch), device=device)
        for index in order.tolist():
            value = loaded[index]
            x, y = value["arrays"][0][1:]
            permutation = torch.randperm(len(y), device=device, generator=generators[index])
            for start in range(0, len(y), args.batch):
                batch_index = permutation[start:start + args.batch]
                logit = forward(model, args.arch, x[batch_index], value["model_index"])
                target = y[batch_index] * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
                if args.arch == "residual" and args.shrinkage > 0:
                    loss = loss + args.shrinkage * model.coefficients.square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                epoch_loss += float(loss.detach()) * len(batch_index)
                epoch_count += len(batch_index)
        val_rows, _, val_loss = evaluate(model, args.arch, loaded, args.batch, 1)
        val_frame = pd.DataFrame(val_rows)
        tail = val_frame[val_frame.tier == "tail"]
        if tail.empty:
            raise ValueError("Tail-RBP validation is empty")
        tail_auprc = float(tail.auprc.mean())
        history.append({"epoch": epoch, "track": track, "train_bce_with_penalty": epoch_loss / epoch_count,
                        "val_bce": val_loss, "val_tail_auprc": tail_auprc,
                        "val_tail_auroc": float(tail.auroc.mean()), "stale_epochs": stale})
        print(f"M1 {track} epoch={epoch} train={epoch_loss / epoch_count:.6f} "
              f"val_tail_auprc={tail_auprc:.6f} stale={stale}", flush=True)
        if tail_auprc > best_score + args.min_delta:
            best_score, best_epoch, stale = tail_auprc, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    val_rows, val_predictions, _ = evaluate(model, args.arch, loaded, args.batch, 1, True)
    test_rows, test_predictions, _ = evaluate(model, args.arch, loaded, args.batch, 2, True)
    return model, pd.DataFrame(history), val_rows + test_rows, val_predictions + test_predictions, {
        "best_epoch": best_epoch, "best_val_tail_auprc": best_score, "epochs_run": len(history),
        "seconds": time.time() - start_time, "n_parameters": sum(p.numel() for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["shared", "independent", "multitask", "residual"])
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--positives-audit", required=True)
    ap.add_argument("--split-clstr", required=True)
    ap.add_argument("--exps-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="rnafm")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tracks", default="neg1c,neg2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--shrinkage", type=float, default=1.0)
    ap.add_argument("--permute-identity", action="store_true")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--log", default="/home/cunyuliu/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    if args.rank not in {4, 8, 16} or args.shrinkage not in {0.0, 0.1, 1.0, 10.0}:
        raise ValueError("Rank/shrinkage outside the preregistered grid")
    if args.patience < 1 or args.max_epochs < args.patience or args.min_delta < 0:
        raise ValueError("Invalid convergence settings")
    os.makedirs(args.out, exist_ok=True)
    if os.listdir(args.out):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.out}")
    lg = RunLogger(args.log, stage="M1", kind=f"function-{args.arch}")
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=args.seed, data_version=args.data_version,
             inputs=[args.split_clstr, args.exps_file], code_version=config["code_sha256"])
    try:
        import torch
        gpu_name, capability = require_cuda()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        loader = TaskLoader(argparse.Namespace(**{**vars(args), "folds": str(args.fold)}))
        all_histories, all_metrics, all_predictions, summaries = [], [], [], {}
        for track in args.tracks.split(","):
            loaded, normalization = load_track(loader, args, track, "cuda")
            model, history, metric_rows, predictions, summary = train_track(loaded, args, track, "cuda")
            history["arch"] = args.arch
            all_histories.append(history)
            all_metrics.extend(metric_rows)
            all_predictions.extend(predictions)
            summaries[track] = summary
            identity_key = "rbp" if args.arch == "residual" else "experiment"
            checkpoint = {"arch": args.arch, "identity_key": identity_key,
                          "tasks": list(dict.fromkeys(v["task"][identity_key] for v in loaded)),
                          "state_dict": model.state_dict(), "normalization": normalization,
                          "config": config, "selection": summary}
            torch.save(checkpoint, os.path.join(args.out, f"model_{track}.pt"))
            del loaded, model
            torch.cuda.empty_cache()
        pd.concat(all_histories).to_csv(os.path.join(args.out, "history.tsv"), sep="\t", index=False)
        pd.DataFrame(all_metrics).to_csv(os.path.join(args.out, "metrics.tsv"), sep="\t", index=False)
        predictions = pd.concat(all_predictions, ignore_index=True)
        predictions["seed"] = args.seed
        for split, frame in predictions.groupby("split"):
            frame.to_csv(os.path.join(args.out, f"{split}_predictions.tsv.gz"), sep="\t",
                         index=False, compression="gzip")
        result = {"gpu": gpu_name, "cuda_capability": list(capability), "tracks": summaries,
                  "n_predictions": len(predictions), "identity_permuted": args.permute_identity}
        with open(os.path.join(args.out, "config.json"), "w") as handle:
            json.dump({**config, "result": result}, handle, indent=2)
        lg.finish(metrics=result, outputs=[args.out])
        print("M1_DONE", json.dumps(result), flush=True)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
