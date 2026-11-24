#!/usr/bin/env python3
"""E1 RNA-FM LoRA/全参微调；与冻结基线共享冻结划分和尾部验证早停协议。"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_g0_models import _fast_tokenizer_setup  # noqa: E402
from e1_frozen_baselines import TaskLoader, metrics, prediction_rows, stable_seed  # noqa: E402
from log_run import RunLogger, sha256_file  # noqa: E402


class SequenceLoader(TaskLoader):
    def __init__(self, args):
        super().__init__(args)
        self.cache = {}

    def sequences(self, name):
        if name not in self.cache:
            pos = pd.read_csv(os.path.join(self.args.windows_dir, f"{name}.windows.tsv"), sep="\t",
                              usecols=["window_id", "seq"])
            neg = pd.read_csv(os.path.join(self.args.neg_dir, f"{name}.negatives.tsv"), sep="\t",
                              usecols=["window_id", "seq"])
            self.cache[name] = dict(zip(pd.concat([pos, neg]).window_id, pd.concat([pos, neg]).seq))
        return self.cache[name]

    def arrays(self, task):
        sequences = self.sequences(task["experiment"])
        return [(ids, [sequences[wid] for wid in ids], y) for ids, y in task["splits"]]


def encode(tokenizer, fast_encode, seqs, device):
    if fast_encode is not None:
        encoded = fast_encode(seqs)
    else:
        encoded = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=103)
    return {key: value.to(device) for key, value in encoded.items()}


def pooled_logits(model, head, encoded):
    import torch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
        return head(pooled).squeeze(1)


def evaluate(model, head, tokenizer, fast_encode, seqs, batch, device):
    import torch
    model.eval(); head.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            encoded = encode(tokenizer, fast_encode, seqs[i:i + batch], device)
            scores.append(pooled_logits(model, head, encoded).float().cpu().numpy())
    return np.concatenate(scores)


def train_task(task, arrays, args, epochs, pass_id, tokenizer, fast_encode, repo, device):
    import torch
    import torch.nn.functional as F
    from multimolecule import AutoModel
    seed = stable_seed(args.seed, task)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device)
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                            lora_dropout=args.lora_dropout, target_modules=["key", "value"], bias="none")
        model = get_peft_model(model, config)
        encoder_params = [p for p in model.parameters() if p.requires_grad]
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        encoder_params = list(model.parameters())
    model.train()
    head = torch.nn.Sequential(torch.nn.Linear(model.config.hidden_size, args.hidden), torch.nn.GELU(),
                               torch.nn.Dropout(args.dropout), torch.nn.Linear(args.hidden, 1)).to(device)
    opt = torch.optim.AdamW([{"params": encoder_params, "lr": args.encoder_lr},
                             {"params": head.parameters(), "lr": args.head_lr}], weight_decay=args.weight_decay)
    (train_ids, train_seqs, ytr), (_, val_seqs, yva), (test_ids, test_seqs, yte) = arrays
    y = torch.from_numpy(ytr.astype(np.float32)).to(device)
    generator = torch.Generator(device=device); generator.manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); head.train()
        order = torch.randperm(len(y), device=device, generator=generator).cpu().numpy()
        total_loss = 0.0
        for i in range(0, len(order), args.batch):
            ii = order[i:i + args.batch]
            encoded = encode(tokenizer, fast_encode, [train_seqs[j] for j in ii], device)
            logits = pooled_logits(model, head, encoded)
            target = y[torch.from_numpy(ii).to(device)]
            target = target * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder_params) + list(head.parameters()), 1.0)
            opt.step(); total_loss += float(loss.detach()) * len(ii)
        val_score = evaluate(model, head, tokenizer, fast_encode, val_seqs, args.batch, device)
        auroc, auprc = metrics(yva, val_score)
        history.append({"epoch": epoch, "val_auroc": auroc, "val_auprc": auprc,
                        "train_loss": total_loss / len(order)})
        print(f"  epoch {epoch} val_auprc={auprc:.5f}", flush=True)
    result = {"val_history": history}
    if pass_id == 2:
        result["val_score"] = evaluate(model, head, tokenizer, fast_encode, val_seqs, args.batch, device)
        result["test_score"] = evaluate(model, head, tokenizer, fast_encode, test_seqs, args.batch, device)
        result.update({"val_y": yva, "test_y": yte})
    del model, head, opt
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--positives-audit", required=True)
    ap.add_argument("--split-clstr", required=True)
    ap.add_argument("--exps-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="multimolecule/rnafm")
    ap.add_argument("--mode", choices=["lora", "full"], required=True)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--tracks", default="neg1c,neg2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--encoder-lr", type=float, default=3e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--data-version", default="e0-v1-id90-frozen")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E1", kind=f"finetune-{args.mode}")
    config = vars(args).copy(); config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    lg.start(config=config, seed=args.seed, data_version=args.data_version,
             inputs=[args.split_clstr, args.exps_file], code_version=config["code_sha256"])
    try:
        import torch
        from multimolecule import AutoTokenizer
        device = "cuda"
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        fast_encode, fast_ok = _fast_tokenizer_setup(tokenizer) if args.model == "multimolecule/rnafm" else (None, False)
        loader = SequenceLoader(args)
        tasks = list(loader.tasks())
        histories = []
        for i, task in enumerate(tasks, 1):
            result = train_task(task, loader.arrays(task), args, args.epochs, 1, tokenizer, fast_encode,
                                args.model, device)
            for row in result["val_history"]:
                histories.append({**{key: task[key] for key in ("experiment", "rbp", "tier", "fold", "track")}, **row})
            print(f"PASS1 {i}/{len(tasks)} {task['experiment']} {task['track']}", flush=True)
        history = pd.DataFrame(histories)
        history.to_csv(os.path.join(args.out, "validation_history.tsv"), sep="\t", index=False)
        selected = history[history.tier == "tail"].groupby(["track", "epoch"]).val_auprc.mean().groupby("track").idxmax()
        best_epoch = {track: int(epoch) for track, epoch in selected}
        print("SELECTED_EPOCH", best_epoch, flush=True)
        rows, predictions = [], []
        for i, task in enumerate(tasks, 1):
            arrays = loader.arrays(task)
            result = train_task(task, arrays, args, best_epoch[task["track"]], 2, tokenizer, fast_encode,
                                args.model, device)
            for split, seqs, y in (("val", arrays[1][1], arrays[1][2]), ("test", arrays[2][1], arrays[2][2])):
                score = result[f"{split}_score"]
                auroc, auprc = metrics(y, score)
                rows.append({**{key: task[key] for key in ("experiment", "rbp", "tier", "fold", "track")},
                             "split": split, "best_epoch": best_epoch[task["track"]], "n_train": len(arrays[0][2]),
                             "n_eval": len(y), "auroc": auroc, "auprc": auprc})
                if split == "test":
                    predictions.append(prediction_rows(task, split, arrays[2][0], y, score))
            print(f"PASS2 {i}/{len(tasks)} {task['experiment']} {task['track']}", flush=True)
        results = pd.DataFrame(rows)
        results.to_csv(os.path.join(args.out, "finetune_results.tsv"), sep="\t", index=False)
        results.groupby(["track", "split", "tier"])[["auroc", "auprc"]].mean().round(6).to_csv(
            os.path.join(args.out, "finetune_summary.tsv"), sep="\t")
        pd.concat(predictions, ignore_index=True).to_csv(os.path.join(args.out, "finetune_test_predictions.tsv.gz"),
                                                         sep="\t", index=False, compression="gzip")
        with open(os.path.join(args.out, "finetune_config.json"), "w") as handle:
            json.dump({**vars(args), "selected_epoch_by_track": best_epoch,
                       "fast_tokenizer_verified": fast_ok}, handle, indent=2)
        test = results[results.split == "test"]
        summary = {"n_tasks": len(tasks), "best_epoch": best_epoch,
                   "test_macro": {track: {"auroc": float(g.auroc.mean()), "auprc": float(g.auprc.mean())}
                                  for track, g in test.groupby("track")}}
        lg.finish(metrics=summary, outputs=[args.out])
        print("E1_FINETUNE_DONE", json.dumps(summary), flush=True)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
