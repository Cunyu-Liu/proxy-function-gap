#!/usr/bin/env python3
"""E5 PAR-CLIP→eCLIP 跨协议描述性评测。

对每个与 eCLIP 主集重叠的 RBP：用 eCLIP 主集 train folds（id90 冻结划分）训练冻结
RNA-FM 现代头，validation fold 选 epoch（不接触 PAR-CLIP）；再以该固定头给 PAR-CLIP
hg38 v2 正例窗口打分，负例参照取同 RBP eCLIP 转录本匹配 neg2/neg1c 窗口。报告逐 RBP
AUROC/AUPRC 与宏平均。这是跨协议描述性信号，负例来自 eCLIP 协议，明确标注，不作 SOTA
或机制结论，不用 PAR-CLIP 调参。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_frozen_baselines import TaskLoader, metrics, stable_seed  # noqa: E402
from log_run import RunLogger  # noqa: E402


def rbp_of_parclip(name):
    return name.split("_")[0]


class MultiExpEmb:
    """按 window_id 把多个 eCLIP 实验的嵌入按原始顺序取回。"""

    def __init__(self, e0_dir, exps, model):
        self.ld = TaskLoader(argparse.Namespace(
            neg_dir=os.path.join(e0_dir, "neg"),
            positives_audit=os.path.join(e0_dir, "positives_audit.tsv"),
            emb_dir=os.path.join(e0_dir, "emb"), model=model,
            windows_dir=os.path.join(e0_dir, "windows"),
            exps_file=os.path.join(e0_dir, "main_exps.txt"), folds="0", tracks="neg1c"))
        self.owner = {}
        for e in exps:
            for fn, kind in ((f"{e}.windows.tsv", "windows"), (f"{e}.negatives.tsv", "neg")):
                d = os.path.join(e0_dir, kind, fn)
                ids = pd.read_csv(d, sep="\t", usecols=["window_id"]).window_id
                for w in ids:
                    self.owner[w] = e
        self.exps = exps

    def fetch(self, ids):
        groups = {e: [] for e in self.exps}
        pos = {e: [] for e in self.exps}
        for i, w in enumerate(ids):
            e = self.owner[w]
            groups[e].append(w)
            pos[e].append(i)
        dim = None
        out = None
        for e in self.exps:
            if not groups[e]:
                continue
            mat = self.ld.embeddings(e, groups[e])
            if out is None:
                dim = mat.shape[1]
                out = np.empty((len(ids), dim), dtype=np.float32)
            out[pos[e]] = mat
        return out


def train_head(xtr, ytr, xva, yva, args, seed):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    mean = xtr.mean(axis=0, keepdims=True)
    std = xtr.std(axis=0, keepdims=True).clip(min=1e-5)
    xtr = (xtr - mean) / std
    xva = (xva - mean) / std
    device = "cuda"
    xtr_t = torch.from_numpy(xtr).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    xva_t = torch.from_numpy(xva).to(device)
    model = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], args.hidden), torch.nn.GELU(),
        torch.nn.Dropout(args.dropout), torch.nn.Linear(args.hidden, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    best = (-1.0, None, 0)
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(xtr_t), device=device, generator=gen)
        for i in range(0, len(order), args.batch):
            ii = order[i:i + args.batch]
            logits = model(xtr_t[ii]).squeeze(1)
            target = ytr_t[ii] * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vs = torch.cat([model(xva_t[i:i + args.batch]).squeeze(1)
                            for i in range(0, len(xva_t), args.batch)]).float().cpu().numpy()
        _, vauprc = metrics(yva, vs)
        hist.append({"epoch": epoch, "val_auprc": vauprc})
        if vauprc == vauprc and vauprc > best[0]:
            best = (vauprc, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
    if best[1] is not None:
        model.load_state_dict(best[1])
    model.eval()
    return model, mean, std, best[2], hist


def score(model, mean, std, x, batch):
    import torch
    x = (x - mean) / std
    xt = torch.from_numpy(x).to("cuda")
    with torch.no_grad():
        s = torch.cat([model(xt[i:i + batch]).squeeze(1)
                       for i in range(0, len(xt), batch)]).float().cpu().numpy()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0-dir", default="/root/rareseq/data/e0")
    ap.add_argument("--parclip-emb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="rnafm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--train-folds", default="2,3,4")
    ap.add_argument("--val-fold", type=int, default=1)
    ap.add_argument("--neg-fold", type=int, default=0)
    ap.add_argument("--tracks", default="neg1c,neg2")
    ap.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E5 PAR-CLIP eval; CPU fallback is forbidden")
    os.makedirs(args.out, exist_ok=True)

    emb_files = [f for f in os.listdir(args.parclip_emb_dir) if f.endswith(f".pos.{args.model}.npy")]
    parclip_rbps = {rbp_of_parclip(f.split(".")[0]): f.split(".")[0] for f in emb_files}
    main_exps = [l.strip() for l in open(os.path.join(args.e0_dir, "main_exps.txt")) if l.strip()]
    main_rbps = {e.rsplit("_", 1)[0] for e in main_exps}
    overlap = sorted(set(parclip_rbps) & main_rbps)

    lg = RunLogger(args.log, stage="E5", kind="parclip-cross-protocol")
    lg.start(config=vars(args), seed=args.seed, data_version="parclip-hg38-v2")

    fold_map = pd.read_csv(os.path.join(args.e0_dir, "neg", "window_fold_map.tsv"),
                           sep="\t", usecols=["window_id", "fold"])
    fold_by_wid = dict(zip(fold_map.window_id, fold_map.fold))
    train_folds = {int(x) for x in args.train_folds.split(",")}

    rows = []
    for rbp in overlap:
        exps = [e for e in main_exps if e.rsplit("_", 1)[0] == rbp]
        pc_name = parclip_rbps[rbp]
        pc_x = np.asarray(np.load(os.path.join(
            args.parclip_emb_dir, f"{pc_name}.pos.{args.model}.npy"), mmap_mode="r"), dtype=np.float32)
        me = MultiExpEmb(args.e0_dir, exps, args.model)
        for track in args.tracks.split(","):
            tr_ids, tr_y, va_ids, va_y, neg_ids = [], [], [], [], []
            for e in exps:
                pos = pd.read_csv(os.path.join(args.e0_dir, "windows", f"{e}.windows.tsv"), sep="\t")
                neg = pd.read_csv(os.path.join(args.e0_dir, "neg", f"{e}.negatives.tsv"), sep="\t")
                pf = pos.window_id.map(fold_by_wid)
                ptr = pos[pf.isin(train_folds)]; pva = pos[pf == args.val_fold]
                ntr = neg[(neg.track == track) & (neg.fold.isin(train_folds))]
                nva = neg[(neg.track == track) & (neg.fold == args.val_fold)]
                nte = neg[(neg.track == track) & (neg.fold == args.neg_fold)]
                tr_ids += ptr.window_id.tolist() + ntr.window_id.tolist()
                tr_y += [1] * len(ptr) + [0] * len(ntr)
                va_ids += pva.window_id.tolist() + nva.window_id.tolist()
                va_y += [1] * len(pva) + [0] * len(nva)
                neg_ids += nte.window_id.tolist()
            if not tr_ids or not va_ids or not neg_ids:
                continue
            xtr = me.fetch(tr_ids); xva = me.fetch(va_ids); xneg = me.fetch(neg_ids)
            ytr = np.array(tr_y, dtype=np.int64); yva = np.array(va_y, dtype=np.int64)
            seed = stable_seed(args.seed, {"experiment": rbp, "fold": 0, "track": track})
            model, mean, std, best_ep, hist = train_head(xtr, ytr, xva, yva, args, seed)
            pos_score = score(model, mean, std, pc_x, args.batch)
            neg_score = score(model, mean, std, xneg, args.batch)
            y = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))])
            s = np.concatenate([pos_score, neg_score])
            auroc, auprc = metrics(y, s)
            rows.append({"rbp": rbp, "track": track, "n_parclip_pos": len(pos_score),
                         "n_eclip_neg": len(neg_score), "best_epoch": best_ep,
                         "auroc": auroc, "auprc": auprc,
                         "eval_protocol": "PARCLIP-pos_vs_eCLIP-neg_descriptive"})
            pd.DataFrame(hist).to_csv(os.path.join(args.out, f"{rbp}_{track}_valhist.tsv"),
                                      sep="\t", index=False)
            print(f"{rbp} {track}: AUROC={auroc:.4f} AUPRC={auprc:.4f} ep{best_ep}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "cross_protocol_results.tsv"), sep="\t", index=False)
    macro = res.groupby("track")[["auroc", "auprc"]].mean()
    macro.to_csv(os.path.join(args.out, "cross_protocol_macro.tsv"), sep="\t")
    lg.finish(metrics={"n_rbp": len(overlap), "macro": macro.to_dict()}, outputs=[args.out])
    print("MACRO\n", macro, flush=True)


if __name__ == "__main__":
    main()