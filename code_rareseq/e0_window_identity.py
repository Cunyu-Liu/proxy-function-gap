#!/usr/bin/env python3
"""E0 — 主集每个正例窗口的 d_hom：到同折训练正例的最近邻 identity（全量，非采样）。
产出 per_window_identity.tsv（window_id/experiment/fold/nn_wid/nn_identity/bucket），
供 §4.7 低同源切片分桶冻结。NN 检索用冻结 RNA-FM 嵌入余弦 top1，identity 用精确全局比对。
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_run import RunLogger

BUCKETS = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0001)]

def bucket(x):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= x < hi:
            return i
    return len(BUCKETS) - 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--model", default="rnafm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exps-file", required=True)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import torch
    from Bio import Align
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1; aligner.mismatch_score = -1
    aligner.open_gap_score = -2; aligner.extend_gap_score = -1

    lg = RunLogger(args.log, stage="E0", kind="window-identity")
    lg.start(config={"model": args.model, "nn_target": "train-positives-same-fold",
                     "buckets": [list(b) for b in BUCKETS]}, seed=None, data_version="e0-v1")
    fmap = pd.read_csv(os.path.join(args.neg_dir, "window_fold_map.tsv"), sep="\t")
    fold_by_wid = dict(zip(fmap.window_id, fmap.fold))
    exps = [l.strip() for l in open(args.exps_file)]
    rows = []
    t0 = time.time()
    for name in exps:
        pt = os.path.join(args.windows_dir, f"{name}.windows.tsv")
        ep = os.path.join(args.emb_dir, f"{name}.pos.{args.model}.npy")
        ip = os.path.join(args.emb_dir, f"{name}.pos.{args.model}.ids")
        if not (os.path.exists(pt) and os.path.exists(ep)):
            continue
        pos = pd.read_csv(pt, sep="\t")
        E = np.load(ep, mmap_mode="r")
        ids = [l.strip() for l in open(ip)]
        row_of = {w: i for i, w in enumerate(ids)}
        seq_of = dict(zip(pos.window_id, pos.seq))
        folds = pos.window_id.map(fold_by_wid)
        # 全实验嵌入一次性上 GPU（≤40k×640×4B≈100MB）
        keep = [w for w in pos.window_id if w in row_of]
        X = np.asarray(E[[row_of[w] for w in keep]], dtype=np.float32)
        Xt = torch.nn.functional.normalize(torch.from_numpy(X).to(device), dim=1)
        fold_of = {w: fold_by_wid.get(w, -1) for w in keep}
        for f in range(5):
            te_idx = [i for i, w in enumerate(keep) if fold_of[w] == f]
            tr_idx = [i for i, w in enumerate(keep) if fold_of[w] != f and fold_of[w] >= 0]
            if not te_idx or not tr_idx:
                continue
            a = Xt[tr_idx]
            nn = []
            for i in range(0, len(te_idx), 4096):
                b = Xt[te_idx[i:i + 4096]]
                nn.append((b @ a.T).argmax(1).cpu().numpy())
            nn = np.concatenate(nn)
            for j, i in enumerate(te_idx):
                w = keep[i]; tw = keep[tr_idx[nn[j]]]
                s1, s2 = seq_of[w], seq_of[tw]
                aln = aligner.align(s1, s2)[0]
                a1, a2 = aln[0], aln[1]
                m = sum(1 for x, y in zip(a1, a2) if x == y and x != "-")
                ident = m / max(len(s1), len(s2))
                rows.append((w, name, f, tw, round(ident, 4), bucket(ident)))
        if len(rows) % 100000 < 40000:
            print(f"{name}: total {len(rows)} rows @ {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["window_id", "experiment", "fold", "nn_wid", "nn_identity", "bucket"])
    df.to_csv(os.path.join(args.out, "per_window_identity.tsv"), sep="\t", index=False)
    dist = df.bucket.value_counts().sort_index()
    summary = {"n_windows": len(df),
               "bucket_counts": {f"{BUCKETS[k][0]:g}-{BUCKETS[k][1]:g}": int(v) for k, v in dist.items()},
               "mean_identity": round(float(df.nn_identity.mean()), 4)}
    import json
    json.dump(summary, open(os.path.join(args.out, "identity_summary.json"), "w"), indent=2)
    lg.finish(metrics=summary, outputs=[args.out])
    print(json.dumps(summary), flush=True)
    print("WINDOW_IDENTITY_DONE", flush=True)

if __name__ == "__main__":
    main()