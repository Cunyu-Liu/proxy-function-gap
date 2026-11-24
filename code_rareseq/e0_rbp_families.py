#!/usr/bin/env python3
"""E0 — RBP 家族分组冻结（§4.7 远OOD家族整组留出用）。
方法: 每 RBP 正例窗口 4-mer 组成谱（每实验最多采样 5000 窗口，seed=20261122）→
RBP 级平均谱 → 余弦距离 → 平均链接层次聚类 → 在候选阈值网格上按 silhouette 选最优切点 → 冻结。
声明: 组成谱是结合 motif 相似性的代理（E0 可得数据内最接近的客观量）；
motif 级精化（MEME/已知家族注释对照）留 E1，若不一致以本冻结为准并记录。
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_run import RunLogger

def kmer_profile(seqs, k=4):
    from itertools import product
    bases = "ACGU"
    kmers = ["".join(p) for p in product(bases, repeat=k)]
    ki = {s: i for i, s in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)), dtype=np.float64)
    for r, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            j = ki.get(s[i:i + k])
            if j is not None:
                X[r, j] += 1
    X /= np.maximum(X.sum(1, keepdims=True), 1)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exps-file", default=None, help="默认主集 main_exps.txt")
    ap.add_argument("--sample", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score
    lg = RunLogger(args.log, stage="E0", kind="rbp-family-freeze")
    lg.start(config={"k": 4, "sample_per_exp": args.sample, "linkage": "average",
                     "metric": "cosine", "threshold_selection": "max-silhouette"},
             seed=args.seed, data_version="e0-v1")
    exps = ([l.strip() for l in open(args.exps_file)] if args.exps_file else
            sorted({f.split(".")[0] for f in os.listdir(args.windows_dir) if f.endswith(".windows.tsv")}))
    rng = np.random.default_rng(args.seed)
    rbp_profiles = {}
    for name in exps:
        rbp = name.rsplit("_", 1)[0]
        pt = os.path.join(args.windows_dir, f"{name}.windows.tsv")
        if not os.path.exists(pt):
            continue
        df = pd.read_csv(pt, sep="\t", usecols=["seq"])
        seqs = df.seq.tolist()
        if len(seqs) > args.sample:
            seqs = [seqs[i] for i in rng.choice(len(seqs), args.sample, replace=False)]
        P = kmer_profile(seqs).mean(0)
        if rbp in rbp_profiles:
            rbp_profiles[rbp].append(P)
        else:
            rbp_profiles[rbp] = [P]
    rbps = sorted(rbp_profiles)
    M = np.stack([np.mean(rbp_profiles[r], axis=0) for r in rbps])
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    D = 1.0 - Mn @ Mn.T
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    best = None
    grid = np.arange(0.05, 0.65, 0.025)
    for t in grid:
        lab = fcluster(Z, t=t, criterion="distance")
        if len(set(lab)) < 2 or len(set(lab)) > len(rbps) // 2:
            continue
        s = silhouette_score(D, lab, metric="precomputed")
        if best is None or s > best[1]:
            best = (float(t), float(s), lab)
    t, s, lab = best
    fam = pd.DataFrame({"rbp": rbps, "family_id": lab})
    fam.to_csv(os.path.join(args.out, "rbp_families.tsv"), sep="\t", index=False)
    np.save(os.path.join(args.out, "rbp_4mer_profiles.npy"), M)
    summary = {"n_rbps": len(rbps), "threshold": round(t, 4), "silhouette": round(s, 4),
               "n_families": int(len(set(lab))),
               "family_sizes": {int(k): int(v) for k, v in zip(*np.unique(lab, return_counts=True))},
               "note": "4-mer composition proxy for binding-motif similarity; frozen for family-holdout OOD"}
    json.dump(summary, open(os.path.join(args.out, "family_summary.json"), "w"), indent=2)
    lg.finish(metrics=summary, outputs=[args.out])
    print(json.dumps(summary), flush=True)
    print("FAMILY_FREEZE_DONE", flush=True)

if __name__ == "__main__":
    main()