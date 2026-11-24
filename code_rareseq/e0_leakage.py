#!/usr/bin/env python3
"""RARE-Seq E0 — 四通道泄漏量化（§4.5）+ 分布统计（§4.4）。

通道1 同源跨split: 簇独占验证（test 窗口簇含 train 成员比例，应≈0）+
      测试→训练最近邻 identity 分布（嵌入 kNN 取 top1 → 精确全局比对 identity）分桶。
通道2 负例定义: neg1f 落入正例±300nt 邻域比例（neg_audit.tsv）+ 正/负例 U/GC 组成差。
通道3 预训练污染: 语料事实声明（RNA-FM=RNAcentral；RiNALMo=ncRNA 语料）+ 缓解切片说明。
通道4 注释转移: RNA 侧 N/A 声明（eCLIP 为实验观测）；蛋白 E6 适用。
通道5 实现伪影: 确定性提取顺序/固定种子记录；跨硬件波动检查计划。
产出: leakage_table.tsv / nn_identity_buckets.tsv / composition_bias.tsv /
      cluster_stats.tsv / leakage_summary.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_run import RunLogger  # noqa: E402


def parse_clstr(path):
    w2c = {}
    cid = None
    with open(path) as f:
        for line in f:
            if line.startswith(">Cluster"):
                cid = int(line.split()[1])
            else:
                w2c[line.split(">")[1].split("...")[0]] = cid
    return w2c


def composition(seqs):
    n = len(seqs)
    if n == 0:
        return {}
    L = np.mean([len(s) for s in seqs])
    u = np.mean([s.count("U") / max(len(s), 1) for s in seqs])
    gc = np.mean([(s.count("G") + s.count("C")) / max(len(s), 1) for s in seqs])
    return {"n": n, "mean_len": round(float(L), 2), "U_frac": round(float(u), 4), "GC_frac": round(float(gc), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--folds-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--model", default="rnafm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--identity-tag", default="id90")
    ap.add_argument("--exps-file", default=None)
    ap.add_argument("--nn-sample", type=int, default=1500, help="每任务精确比对的测试窗口上限")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tracks", default="neg1c,neg2")
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E0", kind="leakage-quant")
    lg.start(config={"identity_tag": args.identity_tag, "nn_sample": args.nn_sample,
                     "model": args.model, "tracks": args.tracks}, seed=None, data_version="e0-v1")

    exps = ([l.strip() for l in open(args.exps_file)] if args.exps_file else
            sorted({f.split(".")[0] for f in os.listdir(args.windows_dir) if f.endswith(".windows.tsv")}))
    w2c = parse_clstr(os.path.join(args.folds_dir, f"clusters.{args.identity_tag}.clstr"))
    fmap = pd.read_csv(os.path.join(args.neg_dir, "window_fold_map.tsv"), sep="\t")
    fold_by_wid = dict(zip(fmap.window_id, fmap.fold))
    tracks = args.tracks.split(",")

    import torch
    from Bio import Align
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -1

    buckets_def = [(0.0, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0001)]
    bucket_rows, comp_rows, clus_rows, leak_rows = [], [], [], []
    for name in exps:
        pt = os.path.join(args.windows_dir, f"{name}.windows.tsv")
        nt = os.path.join(args.neg_dir, f"{name}.negatives.tsv")
        if not os.path.exists(pt):
            continue
        pos = pd.read_csv(pt, sep="\t")
        neg = pd.read_csv(nt, sep="\t") if os.path.exists(nt) else pd.DataFrame()
        pos["fold"] = pos.window_id.map(fold_by_wid)
        # ---- 组成偏差（通道2）----
        comp_rows.append({"experiment": name, "track": "pos", **composition(pos.seq.tolist())})
        for tr in tracks + ["neg1f"]:
            if len(neg):
                sub = neg[neg.track == tr]
                if len(sub):
                    comp_rows.append({"experiment": name, "track": tr, **composition(sub.seq.tolist())})
        # ---- 簇统计（§4.4 n_eff）----
        pc = [w2c[w] for w in pos.window_id if w in w2c]
        clus_rows.append({"experiment": name, "n_pos_windows": len(pos),
                          "n_clusters": len(set(pc)),
                          "n_eff_ratio": round(len(set(pc)) / max(len(pc), 1), 4)})
        # ---- 嵌入索引 ----
        idx = {}
        for kind in ("pos", "neg"):
            p = os.path.join(args.emb_dir, f"{name}.{kind}.{args.model}.npy")
            ip = os.path.join(args.emb_dir, f"{name}.{kind}.{args.model}.ids")
            if os.path.exists(p) and os.path.exists(ip):
                ids = [l.strip() for l in open(ip)]
                for i, wid in enumerate(ids):
                    idx[wid] = (kind, i)
        emb_cache = {}

        def get_mat(wids):
            keep, rows = [], []
            by_kind = defaultdict(list)
            for w in wids:
                if w in idx:
                    by_kind[idx[w][0]].append(w)
            mats = []
            for kind, ws in by_kind.items():
                if kind not in emb_cache:
                    emb_cache[kind] = np.load(os.path.join(args.emb_dir, f"{name}.{kind}.{args.model}.npy"), mmap_mode="r")
                E = emb_cache[kind]
                mats.append(np.asarray(E[[idx[w][1] for w in ws]], dtype=np.float32))
                keep += ws
            return (np.concatenate(mats) if mats else np.zeros((0, 1), np.float32)), keep

        seq_by_wid = dict(zip(pos.window_id, pos.seq))
        if len(neg):
            seq_by_wid.update(dict(zip(neg.window_id, neg.seq)))
        # ---- 每折×轨: 簇泄漏验证 + NN identity ----
        for f in range(args.folds):
            tr_pos = pos[pos.fold != f]
            te_pos = pos[pos.fold == f]
            for tr in tracks:
                ntr = neg[(neg.track == tr) & (neg.fold != f)] if len(neg) else pd.DataFrame()
                nte = neg[(neg.track == tr) & (neg.fold == f)] if len(neg) else pd.DataFrame()
                train_ids = tr_pos.window_id.tolist() + (ntr.window_id.tolist() if len(ntr) else [])
                test_ids = te_pos.window_id.tolist() + (nte.window_id.tolist() if len(nte) else [])
                train_set = set(train_ids)
                # 簇独占验证: test 正例窗口所在簇是否含 train 成员
                n_leak = 0
                n_check = 0
                cluster_train = defaultdict(set)
                for w in train_ids:
                    c = w2c.get(w)
                    if c is not None:
                        cluster_train[c].add(w)
                for w in te_pos.window_id:
                    c = w2c.get(w)
                    if c is None:
                        continue
                    n_check += 1
                    if cluster_train.get(c):
                        n_leak += 1
                leak_rows.append({"experiment": name, "fold": f, "track": tr,
                                  "n_test_pos_checked": n_check, "n_cluster_leak": n_leak,
                                  "cluster_leak_frac": round(n_leak / max(n_check, 1), 5)})
                # NN identity（嵌入 top1 → 精确比对）
                Xtr, tr_keep = get_mat(train_ids)
                if len(test_ids) > args.nn_sample:
                    rng = np.random.default_rng(42 + f)
                    test_ids = list(rng.choice(test_ids, size=args.nn_sample, replace=False))
                Xte, te_keep = get_mat(test_ids)
                if len(Xtr) == 0 or len(Xte) == 0:
                    continue
                a = torch.nn.functional.normalize(torch.from_numpy(Xtr).to(device), dim=1)
                b = torch.nn.functional.normalize(torch.from_numpy(Xte).to(device), dim=1)
                nn_idx = []
                for i in range(0, len(b), 2048):
                    nn_idx.append((b[i:i + 2048] @ a.T).argmax(1).cpu().numpy())
                nn_idx = np.concatenate(nn_idx)
                ids = []
                for w, j in zip(te_keep, nn_idx):
                    tw = tr_keep[j]
                    s1, s2 = seq_by_wid.get(w), seq_by_wid.get(tw)
                    if not s1 or not s2:
                        continue
                    aln = aligner.align(s1, s2)[0]
                    a1, a2 = aln[0], aln[1]
                    matches = sum(1 for x, y in zip(a1, a2) if x == y and x != "-")
                    ident = matches / max(len(s1), len(s2))
                    ids.append(ident)
                hist = [sum(1 for x in ids if lo <= x < hi) / max(len(ids), 1) for lo, hi in buckets_def]
                bucket_rows.append({"experiment": name, "fold": f, "track": tr, "n_sampled": len(ids),
                                    "nn_identity_mean": round(float(np.mean(ids)), 4) if ids else None,
                                    **{f"frac_{lo:g}_{hi:g}": round(h, 4) for (lo, hi), h in zip(buckets_def, hist)}})
        print(f"leakage done: {name}", flush=True)

    pd.DataFrame(leak_rows).to_csv(os.path.join(args.out, "cluster_leak_check.tsv"), sep="\t", index=False)
    bdf = pd.DataFrame(bucket_rows)
    bdf.to_csv(os.path.join(args.out, "nn_identity_buckets.tsv"), sep="\t", index=False)
    cdf = pd.DataFrame(comp_rows)
    cdf.to_csv(os.path.join(args.out, "composition_bias.tsv"), sep="\t", index=False)
    kdf = pd.DataFrame(clus_rows)
    kdf.to_csv(os.path.join(args.out, "cluster_stats.tsv"), sep="\t", index=False)
    # ---- 汇总表 ----
    na = pd.read_csv(os.path.join(args.neg_dir, "neg_audit.tsv"), sep="\t") if os.path.exists(os.path.join(args.neg_dir, "neg_audit.tsv")) else pd.DataFrame()
    comp_pos = cdf[cdf.track == "pos"].set_index("experiment")
    comp_neg = cdf[cdf.track == "neg1c"].set_index("experiment")
    du = float((comp_pos.U_frac - comp_neg.reindex(comp_pos.index).U_frac).mean()) if len(comp_neg) else None
    summary = {
        "channel1_homology": {
            "cluster_leak_frac_mean": round(float(pd.DataFrame(leak_rows).cluster_leak_frac.mean()), 6),
            "nn_identity_mean": round(float(bdf.nn_identity_mean.mean()), 4) if len(bdf) else None,
            "nn_identity_frac_ge0.9": round(float(bdf[[c for c in bdf.columns if c.startswith("frac_0.9")]].sum(1).mean()), 4) if len(bdf) else None,
        },
        "channel2_negative_definition": {
            "neg1f_in_ext300_frac_mean": round(float(na.neg1f_in_ext300_frac.mean()), 4) if len(na) else None,
            "U_frac_pos_minus_neg1c": round(du, 4) if du is not None else None,
            "note": "neg1c→neg2 模型降幅见 probe/overfitnn 结果（技术偏差可学性）",
        },
        "channel3_pretrain_contamination": {
            "status": "declared-not-excludable",
            "rnafm_corpus": "RNAcentral（含人类转录本；简报核验）",
            "rinalmo_corpus": "ncRNA 语料（含人类；简报核验）",
            "mitigation": "跨细胞系/跨协议/整RBP留出切片对污染免疫；主结论以这些切片为准",
        },
        "channel4_annotation_transfer": {
            "status": "N/A-RNA",
            "note": "eCLIP 峰为实验观测，无同源转移注释；蛋白 E6 按证据代码分层",
        },
        "channel5_implementation": {
            "deterministic_extraction": True, "fixed_seeds": {"split": 20261122, "neg": 42},
            "cross_hardware_check": "E1 同一配置复跑波动>1点即排查（单硬件期内声明）",
        },
        "n_eff": {"mean_cluster_ratio": round(float(kdf.n_eff_ratio.mean()), 4)},
    }
    json.dump(summary, open(os.path.join(args.out, "leakage_summary.json"), "w"), indent=2, ensure_ascii=False)
    lg.finish(metrics={"cluster_leak_frac_mean": summary["channel1_homology"]["cluster_leak_frac_mean"],
                       "nn_identity_mean": summary["channel1_homology"]["nn_identity_mean"]},
              outputs=[args.out])
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    print("LEAKAGE_DONE", flush=True)


if __name__ == "__main__":
    main()