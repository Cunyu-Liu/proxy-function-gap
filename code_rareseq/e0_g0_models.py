#!/usr/bin/env python3
"""RARE-Seq E0/G0 — 嵌入缓存 + OverfitNN + 冻结探针。

子命令:
  embed      冻结 FM 前向 → 每实验窗口嵌入 (fp16 npy + ids)
  overfitnn  每实验×折×轨: kNN 标签平均 (嵌入余弦 / 3-mer 频率两特征版), k∈{5,10,20}
  probe      冻结嵌入 + 逻辑回归线性探针 (每实验×折×轨)
环境: HF_HOME=/root/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/root/rareseq/pydeps:/root/rnaft/pypath
      /root/miniconda3/envs/llr_env/bin/python e0_g0_models.py <sub> ...
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_run import RunLogger  # noqa: E402

MODELS = {"rnafm": "multimolecule/rnafm", "rinalmo-giga": "multimolecule/rinalmo-giga"}


def load_windows(windows_dir, exps, with_neg=True):
    """返回 {exp: DataFrame(window_id, seq, label, fold, track)}"""
    fmap_path = os.path.join(windows_dir, "..", "neg", "window_fold_map.tsv")
    out = {}
    for name in exps:
        pt = os.path.join(windows_dir, f"{name}.windows.tsv")
        if not os.path.exists(pt):
            continue
        pos = pd.read_csv(pt, sep="\t")
        pos["label"] = 1
        pos["track"] = "pos"
        out[name] = pos
    return out


def _fast_tokenizer_setup(tok):
    """向量化 RNA 分词（cls+字符映射+eos+右padding），启动时与官方逐位核验，任何不匹配即回退。"""
    import torch
    vocab = tok.get_vocab()
    lut = np.full(256, vocab.get("<unk>", 3), dtype=np.int64)
    for ch, idx in vocab.items():
        if len(ch) == 1:
            lut[ord(ch)] = idx
    cls_id = vocab.get("<cls>", 1)
    eos_id = vocab.get("<eos>", 2)
    pad_id = vocab.get("<pad>", 0)

    def fast_encode(seqs):
        n = len(seqs)
        lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=n)
        L = int(lens.max())
        buf = np.full((n, L), 0, dtype=np.uint8)
        for i, s in enumerate(seqs):
            buf[i, :len(s)] = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        core = lut[buf]
        core[np.arange(n)[:, None], np.arange(L)[None, :]] = np.where(
            np.arange(L)[None, :] < lens[:, None], core, pad_id)
        ids = np.full((n, L + 2), pad_id, dtype=np.int64)
        ids[:, 0] = cls_id
        ids[:, 1:L + 1] = core
        ids[np.arange(n), lens + 1] = eos_id
        mask = (np.arange(L + 2)[None, :] < (lens + 2)[:, None]).astype(np.int64)
        return {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask)}

    # 核验门: 随机 ACGU 序列（含变长）与官方输出逐位一致
    rng = np.random.default_rng(0)
    test_seqs = []
    for _ in range(1500):
        ln = int(rng.integers(80, 102))
        test_seqs.append("".join(rng.choice(list("ACGU"), ln)))
    test_seqs += ["A" * 101, "ACGU" * 25 + "A", "U" * 50, "G" * 101]
    ok = True
    for i in range(0, len(test_seqs), 256):
        chunk = test_seqs[i:i + 256]
        ref = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=103)
        got = fast_encode(chunk)
        for k in ("input_ids", "attention_mask"):
            if not torch.equal(ref[k], got[k]):
                ok = False
                break
        if not ok:
            break
    return (fast_encode if ok else None), ok


def cmd_embed(args):
    import torch
    from multimolecule import AutoModel, AutoTokenizer
    device = "cuda"
    repo = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    fast_encode, fast_ok = (None, False)
    if args.model == "rnafm":
        fast_encode, fast_ok = _fast_tokenizer_setup(tok)
    print(f"fast tokenizer verified: {fast_ok}", flush=True)
    model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage=args.stage, kind="embed-cache")
    lg.start(config={"model": repo, "batch": args.batch, "pool": "mean-masked", "dtype": "fp16",
                     "fast_tokenizer_verified": fast_ok},
             seed=None, data_version=args.data_version)
    files = [(args.windows_dir, f) for f in sorted(os.listdir(args.windows_dir)) if f.endswith(".windows.tsv")]
    if args.neg_dir and os.path.isdir(args.neg_dir):
        files += [(args.neg_dir, f) for f in sorted(os.listdir(args.neg_dir)) if f.endswith(".negatives.tsv")]
    if args.exps_file:
        keep = set(l.strip() for l in open(args.exps_file))
        files = [(d, f) for d, f in files if f.split(".")[0] in keep]
    t0 = time.time()
    n_tot = 0
    for dirpath, fn in files:
        name = fn.split(".")[0]
        kind = "pos" if fn.endswith(".windows.tsv") else "neg"
        dst = os.path.join(args.out, f"{name}.{kind}.{args.model}.npy")
        ids_path = os.path.join(args.out, f"{name}.{kind}.{args.model}.ids")
        if os.path.exists(dst) and not args.force:
            continue
        df = pd.read_csv(os.path.join(dirpath, fn), sep="\t")
        seqs = df.seq.tolist()
        embs = []
        with torch.no_grad():
            for i in range(0, len(seqs), args.batch):
                chunk = seqs[i:i + args.batch]
                if fast_encode is not None:
                    enc = fast_encode(chunk)
                else:
                    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=103)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (h.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                embs.append(pooled.half().cpu().numpy())
        E = np.concatenate(embs) if embs else np.zeros((0, model.config.hidden_size), dtype=np.float16)
        np.save(dst, E)
        with open(ids_path, "w") as f:
            f.write("\n".join(df.window_id.tolist()) + "\n")
        n_tot += len(E)
        if n_tot % 200000 < len(E):
            el = time.time() - t0
            print(f"{name}.{kind}: {len(E)} emb; total {n_tot} @ {n_tot/max(el,1):.0f} seq/s", flush=True)
    lg.finish(metrics={"n_embeddings": n_tot, "seq_per_s": round(n_tot / max(time.time() - t0, 1), 1)},
              outputs=[args.out])
    print("EMBED_DONE", n_tot, flush=True)


_TASK_CACHE = {}


def iter_tasks(windows_dir, neg_dir, exps, folds, tracks):
    """每实验×折×轨: (train_ids, train_labels, test_ids, test_labels) 从嵌入索引取。"""
    fmap = pd.read_csv(os.path.join(neg_dir, "window_fold_map.tsv"), sep="\t")
    fold_by_wid = dict(zip(fmap.window_id, fmap.fold))
    for name in exps:
        if name not in _TASK_CACHE:
            pt = os.path.join(windows_dir, f"{name}.windows.tsv")
            if not os.path.exists(pt):
                _TASK_CACHE[name] = None
                continue
            pos = pd.read_csv(pt, sep="\t")
            pos_fold = pos.window_id.map(fold_by_wid)
            np_ = os.path.join(neg_dir, f"{name}.negatives.tsv")
            neg = pd.read_csv(np_, sep="\t") if os.path.exists(np_) else pd.DataFrame()
            _TASK_CACHE[name] = (pos, pos_fold, neg)
        if _TASK_CACHE[name] is None:
            continue
        pos, pos_fold, neg = _TASK_CACHE[name]
        for f in range(folds):
            for track in tracks:
                tr_pos = pos[pos_fold != f]
                te_pos = pos[pos_fold == f]
                if len(neg):
                    ntr = neg[(neg.track == track) & (neg.fold != f)]
                    nte = neg[(neg.track == track) & (neg.fold == f)]
                else:
                    ntr = nte = pd.DataFrame()
                yield (name, f, track,
                       tr_pos.window_id.tolist() + ntr.window_id.tolist(),
                       [1] * len(tr_pos) + [0] * len(ntr),
                       te_pos.window_id.tolist() + nte.window_id.tolist(),
                       [1] * len(te_pos) + [0] * len(nte))


def load_emb_index(emb_dir, name, model):
    idx = {}
    for kind in ("pos", "neg"):
        p = os.path.join(emb_dir, f"{name}.{kind}.{model}.npy")
        ip = os.path.join(emb_dir, f"{name}.{kind}.{model}.ids")
        if os.path.exists(p) and os.path.exists(ip):
            E = np.load(p, mmap_mode="r")
            ids = [l.strip() for l in open(ip)]
            for i, wid in enumerate(ids):
                idx[wid] = (kind, i)
    return idx


def get_matrix(idx, emb_dir, name, model, ids):
    rows = [idx[w] for w in ids if w in idx]
    if not rows:
        return np.zeros((0, 1), dtype=np.float32), []
    kinds = defaultdict(list)
    for w, (kind, i) in zip([w for w in ids if w in idx], rows):
        kinds[kind].append((w, i))
    mats, keep_ids = [], []
    for kind, wis in kinds.items():
        E = np.load(os.path.join(emb_dir, f"{name}.{kind}.{model}.npy"), mmap_mode="r")
        ii = [i for _, i in wis]
        mats.append(np.asarray(E[ii], dtype=np.float32))
        keep_ids += [w for w, _ in wis]
    return np.concatenate(mats), keep_ids


def kmer_vec(seqs, k=3):
    from itertools import product
    bases = "ACGU"
    kmers = ["".join(p) for p in product(bases, repeat=k)]
    ki = {s: i for i, s in enumerate(kmers)}
    X = np.zeros((len(seqs), len(kmers)), dtype=np.float32)
    for r, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            j = ki.get(s[i:i + k])
            if j is not None:
                X[r, j] += 1
    X /= np.maximum(X.sum(1, keepdims=True), 1)
    return X


def metrics(y, s):
    from sklearn.metrics import roc_auc_score, average_precision_score
    if len(set(y)) < 2:
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def cmd_overfitnn(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exps = [l.strip() for l in open(args.exps_file)] if args.exps_file else None
    if exps is None:
        exps = sorted({f.split(".")[0] for f in os.listdir(args.windows_dir) if f.endswith(".windows.tsv")})
    lg = RunLogger(args.log, stage="E0", kind="overfitnn")
    lg.start(config={"model": args.model, "ks": [5, 10, 20], "features": ["emb", "3mer"],
                     "tracks": args.tracks.split(","), "folds": args.folds}, seed=None,
             data_version=args.data_version)
    seq_cache = {}
    rows = []
    for name, f, track, tr_ids, tr_y, te_ids, te_y in iter_tasks(
            args.windows_dir, args.neg_dir, exps, args.folds, args.tracks.split(",")):
        if not te_ids or not tr_ids:
            continue
        idx = load_emb_index(args.emb_dir, name, args.model)
        Xtr, tr_keep = get_matrix(idx, args.emb_dir, name, args.model, tr_ids)
        Xte, te_keep = get_matrix(idx, args.emb_dir, name, args.model, te_ids)
        ytr = np.array([y for w, y in zip(tr_ids, tr_y) if w in idx])
        yte = np.array([y for w, y in zip(te_ids, te_y) if w in idx])
        if len(yte) == 0 or len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        # 嵌入余弦 kNN（GPU 分块）
        a = torch.from_numpy(Xtr).to(device)
        a = torch.nn.functional.normalize(a, dim=1)
        b = torch.from_numpy(Xte).to(device)
        b = torch.nn.functional.normalize(b, dim=1)
        sims = []
        for i in range(0, len(b), 2048):
            sims.append((b[i:i + 2048] @ a.T).cpu())
        S = torch.cat(sims)
        for k in (5, 10, 20):
            kk = min(k, len(a))
            top = S.topk(kk, dim=1).indices.numpy()
            pred = ytr[top].mean(1)
            au, ap = metrics(yte, pred)
            rows.append({"experiment": name, "fold": f, "track": track, "feature": "emb",
                         "k": k, "n_train": len(ytr), "n_test": len(yte), "auroc": au, "auprc": ap})
        # 3-mer kNN
        if name not in seq_cache:
            seqs = {}
            for fn, lab in ((f"{name}.windows.tsv", None), (f"{name}.negatives.tsv", None)):
                p = os.path.join(args.windows_dir if fn.endswith(".windows.tsv") else args.neg_dir, fn)
                if os.path.exists(p):
                    d = pd.read_csv(p, sep="\t")
                    seqs.update(dict(zip(d.window_id, d.seq)))
            seq_cache[name] = seqs
        sq = seq_cache[name]
        Ktr = kmer_vec([sq[w] for w in tr_keep])
        Kte = kmer_vec([sq[w] for w in te_keep])
        ka = torch.from_numpy(Ktr).to(device)
        kb = torch.from_numpy(Kte).to(device)
        # 余弦
        ka = torch.nn.functional.normalize(ka, dim=1)
        kb = torch.nn.functional.normalize(kb, dim=1)
        Sk = []
        for i in range(0, len(kb), 4096):
            Sk.append((kb[i:i + 4096] @ ka.T).cpu())
        Sk = torch.cat(Sk)
        for k in (5, 10, 20):
            kk = min(k, len(ka))
            top = Sk.topk(kk, dim=1).indices.numpy()
            pred = ytr[top].mean(1)
            au, ap = metrics(yte, pred)
            rows.append({"experiment": name, "fold": f, "track": track, "feature": "3mer",
                         "k": k, "n_train": len(ytr), "n_test": len(yte), "auroc": au, "auprc": ap})
        if len(rows) % 300 < 12:
            print(f"overfitnn rows={len(rows)} last={name} f{f} {track}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "overfitnn_results.tsv"), sep="\t", index=False)
    summ = df.groupby(["track", "feature", "k"])[["auroc", "auprc"]].mean().round(4)
    summ.to_csv(os.path.join(args.out, "overfitnn_summary.tsv"), sep="\t")
    lg.finish(metrics={"n_rows": len(df)}, outputs=[args.out])
    print(summ.to_string(), flush=True)
    print("OVERFITNN_DONE", flush=True)


def cmd_probe(args):
    from sklearn.linear_model import LogisticRegression
    exps = [l.strip() for l in open(args.exps_file)] if args.exps_file else None
    if exps is None:
        exps = sorted({f.split(".")[0] for f in os.listdir(args.windows_dir) if f.endswith(".windows.tsv")})
    lg = RunLogger(args.log, stage="E0", kind="frozen-probe")
    lg.start(config={"model": args.model, "probe": "LogReg-C1", "tracks": args.tracks.split(","),
                     "folds": args.folds}, seed=0, data_version=args.data_version)
    rows = []
    for name, f, track, tr_ids, tr_y, te_ids, te_y in iter_tasks(
            args.windows_dir, args.neg_dir, exps, args.folds, args.tracks.split(",")):
        idx = load_emb_index(args.emb_dir, name, args.model)
        Xtr, _ = get_matrix(idx, args.emb_dir, name, args.model, tr_ids)
        Xte, _ = get_matrix(idx, args.emb_dir, name, args.model, te_ids)
        ytr = np.array([y for w, y in zip(tr_ids, tr_y) if w in idx])
        yte = np.array([y for w, y in zip(te_ids, te_y) if w in idx])
        if len(yte) == 0 or len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=4)
        clf.fit(Xtr, ytr)
        s = clf.decision_function(Xte)
        au, ap = metrics(yte, s)
        rows.append({"experiment": name, "fold": f, "track": track, "model": args.model,
                     "n_train": len(ytr), "n_test": len(yte), "auroc": au, "auprc": ap})
        if len(rows) % 50 == 0:
            print(f"probe rows={len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "probe_results.tsv"), sep="\t", index=False)
    summ = df.groupby(["track", "model"])[["auroc", "auprc"]].mean().round(4)
    summ.to_csv(os.path.join(args.out, "probe_summary.tsv"), sep="\t")
    lg.finish(metrics={"n_rows": len(df)}, outputs=[args.out])
    print(summ.to_string(), flush=True)
    print("PROBE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("embed", "overfitnn", "probe"):
        p = sub.add_parser(nm)
        p.add_argument("--windows-dir", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--model", default="rnafm")
        p.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
        p.add_argument("--data-version", default="e0-v1")
        p.add_argument("--stage", default="E0", choices=["E0", "E1"])
        p.add_argument("--exps-file", default=None)
        if nm == "embed":
            p.add_argument("--batch", type=int, default=256)
            p.add_argument("--force", action="store_true")
            p.add_argument("--neg-dir", default=None)
        else:
            p.add_argument("--neg-dir", required=True)
            p.add_argument("--emb-dir", required=True)
            p.add_argument("--folds", type=int, default=5)
            p.add_argument("--tracks", default="neg1c,neg2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"embed": cmd_embed, "overfitnn": cmd_overfitnn, "probe": cmd_probe}[args.cmd](args)


if __name__ == "__main__":
    main()