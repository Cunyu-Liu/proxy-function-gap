#!/usr/bin/env python3
"""E5 PAR-CLIP→eCLIP 跨协议描述性评测。

eCLIP 主集 fold 2/3/4 训练、fold 1 validation 早停并保存最佳 checkpoint；
fold 0 只提供 eCLIP 参照负例。PAR-CLIP 不参与训练或选参。除旧 eCLIP
neg1c/neg2 参照外，可用其他 RBP 的 PAR-CLIP 位点构造唯一基因匹配、排除
目标位点邻域的同协议 neg2 描述性切片。所有结论均受协议与负例定义限制。
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e1_frozen_baselines import TaskLoader, metrics, stable_seed  # noqa: E402
from log_run import RunLogger, sha256_file  # noqa: E402


class LockedRunLogger(RunLogger):
    def _append(self):
        payload = (json.dumps(self.rec, ensure_ascii=False) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if os.write(fd, payload) != len(payload):
                raise OSError("Incomplete atomic run-log append")
            os.fsync(fd)
        finally:
            os.close(fd)


def malformed_json_lines(path):
    malformed = []
    if not os.path.exists(path):
        return malformed
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                malformed.append(line_number)
    return malformed


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
        out = None
        for e in self.exps:
            if not groups[e]:
                continue
            mat = self.ld.embeddings(e, groups[e])
            if out is None:
                out = np.empty((len(ids), mat.shape[1]), dtype=np.float32)
            out[pos[e]] = mat
        if out is None or not np.isfinite(out).all():
            raise ValueError("Missing or non-finite eCLIP embeddings")
        return out


class ParclipData:
    def __init__(self, emb_dir, windows_dir, model, names):
        self.names = names
        self.tables = {}
        self.arrays = {}
        for name in names:
            table = pd.read_csv(
                os.path.join(windows_dir, f"{name}.windows.tsv"), sep="\t",
                usecols=["window_id", "chrom", "start", "end", "strand"])
            stem = os.path.join(emb_dir, f"{name}.pos.{model}")
            with open(stem + ".ids") as handle:
                ids = [line.strip() for line in handle]
            array = np.load(stem + ".npy", mmap_mode="r")
            if ids != table.window_id.tolist() or len(ids) != array.shape[0]:
                raise ValueError(f"{name}: PAR-CLIP windows/ids/embeddings are not aligned")
            self.tables[name] = table
            self.arrays[name] = array
        self.meta = None

    def fetch(self, rows):
        out = np.empty((len(rows), self.arrays[self.names[0]].shape[1]), dtype=np.float32)
        for name, positions in rows.groupby("name").groups.items():
            source_rows = rows.loc[positions]
            indices = source_rows.row_index.to_numpy(dtype=int)
            out[rows.index.get_indexer(positions)] = np.asarray(
                self.arrays[name][indices], dtype=np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Non-finite PAR-CLIP embeddings")
        return out

    def annotate_genes(self, regions_tsv, transcripts_tsv, work_dir):
        transcripts = pd.read_csv(
            transcripts_tsv, sep="\t", usecols=["transcript_id", "gene_id"])
        tx2gene = dict(zip(transcripts.transcript_id, transcripts.gene_id))
        regions = pd.read_csv(regions_tsv, sep="\t", usecols=["region_id", "transcripts"])
        region_genes = {}
        for row in regions.itertuples(index=False):
            region_genes[row.region_id] = {
                tx2gene[tx] for tx in row.transcripts.split(",") if tx in tx2gene}
        genes = {}
        with tempfile.TemporaryDirectory(prefix="gene_annotation_", dir=work_dir) as tmp:
            crosslinks = os.path.join(tmp, "parclip.crosslinks.bed")
            intersections = os.path.join(tmp, "intersections.bed")
            with open(crosslinks, "w") as handle:
                for name in self.names:
                    for row in self.tables[name].itertuples(index=False):
                        center = row.start + 50
                        handle.write(
                            f"{row.chrom}\t{center}\t{center + 1}\t{row.window_id}\t0\t{row.strand}\n")
            with open(intersections, "w") as handle:
                subprocess.run([
                    "bedtools", "intersect", "-s", "-wa", "-wb", "-a", crosslinks,
                    "-b", regions_tsv.replace(".tsv", ".bed")], stdout=handle, check=True)
            assigned = {}
            with open(intersections) as handle:
                for line in handle:
                    fields = line.rstrip("\n").split("\t")
                    wid, region_id = fields[3], fields[9]
                    assigned.setdefault(wid, set()).update(region_genes.get(region_id, set()))
            genes = {wid: next(iter(values)) for wid, values in assigned.items() if len(values) == 1}
        frames = []
        for name in self.names:
            table = self.tables[name].copy()
            table["name"] = name
            table["rbp"] = rbp_of_parclip(name)
            table["row_index"] = np.arange(len(table))
            table["crosslink"] = table.start + 50
            table["gene"] = table.window_id.map(genes)
            table["coordinate"] = (
                table.chrom + ":" + table.start.astype(str) + ":" + table.strand)
            frames.append(table)
        self.meta = pd.concat(frames, ignore_index=True)
        return {
            "n_windows": len(self.meta),
            "n_unique_gene": int(self.meta.gene.notna().sum()),
            "n_ambiguous_or_unannotated": int(self.meta.gene.isna().sum()),
        }


def exclude_target_neighborhood(candidates, target, distance):
    keep = np.ones(len(candidates), dtype=bool)
    candidates = candidates.reset_index(drop=True)
    for chrom, group in candidates.groupby("chrom").groups.items():
        target_positions = np.sort(target.loc[target.chrom == chrom, "crosslink"].unique())
        if not len(target_positions):
            continue
        positions = group if isinstance(group, np.ndarray) else group.to_numpy()
        query = candidates.loc[positions, "crosslink"].to_numpy()
        right = np.searchsorted(target_positions, query, side="left")
        left = np.maximum(right - 1, 0)
        right_clipped = np.minimum(right, len(target_positions) - 1)
        near = (np.abs(query - target_positions[left]) <= distance) | (
            np.abs(query - target_positions[right_clipped]) <= distance)
        keep[positions[near]] = False
    return candidates.loc[keep].copy()


def match_parclip_neg2(pc_data, target_name, distance=300, seed=42):
    target = pc_data.meta[pc_data.meta.name == target_name].copy()
    target_rbp = rbp_of_parclip(target_name)
    positive_genes = set(target.gene.dropna())
    candidates = pc_data.meta[
        (pc_data.meta.rbp != target_rbp) & pc_data.meta.gene.isin(positive_genes)
    ].drop_duplicates("coordinate").copy()
    candidates = exclude_target_neighborhood(candidates, target, distance)
    pools = {gene: group.index.to_numpy() for gene, group in candidates.groupby("gene")}
    rng = np.random.default_rng(seed)
    positive_parts, negative_parts = [], []
    for gene, positives in target.groupby("gene", sort=False):
        if pd.isna(gene) or gene not in pools or not len(pools[gene]):
            continue
        pool = pools[gene]
        chosen = rng.choice(pool, size=len(positives), replace=True)
        positive_parts.append(positives)
        negative_parts.append(candidates.loc[chosen])
    audit = {
        "rbp": target_rbp,
        "n_target_all": len(target),
        "n_target_unique_gene": int(target.gene.notna().sum()),
        "n_candidate_before_exclusion": int(((pc_data.meta.rbp != target_rbp) &
                                             pc_data.meta.gene.isin(positive_genes)).sum()),
        "n_candidate_after_exclusion": len(candidates),
        "exclusion_distance": distance,
        "matching_seed": seed,
    }
    if not positive_parts:
        audit.update({"n_matched_pos": 0, "n_matched_neg": 0, "n_unique_neg": 0})
        return None, None, audit
    positives = pd.concat(positive_parts)
    negatives = pd.concat(negative_parts)
    audit.update({
        "n_matched_pos": len(positives),
        "n_matched_neg": len(negatives),
        "n_unique_neg": int(negatives.coordinate.nunique()),
    })
    return positives.row_index.to_numpy(dtype=int), negatives.reset_index(drop=True), audit


def train_head(xtr, ytr, xva, yva, args, seed, device):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    mean = xtr.mean(axis=0, keepdims=True)
    std = xtr.std(axis=0, keepdims=True).clip(min=1e-5)
    xtr = (xtr - mean) / std
    xva = (xva - mean) / std
    xtr_t = torch.from_numpy(xtr).to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device)
    xva_t = torch.from_numpy(xva).to(device)
    model = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], args.hidden), torch.nn.GELU(),
        torch.nn.Dropout(args.dropout), torch.nn.Linear(args.hidden, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    best_metric = -np.inf
    best_state = None
    best_epoch = 0
    best_auroc = np.nan
    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(xtr_t), device=device, generator=gen)
        total_loss = 0.0
        for i in range(0, len(order), args.batch):
            ii = order[i:i + args.batch]
            logits = model(xtr_t[ii]).squeeze(1)
            target = ytr_t[ii] * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach()) * len(ii)
        model.eval()
        with torch.no_grad():
            vs = torch.cat([model(xva_t[i:i + args.batch]).squeeze(1)
                            for i in range(0, len(xva_t), args.batch)]).float().cpu().numpy()
        vauroc, vauprc = metrics(yva, vs)
        train_loss = total_loss / len(order)
        if not np.isfinite([vauroc, vauprc, train_loss]).all():
            raise ValueError("Non-finite training/validation metric")
        hist.append({"epoch": epoch, "train_loss": train_loss,
                     "val_auroc": vauroc, "val_auprc": vauprc})
        if vauprc > best_metric + args.min_delta:
            best_metric = vauprc
            best_auroc = vauroc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    if best_state is None:
        raise ValueError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    info = {"best_epoch": best_epoch, "stopped_epoch": epoch,
            "early_stopped": epoch < args.epochs, "best_val_auprc": best_metric,
            "best_val_auroc": best_auroc}
    del xtr_t, ytr_t, xva_t, opt
    return model, mean, std, info, hist, best_state


def score(model, mean, std, x, batch, device):
    import torch
    x = (x - mean) / std
    xt = torch.from_numpy(x).to(device)
    with torch.no_grad():
        s = torch.cat([model(xt[i:i + batch]).squeeze(1)
                       for i in range(0, len(xt), batch)]).float().cpu().numpy()
    if not np.isfinite(s).all():
        raise ValueError("Non-finite evaluation score")
    return s


def save_scores(path, rbp, train_track, protocol, ids, y, scores):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"rbp": rbp, "train_track": train_track, "eval_protocol": protocol,
                  "window_id": ids, "y": y, "score": scores}).to_csv(
        path, sep="\t", index=False, compression="gzip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0-dir", default="/root/rareseq/data/e0")
    ap.add_argument("--parclip-emb-dir", required=True)
    ap.add_argument("--parclip-windows-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="rnafm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=1e-4)
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--parclip-neg2", action="store_true")
    ap.add_argument("--parclip-exclusion-distance", type=int, default=300)
    ap.add_argument("--parclip-matching-seed", type=int, default=42)
    ap.add_argument("--regions-tsv",
                    default="/root/rareseq/data/ref/gencode_v42_parsed/basic_merged_regions.tsv")
    ap.add_argument("--transcripts-tsv",
                    default="/root/rareseq/data/ref/gencode_v42_parsed/transcripts.tsv")
    ap.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    args = ap.parse_args()

    import torch
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E5 PAR-CLIP eval; CPU fallback is forbidden")
    if args.epochs < 1 or args.patience < 1 or args.min_delta < 0:
        raise ValueError("epochs/patience must be positive and min_delta nonnegative")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    os.makedirs(args.out, exist_ok=False)
    os.makedirs(os.path.join(args.out, "checkpoints"))

    emb_files = [f for f in os.listdir(args.parclip_emb_dir) if f.endswith(f".pos.{args.model}.npy")]
    parclip_rbps = {rbp_of_parclip(f.split(".")[0]): f.split(".")[0] for f in emb_files}
    with open(os.path.join(args.e0_dir, "main_exps.txt")) as handle:
        main_exps = [line.strip() for line in handle if line.strip()]
    main_rbps = {e.rsplit("_", 1)[0] for e in main_exps}
    overlap = sorted(set(parclip_rbps) & main_rbps)

    preexisting_malformed = malformed_json_lines(args.log)
    with open(os.path.join(args.out, "log_preflight.json"), "w") as handle:
        json.dump({"preexisting_malformed_lines": preexisting_malformed,
                   "line_111_preexisting_malformed": 111 in preexisting_malformed,
                   "new_write_mode": "fcntl.flock + O_APPEND + single os.write + fsync"}, handle, indent=2)
    lg = LockedRunLogger(args.log, stage="E5", kind="parclip-cross-protocol-contract-p20")
    run_id = lg.start(config=vars(args), seed=args.seed, data_version="parclip-hg38-v2",
                      inputs=[args.parclip_emb_dir, args.parclip_windows_dir,
                              os.path.join(args.e0_dir, "neg", "window_fold_map.tsv")],
                      code_version=sha256_file(os.path.abspath(__file__)))
    with open(os.path.join(args.out, "run_config.json"), "w") as handle:
        json.dump({"run_id": run_id, "config": vars(args), "overlap_rbps": overlap,
                   "cuda_device": torch.cuda.get_device_name(device)}, handle, indent=2)
    try:
        pc_data = None
        if args.parclip_neg2:
            pc_data = ParclipData(args.parclip_emb_dir, args.parclip_windows_dir,
                                  args.model, [parclip_rbps[rbp] for rbp in overlap])
            annotation_stats = pc_data.annotate_genes(args.regions_tsv, args.transcripts_tsv, args.out)
            with open(os.path.join(args.out, "parclip_gene_annotation.json"), "w") as handle:
                json.dump(annotation_stats, handle, indent=2)
        fold_map = pd.read_csv(os.path.join(args.e0_dir, "neg", "window_fold_map.tsv"),
                               sep="\t", usecols=["window_id", "fold"])
        fold_by_wid = dict(zip(fold_map.window_id, fold_map.fold))
        train_folds = {int(x) for x in args.train_folds.split(",")}
        rows, split_rows, matching_rows = [], [], []
        for rbp in overlap:
            exps = [e for e in main_exps if e.rsplit("_", 1)[0] == rbp]
            pc_name = parclip_rbps[rbp]
            pc_x = np.asarray(np.load(os.path.join(
                args.parclip_emb_dir, f"{pc_name}.pos.{args.model}.npy"),
                mmap_mode="r"), dtype=np.float32)
            with open(os.path.join(
                    args.parclip_emb_dir, f"{pc_name}.pos.{args.model}.ids")) as handle:
                pc_ids = [line.strip() for line in handle]
            if len(pc_ids) != len(pc_x) or not np.isfinite(pc_x).all():
                raise ValueError(f"{pc_name}: invalid PAR-CLIP input")
            me = MultiExpEmb(args.e0_dir, exps, args.model)
            matched_pos, matched_neg = None, None
            if pc_data is not None:
                matched_pos, matched_neg, match_audit = match_parclip_neg2(
                    pc_data, pc_name, args.parclip_exclusion_distance,
                    args.parclip_matching_seed)
                matching_rows.append(match_audit)
            for track in args.tracks.split(","):
                tr_ids, tr_y, va_ids, va_y, neg_ids = [], [], [], [], []
                for e in exps:
                    pos = pd.read_csv(os.path.join(args.e0_dir, "windows", f"{e}.windows.tsv"), sep="\t")
                    neg = pd.read_csv(os.path.join(args.e0_dir, "neg", f"{e}.negatives.tsv"), sep="\t")
                    pf = pos.window_id.map(fold_by_wid)
                    if pf.isna().any():
                        raise ValueError(f"{e}: positive windows missing frozen fold assignment")
                    ptr = pos[pf.isin(train_folds)]
                    pva = pos[pf == args.val_fold]
                    ntr = neg[(neg.track == track) & (neg.fold.isin(train_folds))]
                    nva = neg[(neg.track == track) & (neg.fold == args.val_fold)]
                    nte = neg[(neg.track == track) & (neg.fold == args.neg_fold)]
                    tr_ids += ptr.window_id.tolist() + ntr.window_id.tolist()
                    tr_y += [1] * len(ptr) + [0] * len(ntr)
                    va_ids += pva.window_id.tolist() + nva.window_id.tolist()
                    va_y += [1] * len(pva) + [0] * len(nva)
                    neg_ids += nte.window_id.tolist()
                if not tr_ids or not va_ids or not neg_ids:
                    raise ValueError(f"{rbp}/{track}: empty train/validation/eCLIP-negative split")
                if set(tr_ids) & set(va_ids) or set(tr_ids) & set(neg_ids) or set(va_ids) & set(neg_ids):
                    raise ValueError(f"{rbp}/{track}: split overlap detected")
                xtr = me.fetch(tr_ids)
                xva = me.fetch(va_ids)
                xneg = me.fetch(neg_ids)
                ytr = np.array(tr_y, dtype=np.int64)
                yva = np.array(va_y, dtype=np.int64)
                seed = stable_seed(args.seed, {"experiment": rbp, "fold": 0, "track": track})
                model, mean, std, info, hist, best_state = train_head(
                    xtr, ytr, xva, yva, args, seed, device)
                hist_path = os.path.join(args.out, f"{rbp}_{track}_valhist.tsv")
                pd.DataFrame(hist).to_csv(hist_path, sep="\t", index=False)
                torch.save({"model_state": best_state, "mean": mean, "std": std,
                            "input_dim": int(xtr.shape[1]), "hidden": args.hidden,
                            "dropout": args.dropout, "seed": seed, "base_seed": args.seed,
                            "rbp": rbp, "track": track, "train_folds": sorted(train_folds),
                            "val_fold": args.val_fold, "neg_fold": args.neg_fold,
                            "selection": "per-RBP validation AUPRC", **info},
                           os.path.join(args.out, "checkpoints", f"{rbp}_{track}_best.pt"))
                split_rows.append({"rbp": rbp, "track": track, "seed": seed,
                                   "n_train": len(tr_ids), "n_train_pos": int(ytr.sum()),
                                   "n_train_neg": int((1 - ytr).sum()), "n_val": len(va_ids),
                                   "n_val_pos": int(yva.sum()), "n_val_neg": int((1 - yva).sum()),
                                   "n_eclip_eval_neg": len(neg_ids), **info})
                evaluations = [(f"eCLIP-{track}-fold{args.neg_fold}", pc_x, xneg, pc_ids, neg_ids)]
                if track == "neg2" and matched_neg is not None and len(matched_neg):
                    evaluations.append(("PARCLIP-otherRBP-geneMatched-excl300",
                                        pc_x[matched_pos], pc_data.fetch(matched_neg),
                                        np.asarray(pc_ids, dtype=object)[matched_pos].tolist(),
                                        matched_neg.window_id.tolist()))
                for protocol, pos_input, neg_input, pos_ids, negative_ids in evaluations:
                    pos_score = score(model, mean, std, pos_input, args.batch, device)
                    neg_score = score(model, mean, std, neg_input, args.batch, device)
                    y = np.concatenate([np.ones(len(pos_score), dtype=np.int64),
                                        np.zeros(len(neg_score), dtype=np.int64)])
                    s = np.concatenate([pos_score, neg_score])
                    auroc, auprc = metrics(y, s)
                    prevalence = float(y.mean())
                    rows.append({"rbp": rbp, "train_track": track, "eval_protocol": protocol,
                                 "n_parclip_pos": len(pos_score), "n_neg": len(neg_score),
                                 "n_unique_neg": len(set(negative_ids)),
                                 "positive_prevalence": prevalence,
                                 "best_epoch": info["best_epoch"], "stopped_epoch": info["stopped_epoch"],
                                 "auroc": auroc, "auprc": auprc,
                                 "auprc_over_prevalence": auprc / prevalence})
                    save_scores(os.path.join(args.out, "scores", f"{rbp}_{track}_{protocol}.tsv.gz"),
                                rbp, track, protocol, pos_ids + negative_ids, y, s)
                    print(f"{rbp} {track} {protocol}: AUROC={auroc:.6f} "
                          f"AUPRC={auprc:.6f} best={info['best_epoch']} "
                          f"stop={info['stopped_epoch']}", flush=True)
                del model, xtr, xva, xneg
                torch.cuda.empty_cache()
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(args.out, "cross_protocol_results.tsv"), sep="\t", index=False)
        macro = res.groupby("eval_protocol")[["auroc", "auprc"]].mean()
        macro.to_csv(os.path.join(args.out, "cross_protocol_macro.tsv"), sep="\t")
        pd.DataFrame(split_rows).to_csv(os.path.join(args.out, "split_audit.tsv"), sep="\t", index=False)
        if matching_rows:
            pd.DataFrame(matching_rows).to_csv(
                os.path.join(args.out, "parclip_neg2_matching.tsv"), sep="\t", index=False)
        lg.finish(metrics={"n_rbp": len(overlap), "macro": macro.to_dict()}, outputs=[args.out])
        print("RUN_ID", run_id, flush=True)
        print("MACRO\n", macro, flush=True)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
