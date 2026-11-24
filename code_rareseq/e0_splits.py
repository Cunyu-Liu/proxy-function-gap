#!/usr/bin/env python3
"""RARE-Seq E0 — 簇独占 5 折划分冻结（§4.7）。

流程:
  1. 正例窗口（主集 146 实验）→ CD-HIT-EST 聚类（主 90%，敏感性 80%）
  2. 区域链接: 同簇窗口所在 BASIC 合并区域 union；跨区窗口（±50nt 跨界）union
  3. 连通分量按区域数贪心均衡分配到 5 折（种子固定）
  4. 冻结: region_fold_map.tsv + fold-{k}.regions.bed + split_manifest.json
偏差记录: bbad307 原管线用无种子 sort --random-sort + 轮转分折（不可复现）；
本协议改为固定种子 + 簇/区域连通分量整块分配（泄漏受控，预注册后冻结）。
用法: python e0_splits.py --windows-dir ... --regions ... --out ... [--identity 0.9] [--pilot-test N]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_run import RunLogger  # noqa: E402
from rareseq_common import bedtools, read_bed, write_bed  # noqa: E402

SEED = 20261122


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def parse_clstr(path):
    """window_id -> cluster_id（代表序列所在簇）。"""
    w2c = {}
    cid = None
    with open(path) as f:
        for line in f:
            if line.startswith(">Cluster"):
                cid = int(line.split()[1])
            else:
                # 格式: idx\tlen, >name... at identity
                name = line.split(">")[1].split("...")[0]
                w2c[name] = cid
    return w2c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--regions", required=True, help="basic_merged_regions.bed (BED6)")
    ap.add_argument("--regions-tsv", required=True, help="basic_merged_regions.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--identity", type=float, default=0.9)
    ap.add_argument("--exps-file", default=None, help="参与聚类的实验名列表（每行一个）；默认全部")
    ap.add_argument("--pilot-test", type=int, default=0, help=">0 时只取前 N 条窗口测速")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--mem-mb", type=int, default=24000)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = f"id{int(args.identity*100)}"
    lg = RunLogger(args.log, stage="E0", kind="split-freeze")
    lg.start(config={"identity": args.identity, "seed": SEED, "folds": 5,
                     "regions": args.regions, "pilot_test": args.pilot_test},
             seed=SEED, data_version="bbad307-repo-main-2024-01-31+gencode-v42-basic")

    # ---- 1. 合并正例窗口 FASTA + 窗口 BED ----
    exps = None
    if args.exps_file:
        exps = set(l.strip() for l in open(args.exps_file) if l.strip())
    fa_path = os.path.join(args.out, f"pos_windows.{tag}.fasta")
    bed_path = os.path.join(args.out, f"pos_windows.{tag}.bed")
    n_seq = 0
    win_region = {}
    with open(fa_path, "w") as ff, open(bed_path, "w") as fb:
        for fn in sorted(os.listdir(args.windows_dir)):
            if not fn.endswith(".windows.tsv"):
                continue
            name = fn[:-len(".windows.tsv")]
            if exps and name not in exps:
                continue
            df = pd.read_csv(os.path.join(args.windows_dir, fn), sep="\t")
            for r in df.itertuples():
                if args.pilot_test and n_seq >= args.pilot_test:
                    break
                ff.write(f">{r.window_id}\n{r.seq}\n")
                # 交联位点 1bp BED（用于区域归属）
                cl = r.start + 50 if r.strand == "+" else r.end - 51
                fb.write(f"{r.chrom}\t{cl}\t{cl+1}\t{r.window_id}\t0\t{r.strand}\n")
                n_seq += 1
            if args.pilot_test and n_seq >= args.pilot_test:
                break
    print(f"windows for clustering: {n_seq}", flush=True)

    # ---- 2. CD-HIT-EST ----
    clstr_prefix = os.path.join(args.out, f"clusters.{tag}")
    word = 8 if args.identity >= 0.9 else (7 if args.identity >= 0.88 else 5)
    cmd = (f"cd-hit-est -i {fa_path} -o {clstr_prefix} -c {args.identity} -n {word} "
           f"-G 1 -g 1 -d 0 -M {args.mem_mb} -T {args.threads}")
    print("RUN:", cmd, flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True)
        raise RuntimeError("cd-hit-est failed")
    cluster_seconds = time.time() - t0
    print(f"cd-hit-est done in {cluster_seconds:.0f}s", flush=True)
    w2c = parse_clstr(clstr_prefix + ".clstr")
    n_clusters = len(set(w2c.values()))
    print(f"clusters: {n_clusters}", flush=True)

    # ---- 3. 窗口 → 区域归属（同链 intersect）----
    regions = read_bed(args.regions)
    reg_ids = [r[3] for r in regions]
    out = bedtools(f"bedtools intersect -s -wa -wb -a {bed_path} -b {args.regions}")
    win2reg = {}
    span_links = set()
    for line in out.splitlines():
        p = line.split("\t")
        wid, rid = p[3], p[9]
        win2reg.setdefault(wid, set()).add(rid)
    # 跨区窗口: 101nt 窗口区间与区域求交（同链），链接所有覆盖区域
    win101_path = os.path.join(args.out, f"pos_windows101.{tag}.bed")
    with open(win101_path, "w") as f:
        for fn in sorted(os.listdir(args.windows_dir)):
            if not fn.endswith(".windows.tsv"):
                continue
            name = fn[:-len(".windows.tsv")]
            if exps and name not in exps:
                continue
            df = pd.read_csv(os.path.join(args.windows_dir, fn), sep="\t")
            for r in df.itertuples():
                if r.window_id in w2c:
                    f.write(f"{r.chrom}\t{r.start}\t{r.end}\t{r.window_id}\t0\t{r.strand}\n")
    out2 = bedtools(f"bedtools intersect -s -wa -wb -a {win101_path} -b {args.regions}")
    wregs = defaultdict(set)
    for line in out2.splitlines():
        p = line.split("\t")
        wregs[p[3]].add(p[9])
    for wid, rs in wregs.items():
        rs = sorted(rs)
        for a, b in zip(rs, rs[1:]):
            span_links.add((a, b))

    # ---- 4. union-find: 簇链接 + 跨界链接 ----
    uf = UF()
    for rid in reg_ids:
        uf.find(rid)
    c2regs = defaultdict(set)
    for wid, cid in w2c.items():
        for rid in win2reg.get(wid, ()):
            c2regs[cid].add(rid)
    for cid, rs in c2regs.items():
        rs = sorted(rs)
        for a, b in zip(rs, rs[1:]):
            uf.union(a, b)
    for a, b in span_links:
        uf.union(a, b)
    comp = defaultdict(list)
    for rid in reg_ids:
        comp[uf.find(rid)].append(rid)
    print(f"components: {len(comp)} (regions {len(reg_ids)})", flush=True)

    # ---- 5. 分量 → 5 折（窗口负载贪心均衡，种子固定 tie-break）----
    rng = np.random.default_rng(SEED)
    # 每分量窗口负载（窗口经任一归属区域计入其所在分量；union 后同窗口区域必同分量）
    comp_winload = defaultdict(int)
    for wid, rs in win2reg.items():
        any_r = next(iter(rs))
        comp_winload[uf.find(any_r)] += 1
    comp_sizes = sorted(((len(v), k) for k, v in comp.items()), reverse=True)
    # 打乱同负载分量顺序（固定种子）
    jitter = {k: rng.random() for _, k in comp_sizes}
    comp_sizes.sort(key=lambda x: (-comp_winload.get(x[1], 0), -x[0], jitter[x[1]]))
    fold_load = [0] * 5
    fold_regload = [0] * 5
    rid2fold = {}
    cid2fold = {}
    for size, root in comp_sizes:
        f = int(np.argmin(fold_load))
        fold_load[f] += comp_winload.get(root, 0)
        fold_regload[f] += size
        cid2fold[root] = f
        for rid in comp[root]:
            rid2fold[rid] = f
    # 区域详情
    rtx = pd.read_csv(args.regions_tsv, sep="\t")
    rtx["fold"] = rtx["region_id"].map(rid2fold)
    rtx["component"] = rtx["region_id"].map(lambda r: uf.find(r))
    rtx.to_csv(os.path.join(args.out, f"region_fold_map.{tag}.tsv"), sep="\t", index=False)
    for f in range(5):
        rows = [r for r in regions if rid2fold.get(r[3]) == f]
        write_bed(rows, os.path.join(args.out, f"fold-{f}.regions.{tag}.bed"))
    manifest = {
        "identity": args.identity, "seed": SEED, "folds": 5,
        "n_windows_clustered": n_seq, "n_clusters": n_clusters,
        "n_regions": len(reg_ids), "n_components": len(comp),
        "fold_region_counts": fold_regload, "fold_window_counts": fold_load,
        "cluster_seconds": round(cluster_seconds, 1),
        "n_span_links": len(span_links),
        "protocol": "transcript-region-exclusive + CD-HIT-EST cluster-exclusive overlay; components assigned whole to folds (greedy balance)",
        "deviations_from_bbad307": [
            "original uses unseeded sort --random-sort + round-robin (not reproducible); ours fixed seed + component-block assignment",
            "original has no cluster overlay; ours adds CD-HIT-EST per execution doc §4.7",
        ],
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(manifest, open(os.path.join(args.out, f"split_manifest.{tag}.json"), "w"), indent=2)
    lg.finish(metrics={"n_clusters": n_clusters, "n_components": len(comp),
                       "fold_region_counts": fold_regload, "fold_window_counts": fold_load,
                       "cluster_seconds": round(cluster_seconds, 1)},
              outputs=[args.out])
    print(json.dumps({"regions": fold_regload, "windows": fold_load}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()