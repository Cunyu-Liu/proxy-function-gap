#!/usr/bin/env python3
"""RARE-Seq E0 共享工具：BED I/O、bedtools 封装、序列提取（bbad307 协议）。"""
import os
import subprocess
import tempfile

import numpy as np

COMP = str.maketrans("ACGUTRYSWKMBDHVN", "UGCAAYRSWMKVHDBN")


def read_bed(path, ncol=6):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < ncol:
                continue
            rows.append((p[0], int(p[1]), int(p[2]), p[3], float(p[4]) if ncol >= 5 else 0.0, p[5] if ncol >= 6 else "."))
    return rows


def write_bed(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def bedtools(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"bedtools fail: {cmd}\n{r.stderr[:500]}")
    return r.stdout


def intersect_u(a, b, stranded=True, extra=""):
    """A 中与 B 重叠的原始区间（-u -wa）。返回行列表。"""
    s = "-s" if stranded else ""
    out = bedtools(f"bedtools intersect {s} -u -wa {extra} -a {a} -b {b}")
    return [l.split("\t") for l in out.splitlines() if l.strip()]


def slop_bed(in_path, out_path, genome_file, b):
    bedtools(f"bedtools slop -i {in_path} -g {genome_file} -b {b} > {out_path}")


def revcomp_rna(seq):
    return seq.translate(COMP)[::-1]


def dna2rna(seq):
    return seq.upper().replace("T", "U")


def extract_seq(fasta, chrom, start, end, strand):
    """BED 半开区间 → RNA 序列（负链反向互补，T→U）。含非 ACGU 返回 None。"""
    try:
        seq = fasta.fetch(chrom, start, end)
    except (KeyError, ValueError):
        return None
    seq = dna2rna(seq)
    if strand == "-":
        seq = revcomp_rna(seq)
    if any(c not in "ACGU" for c in seq):
        return None
    return seq


def chrom_sizes_from_fai(fai_path, out_path):
    with open(fai_path) as f, open(out_path, "w") as o:
        for line in f:
            p = line.split("\t")
            o.write(f"{p[0]}\t{p[1]}\n")


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)