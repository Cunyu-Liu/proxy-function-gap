#!/usr/bin/env python3
"""RARE-Seq E0 — GENCODE v42 注释解析（v2）。

产出（OUT 目录）:
  transcripts.tsv           每转录本: id/gene/类型/链/区间/外显子总长/全部tag/basic标志
  basic_transcripts.bed     BASIC 转录本 BED6（bbad307 转录本宇宙, 无header, strand第6列）
  basic_merged_regions.bed  BASIC 同链重叠合并区域 BED6（分折单元）
  basic_merged_regions.tsv  合并区域详情（region_id/strand/n_tx/转录本列表）
  all_merged_regions.bed    全注释同链合并 BED6（对照用）
  exons.bed                 外显子区间 BED6
  parse_summary.json

协议依据: bbad307 peaks-cv/Snakefile 使用
  gencode.{v42}.{GRCh38}.basic.annotation.transcripts.bed
  → bedtools merge -s → shuffle → 5 折。
修复: v1 多 tag 只留最后一个（basic 被覆盖）；BED strand 列错位；BED 带 header。
"""
import gzip
import sys
import os
import re
import json
from collections import defaultdict

GTF = sys.argv[1] if len(sys.argv) > 1 else "/root/rareseq/data/ref/gencode.v42.annotation.gtf.gz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/rareseq/data/ref/gencode_v42_parsed"
os.makedirs(OUT, exist_ok=True)

ATTR = re.compile(r'(\S+) "([^"]*)"')


def parse_attrs(s):
    d = {}
    tags = []
    for k, v in ATTR.findall(s):
        if k == "tag":
            tags.append(v)
        else:
            d[k] = v
    d["tags"] = tags
    return d


def merge_spans(spans):
    """spans: list of (start, end, tid) sorted; yield (s, e, [tids]) union of overlaps."""
    cur_s, cur_e, cur_t = None, None, []
    for s, e, tid in sorted(spans):
        if cur_s is None:
            cur_s, cur_e, cur_t = s, e, [tid]
        elif s <= cur_e:
            cur_e = max(cur_e, e)
            cur_t.append(tid)
        else:
            yield cur_s, cur_e, cur_t
            cur_s, cur_e, cur_t = s, e, [tid]
    if cur_s is not None:
        yield cur_s, cur_e, cur_t


def main():
    tx = {}
    exons = defaultdict(list)
    n_lines = 0
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_lines += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, src, feat, start, end, score, strand, phase, attrs = parts
            start, end = int(start), int(end)
            if feat == "transcript":
                a = parse_attrs(attrs)
                tid = a.get("transcript_id", "")
                tx[tid] = {
                    "transcript_id": tid,
                    "gene_id": a.get("gene_id", ""),
                    "gene_name": a.get("gene_name", ""),
                    "transcript_type": a.get("transcript_type", ""),
                    "chrom": chrom, "strand": strand,
                    "start": start - 1, "end": end,
                    "tags": a["tags"],
                    "basic": "basic" in a["tags"],
                }
            elif feat == "exon":
                a = parse_attrs(attrs)
                tid = a.get("transcript_id", "")
                exons[tid].append((chrom, start - 1, end, strand))
    n_basic = sum(t["basic"] for t in tx.values())
    print(f"gtf_lines: {n_lines} transcripts: {len(tx)} basic: {n_basic}", flush=True)

    with open(os.path.join(OUT, "transcripts.tsv"), "w") as fh:
        fh.write("transcript_id\tgene_id\tgene_name\ttranscript_type\tchrom\tstrand\tstart\tend\texon_len\ttags\tbasic\n")
        for tid, t in sorted(tx.items()):
            el = sum(e - s for _, s, e, _ in exons.get(tid, []))
            fh.write(f"{tid}\t{t['gene_id']}\t{t['gene_name']}\t{t['transcript_type']}\t{t['chrom']}\t{t['strand']}\t{t['start']}\t{t['end']}\t{el}\t{','.join(t['tags'])}\t{int(t['basic'])}\n")

    with open(os.path.join(OUT, "exons.bed"), "w") as fh:
        for tid, es in sorted(exons.items()):
            for chrom, s, e, strand in es:
                fh.write(f"{chrom}\t{s}\t{e}\t{tid}\t0\t{strand}\n")

    with open(os.path.join(OUT, "basic_transcripts.bed"), "w") as fh:
        for tid, t in sorted(tx.items()):
            if t["basic"]:
                fh.write(f"{t['chrom']}\t{t['start']}\t{t['end']}\t{tid}\t0\t{t['strand']}\n")

    def write_merged(bed_path, tsv_path, subset, prefix):
        by_cs = defaultdict(list)
        for tid, t in subset.items():
            by_cs[(t["chrom"], t["strand"])].append((t["start"], t["end"], tid))
        n, bp = 0, 0
        with open(bed_path, "w") as fb, open(tsv_path, "w") as ft:
            ft.write("region_id\tchrom\tstart\tend\tstrand\tn_tx\ttranscripts\n")
            for (chrom, strand), spans in sorted(by_cs.items()):
                for s, e, tids in merge_spans(spans):
                    rid = f"{prefix}{n:07d}"
                    fb.write(f"{chrom}\t{s}\t{e}\t{rid}\t0\t{strand}\n")
                    ft.write(f"{rid}\t{chrom}\t{s}\t{e}\t{strand}\t{len(tids)}\t{','.join(tids)}\n")
                    n += 1
                    bp += e - s
        return n, bp

    basic_tx = {k: v for k, v in tx.items() if v["basic"]}
    nb, bpb = write_merged(os.path.join(OUT, "basic_merged_regions.bed"),
                           os.path.join(OUT, "basic_merged_regions.tsv"), basic_tx, "BR")
    na, apb = write_merged(os.path.join(OUT, "all_merged_regions.bed"),
                           os.path.join(OUT, "all_merged_regions.tsv"), tx, "MR")
    summary = {"n_transcripts": len(tx), "n_basic_transcripts": n_basic,
               "n_basic_merged_regions": nb, "basic_merged_total_bp": bpb,
               "n_all_merged_regions": na, "all_merged_total_bp": apb,
               "chroms": sorted({t["chrom"] for t in tx.values()})}
    json.dump(summary, open(os.path.join(OUT, "parse_summary.json"), "w"), indent=2)
    print(json.dumps(summary), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()