#!/usr/bin/env python3
"""Build BASIC-restricted 101 nt windows from lifted PAR-CLIP crosslinks."""
import argparse
import json
import os
from collections import Counter

import pandas as pd
import pysam

from log_run import RunLogger, sha256_file
from rareseq_common import chrom_sizes_from_fai, extract_seq, read_bed, slop_bed, write_bed, bedtools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lifted-dir", required=True)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--regions", required=True, help="GENCODE v42 BASIC same-strand merged BED")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/home/ProxyFunctionGap/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=False)
    if not os.path.exists(args.genome + ".fai"):
        pysam.faidx(args.genome)
    sizes = os.path.join(args.out, "chrom.sizes")
    chrom_sizes_from_fai(args.genome + ".fai", sizes)
    fasta = pysam.FastaFile(args.genome)
    experiments = sorted(name for name in os.listdir(args.lifted_dir)
                         if os.path.isdir(os.path.join(args.lifted_dir, name)))
    lg = RunLogger(args.log, stage="E5", kind="parclip-windows")
    lg.start(config=vars(args), data_version="PAR-CLIP-hg38+GENCODE-v42-BASIC",
             inputs=[args.lifted_dir, args.regions], code_version=sha256_file(os.path.abspath(__file__)))
    stats = []
    try:
        for name in experiments:
            source = os.path.join(args.lifted_dir, name, "peaks.crosslink.bed")
            rows = read_bed(source)
            counts = Counter((c, s, strand) for c, s, e, _, _, strand in rows)
            unique, seen = [], set()
            for c, start, end, _, score, strand in rows:
                key = (c, start, strand)
                if key not in seen:
                    seen.add(key)
                    unique.append((c, start, end, f"{name}|{c}|{start}|{strand}", score, strand))
            raw = os.path.join(args.out, f"{name}.bed")
            slopped = os.path.join(args.out, f"{name}.slop.bed")
            filtered = os.path.join(args.out, f"{name}.basic.bed")
            write_bed(unique, raw)
            slop_bed(raw, slopped, sizes, 50)
            bedtools(f"bedtools intersect -s -u -a {slopped} -b {args.regions} > {filtered}")
            filtered_rows = read_bed(filtered) if os.path.exists(filtered) else []
            n_clip = n_skip = n_final = 0
            output = os.path.join(args.out, f"{name}.windows.tsv")
            with open(output, "w") as handle:
                handle.write("window_id\tchrom\tstart\tend\tstrand\tfc\tseq\tlen\tn_dup\n")
                for c, start, end, wid, score, strand in filtered_rows:
                    if end - start != 101:
                        n_clip += 1
                        continue
                    seq = extract_seq(fasta, c, start, end, strand)
                    if seq is None:
                        n_skip += 1
                        continue
                    crosslink = start + 50 if strand == "+" else end - 51
                    handle.write(f"{wid}\t{c}\t{start}\t{end}\t{strand}\t{score}\t{seq}\t{len(seq)}\t{counts.get((c, crosslink, strand), 1)}\n")
                    n_final += 1
            for path in (raw, slopped, filtered):
                os.unlink(path)
            stats.append({"experiment": name, "rbp": name.split("_", 1)[0], "n_input": len(rows),
                          "n_unique": len(unique), "n_basic_windows": len(filtered_rows),
                          "n_clipped_skipped": n_clip, "n_nonACGU_skipped": n_skip, "n_final": n_final})
            print(name, n_final, flush=True)
        table = pd.DataFrame(stats)
        table.to_csv(os.path.join(args.out, "window_stats.tsv"), sep="\t", index=False)
        totals = {key: int(table[key].sum()) for key in table.columns if key.startswith("n_")}
        with open(os.path.join(args.out, "window_totals.json"), "w") as handle:
            json.dump(totals, handle, indent=2)
        lg.finish(metrics={**totals, "n_experiments": len(table)}, outputs=[args.out])
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
