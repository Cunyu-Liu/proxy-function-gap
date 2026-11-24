#!/usr/bin/env python3
"""按 GitHub 文件树批量下载 Benchmark-RBP 指定文件（raw 通道，重试+大小校验）。"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

TREE = sys.argv[1]
OUT = sys.argv[2]
RAW = "https://raw.githubusercontent.com/mhorlacher/Benchmark-RBP/main/"

CODE_PREFIXES = (
    "data/datasets/ENCODE/Snakefile",
    "data/datasets/ENCODE/config.yml",
    "data/datasets/ENCODE/example.config.yml",
    "data/datasets/ENCODE/info.json",
    "data/datasets/ENCODE/stats.ENCODE.csv",
    "data/datasets/ENCODE/code/",
    "data/datasets/Snakefile",
    "data/datasets/stats.csv",
    "data/datasets/README.md",
    "data/README.md",
    "data/proc/Snakefile",
    "data/proc/config.yml",
    "data/proc/README.md",
    "data/proc/peaks-cv/",
    "data/proc/samples/",
    "data/proc/inputs-2/DeepRiPe-1/Snakefile",
    "data/proc/inputs-2/DeepRiPe-1/scripts/slop50.sh",
    "data/proc/inputs-2/DeepRiPe-1/scripts/bed_to_fasta.py",
    "data/proc/inputs-2/DeepRiPe-1/scripts/generate_labeled_windows_bed.py",
    "data/proc/inputs-2/DeepRiPe-1/scripts/filter_bed.py",
    "data/proc/inputs-2/DeepRiPe-1/scripts/add_rbp_name.sh",
    "data/proc/inputs-2/BERT-RBP/Snakefile",
    "data/proc/inputs-2/BERT-RBP/scripts/tokenize-fasta.py",
    "data/datasets/Mukherjee-PAR-CLIP/Snakefile",
    "data/datasets/Mukherjee-PAR-CLIP/info.json",
    "data/datasets/Mukherjee-PAR-CLIP/processed.full/peaks.counts.csv",
    ".gitmodules",
    "LICENSE",
)
DATA_SUFFIX = ("data/datasets/ENCODE/processed/",)


def wanted(path):
    if path.startswith(CODE_PREFIXES):
        return True
    if path.startswith(DATA_SUFFIX) and (path.endswith("peaks.bed") or path.endswith("peaks.crosslink.bed")):
        return True
    return False


def fetch(item):
    path, size = item
    dst = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return (path, "cached", size)
    err = "unknown"
    for attempt in range(4):
        try:
            r = subprocess.run(
                ["curl", "-sS", "--http1.1", "--max-time", "180", "--retry", "2",
                 "-o", dst, RAW + path],
                capture_output=True, text=True, timeout=400)
            if r.returncode != 0:
                raise IOError(f"curl rc={r.returncode} {r.stderr[:80]}")
            n = os.path.getsize(dst)
            if size and n != size:
                raise IOError(f"size mismatch {n} != {size}")
            return (path, "ok", n)
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 * (attempt + 1))
    return (path, f"FAIL {err}", 0)


def main():
    tree = json.load(open(TREE))["tree"]
    items = [(e["path"], e.get("size", 0)) for e in tree
             if e["type"] == "blob" and wanted(e["path"])]
    total = sum(s for _, s in items)
    print(f"files: {len(items)}  total: {total/1e6:.1f} MB", flush=True)
    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch, it): it[0] for it in items}
        for i, fu in enumerate(as_completed(futs)):
            path, status, n = fu.result()
            if status.startswith("FAIL"):
                fail += 1
                print("FAIL:", path, status, flush=True)
            else:
                ok += 1
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"progress {i+1}/{len(items)} ok={ok} fail={fail} elapsed={el:.0f}s", flush=True)
    print(f"DONE ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()