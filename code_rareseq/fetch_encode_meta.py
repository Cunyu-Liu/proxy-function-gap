#!/usr/bin/env python3
"""RARE-Seq E0 — ENCODE eCLIP 元数据快照抓取 v2。

逐实验抓取 embedded JSON（含文件 cloud_metadata/S3 直链），
输出实验表 + 峰文件表 + 表达定量文件表（TSV），作为数据快照冻结依据。
"""
import json
import os
import sys
import time
import datetime
from collections import Counter, defaultdict

import requests

BASE = "https://www.encodeproject.org"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/rareseq/data/encode_meta"
os.makedirs(OUT, exist_ok=True)
SESS = requests.Session()
SESS.headers.update({"Accept": "application/json", "User-Agent": "rareseq-e0-audit/1.0 (research)"})


def fetch(path, params=None, tries=6):
    url = BASE + path
    for attempt in range(tries):
        try:
            r = SESS.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            print(f"[retry {attempt}] {path} {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def target_label(tgt):
    if isinstance(tgt, dict):
        return (tgt.get("label") or "").replace("-human", "")
    if isinstance(tgt, str):
        return tgt.strip("/").split("/")[-1].replace("-human", "")
    return "?"


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("snapshot_utc:", stamp, flush=True)

    exps = fetch("/search/", {
        "type": "Experiment", "assay_title": "eCLIP",
        "biosample_ontology.term_name": ["K562", "HepG2"],
        "status": "released", "limit": "all", "format": "json", "frame": "object",
    })
    json.dump(exps, open(os.path.join(OUT, "experiments_object.json"), "w"))
    graph = exps.get("@graph", [])
    print("n_experiments:", len(graph), flush=True)

    exp_rows, peak_rows, quant_rows = [], [], []
    raw_dir = os.path.join(OUT, "experiments_embedded")
    os.makedirs(raw_dir, exist_ok=True)
    for i, e in enumerate(graph):
        acc = e["accession"]
        summary = e.get("biosample_summary", "") or ""
        cell = "K562" if "K562" in summary else ("HepG2" if "HepG2" in summary else "?")
        rbp = target_label(e.get("target"))
        detail = fetch(f"/experiments/{acc}/", {"format": "json", "frame": "embedded"})
        json.dump(detail, open(os.path.join(raw_dir, f"{acc}.json"), "w"))
        bo = detail.get("biosample_ontology")
        if isinstance(bo, dict) and bo.get("term_name"):
            cell = bo["term_name"]
        if detail.get("target"):
            rbp = target_label(detail["target"])
        files = detail.get("files", [])
        peaks = [f for f in files if f.get("output_type") == "peaks"
                 and f.get("file_format") == "bed"
                 and f.get("assembly") == "GRCh38"
                 and f.get("status") == "released"]
        quants = [f for f in files if f.get("output_type") == "transcript quantifications"
                  and f.get("file_format") == "tsv" and f.get("status") == "released"]
        exp_rows.append({
            "experiment": acc, "rbp": rbp, "cell_line": cell,
            "lab": (detail.get("lab") or {}).get("title", ""),
            "date_released": detail.get("date_released", ""),
            "n_peak_beds": len(peaks), "n_quant_tsvs": len(quants),
            "description": (detail.get("description") or "")[:120],
        })
        for f in peaks:
            cm = f.get("cloud_metadata") or {}
            peak_rows.append({
                "experiment": acc, "rbp": rbp, "cell_line": cell,
                "file": f.get("accession"),
                "format_type": f.get("file_format_type", ""),
                "reps": ";".join(str(x) for x in (f.get("biological_replicates") or [])),
                "preferred_default": bool(f.get("preferred_default")),
                "size_bytes": f.get("file_size"), "md5sum": f.get("md5sum", ""),
                "s3_url": cm.get("url", ""), "portal_url": BASE + f.get("href", ""),
                "date_released": detail.get("date_released", ""),
            })
        for f in quants:
            cm = f.get("cloud_metadata") or {}
            quant_rows.append({
                "experiment": acc, "rbp": rbp, "cell_line": cell,
                "file": f.get("accession"), "size_bytes": f.get("file_size"),
                "s3_url": cm.get("url", ""), "portal_url": BASE + f.get("href", ""),
            })
        if (i + 1) % 25 == 0:
            print(f"progress {i+1}/{len(graph)}", flush=True)
        time.sleep(0.15)

    def write_tsv(name, rows, cols):
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

    write_tsv("experiments.tsv", sorted(exp_rows, key=lambda r: (r["rbp"], r["cell_line"])),
              ["experiment", "rbp", "cell_line", "lab", "date_released", "n_peak_beds", "n_quant_tsvs", "description"])
    write_tsv("peak_files.tsv", sorted(peak_rows, key=lambda r: (r["rbp"], r["cell_line"], r["reps"])),
              ["experiment", "rbp", "cell_line", "file", "format_type", "reps", "preferred_default",
               "size_bytes", "md5sum", "s3_url", "portal_url", "date_released"])
    write_tsv("quant_files.tsv", quant_rows,
              ["experiment", "rbp", "cell_line", "file", "size_bytes", "s3_url", "portal_url"])

    by_rbp = defaultdict(set)
    for r in exp_rows:
        by_rbp[r["rbp"]].add(r["cell_line"])
    dual = sorted(k for k, v in by_rbp.items() if {"K562", "HepG2"} <= v)
    with open(os.path.join(OUT, "dual_cellline_rbps.txt"), "w") as fh:
        fh.write("\n".join(dual) + "\n")
    summary = {
        "snapshot_utc": stamp, "n_experiments": len(exp_rows),
        "n_peak_bed_files": len(peak_rows), "n_quant_files": len(quant_rows),
        "n_rbp": len(by_rbp), "n_rbp_dual": len(dual),
        "experiments_per_cell": dict(Counter(r["cell_line"] for r in exp_rows)),
        "experiments_with_peaks": sum(1 for r in exp_rows if r["n_peak_beds"] > 0),
    }
    json.dump(summary, open(os.path.join(OUT, "snapshot_info.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()