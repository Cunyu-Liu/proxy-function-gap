#!/usr/bin/env python3
"""E6 下载 GRIMM EC_v2 官方氨基酸 split 并记录哈希。"""
import argparse
import os

import requests

from log_run import RunLogger, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E6", kind="grimm-download")
    lg.start(config={"split": args.split, "release": "EC_v2", "modality": "amino_acids",
                     "mirror": "https://hf-mirror.com"}, data_version="GRIMM-EC_v2")
    metrics = {}
    for split in ("train", "validation", "test1", "test2"):
        name = f"{split}.tsv"
        url = f"https://hf-mirror.com/datasets/HoarfrostLab/GRIMM/resolve/main/EC_v2/amino_acids/split_{args.split}/{name}"
        dst = os.path.join(args.out, name)
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(dst, "wb") as handle:
                for chunk in response.iter_content(1 << 20):
                    handle.write(chunk)
        metrics[name] = {"bytes": os.path.getsize(dst), "sha256": sha256_file(dst), "url": url}
        print(name, metrics[name], flush=True)
    lg.finish(metrics=metrics, outputs=[args.out])


if __name__ == "__main__":
    main()
