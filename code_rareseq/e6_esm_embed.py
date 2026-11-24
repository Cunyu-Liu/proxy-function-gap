#!/usr/bin/env python3
"""E6 GRIMM 序列的官方 ESM-C frozen 氨基酸均值嵌入缓存。"""
import argparse
import os

import numpy as np
import pandas as pd

from log_run import RunLogger, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="esmc_300m", help="官方 ESM 包内部注册名")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--log", default="/root/rareseq/logs/experiments.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="仅用于小规模前向验证")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lg = RunLogger(args.log, stage="E6", kind="esm-embed")
    lg.start(config=vars(args), data_version="GRIMM-EC_v2", code_version=sha256_file(os.path.abspath(__file__)))
    try:
        import time
        import torch
        from esm.models.esmc import ESMC
        device = "cuda"
        model = ESMC.from_pretrained(args.repo, device=torch.device(device)).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        lengths = pd.concat([pd.read_csv(os.path.join(args.data_dir, name + ".tsv"), sep="\t",
                                                     usecols=["Sequence"]).Sequence.str.len()
                             for name in ("train", "validation", "test1", "test2")])
        truncation_rate = float((lengths > args.max_length - 2).mean())
        metrics = {"repo": args.repo, "max_length": args.max_length,
                   "truncation_rate": truncation_rate, "splits": {}}
        for name in ("train", "validation", "test1", "test2"):
            frame = pd.read_csv(os.path.join(args.data_dir, name + ".tsv"), sep="\t")
            if args.limit:
                frame = frame.iloc[:args.limit].copy()
            sequences = frame.Sequence.tolist()
            embeddings = []
            started = time.time()
            with torch.no_grad():
                for i in range(0, len(sequences), args.batch):
                    chunk = [sequence[:args.max_length - 2] for sequence in sequences[i:i + args.batch]]
                    encoded = model._tokenize(chunk)
                    mask = encoded != model.tokenizer.pad_token_id
                    mask[:, 0] = False
                    mask[torch.arange(len(chunk)), mask.sum(1) - 1] = False
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hidden = model(sequence_tokens=encoded).embeddings
                    weights = mask.unsqueeze(-1).float()
                    pooled = (hidden.float() * weights).sum(1) / weights.sum(1).clamp(min=1)
                    embeddings.append(pooled.half().cpu().numpy())
                embeddings = np.concatenate(embeddings)
            assert embeddings.shape == (len(frame), model.embed.embedding_dim)
            assert np.isfinite(embeddings).all()
            np.save(os.path.join(args.out, f"{name}.npy"), embeddings)
            frame[["Entry"]].to_csv(os.path.join(args.out, f"{name}_ids.tsv"), sep="\t", index=False)
            seconds = time.time() - started
            metrics["splits"][name] = {"rows": len(frame), "shape": list(embeddings.shape),
                                        "seconds": seconds, "sequences_per_second": len(frame) / seconds}
            print(name, embeddings.shape, metrics["splits"][name], flush=True)
        metrics["embedding_dim"] = int(embeddings.shape[1])
        lg.finish(metrics=metrics, outputs=[args.out],
                  notes="Official esm 3.2.1 ESMC; amino-acid-only mean pooling; final-layer embeddings")
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
