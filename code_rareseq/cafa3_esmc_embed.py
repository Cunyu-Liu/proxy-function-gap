#!/usr/bin/env python3
"""Frozen ESMC-300M mean embeddings for the CAFA3 temporal holdout."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def read_fasta(path):
    identifiers, sequences = [], []
    identifier, chunks = None, []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if identifier is not None:
                    identifiers.append(identifier)
                    sequences.append("".join(chunks))
                identifier, chunks = line[1:].split()[0], []
            elif line:
                if identifier is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                chunks.append(line)
    if identifier is not None:
        identifiers.append(identifier)
        sequences.append("".join(chunks))
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Empty or duplicate FASTA identifiers in {path}")
    return identifiers, sequences


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embed_split(model, torch, identifiers, sequences, out, split, batch, max_length):
    output_path = out / f"{split}.npy"
    ids_path = out / f"{split}_ids.tsv"
    expected_shape = (len(identifiers), 960)
    if output_path.exists() and ids_path.exists():
        array = np.load(output_path, mmap_mode="r")
        saved_ids = ids_path.read_text(encoding="utf-8").splitlines()
        if array.shape == expected_shape and saved_ids == identifiers and np.isfinite(array).all():
            return {"split": split, "rows": len(identifiers), "shape": list(array.shape),
                    "sha256": sha256_file(output_path), "reused": True}
        raise ValueError(f"Refusing to overwrite invalid or misaligned cache: {output_path}")
    embeddings = []
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(sequences), batch):
            chunk = [sequence[:max_length - 2] for sequence in sequences[start:start + batch]]
            encoded = model._tokenize(chunk)
            if encoded.device.type != "cuda":
                raise RuntimeError("ESMC tokens are not on CUDA")
            mask = encoded != model.tokenizer.pad_token_id
            mask[:, 0] = False
            mask[torch.arange(len(chunk), device=encoded.device), mask.sum(1) - 1] = False
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model(sequence_tokens=encoded).embeddings
            weights = mask.unsqueeze(-1).float()
            pooled = (hidden.float() * weights).sum(1) / weights.sum(1).clamp(min=1)
            embeddings.append(pooled.half().cpu().numpy())
            print(f"EMBED {split} {min(start + batch, len(sequences))}/{len(sequences)}", flush=True)
    array = np.concatenate(embeddings)
    if array.shape != expected_shape or array.dtype != np.float16 or not np.isfinite(array).all():
        raise ValueError(f"Invalid embedding array for {split}: shape={array.shape}, dtype={array.dtype}")
    temporary = out / f".{split}.npy.tmp"
    with open(temporary, "wb") as handle:
        np.save(handle, array)
    os.replace(temporary, output_path)
    temporary_ids = out / f".{split}_ids.tsv.tmp"
    temporary_ids.write_text("\n".join(identifiers) + "\n", encoding="utf-8")
    os.replace(temporary_ids, ids_path)
    seconds = time.time() - started
    return {"split": split, "rows": len(identifiers), "shape": list(array.shape),
            "sha256": sha256_file(output_path), "reused": False, "seconds": seconds,
            "sequences_per_second": len(identifiers) / seconds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default="t0,t1")
    parser.add_argument("--repo", default="esmc_300m")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    args = parser.parse_args()
    import torch
    import esm
    from esm.models.esmc import ESMC
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CAFA3 embedding requires explicit CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((data_dir / "frozen_manifest.json").read_text(encoding="utf-8"))
    frozen_outputs = {item["path"]: item["sha256"] for item in frozen["outputs"]}
    for name in ("t0.fasta", "t1_benchmark.fasta"):
        if sha256_file(data_dir / name) != frozen_outputs[name]:
            raise ValueError(f"Frozen FASTA SHA mismatch: {name}")
    model = ESMC.from_pretrained(args.repo, device=torch.device("cuda")).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("ESMC encoder is not frozen")
    if model.embed.embedding_dim != 960:
        raise ValueError(f"Expected ESMC-300M dimension 960, got {model.embed.embedding_dim}")
    metrics = {"schema_version": "cafa3-esmc300m-embedding-v1", "repo": args.repo,
               "esm_version": esm.__version__, "torch_version": torch.__version__,
               "cuda_device": torch.cuda.get_device_name(0), "embedding_dim": 960,
               "pooling": "final_layer_amino_acid_only_mean_no_bos_eos_pad", "max_length": args.max_length,
               "limit": args.limit, "splits": []}
    for split in args.splits.split(","):
        fasta = data_dir / ("t0.fasta" if split == "t0" else "t1_benchmark.fasta")
        identifiers, sequences = read_fasta(fasta)
        if args.limit:
            identifiers, sequences = identifiers[:args.limit], sequences[:args.limit]
        result = embed_split(model, torch, identifiers, sequences, out, split, args.batch, args.max_length)
        result.update({"fasta_sha256": sha256_file(fasta),
                       "truncated_rows": sum(len(sequence) > args.max_length - 2 for sequence in sequences),
                       "truncation_rate": float(np.mean([len(sequence) > args.max_length - 2 for sequence in sequences]))})
        metrics["splits"].append(result)
        print("SPLIT_DONE", json.dumps(result), flush=True)
    temporary = out / ".embedding_manifest.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, out / "embedding_manifest.json")
    print("CAFA3_EMBED_DONE", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
