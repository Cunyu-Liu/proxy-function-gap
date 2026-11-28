#!/usr/bin/env python3
"""Frozen RiNALMo-giga embeddings for the frozen RNAcompete Ray2013 sequence IDs.

Protocol addendum: RNAcompete_Ray2013_RiNALMo扩展协议_2026-11-28.md.
Same frozen sequences/order/pooling convention as the RNA-FM tree; encoder and
dimension differ (1280d). Outputs are written to a separate artifact root.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from log_run import RunLogger, sha256_file

VERSION = "rnacompete-ray2013-v1.0-id90-strict3train"
REPO = "multimolecule/rinalmo-giga"
DIM = 1280


def ids_sha256(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def atomic_json(path, value):
    with open(path + ".tmp", "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(path + ".tmp", path)


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    x = torch.tensor([1.25, -2.0], device=device, requires_grad=True)
    y = torch.tensor([2.0, 3.0], device=device)
    probe = x * y
    observed = probe.detach().cpu().numpy()
    if not np.array_equal(observed, np.array([2.5, -6.0], dtype=np.float32)):
        raise RuntimeError(f"CUDA numeric probe failed: {observed}")
    probe.sum().backward()
    if not torch.equal(x.grad, y) or not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA backward probe failed")
    torch.optim.SGD([x], lr=0.1).step()
    if not torch.isfinite(x).all():
        raise RuntimeError("CUDA optimizer probe failed")
    if probe.device.type != "cuda" or probe.device.index != 0:
        raise RuntimeError(f"CUDA device probe failed: {probe.device}")
    torch.cuda.synchronize(device)
    info = {"device": "cuda:0", "name": torch.cuda.get_device_name(device),
            "numeric_probe": observed.tolist(), "backward_probe": "passed",
            "optimizer_probe": "passed", "torch": str(torch.__version__)}
    print("DEVICE_PROBE " + json.dumps(info), flush=True)
    del x, y, probe
    torch.cuda.empty_cache()
    return device, info


def validate_cache(npy_path, ids_path, expected_ids):
    with open(ids_path) as handle:
        ids = [line.strip() for line in handle]
    if ids != expected_ids:
        raise ValueError("Embedding IDs differ from frozen sequence_folds order")
    emb = np.load(npy_path, mmap_mode="r")
    if emb.shape != (len(expected_ids), DIM) or emb.dtype != np.float16:
        raise ValueError(f"Invalid embedding cache: {emb.shape} {emb.dtype}")
    for start in range(0, len(emb), 4096):
        if not np.isfinite(np.asarray(emb[start:start + 4096])).all():
            raise ValueError(f"Non-finite embedding at row {start}")
    return {"rows": len(ids), "dim": DIM, "dtype": "float16",
            "ids_sha256": ids_sha256(ids), "npy_sha256": sha256_file(npy_path),
            "ids_file_sha256": sha256_file(ids_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    if args.batch < 1 or args.limit < 0 or not args.out.startswith("/teams/"):
        raise ValueError("Invalid batch/limit or non-/teams output")
    if args.model_path and not args.revision:
        raise ValueError("A local RiNALMo snapshot requires an explicit frozen revision")
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.frozen_root, "split_manifest.json")
    with open(manifest_path) as handle:
        frozen_manifest = json.load(handle)
    if frozen_manifest["version"] != VERSION or frozen_manifest["status"] != "accepted":
        raise ValueError("Frozen benchmark manifest is not accepted")
    sequences = pd.read_csv(os.path.join(args.frozen_root, "sequence_folds.tsv.gz"), sep="\t",
                            usecols=["sequence_id", "rna_seq"])
    if args.limit:
        sequences = sequences.iloc[:args.limit].copy()
    ids = sequences.sequence_id.tolist()
    seqs = sequences.rna_seq.tolist()
    if len(set(ids)) != len(ids) or any(not seq or set(seq) - set("ACGU") for seq in seqs):
        raise ValueError("Invalid or duplicate frozen RNA sequence")
    config = vars(args).copy()
    config.update({"repo": REPO, "pool": "mean-masked", "special_tokens_in_pool": True,
                   "truncation": False, "autocast": "bf16", "cache_dtype": "fp16",
                   "code_sha256": sha256_file(os.path.abspath(__file__)),
                   "frozen_manifest_sha256": sha256_file(manifest_path)})
    logger = RunLogger(args.log, stage="RNAcompete", kind="rinalmo-embed-smoke" if args.limit else "rinalmo-embed")
    logger.start(config=config, data_version=VERSION, inputs=[manifest_path],
                 code_version=config["code_sha256"])
    started = time.time()
    try:
        import torch
        from multimolecule import AutoModel, AutoTokenizer
        device, device_info = require_cuda()
        source = args.model_path or REPO
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, revision=args.revision)
        model = AutoModel.from_pretrained(source, trust_remote_code=True, revision=args.revision).to(device).eval()
        if model.config.hidden_size != DIM:
            raise ValueError(f"Unexpected RiNALMo hidden size: {model.config.hidden_size}")
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError("RiNALMo is not on CUDA")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        max_tokens = max(map(len, seqs)) + 2
        if max_tokens > tokenizer.model_max_length:
            raise ValueError(f"Sequence length {max_tokens} exceeds RiNALMo capacity")
        emb = np.empty((len(seqs), DIM), dtype=np.float16)
        with torch.no_grad():
            for start in range(0, len(seqs), args.batch):
                chunk = seqs[start:start + args.batch]
                encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=False)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model(**encoded).last_hidden_state
                if hidden.device.type != "cuda" or hidden.shape[-1] != DIM:
                    raise RuntimeError("RiNALMo forward did not produce CUDA 1280d states")
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                values = pooled.half().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite RiNALMo output at row {start}")
                emb[start:start + len(chunk)] = values
                if start == 0 or (start // args.batch) % 50 == 0:
                    print(f"RINALMO_EMBED {min(start + len(chunk), len(seqs))}/{len(seqs)}", flush=True)
        npy_path = os.path.join(args.out, "rinalmo.mean_masked.fp16.npy")
        ids_path = os.path.join(args.out, "rinalmo.ids")
        with open(npy_path + ".tmp", "wb") as handle:
            np.save(handle, emb)
            handle.flush()
            os.fsync(handle.fileno())
        with open(ids_path + ".tmp", "w") as handle:
            handle.write("\n".join(ids) + "\n")
        os.replace(npy_path + ".tmp", npy_path)
        os.replace(ids_path + ".tmp", ids_path)
        cache = validate_cache(npy_path, ids_path, ids)
        resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision
        weight_path = None
        if args.model_path:
            for name in ("pytorch_model.bin", "model.safetensors"):
                candidate = os.path.join(source, name)
                if os.path.exists(candidate):
                    weight_path = candidate
                    break
        result = {"status": "passed", "repo": REPO, "resolved_revision": resolved_revision,
                  "model_source": source,
                  "model_weight_sha256": sha256_file(weight_path) if weight_path else None,
                  "pool": "mean-masked", "special_tokens_in_pool": True, "truncation": False,
                  "max_tokens": max_tokens, "cuda": device_info, "cache": cache,
                  "cuda_memory": {"max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                                  "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device))},
                  "frozen_manifest_sha256": config["frozen_manifest_sha256"],
                  "code_sha256": config["code_sha256"], "elapsed_seconds": time.time() - started}
        result_path = os.path.join(args.out, "smoke_cuda.json" if args.limit else "embedding_manifest.json")
        atomic_json(result_path, result)
        logger.finish(metrics={"rows": len(ids), "dim": DIM, "finite": True, "cuda": device_info},
                      outputs=[npy_path, ids_path, result_path])
        print("RNACOMPETE_RINALMO_" + ("SMOKE_OK " if args.limit else "EMBED_OK ") + json.dumps(result), flush=True)
    except Exception as exc:
        logger.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()