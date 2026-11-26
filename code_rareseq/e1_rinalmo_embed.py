#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from e1_rinalmo_frozen_baselines import require_cuda
from e1_locked_logger import E1RunLogger as RunLogger
from log_run import sha256_file


def ids_sha256(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def read_ids(path):
    with open(path) as handle:
        return [line.strip() for line in handle]


def validate_cache(npy_path, ids_path, expected_ids, dim):
    if not os.path.isfile(npy_path) or not os.path.isfile(ids_path):
        raise ValueError("missing cache pair")
    ids = read_ids(ids_path)
    if ids != expected_ids:
        raise ValueError("window IDs differ from frozen input")
    emb = np.load(npy_path, mmap_mode="r")
    if emb.dtype != np.float16 or emb.shape != (len(expected_ids), dim):
        raise ValueError(f"invalid cache shape/dtype: {emb.shape} {emb.dtype}")
    for start in range(0, len(emb), 4096):
        if not np.isfinite(np.asarray(emb[start:start + 4096])).all():
            raise ValueError(f"non-finite embedding at row {start}")
    return {"rows": len(ids), "dim": dim, "dtype": "float16", "ids_sha256": ids_sha256(ids)}


def atomic_copy(src, dst):
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def atomic_save(emb, ids, npy_path, ids_path):
    npy_tmp = npy_path + ".tmp"
    ids_tmp = ids_path + ".tmp"
    with open(npy_tmp, "wb") as handle:
        np.save(handle, emb)
        handle.flush()
        os.fsync(handle.fileno())
    with open(ids_tmp, "w") as handle:
        handle.write("\n".join(ids) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(npy_tmp, npy_path)
    os.replace(ids_tmp, ids_path)


def append_progress(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--neg-dir", required=True)
    parser.add_argument("--exps-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reuse-from")
    parser.add_argument("--model-path")
    parser.add_argument("--log", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=103)
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--data-version", default="e0-v2-id90-strict3train-main")
    args = parser.parse_args()
    if args.batch < 1 or args.max_length != 103 or args.dim != 1280:
        raise ValueError("invalid RiNALMo embedding contract")
    os.makedirs(args.out, exist_ok=True)
    config = vars(args).copy()
    config.update({"repo": "multimolecule/rinalmo-giga", "pool": "mean-masked",
                   "autocast": "bf16", "cache_dtype": "fp16", "frozen": True,
                   "code_sha256": sha256_file(os.path.abspath(__file__))})
    logger = RunLogger(args.log, stage="E1", kind="rinalmo-main-embed")
    logger.start(config=config, data_version=args.data_version,
                 inputs=[args.exps_file], code_version=config["code_sha256"])
    progress_path = os.path.join(args.out, "embedding_progress.jsonl")
    t0 = time.time()
    try:
        import torch
        from multimolecule import AutoModel, AutoTokenizer
        device, device_info = require_cuda()
        repo = args.model_path or "multimolecule/rinalmo-giga"
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(device).eval()
        if model.config.hidden_size != args.dim:
            raise ValueError(f"unexpected hidden size: {model.config.hidden_size}")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        experiments = [line.strip() for line in open(args.exps_file) if line.strip()]
        records = []
        generated = reused = 0
        for experiment in experiments:
            for kind, directory, suffix in (("pos", args.windows_dir, "windows"),
                                            ("neg", args.neg_dir, "negatives")):
                frame = pd.read_csv(os.path.join(directory, f"{experiment}.{suffix}.tsv"), sep="\t")
                expected_ids = frame.window_id.tolist()
                stem = os.path.join(args.out, f"{experiment}.{kind}.rinalmo-giga")
                npy_path, ids_path = stem + ".npy", stem + ".ids"
                source = "generated"
                try:
                    metadata = validate_cache(npy_path, ids_path, expected_ids, args.dim)
                    source = "existing"
                    reused += 1
                except (ValueError, OSError):
                    reuse_stem = (os.path.join(args.reuse_from, f"{experiment}.{kind}.rinalmo-giga")
                                  if args.reuse_from else None)
                    if reuse_stem:
                        try:
                            validate_cache(reuse_stem + ".npy", reuse_stem + ".ids", expected_ids, args.dim)
                            atomic_copy(reuse_stem + ".npy", npy_path)
                            atomic_copy(reuse_stem + ".ids", ids_path)
                            metadata = validate_cache(npy_path, ids_path, expected_ids, args.dim)
                            source = "pilot_cache_reused"
                            reused += 1
                        except (ValueError, OSError):
                            pass
                if source == "generated":
                    seqs = frame.seq.tolist()
                    chunks = []
                    with torch.no_grad():
                        for start in range(0, len(seqs), args.batch):
                            encoded = tokenizer(seqs[start:start + args.batch], return_tensors="pt",
                                                padding=True, truncation=True, max_length=args.max_length)
                            encoded = {key: value.to(device) for key, value in encoded.items()}
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                hidden = model(**encoded).last_hidden_state
                            mask = encoded["attention_mask"].unsqueeze(-1).float()
                            pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                            chunk = pooled.half().cpu().numpy()
                            if not np.isfinite(chunk).all():
                                raise ValueError(f"non-finite model output: {experiment}/{kind}/{start}")
                            chunks.append(chunk)
                    emb = np.concatenate(chunks) if chunks else np.zeros((0, args.dim), dtype=np.float16)
                    atomic_save(emb, expected_ids, npy_path, ids_path)
                    metadata = validate_cache(npy_path, ids_path, expected_ids, args.dim)
                    generated += 1
                record = {"experiment": experiment, "kind": kind, "source": source,
                          "elapsed_seconds": round(time.time() - t0, 3), "cuda": device_info, **metadata}
                append_progress(progress_path, record)
                records.append(record)
                print("RINALMO_EMBED", json.dumps(record), flush=True)
        manifest = {"n_experiments": len(experiments), "n_files": len(records),
                    "n_rows": sum(row["rows"] for row in records), "generated_files": generated,
                    "reused_or_existing_files": reused, "pool": "mean-masked", "autocast": "bf16",
                    "cache_dtype": "fp16", "dim": args.dim, "cuda": device_info,
                    "code_sha256": config["code_sha256"], "files": records}
        manifest_path = os.path.join(args.out, "embedding_manifest.json")
        with open(manifest_path + ".tmp", "w") as handle:
            json.dump(manifest, handle, indent=2)
        os.replace(manifest_path + ".tmp", manifest_path)
        metrics = {"n_experiments": len(experiments), "n_files": len(records),
                   "n_rows": manifest["n_rows"], "generated_files": generated,
                   "reused_or_existing_files": reused, "cuda": device_info,
                   "seconds": round(time.time() - t0, 3)}
        logger.finish(metrics=metrics, outputs=[args.out, manifest_path])
        print("RINALMO_EMBED_DONE", json.dumps(metrics), flush=True)
    except Exception as exc:
        logger.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
