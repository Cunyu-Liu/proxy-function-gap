#!/usr/bin/env python3
"""Concurrency-safe JSONL task evidence for CAFA3 model runs."""
import datetime
import fcntl
import hashlib
import json
import os
import uuid
from pathlib import Path


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def cuda_evidence():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CAFA3 task logging requires explicit CUDA")
    return {"available": True, "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


class TaskLogger:
    def __init__(self, path, code_sha256, data_sha256, cuda):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.code_sha256 = code_sha256
        self.data_sha256 = data_sha256
        self.cuda = cuda

    def append(self, record):
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def start(self, task_type, fold, track, ontology, parent_run_id=None, seed=None):
        run_id = f"cafa3-{task_type}-{uuid.uuid4().hex}"
        event = {"run_id": run_id, "parent_run_id": parent_run_id, "task_type": task_type,
                 "fold": fold, "track": track, "ontology": ontology, "seed": seed,
                 "code_sha256": self.code_sha256, "data_sha256": self.data_sha256,
                 "cuda": self.cuda, "start_utc": utc_now(), "end_utc": None,
                 "status": "running", "artifacts": []}
        self.append(event)
        return event

    def finish(self, event, status, artifacts=(), error=None):
        if status not in {"completed", "failed"}:
            raise ValueError("Terminal task status must be completed or failed")
        record = dict(event)
        record.update({"end_utc": utc_now(), "status": status,
                       "artifacts": [artifact(path) for path in artifacts]})
        if error is not None:
            record["error"] = str(error)
        self.append(record)
        return record
