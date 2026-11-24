import csv
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path


def load_config(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def open_table(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def is_missing(value):
    return str(value).strip().lower() in {"", "na", "nan", "none", "null"}


def normalize_sequence(value):
    sequence = "".join(str(value).upper().split()).replace("U", "T")
    if sequence and set(sequence) <= set("ACGTN"):
        return sequence
    return None


def terminal_50(value):
    sequence = normalize_sequence(value)
    if sequence is None or len(sequence) < 50 or "N" in sequence:
        return None
    return sequence[-50:]


def file_record(path, source=None):
    path = Path(path)
    result = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if source is not None:
        result.update(source)
    return result
