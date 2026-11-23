import csv
import gzip
import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .outer_learner import OuterPool
from .production_pool import train_support
from .production_selection import FreshMemberScorer
from .selection_eval import SelectionRecord, evaluate_strategy, selection_config_hash, write_selection_records
from .trainer.checkpoint import atomic_json_write
from .trainer.data import canonical_row_order_hash, sha256_file


def _sequence(row):
    return row.get("UTR") or row["utr"]


def _write_jsonl(path, rows):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in rows:
                handle.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())


def _write_labels(path, rows):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            import io
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["row_id", "sequence_id", "utr50", "measured_rl"], delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def _hamming1(left, right):
    return len(left) == len(right) and sum(a != b for a, b in zip(left, right)) == 1


def _near_existing(sequence, half_index):
    for position in (0, 1):
        key = sequence[:25] if position == 0 else sequence[25:]
        for other in half_index.get((position, key), []):
            if _hamming1(sequence, other):
                return True
    return False


def _add_half_index(sequence, half_index):
    half_index[(0, sequence[:25])].append(sequence)
    half_index[(1, sequence[25:])].append(sequence)


def build_castillo_context(raw_file, context, sample_role_paths, train_reference, output, pool_count=16, pool_size=64):
    raw_file = Path(raw_file)
    output = Path(output)
    excluded = set()
    sample_hashes = {}
    for path in sample_role_paths:
        path = Path(path)
        sample_hashes[str(path)] = sha256_file(path)
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                excluded.add(row["utr50"])
    candidate_limit = pool_count * pool_size * 20
    heap = []
    with gzip.open(raw_file, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = _sequence(row)
            if len(sequence) != 50 or set(sequence) - set("ACGT") or sequence in excluded or not row["rl"]:
                continue
            value = float(row["rl"])
            if not np.isfinite(value):
                continue
            digest = int.from_bytes(hashlib.sha256(f"{context}|{sequence}".encode()).digest(), "big")
            item = (-digest, sequence)
            if len(heap) < candidate_limit:
                heapq.heappush(heap, item)
            elif digest < -heap[0][0]:
                heapq.heapreplace(heap, item)
    ordered = sorted((sequence for _, sequence in heap), key=lambda sequence: hashlib.sha256(f"{context}|{sequence}".encode()).hexdigest())
    half_index = defaultdict(list)
    selected = []
    for sequence in ordered:
        if not _near_existing(sequence, half_index):
            selected.append(sequence)
            _add_half_index(sequence, half_index)
            if len(selected) == pool_count * pool_size:
                break
    if len(selected) != pool_count * pool_size:
        raise ValueError(f"not enough independent 50nt sequences for {context}")
    selected_set = set(selected)
    labels = defaultdict(list)
    with gzip.open(raw_file, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = _sequence(row)
            if sequence in selected_set:
                labels[sequence].append(float(row["rl"]))
    output.mkdir(parents=True, exist_ok=False)
    (output / "learner").mkdir()
    (output / "evaluator").mkdir()
    pool_hashes = {}
    all_rows = []
    for pool_index in range(pool_count):
        pool_id = f"castillo_{context}_{pool_index:03d}"
        members = selected[pool_index * pool_size:(pool_index + 1) * pool_size]
        pool_hash = hashlib.sha256("\n".join(members).encode()).hexdigest()
        pool_hashes[pool_id] = pool_hash
        rows = [{"pool_id": pool_id, "pool_role": "castillo_outer", "row_id": f"{context}:sequence-{index:06d}", "sequence_id": f"castillo_{hashlib.sha256(sequence.encode()).hexdigest()}", "utr50": sequence, "cluster_id": f"castillo_{hashlib.sha256(sequence.encode()).hexdigest()}", "q_value": 1.0 / pool_size, "pool_hash": pool_hash, "synthetic": False} for index, sequence in enumerate(members)]
        all_rows.extend(rows)
        _write_jsonl(output / "learner" / f"{pool_id}.jsonl.gz", rows)
    _write_labels(output / "evaluator" / "labels.tsv.gz", [{"row_id": row["row_id"], "sequence_id": row["sequence_id"], "utr50": row["utr50"], "measured_rl": repr(float(np.median(labels[row["utr50"]])))} for row in all_rows])
    manifest = {"schema_version": "1.0", "context": context, "raw_file": str(raw_file), "raw_file_sha256": sha256_file(raw_file), "sample_role_hashes": sample_hashes, "pool_count": pool_count, "pool_size": pool_size, "pool_hashes": pool_hashes, "selected_rows": len(all_rows), "labels_median_within_sequence": True, "exact_sample_overlap_excluded": True, "hamming1_excluded": True, "train_reference_sha256": sha256_file(train_reference), "outer_metrics_computed": False, "labels_returned_to_learner": False}
    atomic_json_write(manifest, output / "pool_manifest.json")
    return manifest


def load_castillo_pool(pool_dir, pool_id, train_reference):
    pool_dir = Path(pool_dir)
    manifest = json.loads((pool_dir / "pool_manifest.json").read_text())
    path = pool_dir / "learner" / f"{pool_id}.jsonl.gz"
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(row["pool_id"] != pool_id or row["pool_hash"] != manifest["pool_hashes"][pool_id] for row in rows):
        raise ValueError("Castillo pool identity mismatch")
    if any("measured_rl" in row for row in rows):
        raise PermissionError("Castillo Learner labels are forbidden")
    with np.load(train_reference, allow_pickle=False) as train:
        support = train_support([row["utr50"] for row in rows], train["utr50"].tolist())
    return OuterPool(
        pool_id=pool_id, pool_role="castillo_outer", synthetic=False,
        row_ids=tuple(row["row_id"] for row in rows), sequence_ids=tuple(row["sequence_id"] for row in rows),
        utr50=tuple(row["utr50"] for row in rows), group_ids=tuple(row["cluster_id"] for row in rows),
        support=support, proposal_probs=np.asarray([row["q_value"] for row in rows]),
    )


def run_castillo_grid(pool, inference_config, output):
    members = [FreshMemberScorer(member, pool, inference_config) for member in inference_config.members]
    records = []
    for cap in (1, 2, 4, 8, 16):
        for seed in (101, 202, 303, 404, 505):
            for strategy in ("random", "bon"):
                record = evaluate_strategy(pool, members, np.zeros(len(pool)), strategy, 0, cap, seed)
                if record.regret is not None:
                    raise PermissionError("Castillo Learner cannot compute regret")
                records.append(record)
    metadata = {"pool_id": pool.pool_id, "actual_cuda_inference": True, "member_gpu": [member.gpu for member in members], "outer_labels_read": False, "outer_metrics_computed": False, "protocol": "Random and BoN only; zero-shot frozen fresh ensemble"}
    return write_selection_records(records, output, metadata=metadata)


def evaluate_castillo_pool(pool_dir, pool_id, selection_path, output):
    pool_dir = Path(pool_dir)
    selection_path = Path(selection_path)
    payload = json.loads(selection_path.read_text())
    if payload["execution_metadata"]["outer_labels_read"]:
        raise PermissionError("Castillo Learner boundary mismatch")
    frozen = [SelectionRecord(**row) for row in payload["records"]]
    if selection_config_hash(frozen) != payload["selection_config_sha256"]:
        raise ValueError("Castillo selection hash mismatch")
    labels = {}
    with gzip.open(pool_dir / "evaluator" / "labels.tsv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            labels[row["sequence_id"]] = float(row["measured_rl"])
    with gzip.open(pool_dir / "learner" / f"{pool_id}.jsonl.gz", "rt") as handle:
        members = [json.loads(line) for line in handle if line.strip()]
    pool_max = max(labels[row["sequence_id"]] for row in members)
    records = []
    for row in payload["records"]:
        regret = None if row["abstain"] else float(pool_max - labels[row["selected_id"]])
        records.append({**row, "selector_regret": regret, "regret": regret, "q_applied": False})
    result = {"pool_id": pool_id, "selection_config_sha256": payload["selection_config_sha256"], "selection_sha256": sha256_file(selection_path), "records": records, "outer_metrics_computed": True, "labels_returned_to_learner": False}
    atomic_json_write(result, output)
    return {"pool_id": pool_id, "records": len(records), "labels_returned_to_learner": False}
