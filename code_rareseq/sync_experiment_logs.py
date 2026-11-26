#!/usr/bin/env python3
"""Merge JSONL experiment events without rewriting evidence and audit run traceability."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


TERMINAL_PRECEDENCE = ("invalidated", "completed", "failed")


@dataclass
class Event:
    source: str
    source_path: str
    source_line: int
    raw: bytes
    raw_sha256: str
    semantic_key: str
    record: dict | None
    error: str | None
    merged_line: int | None = None


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_events(path, source):
    path = Path(path)
    events = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), 1):
        raw_sha256 = sha256_bytes(raw)
        try:
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("top-level JSON value is not an object")
            canonical = json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            semantic_key = "json:" + sha256_bytes(canonical)
            error = None
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            record = None
            semantic_key = "raw:" + raw_sha256
            error = str(exc)
        events.append(
            Event(
                source=source,
                source_path=str(path),
                source_line=line_number,
                raw=raw,
                raw_sha256=raw_sha256,
                semantic_key=semantic_key,
                record=record,
                error=error,
            )
        )
    return events


def event_summary(event):
    record = event.record or {}
    summary = {
        "source": event.source,
        "source_line": event.source_line,
        "merged_line": event.merged_line,
        "raw_sha256": event.raw_sha256,
        "semantic_key": event.semantic_key,
        "run_id": record.get("run_id"),
        "stage": record.get("stage"),
        "kind": record.get("kind"),
        "seed": record.get("seed"),
        "status": record.get("status"),
        "start_utc": record.get("start_utc"),
    }
    if event.error is not None:
        summary["error"] = event.error
        summary["raw_preview_utf8"] = event.raw[:240].decode("utf-8", "backslashreplace")
    return summary


def merge_events(local_events, remote_events):
    merged = []
    added = []
    duplicates = []
    seen = {}
    for event in [*local_events, *remote_events]:
        if event.semantic_key in seen:
            duplicates.append(
                {
                    "duplicate": event_summary(event),
                    "retained_merged_line": seen[event.semantic_key].merged_line,
                }
            )
            continue
        event.merged_line = len(merged) + 1
        seen[event.semantic_key] = event
        merged.append(event)
        if event.source != "local":
            added.append(event)
    return {"merged": merged, "added": added, "duplicates": duplicates}


def output_directories(record):
    config = record.get("config") or {}
    directories = []
    if isinstance(config.get("out"), str) and config["out"]:
        directories.append(config["out"])
    for output in record.get("outputs") or []:
        if isinstance(output, str) and output and output not in directories:
            directories.append(output)
    return directories


def build_attempts(events):
    grouped = collections.defaultdict(list)
    for event in events:
        if event.record is None:
            continue
        record = event.record
        config_output = (record.get("config") or {}).get("out")
        key = (record.get("run_id"), record.get("seed"), config_output, record.get("start_utc"))
        grouped[key].append(event)
    attempts = []
    for (run_id, seed, config_output, start_utc), attempt_events in grouped.items():
        running = [e for e in attempt_events if e.record.get("status") == "running"]
        terminal = [e for e in attempt_events if e.record.get("status") != "running"]
        terminal_statuses = sorted({e.record.get("status") for e in terminal})
        effective_status = next(
            (status for status in TERMINAL_PRECEDENCE if status in terminal_statuses),
            "running",
        )
        directories = sorted(
            {directory for e in attempt_events for directory in output_directories(e.record)}
        )
        config_outputs = sorted(
            {
                e.record.get("config", {}).get("out")
                for e in attempt_events
                if e.record.get("config", {}).get("out")
            }
        )
        attempts.append(
            {
                "run_id": run_id,
                "seed": seed,
                "start_utc": start_utc,
                "stage": attempt_events[0].record.get("stage"),
                "kind": attempt_events[0].record.get("kind"),
                "output_directories": directories,
                "run_key": {
                    "run_id": run_id,
                    "output_directories": directories,
                    "seed": seed,
                },
                "merged_lines": [e.merged_line for e in attempt_events],
                "start_lines": [e.merged_line for e in running],
                "finish_lines": [e.merged_line for e in terminal],
                "recorded_statuses": [e.record.get("status") for e in attempt_events],
                "effective_status": effective_status,
                "pairing": "paired" if running and terminal else (
                    "running_only" if running else "terminal_only"
                ),
                "terminal_conflict": len(terminal_statuses) > 1,
                "ambiguous_same_start_outputs": len(config_outputs) > 1,
                "events": attempt_events,
            }
        )
    attempts.sort(
        key=lambda attempt: (
            attempt["start_utc"] or "",
            attempt["run_id"] or "",
            str(attempt["seed"]),
        )
    )
    return attempts


def public_attempt(attempt):
    return {key: value for key, value in attempt.items() if key != "events"}


def completed_events(attempt):
    return [
        event
        for event in attempt["events"]
        if event.record.get("status") == "completed"
    ]


def trace_e1(attempts, manifest_path):
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    rows = []
    for expected in manifest:
        candidates = [
            attempt
            for attempt in attempts
            if attempt["run_id"] == expected["run_id"]
            and expected["path"] in attempt["output_directories"]
        ]
        completed = [
            attempt
            for attempt in candidates
            if attempt["pairing"] == "paired"
            and attempt["effective_status"] == "completed"
            and not attempt["terminal_conflict"]
            and not attempt["ambiguous_same_start_outputs"]
        ]
        matched = len(completed) == 1
        metadata_matches = False
        seed = None
        merged_lines = []
        if matched:
            attempt = completed[0]
            seed = attempt["seed"]
            merged_lines = attempt["merged_lines"]
            finish = completed_events(attempt)[0].record
            metadata_matches = all(
                [
                    finish.get("status") == expected["status"],
                    finish.get("data_version") == expected["data_version"],
                    not expected["code_version"]
                    or finish.get("code_version") == expected["code_version"],
                ]
            )
        rows.append(
            {
                "name": expected["name"],
                "stage": expected["stage"],
                "run_id": expected["run_id"],
                "output_directory": expected["path"],
                "seed": seed,
                "paired_completed_attempts": len(completed),
                "other_attempt_statuses": [
                    attempt["effective_status"]
                    for attempt in candidates
                    if not matched or attempt is not completed[0]
                ],
                "merged_lines": merged_lines,
                "metadata_matches": metadata_matches,
                "traceable": matched and metadata_matches,
            }
        )
    stage_counts = dict(collections.Counter(row["stage"] for row in manifest))
    passed = all(
        [
            len(manifest) == 26,
            stage_counts == {"modern": 15, "knn": 5, "post": 6},
            all(row["traceable"] for row in rows),
        ]
    )
    return {
        "manifest": str(manifest_path),
        "expected_runs": len(manifest),
        "stage_counts": stage_counts,
        "traceable_runs": sum(row["traceable"] for row in rows),
        "missing_or_ambiguous": [row["run_id"] for row in rows if not row["traceable"]],
        "rows": rows,
        "passed": passed,
    }


def trace_e5(attempts, verification_path, per_rbp_path):
    verification = json.loads(Path(verification_path).read_text(encoding="utf-8"))
    with Path(per_rbp_path).open(newline="", encoding="utf-8") as handle:
        per_rbp = list(csv.DictReader(handle, delimiter="\t"))
    source_ids = verification["source_run_ids"]
    pair_methods = collections.defaultdict(set)
    pair_rows = collections.Counter()
    for row in per_rbp:
        pair = (row["holdout"], row["track"], int(row["seed"]))
        pair_methods[pair].add(row["method"])
        pair_rows[row["run_id"]] += 1
    rows = []
    for run_id in source_ids:
        candidates = [attempt for attempt in attempts if attempt["run_id"] == run_id]
        completed = [
            attempt
            for attempt in candidates
            if attempt["pairing"] == "paired"
            and attempt["effective_status"] == "completed"
            and not attempt["terminal_conflict"]
        ]
        continuations = []
        if candidates and not completed:
            for other_id in source_ids:
                if other_id == run_id:
                    continue
                for other in attempts:
                    if (
                        other["run_id"] == other_id
                        and other["pairing"] == "paired"
                        and other["effective_status"] == "completed"
                        and other["seed"] == candidates[0]["seed"]
                        and set(other["output_directories"])
                        & set(candidates[0]["output_directories"])
                    ):
                        continuations.append(other_id)
        rows.append(
            {
                "run_id": run_id,
                "pair_rows": pair_rows[run_id],
                "log_statuses": [attempt["effective_status"] for attempt in candidates],
                "merged_lines": [
                    line for attempt in candidates for line in attempt["merged_lines"]
                ],
                "paired_completed": bool(completed),
                "completed_continuations": sorted(set(continuations)),
                "traceable": bool(candidates)
                and pair_rows[run_id] > 0
                and (bool(completed) or bool(continuations)),
            }
        )
    final_id = verification["run_id"]
    final_completed = [
        attempt
        for attempt in attempts
        if attempt["run_id"] == final_id
        and attempt["pairing"] == "paired"
        and attempt["effective_status"] == "completed"
    ]
    final_metrics = {}
    if len(final_completed) == 1:
        final_metrics = completed_events(final_completed[0])[0].record.get("metrics") or {}
    expected_methods = {"msp", "energy", "deep_knn"}
    passed = all(
        [
            verification.get("status") == "complete",
            verification.get("n_pairs_completed") == 72,
            verification.get("n_pairs_expected") == 72,
            verification.get("missing") == [],
            len(source_ids) == 8,
            set(source_ids) == set(pair_rows),
            len(per_rbp) == 216,
            len(pair_methods) == 72,
            all(methods == expected_methods for methods in pair_methods.values()),
            len(final_completed) == 1,
            final_metrics.get("status") == "complete",
            final_metrics.get("n_rows") == 216,
            final_metrics.get("n_source_runs") == 8,
            final_metrics.get("n_missing") == 0,
            all(row["traceable"] for row in rows),
        ]
    )
    return {
        "verification": str(verification_path),
        "per_rbp": str(per_rbp_path),
        "final_run_id": final_id,
        "final_summary_paired_completed": len(final_completed) == 1,
        "final_summary_metrics": final_metrics,
        "expected_source_runs": len(source_ids),
        "source_runs_with_paired_completion": sum(row["paired_completed"] for row in rows),
        "source_runs_with_historical_open_start": sum(
            bool(row["log_statuses"]) and not row["paired_completed"] for row in rows
        ),
        "pair_rows": len(per_rbp),
        "unique_pairs": len(pair_methods),
        "rows": rows,
        "passed": passed,
    }


def verify_merge(local_events, remote_events, merge_result):
    merged = merge_result["merged"]
    keys = [event.semantic_key for event in merged]
    expected_keys = {event.semantic_key for event in [*local_events, *remote_events]}
    repeated = merge_events(merged, remote_events)
    attempts = build_attempts(merged)
    return {
        "pass1_raw_union_and_local_prefix": len(keys) == len(expected_keys)
        and set(keys) == expected_keys
        and [event.raw for event in merged[: len(local_events)]]
        == [event.raw for event in local_events],
        "pass2_no_duplicate_and_running_never_overrides": len(keys) == len(set(keys))
        and all(
            attempt["effective_status"] == "completed"
            for attempt in attempts
            if any(event.record.get("status") == "completed" for event in attempt["events"])
            and not any(event.record.get("status") == "invalidated" for event in attempt["events"])
        ),
        "pass3_idempotent_remote_replay": len(repeated["merged"]) == len(merged),
    }


def write_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, type=Path, action="append")
    parser.add_argument("--local", type=Path, default=root / "实验日志/experiments.jsonl")
    parser.add_argument(
        "--audit-json", type=Path, default=root / "实验日志/experiments_sync_audit.json"
    )
    parser.add_argument("--e1-manifest", type=Path)
    parser.add_argument("--e5-verification", type=Path)
    parser.add_argument("--e5-per-rbp", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.e5_verification) != bool(args.e5_per_rbp):
        parser.error("--e5-verification and --e5-per-rbp must be supplied together")

    local_events = read_events(args.local, "local")
    remote_events = [
        event
        for remote_path in args.remote
        for event in read_events(remote_path, "remote")
    ]
    result = merge_events(local_events, remote_events)
    merged = result["merged"]
    attempts = build_attempts(merged)
    traces = {}
    if args.e1_manifest:
        traces["e1"] = trace_e1(attempts, args.e1_manifest)
    if args.e5_verification:
        traces["e5"] = trace_e5(attempts, args.e5_verification, args.e5_per_rbp)
    checks = verify_merge(local_events, remote_events, result)
    audit = {
        "schema_version": 1,
        "policy": {
            "raw_records_modified": False,
            "identity": "run_id + output_directories + seed; start_utc separates attempts",
            "pairing": "exact run_id + config.out + seed + start_utc; terminal status has precedence over running",
            "invalid_json": "retained byte-for-byte and deduplicated only by raw SHA-256",
        },
        "sources": {
            "local": {
                "path": str(args.local),
                "sha256": sha256_file(args.local),
                "lines": len(local_events),
            },
            "remote": [
                {
                    "path": str(remote_path),
                    "sha256": sha256_file(remote_path),
                    "lines": len(read_events(remote_path, "remote")),
                }
                for remote_path in args.remote
            ],
        },
        "merge": {
            "merged_lines": len(merged),
            "added_remote_lines": len(result["added"]),
            "duplicate_lines_not_repeated": len(result["duplicates"]),
            "valid_json_lines": sum(event.record is not None for event in merged),
            "invalid_json_lines_preserved": sum(event.record is None for event in merged),
        },
        "checks": checks,
        "invalid_records": [
            event_summary(event) for event in merged if event.record is None
        ],
        "added_events": [event_summary(event) for event in result["added"]],
        "duplicates": result["duplicates"],
        "attempts": [public_attempt(attempt) for attempt in attempts],
        "traces": traces,
    }
    audit["passed"] = all(checks.values()) and all(
        trace["passed"] for trace in traces.values()
    )
    payload = (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if args.dry_run:
        print(json.dumps({"merge": audit["merge"], "checks": checks, "traces": {
            key: {k: v for k, v in trace.items() if k != "rows"}
            for key, trace in traces.items()
        }}, ensure_ascii=False, indent=2))
        return 0 if audit["passed"] else 1
    write_atomic(args.local, b"\n".join(event.raw for event in merged) + b"\n")
    write_atomic(args.audit_json, payload)
    print(json.dumps({"local": str(args.local), "audit": str(args.audit_json), **audit["merge"], "passed": audit["passed"]}, ensure_ascii=False))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
