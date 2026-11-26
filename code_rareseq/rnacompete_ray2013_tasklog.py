#!/usr/bin/env python3
"""Concurrent task JSONL and non-fabricated manifests for RNAcompete runs."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os

from log_run import sha256_file

REPORT_POINTER = "E1_强基线/RNAcompete_Ray2013_RNAFM基线执行报告_2026-11-27.md"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    with open(path + ".tmp", "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(path + ".tmp", path)


def append_locked(path, record):
    payload = (json.dumps(record, sort_keys=True, default=str) + "\n").encode("utf-8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(f"Partial task JSONL append: {written}/{len(payload)}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def data_provenance(args):
    with open(os.path.join(args.frozen_root, "split_manifest.json"), "rb") as handle:
        frozen_sha = hashlib.sha256(handle.read()).hexdigest()
    with open(os.path.join(args.embedding_dir, "embedding_manifest.json")) as handle:
        embedding = json.load(handle)
    provenance = {
        "frozen_manifest_sha256": frozen_sha,
        "label_cache_sha256": sha256_file(args.label_cache),
        "embedding_npy_sha256": embedding["cache"]["npy_sha256"],
        "embedding_ids_sha256": embedding["cache"]["ids_sha256"],
    }
    canonical = json.dumps(provenance, sort_keys=True).encode()
    provenance["combined_data_sha256"] = hashlib.sha256(canonical).hexdigest()
    return provenance


def artifact_paths(command, args):
    track = args.track
    arch = args.arch
    if command in ("select", "train", "evaluate") and track and arch:
        selection = os.path.join(args.out, "selections", f"{track}_{arch}.json")
    elif track:
        selection = os.path.join(args.out, "selections", f"{track}_knn.json")
    else:
        selection = None
    result = {"history": [], "checkpoint": [], "validation_predictions": [],
              "test_predictions": [], "metrics": [], "selection": selection, "gate": None,
              "summary": []}
    if command in ("select", "train") and track and arch:
        if command == "select":
            runs = [os.path.join(args.out, "selection_runs", f"{track}_{arch}_wd{wd:g}")
                    for wd in (1e-4, 1e-2, 1.0)]
        else:
            runs = [os.path.join(args.out, "runs", f"{track}_{arch}_f{args.fold}_s{args.seed}")]
        for run in runs:
            for key, name in (("history", "validation_history.tsv"), ("checkpoint", "best.pt"),
                              ("validation_predictions", "validation_predictions.npz")):
                path = os.path.join(run, name)
                if os.path.isfile(path):
                    result[key].append(path)
    elif command == "evaluate" and track and arch:
        run = os.path.join(args.out, "runs", f"{track}_{arch}_f{args.fold}_s{args.seed}")
        for key, name in (("history", "validation_history.tsv"), ("checkpoint", "best.pt"),
                              ("validation_predictions", "validation_predictions.npz"),
                              ("test_predictions", "test_predictions.npz"),
                              ("metrics", "test_metrics.tsv")):
            path = os.path.join(run, name)
            if os.path.isfile(path):
                result[key].append(path)
    elif command in ("knn-select", "knn-evaluate") and track:
        run = os.path.join(args.out, "knn", f"{track}_f{args.fold if command == 'knn-evaluate' else 0}")
        for key, name in (("validation_predictions", "validation_predictions.npz"),
                          ("test_predictions", "test_predictions.npz"), ("metrics", "test_metrics.tsv")):
            path = os.path.join(run, name)
            if os.path.isfile(path):
                result[key].append(path)
    elif command == "verify-fold0":
        result["gate"] = os.path.join(args.out, "fold0_gate.json")
    elif command == "verify-final":
        result["gate"] = os.path.join(args.out, "final_gate.json")
    elif command == "summarize":
        result["summary"] = [os.path.join(args.out, name) for name in
                             ("sequence_summary.tsv", "rbp_ood_experiment_summary.tsv",
                              "rbp_ood_group_summary.tsv", "all_experiment_metrics.tsv.gz")
                             if os.path.isfile(os.path.join(args.out, name))]
    return result


class TaskLogger:
    def __init__(self, path, args):
        self.path = path
        self.args = args
        self.run_id = None

    def start(self, cuda_required):
        args = self.args
        timestamp = utc_now()
        self.start_utc = timestamp
        self.run_id = "-".join([
            "rnacompete", args.command, args.track or "all", args.arch or "all",
            f"f{args.fold}", f"s{args.seed}", timestamp.replace(":", "").replace("+", "Z"),
            str(os.getpid()),
        ])
        record = {"run_id": self.run_id, "command": args.command,
                  "code_sha256": sha256_file(os.path.abspath(args.prog)),
                  "data_sha256": data_provenance(args), "seed": args.seed, "fold": args.fold,
                  "track": args.track, "arch": args.arch,
                  "cuda": {"required": cuda_required, "device": "cuda:0" if cuda_required else None,
                           "probe": "pending" if cuda_required else "not_required"},
                  "start_utc": timestamp, "end_utc": None, "status": "running",
                  "artifacts": artifact_paths(args.command, args),
                  "derived_from_artifacts": False}
        append_locked(self.path, record)

    def finish(self, status, cuda=None, error=None):
        if self.run_id is None:
            return
        args = self.args
        record = {"run_id": self.run_id, "command": args.command,
                  "code_sha256": sha256_file(os.path.abspath(args.prog)),
                  "data_sha256": data_provenance(args), "seed": args.seed, "fold": args.fold,
                  "track": args.track, "arch": args.arch,
                  "cuda": cuda or {"required": False, "device": None, "probe": "not_required"},
                  "start_utc": self.start_utc, "end_utc": utc_now(), "status": status,
                  "error": None if error is None else f"{type(error).__name__}: {error}",
                  "artifacts": artifact_paths(args.command, args),
                  "derived_from_artifacts": False}
        append_locked(self.path, record)


def read_json_if(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def marker_record(path):
    if not os.path.isfile(path):
        return None
    return {"path": path,
            "mtime_utc": datetime.datetime.fromtimestamp(os.path.getmtime(path),
                                                         datetime.timezone.utc).isoformat()}


def derived_manifest(root, report_pointer=REPORT_POINTER):
    records = []
    for track in ("sequence", "rbp_ood"):
        for arch in ("linear", "mlp", "knn"):
            selection = os.path.join(root, "selections", f"{track}_{arch}.json")
            if os.path.isfile(selection):
                records.append({"run_id": f"derived-selection-{track}-{arch}", "command": "select",
                                "track": track, "arch": arch, "fold": 0, "seed": 0,
                                "status": "derived_frozen", "start_utc": None, "end_utc": None,
                                "derived_from_artifacts": True, "artifacts": {"selection": selection},
                                "observed_artifact": {"path": selection}})
    for kind in ("runs", "knn"):
        base = os.path.join(root, kind)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            run = os.path.join(base, name)
            if not os.path.isdir(run):
                continue
            parts = name.split("_")
            if kind == "runs":
                track = "rbp_ood" if name.startswith("rbp_ood_") else "sequence"
                prefix = track + "_"
                remainder = name[len(prefix):]
                arch = remainder.rsplit("_f", 1)[0]
                fold = int(remainder.rsplit("_f", 1)[1].split("_s")[0])
                seed = int(remainder.rsplit("_s", 1)[1])
            else:
                track = "rbp_ood" if name.startswith("rbp_ood_") else "sequence"
                arch = "knn"
                fold = int(name.rsplit("_f", 1)[1])
                seed = None
            config = read_json_if(os.path.join(run, "eval_config.json"))
            if not config:
                config = read_json_if(os.path.join(run, "config.json"))
            train_metrics = read_json_if(os.path.join(run, "train_metrics.json"))
            train_marker = marker_record(os.path.join(run, ".train_complete"))
            eval_marker = marker_record(os.path.join(run, ".eval_complete"))
            if kind == "runs" and not train_marker and not eval_marker:
                continue
            if kind == "knn" and not eval_marker:
                continue
            records.append({
                "run_id": f"derived-{name}", "command": "evaluate" if eval_marker else "train",
                "track": track, "arch": arch, "fold": fold, "seed": seed,
                "status": "derived_eval_complete" if eval_marker else "derived_train_complete",
                "start_utc": None, "end_utc": None, "derived_from_artifacts": True,
                "code_sha256": config.get("code_sha256"), "data_sha256": {
                    "frozen_manifest_sha256": config.get("frozen_manifest_sha256"),
                    "label_cache_sha256": config.get("label_cache_sha256")},
                "cuda": train_metrics.get("cuda", config.get("cuda")),
                "artifacts": artifact_paths_from_run(run, kind),
                "observed_train_complete": train_marker, "observed_eval_complete": eval_marker,
            })
    return {"manifest_type": "derived_from_artifacts", "derived_from_artifacts": True,
            "historical_start_end_fabricated": False, "start_utc": None, "end_utc": None,
            "generated_utc": utc_now(), "root": root, "report_pointer": report_pointer,
            "n_records": len(records), "records": records}


def artifact_paths_from_run(run, kind):
    result = {"history": [], "checkpoint": [], "validation_predictions": [],
              "test_predictions": [], "metrics": []}
    for key, name in (("history", "validation_history.tsv"), ("checkpoint", "best.pt"),
                      ("validation_predictions", "validation_predictions.npz"),
                      ("test_predictions", "test_predictions.npz"), ("metrics", "test_metrics.tsv")):
        path = os.path.join(run, name)
        if os.path.isfile(path):
            result[key].append(path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-pointer", default=REPORT_POINTER)
    args = parser.parse_args()
    manifest = derived_manifest(args.root, args.report_pointer)
    atomic_json(args.out, manifest)
    print("DERIVED_FROM_ARTIFACTS_MANIFEST_OK", args.out, manifest["n_records"])


if __name__ == "__main__":
    main()
