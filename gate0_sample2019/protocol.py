import csv
import datetime
import fcntl
import gzip
import hashlib
import json
import os
import re
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def parse_utc(value):
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def write_json_new(path, value):
    with open(path, "x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def create_selection(request_dir, run_id, selections, split_manifest, not_before_utc, request_role="query"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("Invalid run_id")
    request_dir = Path(request_dir)
    request_dir.mkdir(parents=True, exist_ok=False)
    selection_path = request_dir / "selection.jsonl"
    with open(selection_path, "x") as handle:
        for item in sorted(selections, key=lambda value: value["sequence_id"]):
            handle.write(json.dumps({"sequence_id": item["sequence_id"]}, sort_keys=True, separators=(",", ":")) + "\n")
    split_manifest = Path(split_manifest)
    split = json.loads(split_manifest.read_text())
    request = {
        "run_id": run_id,
        "request_role": request_role,
        "split_version": split["split_version"],
        "split_hash": split["split_hash"],
        "split_manifest_path": str(split_manifest),
        "selection_path": str(selection_path),
        "selection_hash": sha256_file(selection_path),
        "selection_count": len(selections),
        "not_before_utc": not_before_utc,
        "created_utc": utc_now().isoformat(),
        "E1_enabled": False,
    }
    request_path = request_dir / "request.json"
    write_json_new(request_path, request)
    return request_path


class Learner:
    def __init__(self, allowlist_path):
        self.allowlist_path = Path(allowlist_path)
        self.allowlist = json.loads(self.allowlist_path.read_text())

    def _role_config(self, role, label_access=False):
        key = "label_roles" if label_access else "feature_roles"
        if role not in self.allowlist[key]:
            raise PermissionError(f"Learner cannot read {'labels' if label_access else 'features'} for role {role}")
        return self.allowlist[key][role]

    def _read(self, role, label_access):
        config = self._role_config(role, label_access)
        path = Path(config["path"]).resolve()
        allowed_path = Path(config["path"]).resolve()
        if path != allowed_path or not path.is_file():
            raise PermissionError("Path is not in the Learner allowlist")
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not set(config["columns"]).issubset(reader.fieldnames or []):
                raise PermissionError("Unexpected columns in Learner role view")
            for row in reader:
                yield {column: row[column] for column in config["columns"]}

    def read_features(self, role):
        return self._read(role, False)

    def read_labels(self, role):
        return self._read(role, True)

    def read_arbitrary_path(self, path):
        requested = Path(path).resolve()
        allowed = {Path(config["path"]).resolve() for key in ("feature_roles", "label_roles") for config in self.allowlist[key].values()}
        if requested not in allowed:
            raise PermissionError("Path is not in the Learner allowlist")
        return requested

    def validate_parameter_origin(self, role):
        if role not in self.allowlist["parameter_edit_roles"]:
            raise PermissionError("Threshold and alpha values can only be edited on dev/calibration")
        return True


class Evaluator:
    def __init__(self, evaluator_manifest_path, split_manifest_path, runs_root):
        self.manifest_path = Path(evaluator_manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.split_path = Path(split_manifest_path)
        self.split = json.loads(self.split_path.read_text())
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def evaluate_outer(self, *args, **kwargs):
        raise PermissionError("Outer evaluation and regret are disabled before E1 authorization")

    def _selection(self, request, request_path):
        selection_path = Path(request["selection_path"]).resolve()
        if selection_path.parent != Path(request_path).parent.resolve() or selection_path.name != "selection.jsonl":
            raise PermissionError("Selection path escapes the request directory")
        if sha256_file(selection_path) != request["selection_hash"]:
            raise PermissionError("Selection hash mismatch")
        selections = []
        with open(selection_path) as handle:
            for line in handle:
                item = json.loads(line)
                if set(item) != {"sequence_id"}:
                    raise ValueError("Only sequence_id is allowed in a selection")
                selections.append(item["sequence_id"])
        if len(selections) != request["selection_count"]:
            raise ValueError("Selection count mismatch")
        if not selections or len(set(selections)) != len(selections):
            raise ValueError("Selections must be nonempty and contain unique sequence IDs")
        return selections

    def _query_labels(self, selected):
        wanted = set(selected)
        found = {}
        path = self.manifest["label_vaults"]["query"]["path"]
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sequence_id = row["sequence_id"]
                if sequence_id in wanted:
                    found[sequence_id] = {
                        "sequence_id": sequence_id,
                        "measured_rl": row["measured_rl"],
                        "measured_status": row["measured_status"],
                    }
        if set(found) != wanted:
            raise PermissionError("Selection contains non-query or unknown sequences")
        return found

    def reveal(self, request_path):
        request_path = Path(request_path).resolve()
        request = json.loads(request_path.read_text())
        if request.get("request_role") != "query":
            raise PermissionError("Only query labels can be revealed")
        if request.get("E1_enabled") is not False:
            raise PermissionError("E1 must remain disabled")
        if request["split_version"] != self.split["split_version"] or request["split_hash"] != self.split["split_hash"]:
            raise PermissionError("Split version/hash mismatch")
        if utc_now() < parse_utc(request["not_before_utc"]):
            raise PermissionError("Selection reveal is not yet permitted")
        run_id = request["run_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
            raise ValueError("Invalid run_id")
        selected = self._selection(request, request_path)
        lock_path = self.runs_root / ".reveal.lock"
        with open(lock_path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run_dir = self.runs_root / run_id
            if run_dir.exists():
                raise FileExistsError("run_id already exists; repeated reveal requires a new run_id")
            index_path = self.runs_root / "index.jsonl"
            previous = []
            if index_path.exists():
                with open(index_path) as handle:
                    previous = [json.loads(line) for line in handle if line.strip()]
            repeats = [item["run_id"] for item in previous if item["selection_hash"] == request["selection_hash"]]
            labels = self._query_labels(selected)
            run_dir.mkdir()
            revealed_path = run_dir / "revealed_labels.jsonl"
            with open(revealed_path, "x") as handle:
                for sequence_id in selected:
                    handle.write(json.dumps(labels[sequence_id], sort_keys=True, separators=(",", ":")) + "\n")
            record = {
                "run_id": run_id,
                "request_role": "query",
                "split_version": request["split_version"],
                "split_hash": request["split_hash"],
                "selection_hash": request["selection_hash"],
                "selection_hash_verified_before_reveal": True,
                "request_hash": sha256_file(request_path),
                "not_before_utc": request["not_before_utc"],
                "revealed_utc": utc_now().isoformat(),
                "revealed_label_count": len(selected),
                "revealed_label_hash": sha256_file(revealed_path),
                "revealed_labels_path": str(revealed_path),
                "repeat_of_run_id": repeats[0] if repeats else None,
                "prior_run_preserved": True,
                "outer_labels_returned": False,
                "E1_enabled": False,
            }
            record_path = run_dir / "run_record.json"
            write_json_new(record_path, record)
            with open(index_path, "a") as handle:
                handle.write(json.dumps({
                    "run_id": run_id,
                    "selection_hash": request["selection_hash"],
                    "request_hash": record["request_hash"],
                    "revealed_label_hash": record["revealed_label_hash"],
                    "run_record_path": str(record_path),
                }, sort_keys=True, separators=(",", ":")) + "\n")
            return record


def read_revealed_labels(record):
    if record.get("request_role") != "query" or record.get("outer_labels_returned") is not False:
        raise PermissionError("Only query reveal records are readable")
    path = Path(record["revealed_labels_path"])
    with open(path) as handle:
        for line in handle:
            yield json.loads(line)
