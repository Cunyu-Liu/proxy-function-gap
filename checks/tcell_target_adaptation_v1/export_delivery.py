from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["artifact_root"])
    export = root / "export"
    if export.exists():
        shutil.rmtree(export)
    export.mkdir()
    files = []
    for name in ("prepare_manifest.json", "SPLIT_FROZEN.json", "DEV_REPEAT_SELECTION_FROZEN.json", "TRAINING_FROZEN.json", "POST_EVALUATION_AGGREGATE_FIX.json"):
        files.append(root / name)
    for pattern in ("configs/*.json", "selection/*.json", "target/*.json", "verification/*.json", "logs/*.log", "training/*/seed_*/*.json", "training/*/seed_*/*.jsonl", "training/*completed.json", "failed_attempts/**/*.json", "failed_attempts/**/*.jsonl", "failed_attempts/**/*.log"):
        files.extend(root.glob(pattern))
    workspace = Path(__file__).resolve().parent
    code_files = ["adaptation.py", "prepare.py", "prepare_dev_repeat_labels.py", "train.py", "select_models.py", "evaluate.py", "independent_verify.py", "delivery_verify.py", "export_delivery.py", "fix_learning_curve_aggregate.py", "tests/test_adaptation.py"]
    for source in sorted(set(files)):
        if not source.is_file():
            continue
        relative = source.relative_to(root)
        destination = export / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in code_files:
        destination = export / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / name, destination)
    forbidden = [path for path in export.rglob("*") if path.is_file() and path.suffix in (".npy", ".npz", ".pt", ".gz", ".csv", ".tsv")]
    if forbidden:
        raise RuntimeError(f"forbidden export files: {forbidden}")
    manifest = {"analysis_id": config["analysis_id"], "files": {str(path.relative_to(export)): digest(path) for path in sorted(export.rglob("*")) if path.is_file()}, "forbidden_file_count": 0, "contains_sequence_strings": False, "contains_per_sequence_labels_or_predictions": False, "no_push": True}
    (export / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(manifest["files"]), "forbidden_file_count": 0}, indent=2))


if __name__ == "__main__":
    main()
