from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from adaptation import digest, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["artifact_root"])
    workspace = Path(__file__).resolve().parent
    export = root / "export"
    export.mkdir(exist_ok=True)
    files = []
    def copy(source, relative):
        destination = export / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files.append(destination)

    for name in ("SPLIT_FROZEN.json", "prepare_manifest.json", "TRAINING_FROZEN.json", "TEST_PREDICTIONS_FROZEN.json"):
        copy(root / name, Path(name))
    copy(Path(args.config), Path("configs/listnet_frozen.json"))
    for directory, patterns in (
        ("selection", ("*.json",)), ("verification", ("*.json",)),
        ("target_v2", ("*.json",)), ("training", ("*/seed_*/summary.json", "*/seed_*/history.jsonl", "seed_*_completed.json")),
        ("failed_attempts", ("**/*.json", "**/*.log")),
    ):
        base = root / directory
        if not base.exists():
            continue
        for pattern in patterns:
            for source in sorted(base.glob(pattern)):
                if source.is_file():
                    copy(source, source.relative_to(root))
    for source in sorted(workspace.glob("*.py")):
        copy(source, Path("code") / source.name)
    for source in sorted((workspace / "tests").glob("*.py")):
        copy(source, Path("code/tests") / source.name)
    for source in sorted((workspace / "logs").glob("*.log")):
        copy(source, Path("logs") / source.name)
    forbidden_suffixes = {".npy", ".npz", ".pt", ".pth", ".csv", ".gz", ".tsv"}
    for path in files:
        assert path.suffix.lower() not in forbidden_suffixes
        assert path.suffix.lower() in {".json", ".jsonl", ".py", ".log"}
    manifest = {
        "analysis_id": config["analysis_id"], "file_count": len(files),
        "files": {str(path.relative_to(export)): digest(path) for path in sorted(files)},
        "forbidden_per_sequence_or_binary_exports": 0,
        "sequence_strings_exported": False, "per_sequence_labels_exported": False,
        "per_sequence_predictions_exported": False, "test_selection_indices_exported": False,
    }
    save_json(export / "manifest.json", manifest)
    print(json.dumps({"file_count": len(files), "forbidden_exports": 0, "manifest_sha256": digest(export / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
