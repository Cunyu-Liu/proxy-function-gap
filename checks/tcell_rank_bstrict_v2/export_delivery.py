from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["artifact_root"])
    export = root / "export"
    export.mkdir(exist_ok=True)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or export in path.parents:
            continue
        relative = path.relative_to(root)
        if path.suffix == ".npy" or "target" in relative.parts and path.suffix in {".tsv", ".csv"}:
            continue
        if relative.parts[0] in {"diagnostics", "configs", "logs", "verification"} or path.name in {"selection.json", "fits.json", "gate.json", "pair_manifest.json", "auxiliary_gate.json", "auxiliary_gate_passed.json"}:
            files.append((path, Path("artifacts") / relative))
    for path in sorted(Path(args.workspace).rglob("*")):
        if not path.is_file() or ".pytest_cache" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix == ".py" and "scripts" not in path.parts:
            relative = path.relative_to(args.workspace)
        elif path.name == "rank_frozen.json" and path.parent.name == "configs":
            relative = Path("configs") / path.name
        elif path.name == "diagnose_v1.py":
            relative = Path("scripts") / path.name
        else:
            continue
        if "tests/rank_frozen.json" in str(relative):
            continue
        files.append((path, relative))
    seen = set()
    manifest = []
    for source, relative in files:
        if relative in seen:
            continue
        seen.add(relative)
        destination = export / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        manifest.append({"path": str(relative), "sha256": digest(destination), "bytes": destination.stat().st_size})
    forbidden = []
    for row in manifest:
        path = export / row["path"]
        if path.suffix not in {".json", ".log", ".txt"}:
            continue
        text = path.read_text(errors="replace")
        if "sequence_strings" in text or "per_sequence_labels" in text:
            forbidden.append(row["path"])
    if forbidden:
        raise ValueError(f"forbidden export fields: {forbidden}")
    (export / "artifact_manifest.json").write_text(json.dumps({"files": manifest, "file_count": len(manifest), "forbidden_fields": forbidden}, indent=2) + "\n")
    print(json.dumps({"export": str(export), "file_count": len(manifest), "forbidden_fields": forbidden}))


if __name__ == "__main__":
    main()
