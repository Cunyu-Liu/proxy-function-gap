#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

ROOTS = ["PROTOCOL.md", "REPORT.md", "frozen", "analysis/results", "source_snapshot", "verification", "logs"]
EXCLUDED = {"delivery_manifest.json", "delivery_hashes.sha256", "verification/round3_delivery.json", "verification/round3_local_delivery.json", "logs/round3_delivery.log"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def files(root):
    output = []
    for item in ROOTS:
        path = root / item
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []
        for candidate in candidates:
            relative = candidate.relative_to(root).as_posix()
            if relative not in EXCLUDED and "__pycache__" not in relative:
                output.append((relative, candidate))
    return sorted(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    manifest_path = root / "delivery_manifest.json"
    errors = []
    if args.create:
        entries = [{"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)} for relative, path in files(root)]
        manifest = {"schema_version": "1.0", "analysis": "paper1_sequence_ablation_v1", "files": entries}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (root / "delivery_hashes.sha256").write_text("".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries))
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            errors.append(f"missing: {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            errors.append(f"hash/size mismatch: {entry['path']}")
    actual = {relative for relative, _ in files(root)}
    expected = {entry["path"] for entry in manifest["files"]}
    for relative in sorted(actual - expected):
        errors.append(f"unmanifested file: {relative}")
    for relative in sorted(expected - actual):
        if not (root / relative).exists():
            continue
        errors.append(f"manifested file not enumerated: {relative}")
    report = {
        "round": 3,
        "pass": not errors,
        "manifest_files": len(manifest["files"]),
        "manifest_sha256": sha256(manifest_path),
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
