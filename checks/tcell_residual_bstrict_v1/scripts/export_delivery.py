from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["artifact_root"])
    export = root / "export"
    export.mkdir(exist_ok=False)
    files = {}
    sources = [Path(args.config), root / "raw_preflight.json", root / "auxiliary_prepare_summary.json", root / "auxiliary_gate.json"]
    for direction in ("n50_hek293t_r1_to_defined_hepg2_r1", "defined_hepg2_r1_to_n50_hek293t_r1"):
        base = root / "auxiliary" / direction
        sources.extend([base / "ridge_fits.json", base / "selection.json", base / "gate.json"])
    for context in ("n50_hek293t_r1", "defined_hepg2_r1"):
        base = root / "auxiliary" / context
        sources.extend([base / "prepare_manifest.json", base / "split_manifest.json"])
    for path in (root / "verification").glob("*.json"):
        sources.append(path)
    for path in (root / "logs").glob("*"):
        if path.is_file():
            sources.append(path)
    if (root / "target" / "summary.json").exists():
        sources.extend([root / "target" / "target_freeze.json", root / "target" / "target_selection_freeze.json", root / "target" / "summary.json", root / "target" / "evaluation_complete.json"])
    for source in sources:
        if not source.is_file():
            continue
        relative = Path("artifacts") / source.relative_to(root) if source.is_relative_to(root) else Path("configs") / source.name
        destination = export / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[str(relative)] = sha(source)
    for source in sorted((Path(args.workspace) / "src" / "proxy_function_gap").glob("residual*.py")):
        destination = export / "code" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[str(destination.relative_to(export))] = sha(source)
    for source in sorted((Path(args.workspace) / "tests").glob("test_residual*.py")):
        destination = export / "tests" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[str(destination.relative_to(export))] = sha(source)
    for source in sorted((Path(args.workspace) / "scripts").glob("*.py")):
        destination = export / "scripts" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[str(destination.relative_to(export))] = sha(source)
    (export / "artifact_manifest.json").write_text(json.dumps({"files": files, "sequence_strings": 0, "sequence_level_labels": 0, "sequence_level_predictions": 0}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
