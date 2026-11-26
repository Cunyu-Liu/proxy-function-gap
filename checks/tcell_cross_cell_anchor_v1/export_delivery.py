from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["artifact_root"])
    workspace = Path(__file__).resolve().parent
    export = root / "export"
    export.mkdir(parents=True, exist_ok=False)
    for name in ("prepare_manifest.json", "artifact_manifest.json"):
        shutil.copy2(root / name, export / name)
    for directory in ("configs", "verification", "failed_attempts"):
        shutil.copytree(root / directory, export / directory)
    shutil.copytree(root / "target", export / "target", ignore=shutil.ignore_patterns("*.npy", "*.npz"))
    (export / "code").mkdir(exist_ok=False)
    for name in ("anchor.py", "prepare.py", "evaluate.py", "independent_verify.py", "delivery_verify.py", "export_delivery.py", "verify_local_delivery.py"):
        shutil.copy2(workspace / name, export / "code" / name)
    shutil.copytree(workspace / "tests", export / "code" / "tests", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    shutil.copytree(workspace / "docs", export / "code" / "docs")
    shutil.copytree(workspace / "logs", export / "logs", ignore=shutil.ignore_patterns("__pycache__"))
    forbidden_suffixes = {".npy", ".npz", ".csv", ".tsv", ".gz"}
    for path in export.rglob("*"):
        if path.is_file() and path.suffix in forbidden_suffixes:
            raise ValueError(f"forbidden local export: {path}")
    print(f"Sanitized export complete: {export}")


if __name__ == "__main__":
    main()
