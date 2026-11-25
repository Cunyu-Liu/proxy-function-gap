import json
import platform
import sys
from pathlib import Path

from persist_mrl import CODE, ROOT, dump_json, sha256, write_manifest


def main():
    code_files = sorted(p for p in CODE.rglob("*") if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)
    dump_json({str(p.relative_to(CODE)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in code_files}, ROOT / "verification" / "code_manifest.json")
    dump_json({"python": sys.version, "platform": platform.platform(), "executable": sys.executable, "working_directory": str(CODE), "code_pushed": False}, ROOT / "verification" / "environment.json")
    manifest = write_manifest()
    print(json.dumps({"status": "PASS", "artifact_files": len(manifest)}))


if __name__ == "__main__":
    main()
