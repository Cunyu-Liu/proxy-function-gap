import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


files = {}
for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue
    relative = str(path.relative_to(ROOT))
    if relative in {"artifact_manifest.json", "verification/round3_delivery.json"} or "__pycache__" in relative or path.name == ".DS_Store":
        continue
    files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
manifest = {"schema_version": "scenario_gfp_v1_local_desensitized_delivery", "files": files, "excluded_remote_only": ["raw CSV and ZIP", "learner sequence views", "evaluator sequence/label vaults", "prediction NPZ arrays", "model checkpoints"]}
(ROOT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"file_count": len(files)}, indent=2))
