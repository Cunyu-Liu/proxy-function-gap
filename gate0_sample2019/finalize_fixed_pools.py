import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from .common import load_config, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["output_root"])
    for name in ["code_provenance.json", "integration_manifest.json", "fixed_pool_manifest.json", "delivery_checksums.json"]:
        if (root / name).exists():
            raise FileExistsError(root / name)
    verification = json.loads((root / "verification_summary.json").read_text())
    if not all(verification["rounds"].values()) or verification["E1_enabled"]:
        raise ValueError("Fixed pool verification failed")
    repository = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    if status:
        raise ValueError(f"Staging repository is not clean: {status}")
    build = root / config["build_names"][0]
    source = json.loads((build / "pool_manifest.json").read_text())
    code_files = sorted(list((repository / "gate0_sample2019").glob("*.py")) + list((repository / "tests").glob("*.py")) + list((repository / "configs").glob("*.json")) + [repository / "README.md"])
    provenance = {
        "git_repository": str(repository),
        "git_commit": commit,
        "git_status_clean": True,
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_sha256": sha256_file(Path(args.config)),
        "source_pool_manifest_sha256": sha256_file(build / "pool_manifest.json"),
        "code_sha256": {str(path.relative_to(repository)): sha256_file(path) for path in code_files},
    }
    (root / "code_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    manifest = dict(source)
    manifest["git_commit"] = commit
    manifest["selected_build"] = str(build)
    manifest["verification_rounds"] = verification["rounds"]
    (root / "fixed_pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    integration = {
        "staging_repository": str(repository),
        "staging_branch": "staging",
        "staging_commit": commit,
        "code_files": [str(path.relative_to(repository)) for path in code_files],
        "fixed_pool_manifest": str(root / "fixed_pool_manifest.json"),
        "verification_summary": str(root / "verification_summary.json"),
        "code_provenance": str(root / "code_provenance.json"),
        "E1_enabled": False,
        "outer_scoring_performed": False,
    }
    (root / "integration_manifest.json").write_text(json.dumps(integration, indent=2) + "\n")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "delivery_checksums.json":
            files[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (root / "delivery_checksums.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps({"fixed_pool_manifest": str(root / "fixed_pool_manifest.json"), "git_commit": commit, "checksummed_files": len(files)}))


if __name__ == "__main__":
    main()
