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
    for name in ["code_provenance.json", "integration_manifest.json", "pack_manifest.json", "delivery_checksums.json"]:
        if (root / name).exists():
            raise FileExistsError(root / name)
    verification = json.loads((root / "verification_summary.json").read_text())
    if not all(verification["rounds"].values()):
        raise ValueError("Fresh proxy verification failed")
    repository = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    if status:
        raise ValueError(f"Staging repository is not clean: {status}")
    build = root / config["build_names"][0]
    source_pack = json.loads((build / "pack_manifest.json").read_text())
    code_files = sorted(list((repository / "gate0_sample2019").glob("*.py")) + list((repository / "tests").glob("*.py")) + list((repository / "configs").glob("*.json")) + [repository / "README.md"])
    provenance = {
        "git_repository": str(repository),
        "git_commit": commit,
        "git_status_clean": True,
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_sha256": sha256_file(Path(args.config)),
        "build_manifest_sha256": sha256_file(build / "pack_manifest.json"),
        "code_sha256": {str(path.relative_to(repository)): sha256_file(path) for path in code_files},
    }
    (root / "code_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    pack = {
        "schema_version": "1.0",
        "pack_version": config["pack_version"],
        "normalization_version": config["normalization_version"],
        "split_version": config["parent_split_version"],
        "split_manifest_sha256": source_pack["split_manifest_sha256"],
        "data_manifest_sha256": source_pack["data_manifest_sha256"],
        "git_commit": commit,
        "selected_build": str(build),
        "role_counts": source_pack["role_counts"],
        "files": source_pack["files"],
        "source_access_log": str(build / "source_access_log.json"),
        "replicate_summary_rule": config["replicate_summary_rule"],
        "train_statistics": config["train_statistics"],
        "verification_rounds": verification["rounds"],
        "double_build_verified": True,
        "outer_labels_read_or_packaged": False,
        "outer_learner_package_created": False,
        "outer_metrics_computed": False,
        "proxy_training_performed": False,
    }
    (root / "pack_manifest.json").write_text(json.dumps(pack, indent=2) + "\n")
    integration = {
        "staging_repository": str(repository),
        "staging_branch": "staging",
        "staging_commit": commit,
        "code_files": [str(path.relative_to(repository)) for path in code_files],
        "pack_manifest": str(root / "pack_manifest.json"),
        "verification_summary": str(root / "verification_summary.json"),
        "code_provenance": str(root / "code_provenance.json"),
        "learner_packages": [str(build / "learner/train.tsv.gz"), str(build / "learner/dev.tsv.gz"), str(build / "learner/calibration.tsv.gz"), str(build / "learner/query_unlabeled.tsv.gz")],
        "outer_learner_package_created": False,
        "outer_labels_read_or_packaged": False,
        "train_statistics": config["train_statistics"],
    }
    (root / "integration_manifest.json").write_text(json.dumps(integration, indent=2) + "\n")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "delivery_checksums.json":
            files[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (root / "delivery_checksums.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps({"pack_version": pack["pack_version"], "git_commit": commit, "role_counts": pack["role_counts"], "checksummed_files": len(files)}))


if __name__ == "__main__":
    main()
