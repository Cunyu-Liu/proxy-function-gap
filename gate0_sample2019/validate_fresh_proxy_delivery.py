import argparse
import json
import subprocess
from pathlib import Path

from .common import load_config, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["output_root"])
    target = root / "final_validation.json"
    if target.exists():
        raise FileExistsError(target)
    checksums = json.loads((root / "delivery_checksums.json").read_text())
    mismatches = []
    for relative, expected in checksums.items():
        path = root / relative
        if not path.exists() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            mismatches.append(relative)
    provenance = json.loads((root / "code_provenance.json").read_text())
    repository = Path(provenance["git_repository"])
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    pack = json.loads((root / "pack_manifest.json").read_text())
    split = json.loads((Path(config["parent_split_dir"]) / "split_manifest_v2.json").read_text())
    result = {
        "checksummed_files": len(checksums),
        "checksum_mismatches": mismatches,
        "staging_commit": commit,
        "provenance_commit_matches": commit == provenance["git_commit"],
        "staging_status_clean": not status,
        "split_manifest_hash_matches": sha256_file(Path(config["parent_split_dir"]) / "split_manifest_v2.json") == split["parent_split_manifest_sha256"],
        "data_manifest_hash_matches": sha256_file(Path(config["input_manifest"])) == pack["data_manifest_sha256"],
        "outer_labels_read_or_packaged": pack["outer_labels_read_or_packaged"],
        "outer_learner_package_created": pack["outer_learner_package_created"],
        "outer_metrics_computed": pack["outer_metrics_computed"],
        "proxy_training_performed": pack["proxy_training_performed"],
        "train_statistics": pack["train_statistics"],
    }
    result["pass"] = not mismatches and result["provenance_commit_matches"] and result["staging_status_clean"] and result["split_manifest_hash_matches"] and result["data_manifest_hash_matches"] and not result["outer_labels_read_or_packaged"] and not result["outer_learner_package_created"] and not result["outer_metrics_computed"] and not result["proxy_training_performed"] and result["train_statistics"] == "DEFERRED_TO_TRAINER"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["pass"]:
        raise ValueError("Fresh proxy final delivery validation failed")


if __name__ == "__main__":
    main()
