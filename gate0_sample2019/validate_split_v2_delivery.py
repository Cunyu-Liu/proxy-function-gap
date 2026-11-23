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
    output = Path(config["output_dir"])
    target = output / "final_validation_v2.json"
    if target.exists():
        raise FileExistsError(target)
    checksums = json.loads((output / "delivery_checksums.json").read_text())
    mismatches = []
    for relative, expected in checksums.items():
        path = output / relative
        if not path.exists() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            mismatches.append(relative)
    provenance = json.loads((output / "code_provenance.json").read_text())
    repository = Path(provenance["git_repository"])
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    delivery = json.loads((output / "split_delivery_v2.json").read_text())
    split = json.loads((output / "split_manifest_v2.json").read_text())
    result = {
        "checksummed_files": len(checksums),
        "checksum_mismatches": mismatches,
        "staging_commit": commit,
        "provenance_commit_matches": commit == provenance["git_commit"],
        "staging_status_clean": not status,
        "split_manifest_hash_matches": sha256_file(output / "split_manifest_v2.json") == provenance["split_manifest_sha256"],
        "data_manifest_hash_matches": sha256_file(Path(config["input_manifest"])) == split["data_manifest_sha256"],
        "parent_split_immutable": sha256_file(Path(config["parent_split_dir"]) / "split_manifest.json") == split["parent_split_manifest_sha256"],
        "verification_rounds": delivery["verification_rounds"],
        "original_E1_authorized": delivery["original_E1_authorized"],
        "retrained_E1_authorized": delivery["retrained_E1_authorized"],
        "outer_metrics_computed": delivery["outer_metrics_computed"],
        "outer_labels_returned": delivery["outer_labels_returned"],
    }
    result["pass"] = not mismatches and result["provenance_commit_matches"] and result["staging_status_clean"] and result["split_manifest_hash_matches"] and result["data_manifest_hash_matches"] and result["parent_split_immutable"] and all(result["verification_rounds"].values()) and result["original_E1_authorized"] and not result["retrained_E1_authorized"] and not result["outer_metrics_computed"] and not result["outer_labels_returned"]
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["pass"]:
        raise ValueError("Final V2 delivery validation failed")


if __name__ == "__main__":
    main()
