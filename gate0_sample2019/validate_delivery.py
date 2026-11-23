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
    target = output / "final_validation.json"
    if target.exists():
        raise FileExistsError(target)
    checksums = json.loads((output / "delivery_checksums.json").read_text())
    mismatches = []
    for name, expected in checksums.items():
        path = output / name
        if not path.exists() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            mismatches.append(name)
    manifest = json.loads(Path(config["input_manifest"]).read_text())
    raw_mismatches = [item["name"] for item in manifest["files"] if sha256_file(Path(item["path"])) != item["sha256"]]
    provenance = json.loads((output / "code_provenance.json").read_text())
    repository = Path(provenance["git_repository"])
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    delivery = json.loads((output / "gate0_delivery.json").read_text())
    result = {
        "derived_checksum_count": len(checksums),
        "derived_checksum_mismatches": mismatches,
        "raw_hash_mismatches": raw_mismatches,
        "staging_commit": commit,
        "provenance_commit_matches": commit == provenance["git_commit"],
        "staging_status_clean": not status,
        "verification_rounds": delivery["verification_rounds"],
        "E1_authorized": delivery["E1_authorized"],
        "outer_results_accessed": delivery["outer_results_accessed"],
        "proxy_training_performed": delivery["proxy_training_performed"],
        "E1_evaluation_performed": delivery["E1_evaluation_performed"],
    }
    result["pass"] = not mismatches and not raw_mismatches and result["provenance_commit_matches"] and result["staging_status_clean"] and all(result["verification_rounds"].values()) and not result["E1_authorized"] and not result["outer_results_accessed"] and not result["proxy_training_performed"] and not result["E1_evaluation_performed"]
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["pass"]:
        raise ValueError("Final delivery validation failed")


if __name__ == "__main__":
    main()
