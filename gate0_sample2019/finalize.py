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
    output = Path(config["output_dir"])
    for name in ["code_provenance.json", "gate0_delivery.json", "delivery_checksums.json"]:
        if (output / name).exists():
            raise FileExistsError(output / name)
    verification = json.loads((output / "verification_summary.json").read_text())
    if not all(verification["rounds"].values()):
        raise ValueError("Verification rounds are not all passing")
    repository = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    if status:
        raise ValueError(f"Staging repository is not clean: {status}")
    code_files = sorted(list((repository / "gate0_sample2019").glob("*.py")) + list((repository / "tests").glob("*.py")) + list((repository / "configs").glob("*.json")) + [repository / "README.md", Path(args.config).resolve()])
    provenance = {
        "git_repository": str(repository),
        "git_commit": commit,
        "git_status_clean": True,
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_sha256": sha256_file(Path(args.config)),
        "input_manifest_sha256": sha256_file(Path(config["input_manifest"])),
        "code_sha256": {str(path.relative_to(repository)): sha256_file(path) for path in code_files},
        "verification_summary_sha256": sha256_file(output / "verification_summary.json"),
    }
    (output / "code_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    delivery = {
        "status": "GATE0_SPLIT_READY_ARTIFACTS_VERIFIED_PROPOSED_NOT_FROZEN",
        "git_commit": commit,
        "normalization_version": config["normalization_version"],
        "verification_rounds": verification["rounds"],
        "rows": verification["rows"],
        "missing_rl_excluded": verification["missing_rl_excluded"],
        "unique_normalized_sequences": verification["unique_normalized_sequences"],
        "split_ready_groups": verification["split_ready_groups"],
        "near_duplicate_edges": verification["near_duplicate_edges"],
        "outer_results_accessed": False,
        "proxy_training_performed": False,
        "E1_evaluation_performed": False,
        "E1_authorized": False,
        "remaining_blockers": verification["remaining_blockers"],
    }
    (output / "gate0_delivery.json").write_text(json.dumps(delivery, indent=2) + "\n")
    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "delivery_checksums.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (output / "delivery_checksums.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps(delivery))


if __name__ == "__main__":
    main()
