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
    for name in ["code_provenance.json", "integration_manifest.json", "split_delivery.json", "delivery_checksums.json"]:
        if (output / name).exists():
            raise FileExistsError(output / name)
    verification = json.loads((output / "verification_summary.json").read_text())
    if not all(verification["rounds"].values()) or verification["E1_authorized"]:
        raise ValueError("Split verification is incomplete or E1 was authorized")
    repository = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    if status:
        raise ValueError(f"Staging repository is not clean: {status}")
    code_files = sorted(list((repository / "gate0_sample2019").glob("*.py")) + list((repository / "tests").glob("*.py")) + list((repository / "configs").glob("*.json")) + [repository / "README.md"])
    provenance = {
        "git_repository": str(repository),
        "git_commit": commit,
        "git_status_clean": True,
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_sha256": sha256_file(Path(args.config)),
        "split_manifest_sha256": sha256_file(output / "split_manifest.json"),
        "code_sha256": {str(path.relative_to(repository)): sha256_file(path) for path in code_files},
    }
    (output / "code_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    integration = {
        "staging_repository": str(repository),
        "staging_branch": "staging",
        "staging_commit": commit,
        "code_files": [str(path.relative_to(repository)) for path in code_files],
        "derived_root": str(output),
        "required_manifests": [
            "split_manifest.json", "role_manifest.json", "learner_allowlist.json",
            "evaluator_manifest.json", "verification_summary.json", "code_provenance.json",
        ],
        "protocol_entrypoints": [
            "gate0_sample2019.protocol.Learner",
            "gate0_sample2019.protocol.Evaluator",
            "gate0_sample2019.protocol.create_selection",
        ],
        "E1_authorized": False,
    }
    (output / "integration_manifest.json").write_text(json.dumps(integration, indent=2) + "\n")
    delivery = {
        "status": "SPLIT_AND_VAULT_FROZEN_VERIFIED_E1_NOT_AUTHORIZED",
        "split_version": verification["split_version"],
        "split_hash": verification["split_hash"],
        "git_commit": commit,
        "verification_rounds": verification["rounds"],
        "role_counts": verification["role_counts"],
        "limitations": verification["limitations"],
        "outer_regret_computed": False,
        "outer_labels_returned": False,
        "outer_results_accessed": False,
        "proxy_training_performed": False,
        "E1_authorized": False,
        "remaining_blockers": [
            "E1 remains disabled; no outer regret or scientific evaluation has been run.",
            "Predicted MRL and any proxy/selector must be produced under the approved protocol before outer evaluation.",
            "Threshold and alpha selection is restricted to dev/calibration.",
            "Figure 3 reproduction remains pending.",
        ],
    }
    (output / "split_delivery.json").write_text(json.dumps(delivery, indent=2) + "\n")
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "delivery_checksums.json" and ".reveal.lock" not in path.name:
            files[str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (output / "delivery_checksums.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps({"status": delivery["status"], "split_version": delivery["split_version"], "split_hash": delivery["split_hash"], "git_commit": commit, "checksummed_files": len(files)}))


if __name__ == "__main__":
    main()
