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
    for name in ["code_provenance.json", "integration_manifest_v2.json", "split_delivery_v2.json", "delivery_checksums.json"]:
        if (output / name).exists():
            raise FileExistsError(output / name)
    verification = json.loads((output / "verification_summary_v2.json").read_text())
    if not all(verification["rounds"].values()) or not verification["original_E1_authorized"] or verification["retrained_E1_authorized"]:
        raise ValueError("V2 verification or model-specific authorization is invalid")
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
        "split_manifest_sha256": sha256_file(output / "split_manifest_v2.json"),
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
            "split_manifest_v2.json", "original_evolution_role_manifest.json",
            "retrained_evolution_role_manifest.json", "e1_pool_manifest.json",
            "e1_authorization.json", "verification_summary_v2.json", "code_provenance.json",
        ],
        "model_specific_files": ["role_assignments_v2.tsv.gz", "model_specific_allocation_units.tsv.gz", "control_roles.tsv.gz", "replicate_context_summary.tsv"],
        "original_E1_authorized": True,
        "retrained_E1_authorized": False,
        "outer_metrics_computed": False,
    }
    (output / "integration_manifest_v2.json").write_text(json.dumps(integration, indent=2) + "\n")
    delivery = {
        "status": "MODEL_SPECIFIC_SPLIT_V2_VERIFIED_NO_OUTER_METRICS",
        "split_version": verification["split_version"],
        "split_hash": verification["split_hash"],
        "git_commit": commit,
        "verification_rounds": verification["rounds"],
        "original_E1_authorized": True,
        "original_E1_scope": "Fixed-pool eGFP unmod designed single-assay, original_evolution, historical measured-state only",
        "original_Paper1_historical_replay_authorized": verification["original_Paper1_historical_replay_authorized"],
        "retrained_E1_authorized": False,
        "retrained_NO_GO_reason": "2695 retraining sequence list and linked lineage unresolved",
        "outer_metrics_computed": False,
        "outer_labels_returned": False,
        "proxy_training_performed": False,
        "remaining_blockers": [
            "No outer metric has been computed; original authorization is limited to the declared fixed-pool historical measured-state context.",
            "Retrained remains NO-GO until the 2695 sequence list and all linked components are resolved.",
            "Rep2, modified RNA and mCherry are outside original fixed-pool E1.",
            "Step sequences are Paper1 historical measured-state replay only, never never-evaluated.",
        ],
    }
    (output / "split_delivery_v2.json").write_text(json.dumps(delivery, indent=2) + "\n")
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "delivery_checksums.json":
            files[str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (output / "delivery_checksums.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps({"status": delivery["status"], "split_version": delivery["split_version"], "split_hash": delivery["split_hash"], "git_commit": commit, "checksummed_files": len(files)}))


if __name__ == "__main__":
    main()
