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
    for name in ["code_provenance.json", "integration_manifest.json", "child_manifest.json", "delivery_checksums.json"]:
        if (root / name).exists():
            raise FileExistsError(root / name)
    verification = json.loads((root / "verification_summary.json").read_text())
    if not all(verification["rounds"].values()) or not verification["formal_training_start_authorized"] or verification["E1_enabled"]:
        raise ValueError("Fresh CNN v2 verification failed")
    repository = Path(__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    if status:
        raise ValueError(f"Staging repository is not clean: {status}")
    build = root / config["build_names"][0]
    candidate = json.loads((build / "candidate_manifest.json").read_text())
    outer = json.loads((build / "outer_manifest.json").read_text())
    production = json.loads((build / "data_manifest.json").read_text())
    code_files = sorted(list((repository / "gate0_sample2019").glob("*.py")) + list((repository / "tests").glob("*.py")) + list((repository / "configs").glob("*.json")) + [repository / "README.md"])
    provenance = {
        "git_repository": str(repository),
        "git_commit": commit,
        "git_status_clean": True,
        "python_version": sys.version,
        "platform": platform.platform(),
        "config_sha256": sha256_file(Path(args.config)),
        "candidate_manifest_sha256": sha256_file(build / "candidate_manifest.json"),
        "production_data_manifest_sha256": sha256_file(build / "data_manifest.json"),
        "code_sha256": {str(path.relative_to(repository)): sha256_file(path) for path in code_files},
    }
    (root / "code_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    child = {
        "schema_version": "2.0",
        "child_split_version": config["child_split_version"],
        "child_lineage_version": config["child_lineage_version"],
        "lineage_hash": candidate["lineage_hash"],
        "git_commit": commit,
        "parent_pack_commit": config["parent_pack_commit"],
        "parent_pack_manifest_sha256": candidate["parent_pack_manifest_sha256"],
        "parent_split_v1_manifest_sha256": candidate["parent_split_v1_manifest_sha256"],
        "parent_split_v2_manifest_sha256": candidate["parent_split_v2_manifest_sha256"],
        "data_manifest_sha256": candidate["data_manifest_sha256"],
        "selected_build": str(build),
        "train_scaler_rows": candidate["train_scaler_rows"],
        "train_scaler_row_hash": candidate["train_scaler_row_hash"],
        "dev_rows": candidate["dev_rows"],
        "calibration_rows": candidate["calibration_rows"],
        "query_rows": candidate["query_rows"],
        "outer_manifest": str(build / "outer_manifest.json"),
        "outer_row_hashes": outer["outer_row_hashes"],
        "outer_strata": outer["outer_strata"],
        "fresh_main_random_unmod_rep1_rows": outer["fresh_main_random_unmod_rep1_rows"],
        "designed_outer_rows": outer["designed_outer_rows"],
        "random_outer_rows": outer["random_outer_rows"],
        "production_data_manifest": str(build / "data_manifest.json"),
        "production_files": production["production_files"],
        "model_lineage_template": str(build / "model_lineage_template.json"),
        "learner_allowlist": str(build / "learner_allowlist.json"),
        "initialization": "fresh_random",
        "official_weights": False,
        "distillation": False,
        "independence_rule": "Fresh independently re-reviews random outer; original/retrained manifests are unchanged and their blanket random exclusions are not inherited.",
        "formal_training_start_authorized": True,
        "E1_enabled": False,
        "authorization_conditions": "Training may start after this verified manifest. E1 remains disabled until model hash, epoch, convergence, V/Q calibration and configuration freeze are all evidenced.",
        "historical_evaluation_status": "historical_measured_state",
        "never_evaluated": False,
        "outer_metrics_computed": False,
        "proxy_training_performed": False,
        "synthetic": False,
        "train_statistics": config["train_statistics"],
    }
    (root / "child_manifest.json").write_text(json.dumps(child, indent=2) + "\n")
    integration = {
        "staging_repository": str(repository),
        "staging_branch": "staging",
        "staging_commit": commit,
        "code_files": [str(path.relative_to(repository)) for path in code_files],
        "child_manifest": str(root / "child_manifest.json"),
        "production_data_manifest": str(build / "data_manifest.json"),
        "verification_summary": str(root / "verification_summary.json"),
        "code_provenance": str(root / "code_provenance.json"),
        "formal_training_start_authorized": True,
        "E1_enabled": False,
        "outer_metrics_computed": False,
        "synthetic": False,
    }
    (root / "integration_manifest.json").write_text(json.dumps(integration, indent=2) + "\n")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "delivery_checksums.json":
            files[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (root / "delivery_checksums.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps({"child_manifest": str(root / "child_manifest.json"), "git_commit": commit, "formal_training_start_authorized": True, "E1_enabled": False, "checksummed_files": len(files)}))


if __name__ == "__main__":
    main()
