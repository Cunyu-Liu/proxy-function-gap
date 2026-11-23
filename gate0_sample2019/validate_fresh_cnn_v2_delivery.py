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
    child = json.loads((root / "child_manifest.json").read_text())
    production = json.loads((Path(child["selected_build"]) / "data_manifest.json").read_text())
    template = json.loads((Path(child["selected_build"]) / "model_lineage_template.json").read_text())
    result = {
        "checksummed_files": len(checksums),
        "checksum_mismatches": mismatches,
        "staging_commit": commit,
        "provenance_commit_matches": commit == provenance["git_commit"],
        "staging_status_clean": not status,
        "split_v1_manifest_hash_matches": sha256_file(Path(config["parent_split_v1_dir"]) / "split_manifest.json") == child["parent_split_v1_manifest_sha256"],
        "split_v2_manifest_hash_matches": sha256_file(Path(config["parent_split_v2_dir"]) / "split_manifest_v2.json") == child["parent_split_v2_manifest_sha256"],
        "data_manifest_hash_matches": sha256_file(Path(config["input_manifest"])) == child["data_manifest_sha256"],
        "production_manifest_hash_matches": sha256_file(Path(child["selected_build"]) / "data_manifest.json") == provenance["production_data_manifest_sha256"],
        "formal_training_start_authorized": child["formal_training_start_authorized"],
        "E1_enabled": child["E1_enabled"],
        "official_weights": child["official_weights"],
        "distillation": child["distillation"],
        "synthetic": production["synthetic"],
        "model_evidence_not_prefilled": template["training_status"] == "NOT_STARTED" and template["model_hash"] is None and template["epoch"] is None and template["convergence"] is None and template["V_calibration"] is None and template["Q_calibration"] is None and not template["configuration_frozen"],
        "outer_metrics_computed": child["outer_metrics_computed"],
    }
    result["pass"] = not mismatches and result["provenance_commit_matches"] and result["staging_status_clean"] and result["split_v1_manifest_hash_matches"] and result["split_v2_manifest_hash_matches"] and result["data_manifest_hash_matches"] and result["production_manifest_hash_matches"] and result["formal_training_start_authorized"] and not result["E1_enabled"] and not result["official_weights"] and not result["distillation"] and result["synthetic"] is False and result["model_evidence_not_prefilled"] and not result["outer_metrics_computed"]
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["pass"]:
        raise ValueError("Fresh CNN v2 final delivery validation failed")


if __name__ == "__main__":
    main()
