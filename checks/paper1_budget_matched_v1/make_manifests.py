#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    source = args.artifact / "source_snapshot"
    source.mkdir(exist_ok=True)
    for name in ("run_analysis.py", "verify_independent.py", "build_report.py", "export_source_data.py", "make_manifests.py", "verify_delivery.py"):
        source.joinpath(name).write_bytes((args.workspace / "checks/paper1_budget_matched_v1" / name).read_bytes())
    source.joinpath("test_budget_matched.py").write_bytes((args.workspace / "tests/paper1_budget_matched_v1/test_budget_matched.py").read_bytes())
    args.artifact.joinpath("frozen_config.json").write_bytes((args.workspace / "checks/paper1_budget_matched_v1/frozen_config.json").read_bytes())
    artifact_files = sorted(path for path in args.artifact.rglob("*") if path.is_file() and path.name not in {"artifact_manifest.json", "delivery_manifest.json"})
    artifact_manifest = {
        "schema_version": "1.0",
        "artifact_root": str(args.artifact),
        "workspace": str(args.workspace),
        "files": [{"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size} for path in artifact_files],
    }
    args.artifact.joinpath("artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2) + "\n")
    delivery = []
    local_root = "deliverables/reports/source_data/paper1_budget_matched_v1"
    for path in artifact_files:
        relative = path.relative_to(args.artifact)
        if relative == Path("analysis/paired_records.tsv") or relative.parts[0] == "analysis":
            continue
        if relative == Path("PAPER1_BUDGET_MATCHED_V1.md"):
            local = "deliverables/reports/docs/PAPER1_BUDGET_MATCHED_V1.md"
        elif relative.parts[0] == "export":
            local = f"{local_root}/{'/'.join(relative.parts[1:])}"
        else:
            local = f"{local_root}/{relative.as_posix()}"
        delivery.append({"remote_path": str(path), "local_path": local, "sha256": digest(path), "bytes": path.stat().st_size})
    payload = {
        "schema_version": "1.0",
        "evidence_boundary": "retrospective_sensitivity_not_new_blind_test",
        "contains_sequence_strings": False,
        "contains_sequence_level_measured_labels": False,
        "remote_only_selected_id_regret_audit": "analysis/paired_records.tsv",
        "files": sorted(delivery, key=lambda row: row["local_path"]),
    }
    args.artifact.joinpath("delivery_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "artifact_files": len(artifact_files), "delivery_files": len(delivery)}))


if __name__ == "__main__":
    main()
