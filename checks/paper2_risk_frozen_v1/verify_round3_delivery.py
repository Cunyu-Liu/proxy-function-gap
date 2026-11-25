from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    remote_manifest = json.loads((args.artifact_root / "delivery_manifest.json").read_text())
    local_manifest = json.loads((args.local_root / "delivery_manifest.json").read_text())
    if remote_manifest != local_manifest:
        raise ValueError("delivery manifests differ")
    checked = []
    for item in remote_manifest["files"]:
        relative = item["path"]
        local_path = args.report if relative == "PAPER2_RISK_FROZEN_V1.md" else args.local_root / relative
        if digest(local_path) != item["sha256"]:
            raise ValueError(f"local delivery hash mismatch: {relative}")
        checked.append(relative)
    frozen = json.loads((args.local_root / "frozen_protocol.json").read_text())
    report_text = args.report.read_text()
    required = [
        f"{frozen['selected_threshold']}",
        f"{frozen['calibration']['regret_reduction']:.6f}",
        "calibration 门禁失败，未执行 outer 评分",
    ]
    if any(value not in report_text for value in required):
        raise ValueError("report does not match frozen result")
    forbidden = ["outer_result.json", "outer_designed_decisions.csv", "outer_random_decisions.csv"]
    if any((args.local_root / name).exists() for name in forbidden):
        raise ValueError("unexpected outer output after failed gate")
    result = {
        "schema_version": "paper2-risk-frozen-v1-verification-round3",
        "pass": True,
        "manifest_files_checked": len(checked),
        "local_source_data_matches_teams_manifest": True,
        "local_report_matches_teams_report": True,
        "report_key_values_match_frozen_json": True,
        "outer_scored": False,
        "paper3_or_main_paper_modified_by_this_package": False,
    }
    output = args.local_root / "verification_round3_delivery.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
