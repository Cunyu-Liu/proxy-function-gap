from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    prepare = json.loads((root / "prepare_sample_manifest.json").read_text())
    auxiliary = json.loads((root / "prepare_auxiliary_manifest.json").read_text())
    independent = json.loads((root / "verification" / "round2_independent.json").read_text())
    assert prepare["status"] == auxiliary["status"] == independent["status"] == "PASS"
    assert prepare["config_sha256"] == auxiliary["config_sha256"] == digest(config_path)
    assert not prepare["target_mrl_read"] and not auxiliary["target_mrl_read"]
    assert independent["training_runs"] == 21 and independent["maximum_absolute_difference"] <= 1e-12
    for name, expected in prepare["files"].items():
        assert digest(root / name) == expected
    gate = json.loads((root / "auxiliary_gate.json").read_text())
    target = root / "target"
    if gate["both_directions_pass"]:
        assert (target / "ONE_EVALUATION_COMPLETED.json").is_file()
        target_count = 1
    else:
        assert not target.exists()
        target_count = 0
    workspace = Path(__file__).resolve().parent
    test = subprocess.run(["/root/miniconda3/envs/lucaone_cu128/bin/python", "-m", "pytest", "-q", "tests"], cwd=workspace, text=True, capture_output=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "round3_tests.log").write_text(test.stdout + test.stderr)
    assert test.returncode == 0
    report = {"status": "PASS", "round": 3, "config_sha256": digest(config_path), "sample_and_auxiliary_hashes": True, "training_runs": 21, "independent_maximum_absolute_difference": independent["maximum_absolute_difference"], "focused_tests": test.stdout.strip(), "destination_evaluation_count_per_direction": 1, "target_evaluation_count": target_count, "fail_closed": target_count in (0, 1), "no_push": True, "evidence_boundary": config["evidence_boundary"]}
    (root / "verification" / "round3_delivery.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
