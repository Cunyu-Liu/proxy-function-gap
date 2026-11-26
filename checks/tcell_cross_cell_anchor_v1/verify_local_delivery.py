from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    forbidden = []
    allowed = {".json", ".py", ".md", ".log", ".txt"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix not in allowed:
            forbidden.append(str(path.relative_to(root)))
    if forbidden:
        raise ValueError(f"non-sanitized export files: {forbidden}")
    results = json.loads((root / "target" / "results.json").read_text())
    assert results["evaluation_count"] == 1
    assert results["primary"]["success"] is False
    assert results["sequence_independent"] is False and results["new_blind_test"] is False
    rounds = [json.loads((root / "verification" / name).read_text())["status"] for name in ("round1_preflight.json", "round2_independent.json", "round3_delivery.json")]
    assert rounds == ["PASS", "PASS", "PASS"]
    report = {
        "status": "PASS", "round": "3_local",
        "forbidden_sequence_array_or_raw_files": [],
        "three_remote_rounds_pass": True,
        "negative_result_preserved": True,
        "evidence_boundary_preserved": True,
        "local_contains_only_aggregate_json_code_logs_and_verification": True,
    }
    output = root / "verification" / "round3_local_delivery.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
