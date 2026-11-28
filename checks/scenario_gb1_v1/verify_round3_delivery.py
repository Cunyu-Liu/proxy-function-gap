#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    package = Path(args.package)
    report_path = Path(args.report)
    manifest = json.loads((package / "delivery_manifest.json").read_text())
    checks = {}
    file_checks = {}
    for relative, record in manifest["files"].items():
        path = package / relative
        file_checks[relative] = path.exists() and path.stat().st_size == record["bytes"] and file_hash(path) == record["sha256"]
    report_record = manifest["report"]
    file_checks["report"] = report_path.stat().st_size == report_record["bytes"] and file_hash(report_path) == report_record["sha256"]
    checks["delivery_hashes"] = all(file_checks.values())
    aggregate = json.loads((package / "aggregate/test_aggregate.json").read_text())
    round1 = json.loads((package / "verification/round1_raw.json").read_text())
    round2 = json.loads((package / "verification/round2_independent.json").read_text())
    checks["round1_pass"] = round1["pass"]
    checks["round2_pass"] = round2["pass"]
    checks["lnw_not_kd"] = aggregate["label"] == "lnW" and "lnW" in report_path.read_text() and "不是 `Kd`" in report_path.read_text()
    text = report_path.read_text()
    checks["retrospective_not_prospective"] = "回顾性" in text and "新盲测" in text
    primary = aggregate["primary_delta_regret"]
    checks["signed_result_matches_report"] = ("BoN 的 cap16 regret 低于 RandomN" in text) if primary < 0 else ("BoN 的 cap16 regret 未低于 RandomN" in text)
    checks["all_budgets_reported"] = all(f"| {budget} |" in text for budget in (1, 4, 16, 64))
    checks["auxiliary_metrics_reported"] = all(term in text for term in ("MAE", "Spearman", "top16", "top-decile"))
    forbidden_suffixes = {".csv", ".npy", ".npz", ".fasta", ".fa"}
    local_payloads = [path for path in package.rglob("*") if path.is_file() and path.suffix in forbidden_suffixes]
    checks["no_sequence_level_payloads"] = not local_payloads
    checks["negative_or_null_preserved"] = "不预设" in text and "单池" in text
    report = {"round": 3, "scope": "local sanitized delivery, hashes, narrative and aggregate consistency", "pass": all(checks.values()), "checks": checks, "file_checks": file_checks, "forbidden_payloads": list(map(str, local_payloads))}
    (package / "verification/round3_delivery.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
