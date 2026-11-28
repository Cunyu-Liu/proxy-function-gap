import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return math.fsum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["manifest", "verify"])
    parser.add_argument("--delivery-root", required=True)
    args = parser.parse_args()
    root = Path(args.delivery_root)
    manifest_path = root / "artifact_manifest.json"
    if args.command == "manifest":
        files = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in {"artifact_manifest.json", "round3_delivery.json"}:
                continue
            relative = path.relative_to(root).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest = {
            "schema_version": "1.0",
            "experiment": "scenario_aav_v1",
            "raw_sequences_included": False,
            "raw_source_csv_included": False,
            "checkpoints_included": False,
            "files": files,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"files": len(files), "manifest": str(manifest_path)}, indent=2))
        return

    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        path = root / relative
        assert path.is_file(), relative
        assert path.stat().st_size == expected["bytes"], relative
        assert sha256(path) == expected["sha256"], relative
    assert not manifest["raw_sequences_included"]
    assert not any(relative.startswith("raw/") for relative in manifest["files"])
    assert not any(relative.endswith(".pt") for relative in manifest["files"])
    assert not any("test_predictions" in relative or "test_features" in relative or "test_labels" in relative for relative in manifest["files"])

    results = json.loads((root / "results" / "main_results.json").read_text())
    audit = json.loads((root / "results" / "data_audit.json").read_text())
    round1 = json.loads((root / "verification" / "round1_raw.json").read_text())
    round2 = json.loads((root / "verification" / "round2_independent.json").read_text())
    assert round1["status"] == "PASS" and all(round1["checks"].values())
    assert round2["status"] == "PASS" and all(round2["checks"].values())
    assert audit["raw"]["exact_overlap"] == 9993
    assert audit["split"]["cross_split_hamming_leq1"] == 0
    assert results["primary_cap"] == 16
    assert results["retrospective_public_data"] and not results["new_blinded_validation"]
    assert not results["meeting6_used"]
    assert "not tropism" in results["endpoint_semantics"]

    cap_checks = {}
    for cap_text, expected in results["caps"].items():
        path = root / "evaluation" / f"cap{cap_text}_pools.tsv"
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            forbidden = {"AA", "aa", "sequence", "sequences", "CodonRep1", "CodonRep2"}
            assert not forbidden.intersection(reader.fieldnames or [])
            rows = list(reader)
        assert len(rows) == expected["pools"]
        bon = [float(row["bon_regret"]) for row in rows]
        random = [float(row["random_regret"]) for row in rows]
        delta = [b - r for b, r in zip(bon, random)]
        assert abs(mean(bon) - expected["bon_mean_regret"]) < 1e-9
        assert abs(mean(random) - expected["random_mean_regret"]) < 1e-9
        assert abs(mean(delta) - expected["paired_delta_bon_minus_random"]) < 1e-9
        for row in rows:
            assert abs(float(row["oracle_measured_production"]) - float(row["bon_measured_production"]) - float(row["bon_regret"])) < 1e-9
            assert abs(float(row["oracle_measured_production"]) - float(row["random_measured_production"]) - float(row["random_regret"])) < 1e-9
        cap_checks[cap_text] = {"rows": len(rows), "mean_arithmetic_verified": True}

    report = (root / "REPORT.md").read_text().replace("−", "-")
    primary_delta = results["caps"]["16"]["paired_delta_bon_minus_random"]
    assert f"{primary_delta:.6f}" in report
    assert "回顾性" in report and "不是 tropism" in report
    verification = {
        "round": "delivery",
        "status": "PASS",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "manifest_files_verified": len(manifest["files"]),
        "raw_or_codon_sequences_in_local_delivery": False,
        "round1_and_round2_pass_required": True,
        "cap_arithmetic": cap_checks,
        "primary_cap16_delta": primary_delta,
        "report_boundaries_verified": True,
    }
    (root / "verification" / "round3_delivery.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
