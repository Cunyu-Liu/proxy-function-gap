#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
OUT = "/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_error_diagnosis_v1"
CODE = "/home/proxy-function-gap-tcell-diagnosis"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(remote):
    if remote.startswith(CODE + "/"):
        return HERE / remote[len(CODE) + 1:]
    if remote == OUT + "/REPORT.md":
        return WORKSPACE / "TCELL_ERROR_DIAGNOSIS_V1.md"
    if remote == OUT + "/PROTOCOL.md":
        return HERE / "PROTOCOL.md"
    if remote.startswith(OUT + "/export/"):
        return HERE / "analysis" / remote[len(OUT + "/export/"):]
    for directory in ("raw_audit", "verification", "logs"):
        if remote.startswith(OUT + "/" + directory + "/"):
            return HERE / directory / remote[len(OUT + "/" + directory + "/"):]
    return None


def main():
    manifest = json.loads((HERE / "artifact_manifest.json").read_text())
    assert manifest["status"] == "PASS"
    checked = 0
    for remote, record in manifest["files"].items():
        path = local_path(remote)
        if path is None:
            assert "/analysis/" in remote, remote
            continue
        assert path.is_file(), path
        assert path.stat().st_size == record["bytes"], path
        assert sha(path) == record["sha256"], path
        checked += 1
    round3 = json.loads((HERE / "verification/round3_delivery.json").read_text())
    assert round3["status"] == "PASS" and round3["rounds"]["round3_remote"] == "PASS"
    for path in HERE.rglob("*"):
        if path.is_file():
            assert path.suffix not in {".fasta", ".fastq", ".fa", ".fq"}, path
            if path.suffix in {".tsv", ".csv"}:
                header = path.open().readline()
                for forbidden in ("sequence_id", "utr50", "selected_id", "selected_index", "\ty\t"):
                    assert forbidden not in header, (path, forbidden)
    result = {"status": "PASS", "local_hash_checks": checked,
              "remote_only_analysis_files_not_exported": True,
              "candidate_labels_or_sequences_present": False,
              "round3_remote_status": round3["status"]}
    (HERE / "verification/round3_local_delivery.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
