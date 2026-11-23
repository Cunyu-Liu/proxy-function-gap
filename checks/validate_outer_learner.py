import argparse
import json
from pathlib import Path

from proxy_function_gap.selection_eval import SelectionRecord, selection_config_hash
from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.selection_dir)
    reports = []
    for index in range(args.count):
        pool_id = f"{args.role}_{index:03d}"
        path = root / f"{pool_id}.json"
        payload = json.loads(path.read_text())
        records = [SelectionRecord(**row) for row in payload["records"]]
        assert len(records) == 875 and selection_config_hash(records) == payload["selection_config_sha256"]
        assert all(row.regret is None for row in records)
        metadata = payload["execution_metadata"]
        assert metadata["actual_cuda_inference"] and not metadata["outer_labels_read"] and not metadata["outer_metrics_computed"]
        assert metadata["q_feature_member_item_forwards"] == 3 * (64 if args.role == "outer_designed" else 256)
        q_path = root / f"{pool_id}_q.json"
        decisions = json.loads(q_path.read_text())
        assert decisions["selection_config_sha256"] == payload["selection_config_sha256"] and not decisions["outer_labels_read"]
        assert len(decisions["decisions"]) == 5
        assert all(0 <= row["q_probability"] <= 1 for row in decisions["decisions"])
        reports.append({"pool_id": pool_id, "selection_sha256": sha256_file(path), "q_sha256": sha256_file(q_path), "records": len(records)})
    report = {"pass": True, "pools": reports, "records": sum(row["records"] for row in reports), "outer_labels_read": False, "outer_metrics_computed": False}
    (root / "learner_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": True, "pools": len(reports), "records": report["records"], "outer_labels_read": False}))


if __name__ == "__main__":
    main()
