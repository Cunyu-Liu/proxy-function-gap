import argparse
import json
from pathlib import Path

import numpy as np

from proxy_function_gap.pool_manifest import load_pool_manifest
from proxy_function_gap.selection_eval import SelectionRecord, selection_config_hash
from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--role", choices=["dev", "calibration"], default="dev")
    args = parser.parse_args()
    root = Path(args.selection_dir)
    summaries = []
    errors = []
    for index in range(args.count):
        pool_id = f"{args.role}_{index:03d}"
        path = root / f"{pool_id}.json"
        pool = load_pool_manifest(Path(args.pool_root) / pool_id / "pool_manifest.json")
        payload = json.loads(path.read_text())
        records = [SelectionRecord(**{key: value for key, value in row.items() if key in SelectionRecord.__dataclass_fields__}) for row in payload["records"]]
        if len(records) != 875 or selection_config_hash(records) != payload["selection_config_sha256"]:
            errors.append(f"record_count_or_hash:{pool_id}")
        identities = [(row.pool_id, row.strategy, row.alpha, row.proposal_cap, row.seed) for row in records]
        if len(set(identities)) != len(identities):
            errors.append(f"duplicate_record:{pool_id}")
        metadata = payload["execution_metadata"]
        if metadata["pool_manifest_sha256"] != pool.manifest_sha256 or metadata["outer_labels_read"] or metadata["outer_metrics_computed"] or not metadata["actual_cuda_inference"]:
            errors.append(f"metadata_boundary:{pool_id}")
        if metadata.get("cost_ledger_schema_version") != "2.0" or metadata.get("actual_cost_spent") != sum(row.total_model_item_forwards for row in records):
            errors.append(f"cost_ledger_metadata:{pool_id}")
        for row in records:
            if row.ensemble_member_item_forwards != 3 * (row.proxy_item_forwards + row.verifier_item_forwards):
                errors.append(f"member_forward_count:{pool_id}")
            if row.proxy_member_item_forwards != 3 * row.proxy_item_forwards or row.verifier_member_item_forwards != 3 * row.verifier_item_forwards:
                errors.append(f"split_member_forward_count:{pool_id}")
            if row.total_model_item_forwards != row.ensemble_member_item_forwards + row.v_model_item_forwards + row.q_model_item_forwards:
                errors.append(f"total_forward_count:{pool_id}")
            if row.v_model_item_forwards != 4 * row.verifier_item_forwards:
                errors.append(f"v_forward_count:{pool_id}")
            if row.abstain:
                if row.regret is not None:
                    errors.append(f"abstain_regret:{pool_id}")
            else:
                expected = float(np.max(pool.measured) - pool.measured[row.selected_index])
                if row.selected_id != pool.sequence_ids[row.selected_index] or not np.isclose(row.regret, expected, rtol=0, atol=1e-12):
                    errors.append(f"regret:{pool_id}")
        summaries.append({"pool_id": pool_id, "sha256": sha256_file(path), "records": len(records)})
    report = {"pass": not errors, "pools": summaries, "records": sum(item["records"] for item in summaries), "errors": errors, "outer_labels_read": False, "scientific_result": False}
    (root / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "pools": len(summaries), "records": report["records"], "errors": errors[:10]}))
    if errors:
        raise ValueError("production grid independent validation failed")


if __name__ == "__main__":
    main()
