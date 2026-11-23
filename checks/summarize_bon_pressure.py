import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pool-prefix")
    args = parser.parse_args()
    records = []
    for index in range(args.count):
        pool_id = f"{args.pool_prefix}_{index:03d}" if args.pool_prefix else f"{args.role}_{index:03d}"
        payload = json.loads((Path(args.evaluation_dir) / f"{pool_id}.json").read_text())
        records.extend(row for row in payload["records"] if row["strategy"] == "bon" and row["alpha"] == 0)
    rows = []
    for cap in (1, 2, 4, 8, 16):
        per_pool = []
        costs = []
        for index in range(args.count):
            pool_id = f"{args.pool_prefix}_{index:03d}" if args.pool_prefix else f"{args.role}_{index:03d}"
            current = [row for row in records if row["pool_id"] == pool_id and row["proposal_cap"] == cap]
            per_pool.append(float(np.mean([row["regret"] for row in current])))
            costs.extend(row["ensemble_member_item_forwards"] for row in current)
        rng = np.random.default_rng(20261124 + cap)
        values = np.asarray(per_pool)
        bootstrap = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(10000)]
        rows.append({"proposal_cap": cap, "mean_regret": float(np.mean(values)), "cluster_bootstrap_95_percent_interval": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))], "mean_member_item_forwards": float(np.mean(costs)), "outer_pools": len(values)})
    report = {"pool_role": args.role, "analysis": "Fixed-pool best-of-N pressure; pool generation is unchanged", "rows": rows, "evidence_boundary": "Secondary analysis of already evaluated pilot outer pools, not a new independent blind test", "labels_returned_to_learner": False}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
