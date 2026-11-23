import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from proxy_function_gap.selection_eval import SelectionRecord, selection_config_hash
from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.selection_dir)
    records = []
    sources = []
    for index in range(args.count):
        path = root / f"dev_{index:03d}.json"
        payload = json.loads(path.read_text())
        current = [SelectionRecord(**row) for row in payload["records"]]
        records.extend(current)
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    groups = defaultdict(list)
    for row in records:
        groups[(row.strategy, row.alpha, row.proposal_cap)].append(row)
    summary = []
    for (strategy, alpha, cap), rows in sorted(groups.items()):
        per_pool = []
        for index in range(args.count):
            values = [row.regret for row in rows if row.pool_id == f"dev_{index:03d}" and not row.abstain]
            per_pool.append(float(np.mean(values)) if values else None)
        complete = all(value is not None for value in per_pool)
        summary.append({
            "strategy": strategy,
            "alpha": alpha,
            "proposal_cap": cap,
            "mean_regret": float(np.mean(per_pool)) if complete else None,
            "pool_standard_error": float(np.std(per_pool, ddof=1) / np.sqrt(args.count)) if complete else None,
            "coverage": float(np.mean([not row.abstain for row in rows])),
            "mean_member_item_forwards": float(np.mean([row.ensemble_member_item_forwards for row in rows])),
            "mean_v_model_item_forwards": float(np.mean([row.v_model_item_forwards for row in rows])),
        })
    candidates = [row for row in summary if row["strategy"] == "fp-rej" and row["coverage"] == 1.0]
    if not candidates:
        raise ValueError("no FP-REJ development configuration has full coverage")
    selected = min(candidates, key=lambda row: (row["mean_regret"], row["mean_member_item_forwards"], row["proposal_cap"], row["alpha"]))
    report = {
        "selection_rule": "Among predeclared FP-REJ configurations with full dev coverage, minimize mean dev regret; ties use lower actual member cost, cap and alpha. No calibration or outer result used.",
        "selected": selected,
        "all_configurations": summary,
        "selection_sources": sources,
        "combined_selection_config_sha256": selection_config_hash(records),
        "dev_pools": args.count,
        "outer_labels_read": False,
        "scientific_result": False,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
