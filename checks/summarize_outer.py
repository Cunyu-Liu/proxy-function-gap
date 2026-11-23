import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    for index in range(args.count):
        pool_id = f"{args.role}_{index:03d}"
        payload = json.loads((Path(args.evaluation_dir) / f"{pool_id}.json").read_text())
        for row in payload["records"]:
            groups[(row["strategy"], row["alpha"], row["proposal_cap"])].append(row)
    summaries = []
    for (strategy, alpha, cap), rows in sorted(groups.items()):
        per_pool = []
        coverage = []
        selector_regret = []
        for index in range(args.count):
            current = [row for row in rows if row["pool_id"] == pool_id_format(args.role, index)]
            values = [row["regret"] for row in current if not row["abstain"]]
            per_pool.append(float(np.mean(values)) if values else None)
            coverage.append(float(np.mean([not row["abstain"] for row in current])))
            selector_regret.extend(row["selector_regret"] for row in current if row["selector_regret"] is not None)
        complete = all(value is not None for value in per_pool)
        summaries.append({
            "strategy": strategy, "alpha": alpha, "proposal_cap": cap,
            "mean_regret": float(np.mean(per_pool)) if complete else None,
            "coverage": float(np.mean(coverage)),
            "selector_mean_regret_before_q": float(np.mean(selector_regret)) if selector_regret else None,
            "mean_member_item_forwards": float(np.mean([row["ensemble_member_item_forwards"] for row in rows])),
            "mean_v_model_item_forwards": float(np.mean([row["v_model_item_forwards"] for row in rows])),
        })
    selected_rows = groups[("fp-rej", 20, 16)]
    bon_rows = groups[("bon", 0, 16)]
    differences = []
    for index in range(args.count):
        pool_id = pool_id_format(args.role, index)
        selected_values = [row["regret"] for row in selected_rows if row["pool_id"] == pool_id and not row["abstain"]]
        bon_values = [row["regret"] for row in bon_rows if row["pool_id"] == pool_id and not row["abstain"]]
        if selected_values and bon_values:
            differences.append(float(np.mean(selected_values) - np.mean(bon_values)))
    rng = np.random.default_rng(20261124)
    bootstrap = []
    if differences:
        values = np.asarray(differences)
        for _ in range(10000):
            bootstrap.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
    comparison = {"difference": "FP-REJ+Q minus BoN cap16; negative favors FP-REJ+Q", "complete_outer_pools": len(differences), "mean_regret_difference": float(np.mean(differences)) if differences else None, "cluster_bootstrap_95_percent_interval": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))] if bootstrap else None}
    selector_differences = []
    for index in range(args.count):
        pool_id = pool_id_format(args.role, index)
        selected_values = [row["selector_regret"] for row in selected_rows if row["pool_id"] == pool_id and row["selector_regret"] is not None]
        bon_values = [row["selector_regret"] for row in bon_rows if row["pool_id"] == pool_id and row["selector_regret"] is not None]
        if selected_values and bon_values:
            selector_differences.append(float(np.mean(selected_values) - np.mean(bon_values)))
    selector_values = np.asarray(selector_differences)
    selector_bootstrap = [float(np.mean(rng.choice(selector_values, size=len(selector_values), replace=True))) for _ in range(10000)] if selector_differences else []
    selector_comparison = {"difference": "FP-REJ without Q minus BoN cap16; negative favors FP-REJ", "complete_outer_pools": len(selector_differences), "mean_regret_difference": float(np.mean(selector_values)) if selector_differences else None, "cluster_bootstrap_95_percent_interval": [float(np.percentile(selector_bootstrap, 2.5)), float(np.percentile(selector_bootstrap, 97.5))] if selector_bootstrap else None}
    report = {"pool_role": args.role, "outer_pools": args.count, "primary_comparison": comparison, "selector_comparison_before_q": selector_comparison, "configurations": summaries, "evidence_boundary": "Disjoint same-study fixed pools; not independent external studies.", "labels_returned_to_learner": False}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    selected = [row for row in summaries if row["strategy"] == "fp-rej" and row["alpha"] == 20 and row["proposal_cap"] == 16][0]
    bon = [row for row in summaries if row["strategy"] == "bon" and row["proposal_cap"] == 16][0]
    print(json.dumps({"frozen_fp_rej_q": selected, "bon_cap16": bon}, indent=2))


def pool_id_format(role, index):
    return f"{role}_{index:03d}"


if __name__ == "__main__":
    main()
