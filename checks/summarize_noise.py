import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for index in range(args.count):
        pool_id = f"castillo_{args.context}_{index:03d}"
        records.extend(json.loads((Path(args.evaluation_dir) / f"{pool_id}.json").read_text())["records"])
    rows = []
    for sigma in (0, 0.1, 0.25, 0.5, 1.0):
        for cap in (1, 4, 16):
            current = [row for row in records if row["strategy"] == "bon" and row["proposal_cap"] == cap and row.get("noise_sigma", 0) == sigma]
            baseline = [row for row in records if row["strategy"] == "bon" and row["proposal_cap"] == cap and row.get("noise_sigma", 0) == 0]
            differences = []
            for index in range(args.count):
                pool_id = f"castillo_{args.context}_{index:03d}"
                noisy_values = [row["regret"] for row in current if row["pool_id"] == pool_id]
                base_values = [row["regret"] for row in baseline if row["pool_id"] == pool_id]
                differences.append(float(np.mean(noisy_values) - np.mean(base_values)))
            rows.append({"noise_sigma": sigma, "proposal_cap": cap, "mean_regret": float(np.mean([row["regret"] for row in current])), "mean_difference_from_sigma0": float(np.mean(differences)), "mean_member_item_forwards": float(np.mean([row["ensemble_member_item_forwards"] for row in current]))})
    report = {"context": args.context, "noise_definition": "Gaussian perturbation of normalized CNN output before train-only inverse transform", "rows": rows, "labels_returned_to_learner": False}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
