import argparse
import json
from pathlib import Path

import numpy as np

from proxy_function_gap.mechanism import evaluate_mechanism


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for index in range(args.count):
        pool_id = f"castillo_{args.context}_{index:03d}"
        reports.append(evaluate_mechanism(args.pool_dir, pool_id, Path(args.scores_dir) / f"{pool_id}.json", Path(args.selection_dir) / f"{pool_id}.json", output / f"{pool_id}.json"))
    differences = np.asarray([row["score_matched"]["abs_error_difference"] for row in reports])
    rng = np.random.default_rng(20261124)
    bootstrap = [float(np.mean(rng.choice(differences, size=len(differences), replace=True))) for _ in range(10000)]
    summary = {
        "context": args.context,
        "pools": args.count,
        "score_matched_mean_abs_error_difference": float(differences.mean()),
        "score_matched_cluster_bootstrap_95_percent_interval": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
        "optimized_mean_abs_error": float(np.mean([row["score_matched"]["optimized_mean_abs_error"] for row in reports])),
        "matched_mean_abs_error": float(np.mean([row["score_matched"]["matched_mean_abs_error"] for row in reports])),
        "mean_proxy_gap": float(np.mean([row["score_matched"]["mean_proxy_gap"] for row in reports])),
        "low_spread_high_error_count": int(sum(row["common_proxy"]["low_spread_high_error_count"] for row in reports)),
        "low_spread_mean_error": float(np.mean([row["common_proxy"]["low_spread_mean_error"] for row in reports])),
        "other_spread_mean_error": float(np.mean([row["common_proxy"]["other_spread_mean_error"] for row in reports])),
        "sequence_families": {
            name: {"count": int(sum(row["sequence_families"][name]["count"] for row in reports)), "mean_error": float(np.mean([row["sequence_families"][name]["mean_error"] for row in reports]))}
            for name in reports[0]["sequence_families"]
        },
        "labels_returned_to_learner": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
