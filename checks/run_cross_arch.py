import argparse
import json
from pathlib import Path

import numpy as np

from proxy_function_gap.castillo import load_castillo_pool
from proxy_function_gap.kmer_proxy import evaluate_cross_architecture, score_kmer_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--train-reference", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--cnn-scores-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for index in range(args.count):
        pool_id = f"castillo_{args.context}_{index:03d}"
        pool = load_castillo_pool(args.pool_dir, pool_id, args.train_reference)
        kmer_path = output / f"{pool_id}_kmer.json"
        score_kmer_pool(pool, args.checkpoints, kmer_path)
        summaries.append(evaluate_cross_architecture(args.pool_dir, pool_id, Path(args.cnn_scores_dir) / f"{pool_id}.json", kmer_path, output / f"{pool_id}.json"))
    result = {"context": args.context, "pools": args.count, "fresh_mean_abs_error": float(np.mean([row["fresh_mean_abs_error"] for row in summaries])), "kmer_mean_abs_error": float(np.mean([row["kmer_mean_abs_error"] for row in summaries])), "mean_architecture_gap": float(np.mean([row["mean_architecture_gap"] for row in summaries])), "architecture_error_correlation": float(np.mean([row["architecture_error_correlation"] for row in summaries])), "low_gap_high_error_count": int(sum(row["low_gap_high_error_count"] for row in summaries)), "low_gap_mean_max_error": float(np.mean([row["low_gap_mean_max_error"] for row in summaries])), "other_gap_mean_max_error": float(np.mean([row["other_gap_mean_max_error"] for row in summaries])), "labels_returned_to_learner": False}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
