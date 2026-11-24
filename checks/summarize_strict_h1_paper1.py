import argparse
import json
from pathlib import Path

import numpy as np

from proxy_function_gap.trainer.data import sha256_file


def cluster_bootstrap(values, seed, replicates=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def summarize(config):
    root = Path(config["analysis_root"])
    seed = config["bootstrap"]["seed"]
    replicates = config["bootstrap"]["replicates"]
    contexts = {}
    for offset, context in enumerate(config["contexts"]):
        context_root = root / context
        pressure_rows = []
        pressure_per_pool = []
        for cap_index, cap in enumerate(config["selection"]["proposal_caps"]):
            per_pool = []
            for index in range(config["pool_count"]):
                pool_id = f"castillo_{context}_{index:03d}"
                payload = json.loads((context_root / "evaluation" / f"{pool_id}.json").read_text())
                records = [row for row in payload["records"] if row["alpha"] == 0 and row["proposal_cap"] == cap]
                random = [row for row in records if row["strategy"] == "random"]
                bon = [row for row in records if row["strategy"] == "bon"]
                random_by_seed = {row["seed"]: row["regret"] for row in random}
                bon_by_seed = {row["seed"]: row["regret"] for row in bon}
                if set(random_by_seed) != set(config["selection"]["seeds"]) or set(bon_by_seed) != set(random_by_seed):
                    raise ValueError(f"paired pressure seeds missing for {pool_id} cap {cap}")
                random_mean = float(np.mean(list(random_by_seed.values())))
                bon_mean = float(np.mean(list(bon_by_seed.values())))
                paired = float(np.mean([bon_by_seed[seed_value] - random_by_seed[seed_value] for seed_value in config["selection"]["seeds"]]))
                row = {"pool_id": pool_id, "proposal_cap": cap, "random_mean_regret": random_mean, "bon_mean_regret": bon_mean, "paired_bon_minus_random": paired}
                pressure_per_pool.append(row)
                per_pool.append(row)
            effects = [row["paired_bon_minus_random"] for row in per_pool]
            pressure_rows.append({
                "proposal_cap": cap,
                "random_mean_regret": float(np.mean([row["random_mean_regret"] for row in per_pool])),
                "bon_mean_regret": float(np.mean([row["bon_mean_regret"] for row in per_pool])),
                "paired_bon_minus_random": float(np.mean(effects)),
                "paired_cluster_bootstrap_95_percent_interval": cluster_bootstrap(effects, seed + offset * 100 + cap_index, replicates),
                "outer_pools": len(per_pool),
            })
        score_per_pool = []
        common_per_pool = []
        for index in range(config["pool_count"]):
            pool_id = f"castillo_{context}_{index:03d}"
            score = json.loads((context_root / "mechanism" / f"{pool_id}.json").read_text())["score_matched"]
            score_per_pool.append({
                "pool_id": pool_id,
                "optimized_mean_abs_error": score["optimized_mean_abs_error"],
                "matched_mean_abs_error": score["matched_mean_abs_error"],
                "paired_optimized_minus_matched": score["abs_error_difference"],
                "mean_proxy_gap": score["mean_proxy_gap"],
            })
            common = json.loads((context_root / "cross_architecture" / f"{pool_id}.json").read_text())
            common_per_pool.append({
                "pool_id": pool_id,
                "cnn_mean_abs_error": common["fresh_mean_abs_error"],
                "kmer_mean_abs_error": common["kmer_mean_abs_error"],
                "paired_cnn_minus_kmer": common["fresh_mean_abs_error"] - common["kmer_mean_abs_error"],
                "architecture_error_correlation": common["architecture_error_correlation"],
                "mean_architecture_gap": common["mean_architecture_gap"],
                "low_gap_high_error_count": common["low_gap_high_error_count"],
            })
        score_effects = [row["paired_optimized_minus_matched"] for row in score_per_pool]
        common_effects = [row["paired_cnn_minus_kmer"] for row in common_per_pool]
        correlations = [row["architecture_error_correlation"] for row in common_per_pool]
        contexts[context] = {
            "pressure": {"rows": pressure_rows, "per_pool": pressure_per_pool},
            "score_matching": {
                "optimized_mean_abs_error": float(np.mean([row["optimized_mean_abs_error"] for row in score_per_pool])),
                "matched_mean_abs_error": float(np.mean([row["matched_mean_abs_error"] for row in score_per_pool])),
                "paired_optimized_minus_matched": float(np.mean(score_effects)),
                "paired_cluster_bootstrap_95_percent_interval": cluster_bootstrap(score_effects, seed + 1000 + offset, replicates),
                "per_pool": score_per_pool,
            },
            "common_error": {
                "cnn_mean_abs_error": float(np.mean([row["cnn_mean_abs_error"] for row in common_per_pool])),
                "kmer_mean_abs_error": float(np.mean([row["kmer_mean_abs_error"] for row in common_per_pool])),
                "paired_cnn_minus_kmer": float(np.mean(common_effects)),
                "paired_cluster_bootstrap_95_percent_interval": cluster_bootstrap(common_effects, seed + 2000 + offset, replicates),
                "mean_architecture_error_correlation": float(np.mean(correlations)),
                "correlation_cluster_bootstrap_95_percent_interval": cluster_bootstrap(correlations, seed + 3000 + offset, replicates),
                "per_pool": common_per_pool,
            },
            "source_hashes": {
                "context_manifest": sha256_file(context_root / "context_manifest.json"),
                "evaluation_manifest": sha256_file(context_root / "evaluation" / "evaluation_manifest.json"),
                "common_error_summary": sha256_file(context_root / "cross_architecture" / "summary.json"),
            },
        }
    return {
        "schema_version": "1.0",
        "protocol": config["protocol"],
        "evidence_boundary": config["evidence_boundary"],
        "new_blind_test": False,
        "bootstrap": config["bootstrap"],
        "effect_signs": {
            "pressure": "BoN regret minus paired Random regret; negative favors BoN",
            "score_matching": "optimized absolute error minus score-matched absolute error; positive indicates higher optimized error",
            "common_error": "CNN absolute error minus 4-mer MLP absolute error; negative favors CNN",
        },
        "contexts": contexts,
        "contains_sequence_level_labels": False,
        "contains_sequences": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    report = summarize(config)
    output = Path(args.output) if args.output else Path(config["analysis_root"]) / "strict_h1_paired_summary.json"
    if args.validate:
        expected = json.loads(output.read_text())
        passed = report == expected
        validation = {"round": args.round, "pass": passed, "contexts": len(report["contexts"]), "independent_recomputation": True, "contains_sequence_level_labels": False}
        target = output.parent / f"summary_validation_round{args.round}.json"
        target.write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps(validation))
        if not passed:
            raise ValueError("strict H1 summary validation failed")
    else:
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"pass": True, "contexts": len(report["contexts"]), "output": str(output)}))


if __name__ == "__main__":
    main()
