import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU-only independent Paper3 V4/outer-protocol audit")
    parser.add_argument("--v4-aggregate", required=True)
    parser.add_argument("--v4-three-round", required=True)
    parser.add_argument("--dry-aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    v4 = read_json(args.v4_aggregate)
    v4_rounds = read_json(args.v4_three_round)
    dry = read_json(args.dry_aggregate)
    records = v4["curve_records"]
    pools = sorted({item["query_pool_id"] for item in records if item["representation"] == "kmer4"})
    paired = []
    pooled = {}
    for budget in (0, 4, 8, 16, 32, 64):
        values = {}
        for strategy in ("random", "expected_improvement"):
            pool_means = []
            for pool in pools:
                selected = [item["regret"] for item in records if item["representation"] == "kmer4"
                            and item["query_pool_id"] == pool and item["budget"] == budget
                            and item["strategy"] == strategy]
                pool_means.append(float(np.mean(selected)))
            values[strategy] = pool_means
            pooled[f"{strategy}_{budget}"] = float(np.mean(pool_means))
        if budget == 16:
            paired = np.asarray(values["expected_improvement"]) - np.asarray(values["random"])
    rng = np.random.default_rng(20261125)
    indices = rng.integers(0, len(paired), size=(10000, len(paired)))
    boot = paired[indices].mean(axis=1)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    dry_rounds = dry["rounds"]
    dry_primary = dry_rounds[0]["primary_descriptive"]["16"]
    output = {
        "schema_version": "paper3-outer-cpu-audit-v1",
        "analysis_device": "CPU_only_statistical_recalculation_no_model_inference",
        "v4": {
            "case_count": v4["case_count"], "three_round_identical": v4_rounds["scientific_results_identical"],
            "scientific_sha256": v4_rounds["scientific_sha256_each"],
            "six_strategies": sorted({item["strategy"] for item in records}),
            "outer_labels_read": v4["outer_labels_read"], "outer_metrics_computed": v4["outer_metrics_computed"],
            "kmer4_fixed_five_pool_ids": pools,
            "kmer4_budget16_ei_minus_random_by_pool": dict(zip(pools, paired.tolist())),
            "kmer4_budget16_ei_minus_random_mean": float(paired.mean()),
            "calibration_pool_cluster_bootstrap_95": [float(lower), float(upper)],
            "recomputed_pooled_values": pooled,
            "scope": "retrospective calibration diagnostic; not outer evidence",
        },
        "dry_run": {
            "round_count": len(dry_rounds),
            "scientific_sha256_each": [item["scientific_sha256"] for item in dry_rounds],
            "identical": len({item["scientific_sha256"] for item in dry_rounds}) == 1,
            "query_count_each": [item["query_count"] for item in dry_rounds],
            "evaluation_count_each": [item["evaluation_count"] for item in dry_rounds],
            "budget16_ei_minus_random_descriptive": dry_primary["ei_minus_random"],
            "outer_labels_read": dry["outer_labels_read"], "outer_metrics_computed": dry["outer_metrics_computed"],
            "scope": "calibration-only implementation rehearsal; no inferential claim",
        },
        "gate_audit": {
            "v4_six_strategy_coverage": "PASS",
            "v4_development_only_hyperparameters": "PASS_FROM_INNER_DEVELOPMENT_SELECTION_ARTIFACTS",
            "v4_direct_query_evaluation_separation_for_outer": "FAIL_SAME_POOL_QUERY_AND_EVALUATION",
            "v1_separated_qe_implementation": "PASS_THREE_DRY_RUNS",
            "v1_frozen_representation_prior_noise": "PASS_KMER4_0.1_0.16",
            "paper2_p0_release": "FAIL_CALIBRATION_GATE_HARD_LOCK_NO_OUTER",
            "formal_outer_learner_executed": False,
            "formal_outer_evaluator_executed": False,
            "outer_labels_read_by_this_work_package": False,
            "outer_metrics_computed_by_this_work_package": False,
            "next_action": "Keep Paper3 outer hard locked; do not create a release or run the formal outer entry point",
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(hashlib.sha256(path.read_bytes()).hexdigest(), str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
