import argparse
import json
from pathlib import Path

import numpy as np

from .paper3_headroom_v4 import STRATEGIES
from .paper3_outer import OuterProtocolConfig, canonical, digest
from .pool_manifest import load_pool_manifest
from .trainer.checkpoint import atomic_json_write


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_round(directory):
    directory = Path(directory)
    config = OuterProtocolConfig.from_json(directory / "frozen_config.json")
    learner = read_json(directory / "learner_frozen.json")
    evaluator = read_json(directory / "evaluator_dry_run.json")
    summary = read_json(directory / "summary.json")
    if learner["config_sha256"] != config.sha256() or learner["outer_labels_read"] or learner["outer_metrics_computed"]:
        raise PermissionError("dry-run config or outer boundary failed")
    query_ids, evaluation_ids = learner["query_ids"], learner["evaluation_ids"]
    if len(query_ids) != 32 or len(evaluation_ids) != 32 or set(query_ids) & set(evaluation_ids):
        raise PermissionError("Q/E coverage or identity isolation failed")
    query_sequences = dict(zip(query_ids, learner["query_sequences"]))
    evaluation_sequences = dict(zip(evaluation_ids, learner["evaluation_sequences"]))
    for query_id, query_sequence in query_sequences.items():
        for evaluation_id, evaluation_sequence in evaluation_sequences.items():
            if sum(a != b for a, b in zip(query_sequence, evaluation_sequence)) <= 1:
                raise PermissionError("Q/E exact or Hamming-1 conflict")
    source = config.v4()
    calibration = load_pool_manifest(source.pool_root / config.dry_run_pool_id / "pool_manifest.json")
    labels = dict(zip(calibration.row_ids, calibration.measured))
    evaluation_labels = np.asarray([labels[item] for item in evaluation_ids])
    oracle = float(evaluation_labels.max())
    if len(learner["curves"]) != 8 or len(evaluator["curves"]) != 8:
        raise ValueError("three random and five non-random curves are required")
    if {curve["strategy"] for curve in learner["curves"]} != set(STRATEGIES):
        raise ValueError("strategy coverage failed")
    full_q_selections = set()
    for frozen, scored in zip(learner["curves"], evaluator["curves"]):
        if frozen["strategy"] != scored["strategy"] or frozen["seed"] != scored["seed"]:
            raise ValueError("Learner/Evaluator curve identity mismatch")
        if frozen["evaluation_labels_read"] or frozen["evaluation_scores_returned_to_learner"]:
            raise PermissionError("Learner evaluation boundary failed")
        if len(frozen["query_order"]) != 32 or len(set(frozen["query_order"])) != 32 or not set(frozen["query_order"]) <= set(query_ids):
            raise ValueError("unique query ledger failed")
        if [item["budget"] for item in frozen["records"]] != list(config.budgets):
            raise ValueError("budget grid failed")
        for record, score in zip(frozen["records"], scored["records"]):
            if record["unique_queries_used"] != record["budget"] or record["selected_id"] not in evaluation_ids:
                raise ValueError("record budget or evaluation identity failed")
            if record["training_device"] != "cuda:0" or record["inference_device"] != "cuda:0":
                raise RuntimeError("dry-run fit/inference left CUDA")
            expected = oracle - labels[record["selected_id"]]
            if not np.isclose(score["regret"], expected, rtol=0, atol=1e-12):
                raise ValueError("independent measured-MRL regret reconstruction failed")
        for step, item in enumerate(frozen["acquisition_metadata"]):
            if item["step"] != step or item["chosen_query_id"] != frozen["query_order"][step] or item["unique_queries_used"] != step + 1:
                raise ValueError("acquisition ledger reconstruction failed")
            if item["chosen_query_id"] not in query_ids:
                raise PermissionError("an evaluation ID entered the query ledger")
            if frozen["strategy"] == "regret_analytic":
                if item["device"] != "cuda:0" or item["posterior_samples"] or item["label_monte_carlo_samples"]:
                    raise RuntimeError("regret-analytic CUDA/no-MC boundary failed")
                levels = item["levels"]
                if [(level["label_quadrature_nodes"], level["value_quadrature_nodes"]) for level in levels] != [(64, 32), (64, 64), (64, 128)]:
                    raise ValueError("regret-analytic quadrature ladder failed")
        full_q_selections.add(frozen["records"][-1]["selected_id"])
    if len(full_q_selections) != 1 or not evaluator["full_q_same_selection"]:
        raise ValueError("full-Q information-set selection mismatch")
    scientific_sha256 = digest({"learner": learner["curves"], "evaluator": evaluator["curves"]})
    if scientific_sha256 != summary["scientific_sha256"]:
        raise ValueError("scientific hash mismatch")
    primary = {}
    for budget in config.budgets:
        ei_curve = next(curve for curve in evaluator["curves"] if curve["strategy"] == "expected_improvement")
        ei = next(record["regret"] for record in ei_curve["records"] if record["budget"] == budget)
        random_values = [next(record["regret"] for record in curve["records"] if record["budget"] == budget)
                         for curve in evaluator["curves"] if curve["strategy"] == "random"]
        primary[str(budget)] = {"expected_improvement_regret": ei,
                                "random_seed_mean_regret": float(np.mean(random_values)),
                                "ei_minus_random": float(ei - np.mean(random_values))}
    return {"directory": str(directory), "scientific_sha256": scientific_sha256,
            "curve_count": 8, "query_count": 32, "evaluation_count": 32,
            "primary_descriptive": primary, "outer_labels_read": False,
            "outer_metrics_computed": False, "scientific_conclusion": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independently validate three Paper3 calibration-only dry rounds")
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rounds = [validate_round(item) for item in args.directories]
    identical = len(rounds) == 3 and len({item["scientific_sha256"] for item in rounds}) == 1
    comparison = {"round_count": len(rounds),
                  "scientific_sha256_each": [item["scientific_sha256"] for item in rounds],
                  "scientific_results_identical": identical,
                  "outer_labels_read": False, "outer_metrics_computed": False}
    if not identical:
        raise ValueError("three identical dry rounds are required")
    output = Path(args.output)
    atomic_json_write(comparison, output / "three_round_validation.json")
    atomic_json_write({"schema_version": "paper3-outer-dry-run-aggregate-v1",
                       "rounds": rounds, "scope": "calibration-only implementation rehearsal; not an outer result",
                       "outer_labels_read": False, "outer_metrics_computed": False,
                       "acquisition_superiority_claimed": False}, output / "aggregate.json")
    print(canonical(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
