import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .trainer.checkpoint import atomic_json_write


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def validate_round(directory):
    directory = Path(directory)
    config = read_json(directory / "frozen_config.json")
    summary = read_json(directory / "summary.json")
    numerical = read_json(directory / "numerical_validation.json")
    if not numerical["passed"] or numerical["outer_labels_read"]:
        raise ValueError("V4 numerical validation or outer boundary failed")
    if numerical["training_device"] != "cuda:0" or numerical["analytic_metadata"]["device"] != "cuda:0" or numerical["nested_mc_metadata"]["device"] != "cuda:0":
        raise ValueError("V4 numerical validation left CUDA")
    if max(numerical["exhaustive_absolute_errors"]) > numerical["exhaustive_tolerance"] or max(numerical["two_candidate_absolute_errors"]) > numerical["two_candidate_tolerance"]:
        raise ValueError("V4 exhaustive/closed-form tolerances failed")
    if any(error > max(0.02, 5 * se) for error, se in zip(numerical["nested_mc_absolute_errors"], numerical["nested_mc_nominal_standard_errors"])):
        raise ValueError("V4 high-sample MC tolerance failed")
    expected = {(representation, query) for representation in config["representations"] for query in config["query_pool_ids"]}
    observed = {(item["representation"], item["query_pool_id"]) for item in summary["cases"]}
    if observed != expected or len(summary["cases"]) != 48:
        raise ValueError("V4 does not contain all 48 frozen gates")
    cohort = {(item["representation"], item["query_pool_id"]) for item in config["curve_cohort"]}
    if len(cohort) != 6:
        raise ValueError("V4 fixed curve cohort is not six cases")
    scientific_cases = []
    curve_records = []
    convergence_records = []
    selections = {}
    for representation in config["representations"]:
        selection = read_json(directory / representation / "inner_development_selection.json")
        if selection["query_labels_used"] or selection["outer_labels_read"] or len(selection["grid"]) != len(config["prior_variance_grid"]) * len(config["noise_variance_grid"]):
            raise PermissionError("invalid inner development selection boundary or coverage")
        if any(len(item["folds"]) != config["inner_cv_folds"] for item in selection["grid"]):
            raise ValueError("inner CV fold coverage failed")
        if any(fold["training_device"] != "cuda:0" or fold["inference_device"] != "cuda:0" for item in selection["grid"] for fold in item["folds"]):
            raise ValueError("inner CV left CUDA")
        selections[representation] = selection
    for representation, query_id in sorted(expected):
        case_dir = directory / representation / query_id
        result = read_json(case_dir / "result.json")
        gate = read_json(case_dir / "gate.json")
        if not np.isclose(result["repair_headroom"], result["zero_label_regret"] - result["full_q_regret"], rtol=0, atol=1e-10):
            raise ValueError("headroom arithmetic mismatch")
        passed = result["repair_headroom"] > config["min_repair_margin"]
        if passed != result["gate"]["pass"] or result["outer_labels_read"] or result["outer_metrics_computed"]:
            raise ValueError("gate arithmetic or outer boundary failed")
        if result["brute_force_refit"]["weight_max_absolute_error"] > 1e-8 or not result["brute_force_refit"]["argmax_matches"]:
            raise ValueError("brute-force refit check failed")
        if gate["zero_label_selected_id"] != result["zero_label_selected_id"] or gate["full_q_selected_id"] != result["full_q_selected_id"]:
            raise ValueError("gate and final selected IDs differ")
        curves = result["curves"]
        if (representation, query_id) in cohort:
            if result["curve_status"] != "RUN_FIXED_V2_COHORT" or len(curves) != 8:
                raise ValueError("a fixed cohort case must have three random and five other curves")
            if sorted(item["seed"] for item in curves if item["strategy"] == "random") != sorted(config["random_seeds"]):
                raise ValueError("random seeds differ from frozen protocol")
            if {item["strategy"] for item in curves} != set(config["strategies"]):
                raise ValueError("strategy coverage differs from frozen protocol")
            for curve in curves:
                if [item["budget"] for item in curve["records"]] != config["budgets"]:
                    raise ValueError("curve budgets differ from protocol")
                if len(set(curve["query_order"])) != 64 or curve["evaluation_scores_returned_to_learner"]:
                    raise ValueError("curve query ledger or evaluator isolation failed")
                for record in curve["records"]:
                    if record["unique_queries_used"] != record["budget"] or not np.isclose(record["regret"], record["regret_scaled"] * result["feature_scaler"]["target_scale"], rtol=0, atol=1e-10):
                        raise ValueError("curve label budget or target-scale reconstruction failed")
                    if record["training_device"] != "cuda:0" or record["inference_device"] != "cuda:0":
                        raise ValueError("curve left CUDA")
                    curve_records.append({"representation": representation, "query_pool_id": query_id, "strategy": curve["strategy"], "seed": curve["seed"], **record})
                if curve["records"][-1]["selected_index"] != result["full_q"]["selected_index"]:
                    raise ValueError("full-Q curve and gate selections differ")
                for item in curve["acquisition_metadata"]:
                    step = item["step"]
                    batch_start = (step // config["batch_size"]) * config["batch_size"]
                    chosen = item["chosen_index"]
                    for previous in curve["query_order"][batch_start:step]:
                        distance = sum(a != b for a, b in zip(result["query_sequences"][previous], result["query_sequences"][chosen]))
                        if distance <= config["near_duplicate_max_hamming"]:
                            raise ValueError("within-batch near-duplicate constraint failed")
                    if curve["strategy"] == "regret_analytic":
                        levels = item["levels"]
                        if [(level["label_quadrature_nodes"], level["value_quadrature_nodes"]) for level in levels] != list(zip(config["label_quadrature_ladder"], config["value_quadrature_ladder"])):
                            raise ValueError("V4 label/value quadrature ladder coverage failed")
                        if item["posterior_samples"] != 0 or item["label_monte_carlo_samples"] != 0 or item["device"] != "cuda:0":
                            raise ValueError("V4 analytic no-MC/CUDA contract failed")
                        if item["posterior_dependence"] != "diagonal_independent_fantasy_posterior_approximation" or item["joint_gaussian_multivariate_max_claimed"]:
                            raise ValueError("V4 independence claim is invalid")
                        convergence_records.append({
                            "representation": representation, "query_pool_id": query_id, "step": step,
                            "converged": item["converged"],
                            "last_choice_stable": levels[-2]["chosen_available_position"] == levels[-1]["chosen_available_position"],
                            "max_absolute_score_delta": levels[-1]["max_absolute_score_delta_from_previous"],
                        })
        elif curves or (case_dir / "curves.json").exists():
            raise ValueError("a non-cohort case has acquisition curves")
        scientific_cases.append({
            "representation": representation, "query_pool_id": query_id,
            "zero_label_selected_id": result["zero_label_selected_id"], "full_q_selected_id": result["full_q_selected_id"],
            "zero_label_regret": result["zero_label_regret"], "full_q_regret": result["full_q_regret"],
            "repair_headroom": result["repair_headroom"], "gate_pass": result["gate"]["pass"],
            "curve_status": result["curve_status"], "stop_reason": result["stop_reason"],
            "selected_prior_variance": result["selected_prior_variance"], "selected_noise_variance": result["selected_noise_variance"],
            "fixed_v2_curve_cohort_member": result["fixed_v2_curve_cohort_member"],
            "posterior_change": result["posterior_change"], "brute_force_refit": result["brute_force_refit"],
            "curves": curves,
        })
    return {
        "directory": str(directory), "config_sha256": summary["config_sha256"], "case_count": 48,
        "scientific_sha256": digest({"numerical_validation": numerical, "selections": selections, "cases": scientific_cases}),
        "scientific_cases": scientific_cases, "curve_records": curve_records,
        "convergence_records": convergence_records, "selections": selections,
        "numerical_validation": numerical,
    }


def aggregate(validated):
    cases = [{key: value for key, value in item.items() if key not in ("posterior_change", "brute_force_refit", "curves")} for item in validated["scientific_cases"]]
    representations = []
    for representation in sorted({item["representation"] for item in cases}):
        selected = [item for item in cases if item["representation"] == representation]
        headroom = np.asarray([item["repair_headroom"] for item in selected])
        representations.append({
            "representation": representation, "pool_count": 16,
            "headroom_present_count": int(np.sum(headroom > 0)), "headroom_absent_count": int(np.sum(headroom <= 0)),
            "headroom_mean": float(headroom.mean()), "headroom_median": float(np.median(headroom)),
            "headroom_min": float(headroom.min()), "headroom_max": float(headroom.max()),
            "zero_regret_mean": float(np.mean([item["zero_label_regret"] for item in selected])),
            "full_q_regret_mean": float(np.mean([item["full_q_regret"] for item in selected])),
            "selected_prior_variance": selected[0]["selected_prior_variance"],
            "selected_noise_variance": selected[0]["selected_noise_variance"],
            "query_pool_ids": [item["query_pool_id"] for item in selected],
        })
    strategies = ("random", "max_variance", "diversity", "expected_improvement", "top_proxy", "regret_analytic")
    curves = []
    per_case_curves = []
    for representation in sorted({item["representation"] for item in cases}):
        eligible = {item["query_pool_id"] for item in cases if item["representation"] == representation and item["fixed_v2_curve_cohort_member"]}
        for query_id in sorted(eligible):
            for budget in (0, 4, 8, 16, 32, 64):
                for strategy in strategies:
                    records = [item for item in validated["curve_records"] if item["representation"] == representation and item["query_pool_id"] == query_id and item["budget"] == budget and item["strategy"] == strategy]
                    per_case_curves.append({
                        "representation": representation, "query_pool_id": query_id, "budget": budget, "strategy": strategy,
                        "random_seed_count": len(records), "regret": float(np.mean([item["regret"] for item in records])),
                    })
        for budget in (0, 4, 8, 16, 32, 64):
            for strategy in strategies:
                records = [item for item in validated["curve_records"] if item["representation"] == representation and item["query_pool_id"] in eligible and item["budget"] == budget and item["strategy"] == strategy]
                if not records:
                    continue
                by_pool = {}
                for record in records:
                    by_pool.setdefault(record["query_pool_id"], []).append(record["regret"])
                pool_means = [float(np.mean(values)) for _, values in sorted(by_pool.items())]
                curves.append({
                    "representation": representation, "budget": budget, "strategy": strategy,
                    "fixed_cohort_pool_count": len(eligible), "random_seed_count": 3 if strategy == "random" else 1,
                    "pool_mean_regret_mean": float(np.mean(pool_means)),
                    "pool_mean_regret_min": float(np.min(pool_means)), "pool_mean_regret_max": float(np.max(pool_means)),
                    "pool_mean_regrets": dict(zip(sorted(by_pool), pool_means)),
                })
    convergence = validated["convergence_records"]
    by_case = []
    for representation, query_id in sorted({(item["representation"], item["query_pool_id"]) for item in convergence}):
        selected = [item for item in convergence if item["representation"] == representation and item["query_pool_id"] == query_id]
        by_case.append({
            "representation": representation, "query_pool_id": query_id,
            "converged_count": sum(item["converged"] for item in selected),
            "converged_fraction": float(np.mean([item["converged"] for item in selected])),
            "last_choice_stable_fraction": float(np.mean([item["last_choice_stable"] for item in selected])),
            "max_absolute_score_delta_median": float(np.median([item["max_absolute_score_delta"] for item in selected])),
            "max_absolute_score_delta_max": float(np.max([item["max_absolute_score_delta"] for item in selected])),
        })
    convergence_summary = {
        "acquisition_step_count": len(convergence),
        "converged_count": sum(item["converged"] for item in convergence),
        "converged_fraction": float(np.mean([item["converged"] for item in convergence])) if convergence else None,
        "last_choice_stable_fraction": float(np.mean([item["last_choice_stable"] for item in convergence])) if convergence else None,
        "max_absolute_score_delta_median": float(np.median([item["max_absolute_score_delta"] for item in convergence])) if convergence else None,
        "max_absolute_score_delta_max": float(np.max([item["max_absolute_score_delta"] for item in convergence])) if convergence else None,
        "by_case": by_case, "records": convergence,
    }
    numerical = validated["numerical_validation"]
    numerical_summary = {
        "exhaustive_absolute_error_max": max(numerical["exhaustive_absolute_errors"]),
        "two_candidate_absolute_error_max": max(numerical["two_candidate_absolute_errors"]),
        "nested_mc_absolute_errors": numerical["nested_mc_absolute_errors"],
        "nested_mc_nominal_standard_errors": numerical["nested_mc_nominal_standard_errors"],
        "nested_mc_metadata": numerical["nested_mc_metadata"], "passed": numerical["passed"],
    }
    return {
        "schema_version": "paper3-headroom-v4-aggregate-v1", "case_count": 48,
        "cases": cases, "representation_summary": representations, "curve_summary": curves,
        "per_case_curve_summary": per_case_curves, "curve_records": validated["curve_records"],
        "regret_convergence_summary": convergence_summary, "numerical_validation_summary": numerical_summary,
        "outer_labels_read": False, "outer_metrics_computed": False,
        "acquisition_superiority_claimed": False,
        "posterior_dependence": "diagonal_independent_fantasy_posterior_approximation",
        "scope": "fixed retrospective development/calibration V2 cohort; not a new blind test",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independently validate and aggregate Paper3 V4 rounds")
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rounds = [validate_round(item) for item in args.directories]
    comparison = {
        "round_count": len(rounds), "case_count_each": [item["case_count"] for item in rounds],
        "scientific_sha256_each": [item["scientific_sha256"] for item in rounds],
        "scientific_results_identical": len({item["scientific_sha256"] for item in rounds}) == 1,
    }
    if len(rounds) != 3 or not comparison["scientific_results_identical"]:
        raise ValueError("three identical V4 scientific validation rounds are required")
    output = Path(args.output)
    atomic_json_write(comparison, output / "three_round_validation.json")
    aggregate_payload = aggregate(rounds[0])
    atomic_json_write(aggregate_payload, output / "aggregate.json")
    print(json.dumps({"validation": comparison, "representation_summary": aggregate_payload["representation_summary"], "convergence": {key: value for key, value in aggregate_payload["regret_convergence_summary"].items() if key not in ("records", "by_case")}, "numerical_validation": aggregate_payload["numerical_validation_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
