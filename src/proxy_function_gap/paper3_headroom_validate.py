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
    expected = {(representation, query) for representation in config["representations"] for query in config["query_pool_ids"]}
    observed = {(item["representation"], item["query_pool_id"]) for item in summary["cases"]}
    if observed != expected or len(summary["cases"]) != len(expected):
        raise ValueError("preregistered case coverage differs from the frozen configuration")
    if summary["case_count"] != len(expected) or summary["headroom_present_count"] + summary["headroom_absent_count"] != len(expected):
        raise ValueError("summary case accounting failed")
    scientific_cases = []
    curve_records = []
    for representation, query_id in sorted(expected):
        case_dir = directory / representation / query_id
        result = read_json(case_dir / "result.json")
        gate = read_json(case_dir / "gate.json")
        if not np.isclose(result["repair_headroom"], result["zero_label_regret"] - result["full_q_regret"], rtol=0, atol=1e-10):
            raise ValueError("headroom arithmetic mismatch")
        passed = result["repair_headroom"] > config["min_repair_margin"]
        if passed != result["gate"]["pass"] or passed != (result["curve_status"] == "RUN_HEADROOM_PRESENT"):
            raise ValueError("gate and curve status differ")
        if result["brute_force_refit"]["weight_max_absolute_error"] > 1e-8 or not result["brute_force_refit"]["argmax_matches"]:
            raise ValueError("brute-force refit check failed")
        if result["outer_labels_read"] or result["outer_metrics_computed"]:
            raise PermissionError("outer access was recorded")
        if gate["zero_label_selected_id"] != result["zero_label_selected_id"] or gate["full_q_selected_id"] != result["full_q_selected_id"]:
            raise ValueError("gate and final selected IDs differ")
        curves = result["curves"]
        if passed:
            if len(curves) != 5 or (case_dir / "curves.json") not in list(case_dir.iterdir()):
                raise ValueError("an eligible case does not have exactly three random and two deterministic curves")
            if sorted(item["seed"] for item in curves if item["strategy"] == "random") != sorted(config["random_seeds"]):
                raise ValueError("random curve seeds differ from preregistration")
            for curve in curves:
                if [item["budget"] for item in curve["records"]] != config["budgets"]:
                    raise ValueError("curve budgets differ from preregistration")
                if len(set(curve["query_order"])) != config["budgets"][-1] or curve["evaluation_scores_returned_to_learner"]:
                    raise ValueError("curve query ledger or evaluator isolation failed")
                for record in curve["records"]:
                    if record["unique_queries_used"] != record["budget"] or not np.isclose(record["regret"], record["regret_scaled"] * result["feature_scaler"]["target_scale"], rtol=0, atol=1e-10):
                        raise ValueError("curve label budget or target-scale reconstruction failed")
                    if record["training_device"] != "cuda:0" or record["inference_device"] != "cuda:0":
                        raise ValueError("curve left CUDA")
                    curve_records.append({"representation": representation, "query_pool_id": query_id, "strategy": curve["strategy"], "seed": curve["seed"], **record})
                if curve["records"][-1]["selected_index"] != result["full_q"]["selected_index"]:
                    raise ValueError("full-Q curve and gate selections differ")
        elif curves or (case_dir / "curves.json").exists():
            raise ValueError("a stopped case has acquisition curves")
        scientific_cases.append({
            "representation": representation, "query_pool_id": query_id,
            "zero_label_selected_id": result["zero_label_selected_id"], "full_q_selected_id": result["full_q_selected_id"],
            "zero_label_regret": result["zero_label_regret"], "full_q_regret": result["full_q_regret"],
            "repair_headroom": result["repair_headroom"], "gate_pass": result["gate"]["pass"],
            "curve_status": result["curve_status"], "stop_reason": result["stop_reason"],
            "posterior_change": result["posterior_change"], "brute_force_refit": result["brute_force_refit"],
            "curves": curves,
        })
    return {
        "directory": str(directory), "config_sha256": summary["config_sha256"],
        "case_count": len(expected), "scientific_sha256": digest(scientific_cases),
        "scientific_cases": scientific_cases, "curve_records": curve_records,
    }


def aggregate(validated):
    cases = [{
        key: value for key, value in item.items() if key not in ("posterior_change", "brute_force_refit", "curves")
    } for item in validated["scientific_cases"]]
    representations = []
    for representation in sorted({item["representation"] for item in cases}):
        selected = [item for item in cases if item["representation"] == representation]
        headroom = np.asarray([item["repair_headroom"] for item in selected])
        representations.append({
            "representation": representation, "pool_count": len(selected),
            "headroom_present_count": int(np.sum(headroom > 0)), "headroom_absent_count": int(np.sum(headroom <= 0)),
            "headroom_negative_count": int(np.sum(headroom < 0)),
            "headroom_mean": float(headroom.mean()), "headroom_median": float(np.median(headroom)),
            "headroom_min": float(headroom.min()), "headroom_max": float(headroom.max()),
            "zero_regret_mean": float(np.mean([item["zero_label_regret"] for item in selected])),
            "full_q_regret_mean": float(np.mean([item["full_q_regret"] for item in selected])),
            "query_pool_ids": [item["query_pool_id"] for item in selected],
        })
    curves = []
    for representation in sorted({item["representation"] for item in cases}):
        eligible = {item["query_pool_id"] for item in cases if item["representation"] == representation and item["gate_pass"]}
        for budget in sorted({item["budget"] for item in validated["curve_records"]}):
            for strategy in ("random", "max_variance", "regret"):
                records = [item for item in validated["curve_records"] if item["representation"] == representation and item["query_pool_id"] in eligible and item["budget"] == budget and item["strategy"] == strategy]
                if not records:
                    continue
                by_pool = {}
                for record in records:
                    by_pool.setdefault(record["query_pool_id"], []).append(record["regret"])
                pool_means = [float(np.mean(values)) for _, values in sorted(by_pool.items())]
                curves.append({
                    "representation": representation, "budget": budget, "strategy": strategy,
                    "eligible_pool_count": len(eligible), "random_seed_count": 3 if strategy == "random" else 1,
                    "pool_mean_regret_mean": float(np.mean(pool_means)),
                    "pool_mean_regret_min": float(np.min(pool_means)), "pool_mean_regret_max": float(np.max(pool_means)),
                    "pool_mean_regrets": dict(zip(sorted(by_pool), pool_means)),
                })
    return {
        "schema_version": "paper3-headroom-aggregate-v1", "case_count": len(cases),
        "cases": cases, "representation_summary": representations, "curve_summary": curves,
        "outer_labels_read": False, "outer_metrics_computed": False,
        "acquisition_superiority_claimed": False,
        "scope": "retrospective development/query protocol sensitivity; not a new blind test",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independently validate and aggregate Paper3 headroom rounds")
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rounds = [validate_round(item) for item in args.directories]
    comparison = {
        "round_count": len(rounds), "case_count_each": [item["case_count"] for item in rounds],
        "scientific_sha256_each": [item["scientific_sha256"] for item in rounds],
        "scientific_results_identical": len({item["scientific_sha256"] for item in rounds}) == 1,
    }
    if len(rounds) > 1 and not comparison["scientific_results_identical"]:
        raise ValueError("scientific results differ across validation rounds")
    output = Path(args.output)
    atomic_json_write(comparison, output / "three_round_validation.json")
    aggregate_payload = aggregate(rounds[0])
    atomic_json_write(aggregate_payload, output / "aggregate.json")
    print(json.dumps({"validation": comparison, "representation_summary": aggregate_payload["representation_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
