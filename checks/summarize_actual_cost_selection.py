import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from proxy_function_gap.selection_eval import SelectionRecord, selection_config_hash
from proxy_function_gap.small_model import log_loss
from proxy_function_gap.trainer.data import sha256_file


COMMON_CNN_BUDGET = 48
PRIMARY_ALPHA = {"random": 0, "bon": 0, "lcb": 0, "wsc": 20, "greedy": 20, "fp-rej": 20, "fp-tilt": 20}


def _records(payload):
    return [SelectionRecord(**{key: value for key, value in row.items() if key in SelectionRecord.__dataclass_fields__}) for row in payload["records"]]


def _mean(values):
    observed = [value for value in values if value is not None]
    return float(np.mean(observed)) if observed else None


def _pool_summary(values):
    return {
        "mean_regret": _mean([value for value in values if value is not None]),
        "pool_standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 and all(value is not None for value in values) else None,
    }


def _risk_coverage(probabilities, regrets, thresholds):
    rows = []
    valid = np.isfinite(regrets)
    for threshold in thresholds:
        accepted = (probabilities < threshold) & valid
        rows.append({
            "threshold": float(threshold),
            "threshold_domain": "platt_calibrated_probability",
            "coverage": float(np.mean(accepted)),
            "accepted": int(np.sum(accepted)),
            "mean_regret": float(np.mean(regrets[accepted])) if accepted.any() else None,
        })
    return rows


def _comparison(groups, count):
    rng = np.random.default_rng(20261124)
    comparisons = []
    for strategy in PRIMARY_ALPHA:
        if strategy == "bon":
            continue
        for ablation in ("v_only", "v_plus_q"):
            if ablation == "v_plus_q" and strategy != "fp-rej":
                continue
            differences = []
            for index in range(count):
                current = _mean(groups[strategy][index][ablation])
                reference = _mean(groups["bon"][index]["v_only"])
                if current is not None and reference is not None:
                    differences.append(current - reference)
            bootstrap = []
            if differences:
                values = np.asarray(differences)
                for _ in range(10000):
                    bootstrap.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
            comparisons.append({
                "strategy": strategy,
                "ablation": ablation,
                "reference": "bon_v_only_cap16",
                "mean_regret_difference": _mean(differences),
                "complete_pools": len(differences),
                "cluster_bootstrap_95_percent_interval": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))] if bootstrap else None,
                "direction": "negative favors strategy",
            })
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--evaluation-dir")
    parser.add_argument("--q-artifacts", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--role", choices=["dev", "calibration", "outer_designed", "outer_random"], required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.selection_dir)
    q_root = Path(args.q_artifacts)
    q_manifest = json.loads((q_root / "q_manifest.json").read_text())
    frozen = json.loads(Path(args.frozen_config).read_text())["selected"]
    if (frozen["strategy"], frozen["alpha"], frozen["proposal_cap"]) != ("fp-rej", 20, 16):
        raise ValueError("actual-cost report requires the frozen FP-REJ alpha20 cap16 configuration")
    groups = {strategy: defaultdict(lambda: {"v_only": [], "v_plus_q": []}) for strategy in PRIMARY_ALPHA}
    costs = {strategy: [] for strategy in PRIMARY_ALPHA}
    sources = []
    decision_values = []
    decision_regrets = []
    q_feature_costs = []
    for index in range(args.count):
        pool_id = f"{args.role}_{index:03d}"
        selection_path = root / f"{pool_id}.json"
        selection = json.loads(selection_path.read_text())
        records = _records(selection)
        if selection_config_hash(records) != selection["selection_config_sha256"]:
            raise ValueError(f"selection hash mismatch: {pool_id}")
        evaluated = None
        if args.evaluation_dir:
            evaluated = json.loads((Path(args.evaluation_dir) / f"{pool_id}.json").read_text())["records"]
        q_path = root / f"{pool_id}_q.json"
        q_payload = json.loads(q_path.read_text())
        if q_payload["selection_config_sha256"] != selection["selection_config_sha256"] or q_payload["threshold_domain"] != q_manifest["threshold_domain"]:
            raise ValueError(f"Q decision lineage mismatch: {pool_id}")
        decisions = {(row["pool_id"], row["seed"]): row for row in q_payload["decisions"]}
        q_feature_cost = q_payload["q_feature_member_item_forwards"]
        q_feature_costs.append(q_feature_cost)
        sources.append({"pool_id": pool_id, "selection_sha256": sha256_file(selection_path), "q_sha256": sha256_file(q_path)})
        for record in records:
            if record.proposal_cap != 16 or record.strategy not in PRIMARY_ALPHA or record.alpha != PRIMARY_ALPHA[record.strategy]:
                continue
            scored = None
            if evaluated is not None:
                scored = next(row for row in evaluated if row["strategy"] == record.strategy and row["alpha"] == record.alpha and row["proposal_cap"] == record.proposal_cap and row["seed"] == record.seed)
            selector_abstain = scored["selector_abstain"] if scored is not None else record.abstain
            selector_regret = scored["selector_regret"] if scored is not None else record.regret
            groups[record.strategy][index]["v_only"].append(selector_regret)
            q_applied = record.strategy == "fp-rej"
            if q_applied:
                decision = decisions.get((pool_id, record.seed))
                if decision is None:
                    raise ValueError(f"missing Q decision: {pool_id}/{record.seed}")
                final_abstain = scored["abstain"] if scored is not None else selector_abstain or decision["abstain"]
                final_regret = scored["regret"] if scored is not None else None if final_abstain else selector_regret
                groups[record.strategy][index]["v_plus_q"].append(final_regret)
                decision_values.append((decision["raw_q_probability"], decision["q_probability"], selector_regret))
                decision_regrets.append(selector_regret)
            else:
                final_abstain = selector_abstain
                final_regret = selector_regret
            q_model_cost = decisions[(pool_id, record.seed)]["q_model_item_forwards"] if q_applied else 0
            costs[record.strategy].append({
                "pool_id": pool_id,
                "seed": record.seed,
                "selector_abstain": selector_abstain,
                "final_abstain": final_abstain,
                "selector_regret": selector_regret,
                "final_regret": final_regret,
                "cnn": record.ensemble_member_item_forwards,
                "v": record.v_model_item_forwards,
                "q": q_model_cost,
                "q_feature_per_pool": q_feature_cost if q_applied else 0,
                "proxy_member": record.proxy_member_item_forwards,
                "verifier_member": record.verifier_member_item_forwards,
            })
    summaries = []
    for strategy in PRIMARY_ALPHA:
        rows = costs[strategy]
        if len(rows) != args.count * 5:
            raise ValueError(f"incomplete primary rows for {strategy}: {len(rows)}")
        v_only_pools = [_mean(groups[strategy][index]["v_only"]) for index in range(args.count)]
        vq_pools = [_mean(groups[strategy][index]["v_plus_q"]) for index in range(args.count)] if strategy == "fp-rej" else []
        final_pools = vq_pools if strategy == "fp-rej" else v_only_pools
        summary = {
            "strategy": strategy,
            "alpha": PRIMARY_ALPHA[strategy],
            "proposal_cap": 16,
            "v_only": {**_pool_summary(v_only_pools), "coverage": float(np.mean([not row["selector_abstain"] for row in rows]))},
            "v_plus_q": {**_pool_summary(vq_pools), "coverage": float(np.mean([not row["final_abstain"] for row in rows]))} if strategy == "fp-rej" else None,
            "mean_regret": _pool_summary(final_pools)["mean_regret"],
            "coverage": float(np.mean([not row["final_abstain"] for row in rows])),
            "mean_cnn_member_item_forwards": float(np.mean([row["cnn"] for row in rows])),
            "mean_proxy_member_item_forwards": float(np.mean([row["proxy_member"] for row in rows])),
            "mean_verifier_member_item_forwards": float(np.mean([row["verifier_member"] for row in rows])),
            "mean_v_model_item_forwards": float(np.mean([row["v"] for row in rows])),
            "mean_q_model_item_forwards": float(np.mean([row["q"] for row in rows])),
            "q_feature_member_item_forwards_per_pool": float(np.mean([row["q_feature_per_pool"] for row in rows])),
            "mean_total_model_item_forwards_including_amortized_q_features": float(np.mean([row["cnn"] + row["v"] + row["q"] + row["q_feature_per_pool"] / 5 for row in rows])),
            "common_cnn_budget": COMMON_CNN_BUDGET,
            "selection_cnn_within_common_budget": all(row["cnn"] <= COMMON_CNN_BUDGET for row in rows),
            "max_cnn_member_item_forwards": max(row["cnn"] for row in rows),
        }
        summaries.append(summary)
    if args.role == "dev":
        saved = np.load(q_root / "q_oof_predictions.npz", allow_pickle=False)
        raw, calibrated, regrets, targets = saved["raw_q"], saved["calibrated_q"], saved["regret"], saved["target"]
        q_basis = "grouped_dev_oof"
    elif args.role == "calibration":
        saved = np.load(q_root / "q_calibration_predictions.npz", allow_pickle=False)
        raw, calibrated, regrets, targets = saved["raw_q"], saved["calibrated_q"], saved["regret"], saved["target"]
        q_basis = "held_out_calibration"
    else:
        raw = np.asarray([row[0] for row in decision_values])
        calibrated = np.asarray([row[1] for row in decision_values])
        regrets = np.asarray([np.nan if row is None else row for row in decision_regrets], dtype=np.float64)
        targets = np.asarray([float(row is not None and row > q_manifest["delta_regret"]) for row in decision_regrets])
        q_basis = "retrospective_outer_frozen_q"
    q_diagnostics = {
        "basis": q_basis,
        "threshold_domain": q_manifest["threshold_domain"],
        "selected_threshold": q_manifest["selected_threshold"],
        "raw_q": {"mean": float(np.mean(raw)), "min": float(np.min(raw)), "max": float(np.max(raw)), "log_loss": log_loss(targets, raw)},
        "calibrated_q": {"mean": float(np.mean(calibrated)), "min": float(np.min(calibrated)), "max": float(np.max(calibrated)), "log_loss": log_loss(targets, calibrated)},
        "min_coverage_gate": q_manifest["min_coverage"],
        "risk_coverage": _risk_coverage(calibrated, regrets, q_manifest["threshold_grid"]),
    }
    outer = args.role.startswith("outer_")
    report = {
        "schema_version": "1.0",
        "pool_role": args.role,
        "pools": args.count,
        "paired_seeds": [101, 202, 303, 404, 505],
        "evidence_status": "exploratory_sensitivity_retrospective_previously_viewed_outer_not_a_new_blind_test" if outer else "development_only_not_scientific_result",
        "hyperparameter_source": "dev_only_frozen_before_outer",
        "frozen_dev_config_sha256": sha256_file(args.frozen_config),
        "q_manifest_sha256": sha256_file(q_root / "q_manifest.json"),
        "common_main_budget": {"definition": "BoN cap16 three-member CNN item-forwards", "cnn_member_item_forwards": COMMON_CNN_BUDGET, "v_and_q_costs_reported_separately": True, "fp_tilt_full_support_not_truncated": True},
        "configurations": summaries,
        "paired_comparisons": _comparison(groups, args.count),
        "q_diagnostics": q_diagnostics,
        "sources": sources,
        "outer_labels_read": outer,
        "labels_returned_to_learner": False,
        "scientific_result": False,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pool_role": args.role, "pools": args.count, "evidence_status": report["evidence_status"], "configurations": summaries, "q_diagnostics": q_diagnostics}, indent=2))


if __name__ == "__main__":
    main()
