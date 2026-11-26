from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .residual import cluster_bootstrap, cnn_predict, digest, load_json, proposal_indices, save_json, simple_features


def code_hashes(workspace):
    root = Path(workspace)
    files = sorted(list((root / "src").rglob("*.py")) + list((root / "tests").glob("test_residual*.py")) + list((root / "scripts").glob("*.py")))
    return {str(path.relative_to(root)): digest(path) for path in files}


def verify_frozen_inputs(config, workspace, root):
    freeze = load_json(root / "target_freeze.json")
    if digest(config) != freeze["config_sha256"] or code_hashes(workspace) != freeze["code_sha256"]:
        raise PermissionError("frozen code/config identity changed")
    for row in freeze["auxiliary_selections"]:
        if digest(row["path"]) != row["sha256"]:
            raise PermissionError("auxiliary coefficient identity changed")
    return freeze


def freeze_target(config_path, workspace):
    config_path = Path(config_path)
    config = load_json(config_path)
    root = Path(config["artifact_root"])
    gate = load_json(root / "auxiliary_gate_passed.json")
    if gate.get("pass") is not True:
        raise PermissionError("auxiliary gate did not pass")
    directions = ("n50_hek293t_r1_to_defined_hepg2_r1", "defined_hepg2_r1_to_n50_hek293t_r1")
    selections = []
    for direction in directions:
        path = root / "auxiliary" / direction / "selection.json"
        if digest(path) != gate["selection_sha256"][direction]:
            raise ValueError("auxiliary selection hash mismatch")
        value = load_json(path)
        if not value["gate_pass"]:
            raise PermissionError("cannot freeze failed auxiliary direction")
        selections.append({"direction": direction, "path": str(path), "sha256": digest(path), "lambda": value["chosen"]["lambda"], "strength": value["chosen"]["strength"], "coefficients": value["coefficients"], "affine_coefficients": value["affine_only"]["fit"]["coefficients"]})
    freeze = {
        "schema_version": "1.0", "analysis_id": config["analysis_id"], "frozen_before_target_inference_and_labels": True,
        "config_path": str(config_path), "config_sha256": digest(config_path), "workspace": str(workspace),
        "code_sha256": code_hashes(workspace), "auxiliary_selections": selections,
        "final_combination": config["selection"]["final_combination"], "sample_train_sd": config["sample_train_sd"],
        "target_mrl_read": False, "target_evaluation_count_before_freeze": 0,
        "evidence_boundary": config["evidence_boundary"],
    }
    save_json(root / "target_freeze.json", freeze)
    return freeze


def load_learner_pools(pool_root, context):
    result = []
    for path in sorted((Path(pool_root) / context / "learner").glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != 64 or any("measured_rl" in row for row in rows):
            raise ValueError("invalid or leaky learner pool")
        result.append(rows)
    if len(result) != 16:
        raise ValueError("exactly 16 target pools required")
    return result


def select_target(config_path, workspace):
    config_path = Path(config_path)
    config = load_json(config_path)
    root = Path(config["artifact_root"])
    freeze = verify_frozen_inputs(config_path, workspace, root)
    records = {}
    for context in config["target"]["contexts"]:
        pools = load_learner_pools(config["pool_root"], context)
        rows = [row for pool in pools for row in pool]
        sequences = np.asarray([row["utr50"] for row in rows], dtype="S50")
        p, member_predictions, inference = cnn_predict(sequences, config["inference_config"])
        if abs(inference["sample_train_sd"] - config["sample_train_sd"]) > 1e-15:
            raise ValueError("deployment Sample train SD mismatch")
        x = simple_features(sequences, p)
        correction = np.zeros(len(rows))
        affine_correction = np.zeros(len(rows))
        for selected in freeze["auxiliary_selections"]:
            correction += 0.5 * selected["strength"] * (x @ np.asarray(selected["coefficients"]))
            affine_correction += 0.5 * selected["strength"] * (x[:, [0, 5]] @ np.asarray(selected["affine_coefficients"]))
        affine_transformed = p + config["sample_train_sd"] * affine_correction
        scores = {
            "frozen_cnn": p,
            "corrected_cnn": p + config["sample_train_sd"] * correction,
            "affine_only": p,
        }
        effective_slope = 1.0 + config["sample_train_sd"] * sum(0.5 * selected["strength"] * selected["affine_coefficients"][0] for selected in freeze["auxiliary_selections"])
        if effective_slope <= 0:
            raise ValueError("final affine slope is not positive")
        context_dir = root / "target" / context
        context_dir.mkdir(parents=True, exist_ok=False)
        np.savez(context_dir / "scores.npz", sequences=sequences, sequence_ids=np.asarray([row["sequence_id"] for row in rows]), **scores, affine_transformed=affine_transformed, member_predictions=member_predictions)
        context_records = []
        for pool_index, pool in enumerate(pools):
            offset = pool_index * 64
            affine_order = np.lexsort((np.arange(64), -scores["affine_only"][offset:offset + 64]))
            frozen_order = np.lexsort((np.arange(64), -p[offset:offset + 64]))
            if not np.array_equal(affine_order, frozen_order):
                raise ValueError("affine negative control changed target ranking")
            for cap in config["target"]["caps"]:
                for seed in config["target"]["seeds"]:
                    proposals = proposal_indices(64, seed, cap)
                    for model, values in scores.items():
                        selected = max(proposals, key=lambda index: (-values[offset + index], index))
                        if model == "affine_only":
                            frozen_selected = max(proposals, key=lambda index: (-p[offset + index], index))
                            if selected != frozen_selected:
                                raise ValueError("affine negative control changed target selection")
                        context_records.append({
                            "pool_id": pool[0]["pool_id"], "pool_index": pool_index, "model": model,
                            "proposal_cap": cap, "seed": seed, "proposal_indices": proposals,
                            "selected_index": selected, "selected_id": pool[selected]["sequence_id"],
                            "regret": None, "selector_regret": None,
                        })
        if any(row["regret"] is not None for row in context_records):
            raise PermissionError("selection regret must be null")
        selection = {
            "schema_version": "1.0", "context": context, "records": context_records,
            "execution_metadata": {"outer_labels_read": False, "outer_metrics_computed": False, "actual_cuda_inference": True, "device": "cuda:0", "cpu_fallback": False, "target_freeze_sha256": digest(root / "target_freeze.json"), "affine_effective_slope": effective_slope},
            "inference": inference, "scores_sha256": digest(context_dir / "scores.npz"),
        }
        path = context_dir / "selection.json"
        save_json(path, selection)
        records[context] = {"selection_sha256": digest(path), "records": len(context_records), "regret_null": True}
    save_json(root / "target" / "target_selection_freeze.json", {"target_freeze_sha256": digest(root / "target_freeze.json"), "contexts": records, "labels_read": False, "evaluation_performed": False})
    return records


def holm(rows):
    ordered = sorted(rows, key=lambda row: (row["p_two_sided"], row["context"], row["cap"]))
    running = 0.0
    for rank, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * row["p_two_sided"]))
        row["holm_p"] = running


def evaluate_target(config_path, workspace):
    config_path = Path(config_path)
    config = load_json(config_path)
    root = Path(config["artifact_root"])
    verify_frozen_inputs(config_path, workspace, root)
    marker = load_json(root / "target" / "target_selection_freeze.json")
    if marker["labels_read"] or marker["evaluation_performed"]:
        raise PermissionError("target evaluation may run only once")
    if (root / "target" / "evaluation_started.json").exists():
        raise PermissionError("target evaluation has already started and cannot be repeated")
    save_json(root / "target" / "evaluation_started.json", {"evaluation_count": 1, "target_selection_freeze_sha256": digest(root / "target" / "target_selection_freeze.json")})
    for context, item in marker["contexts"].items():
        path = root / "target" / context / "selection.json"
        if digest(path) != item["selection_sha256"]:
            raise PermissionError("target selection hash mismatch")
        selection = load_json(path)
        if selection["execution_metadata"]["outer_labels_read"] or any(row["regret"] is not None for row in selection["records"]):
            raise PermissionError("invalid pre-evaluation selection")
    all_summaries = []
    per_pool_summaries = []
    secondary = []
    cap16_pool_effects = []
    baseline_cap16 = []
    for context_index, context in enumerate(config["target"]["contexts"]):
        selection = load_json(root / "target" / context / "selection.json")
        labels = {}
        with gzip.open(Path(config["pool_root"]) / context / "evaluator" / "labels.tsv.gz", "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                labels[row["sequence_id"]] = float(row["measured_rl"])
        pools = load_learner_pools(config["pool_root"], context)
        pool_max = {pool[0]["pool_id"]: max(labels[row["sequence_id"]] for row in pool) for pool in pools}
        evaluated = []
        for row in selection["records"]:
            regret = float(pool_max[row["pool_id"]] - labels[row["selected_id"]])
            evaluated.append({**row, "regret": regret, "selector_regret": regret})
        save_json(root / "target" / context / "evaluation.json", {"context": context, "records": evaluated, "outer_metrics_computed": True, "labels_returned_to_learner": False, "target_freeze_sha256": digest(root / "target_freeze.json")})
        for cap in config["target"]["caps"]:
            for pool_index in range(16):
                pool_id = f"castillo_{context}_{pool_index:03d}"
                for model in config["target"]["models"]:
                    values = [row["regret"] for row in evaluated if row["proposal_cap"] == cap and row["pool_index"] == pool_index and row["model"] == model]
                    per_pool_summaries.append({"context": context, "pool_id": pool_id, "pool_index": pool_index, "cap": cap, "model": model, "mean_regret": float(np.mean(values)), "seed_regret": values})
            for model in config["target"]["models"]:
                values = [row["regret"] for row in evaluated if row["proposal_cap"] == cap and row["model"] == model]
                all_summaries.append({"context": context, "cap": cap, "model": model, "mean_regret": float(np.mean(values)), "records": len(values)})
            for model in ("corrected_cnn", "affine_only"):
                effects = []
                pool_effects = []
                for pool_index in range(16):
                    frozen = [row["regret"] for row in evaluated if row["proposal_cap"] == cap and row["pool_index"] == pool_index and row["model"] == "frozen_cnn"]
                    changed = [row["regret"] for row in evaluated if row["proposal_cap"] == cap and row["pool_index"] == pool_index and row["model"] == model]
                    effects.extend(np.asarray(changed) - np.asarray(frozen))
                    pool_effects.append(float(np.mean(changed) - np.mean(frozen)))
                result = cluster_bootstrap(pool_effects, config["target"]["bootstrap_seed"] + context_index, config["target"]["bootstrap_replicates"])
                row = {"context": context, "cap": cap, "model": model, "contrast": f"{model}_minus_frozen_cnn", **result, "pool_effects": pool_effects}
                all_summaries.append(row)
                if model == "corrected_cnn" and cap in (2, 4, 8):
                    secondary.append(row)
                if model == "corrected_cnn" and cap == 16:
                    cap16_pool_effects.extend(pool_effects)
                    baseline_cap16.append(next(item["mean_regret"] for item in all_summaries if item.get("context") == context and item.get("cap") == 16 and item.get("model") == "frozen_cnn"))
    combined = cluster_bootstrap(cap16_pool_effects, config["target"]["bootstrap_seed"], config["target"]["bootstrap_replicates"])
    baseline_mean = float(np.mean(baseline_cap16))
    threshold = max(0.1, 0.05 * baseline_mean)
    repeat_rows = [row for row in all_summaries if row.get("model") == "corrected_cnn" and row.get("cap") == 16 and "contrast" in row]
    success = all(row["effect"] < 0 for row in repeat_rows) and combined["ci_high"] < 0 and -combined["effect"] >= threshold
    holm(secondary)
    summary = {
        "schema_version": "1.0", "analysis_id": config["analysis_id"], "results": all_summaries, "per_pool_results": per_pool_summaries,
        "secondary_holm": [{k: v for k, v in row.items() if k != "pool_effects"} for row in secondary],
        "primary": {"cap": 16, "combined_corrected_minus_frozen": combined, "combined_frozen_mean_regret": baseline_mean, "minimum_absolute_improvement": threshold, "absolute_improvement": -combined["effect"], "both_repeats_improved": all(row["effect"] < 0 for row in repeat_rows), "main_success": success},
        "evaluation_count": 1, "labels_returned_to_learner": False, "evidence_boundary": config["evidence_boundary"],
    }
    save_json(root / "target" / "summary.json", summary)
    save_json(root / "target" / "evaluation_complete.json", {"evaluation_count": 1, "summary_sha256": digest(root / "target" / "summary.json"), "main_success": success})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "select", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_target(args.config, args.workspace)
    elif args.command == "select":
        select_target(args.config, args.workspace)
    else:
        evaluate_target(args.config, args.workspace)


if __name__ == "__main__":
    main()
