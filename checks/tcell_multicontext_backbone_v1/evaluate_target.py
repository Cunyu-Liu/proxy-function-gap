from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from backbone import cap_regret, cluster_bootstrap, digest, holm_adjust, save_json, valid_sequence
from select_and_gate import load_ensemble, predict


def selected_labels(path, sequences):
    wanted = set(value.decode("ascii") for value in sequences)
    labels = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = row.get("UTR") or row["utr"]
            if sequence in wanted and valid_sequence(sequence):
                if not row["rl"]:
                    raise ValueError("missing selected target MRL")
                labels[sequence] = float(row["rl"])
    if set(labels) != wanted or not np.isfinite(list(labels.values())).all():
        raise ValueError("incomplete/nonfinite target labels")
    return np.asarray([labels[value.decode("ascii")] for value in sequences], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    gate = json.loads((root / "auxiliary_gate.json").read_text())
    if not gate["both_directions_pass"] or not gate["target_evaluation_permitted"] or gate["target_evaluation_count"] != 0:
        raise PermissionError("dual auxiliary gate did not pass; target forbidden")
    target = root / "target"
    target.mkdir(parents=True, exist_ok=False)
    with (target / "ONE_EVALUATION_STARTED.json").open("x") as handle:
        handle.write(json.dumps({"config_sha256": digest(config_path), "evaluation_count": 1}, indent=2) + "\n")
    anchor = Path(config["target_if_and_only_if_dual_gate_pass"]["anchor_root"])
    sequences = np.load(anchor / "sequences.npy", allow_pickle=False)
    pools = np.load(anchor / "pools.npy", allow_pickle=False)
    proposals = np.load(anchor / "proposals.npy", allow_pickle=False)
    scores = {"baseline": np.load(anchor / "cnn_predictions.npy", allow_pickle=False).astype(np.float64)}
    for variant in ("main", "original_mse_tail_rank"):
        selection = json.loads((root / "selection" / f"{variant}.json").read_text())["selected"]
        models, _ = load_ensemble(root, selection["name"], config)
        scores[variant] = predict(models, sequences)
    models, _ = load_ensemble(root, "extended_mse_only", config)
    scores["extended_mse_only"] = predict(models, sequences)
    for name, values in scores.items():
        np.save(target / f"{name}_predictions.npy", values)
    caps = config["target_if_and_only_if_dual_gate_pass"]["caps"]
    regrets = {}
    results = {"analysis_id": config["analysis_id"], "evaluation_count": 1, "by_repeat": {}, "evidence_boundary": config["evidence_boundary"], "new_blind_test": False}
    for repeat_index, repeat in enumerate(config["target_if_and_only_if_dual_gate_pass"]["contexts"]):
        raw = config["target_if_and_only_if_dual_gate_pass"]["raw"][repeat]
        if digest(raw) != config["target_if_and_only_if_dual_gate_pass"]["raw_sha256"][repeat]:
            raise ValueError("target raw hash changed")
        labels = selected_labels(raw, sequences)
        np.save(target / f"{repeat}_labels.npy", labels)
        results["by_repeat"][repeat] = {}
        for model, values in scores.items():
            regrets[repeat, model] = []
            for cap in caps:
                regret, selected = cap_regret(values, labels, pools, proposals, cap)
                regrets[repeat, model].append(regret)
                results["by_repeat"][repeat].setdefault(str(cap), {})[model] = float(regret.mean())
            regrets[repeat, model] = np.stack(regrets[repeat, model])
            np.save(target / f"{repeat}_{model}_regret.npy", regrets[repeat, model])
    p_values = []
    p_keys = []
    for repeat_index, repeat in enumerate(config["target_if_and_only_if_dual_gate_pass"]["contexts"]):
        for cap in (2, 4, 8):
            cap_index = caps.index(cap)
            effect = regrets[repeat, "main"][cap_index] - regrets[repeat, "baseline"][cap_index]
            boot = cluster_bootstrap(effect, 20261124 + repeat_index * 1000 + cap * 10)
            p_values.append(boot["p_two_sided"])
            p_keys.append((repeat, cap))
            results["by_repeat"][repeat][str(cap)]["main_secondary_effect"] = boot
    for (repeat, cap), value in zip(p_keys, holm_adjust(p_values)):
        results["by_repeat"][repeat][str(cap)]["main_secondary_effect"]["p_holm_6"] = float(value)
    primary_index = caps.index(16)
    repeat_effects = {}
    combined_parts = []
    baseline_parts = []
    for repeat in config["target_if_and_only_if_dual_gate_pass"]["contexts"]:
        effect = regrets[repeat, "main"][primary_index] - regrets[repeat, "baseline"][primary_index]
        repeat_effects[repeat] = float(effect.mean())
        combined_parts.append(effect)
        baseline_parts.append(regrets[repeat, "baseline"][primary_index])
    combined = np.mean(np.stack(combined_parts), axis=0)
    boot = cluster_bootstrap(combined, config["target_if_and_only_if_dual_gate_pass"]["bootstrap_seed"], config["target_if_and_only_if_dual_gate_pass"]["bootstrap_replicates"])
    baseline = float(np.mean(baseline_parts))
    threshold = max(0.1, 0.05 * baseline)
    training_pass = True
    for name in [gate["selected_main"]["name"]]:
        for seed in config["model"]["seeds"]:
            summary = json.loads((root / "training" / name / f"seed_{seed}" / "summary.json").read_text())
            training_pass &= summary["converged_dev_plateau"] and summary["finite_losses"] and summary["gpu"]["actual_cuda"] and not summary["gpu"]["cpu_fallback"] and summary["best_epoch"] > 0
    primary = {"repeat_main_minus_baseline_effects": repeat_effects, "combined_main_minus_baseline": boot, "combined_baseline_regret": baseline, "minimum_required_absolute_reduction": threshold, "absolute_reduction": -boot["effect"], "both_repeats_improve": all(value < 0 for value in repeat_effects.values()), "training_numerical_cuda_pass": bool(training_pass), "success": bool(all(value < 0 for value in repeat_effects.values()) and boot["ci_high"] < 0 and -boot["effect"] >= threshold and training_pass), "evaluation_count": 1, "post_evaluation_protocol_changes": 0}
    results["primary"] = primary
    save_json(target / "results.json", results)
    save_json(target / "primary.json", primary)
    save_json(target / "ONE_EVALUATION_COMPLETED.json", {"evaluation_count": 1, "results_sha256": digest(target / "results.json")})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
