from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from adaptation import (
    FreshSampleCNN, cap_regret, cluster_bootstrap, codes_to_one_hot,
    configure_cuda, digest, encode_codes, holm_adjust, random_regret,
    save_json, spearman, valid_sequence,
)


def sequence(row):
    return row.get("UTR") or row["utr"]


def read_test_labels(path, selected):
    wanted = set(selected)
    labels = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            value = sequence(row)
            if value in wanted:
                if not valid_sequence(value) or value in labels or not row["rl"]:
                    raise ValueError("invalid selected test row")
                label = float(row["rl"])
                if not np.isfinite(label):
                    raise ValueError("nonfinite test MRL")
                labels[value] = label
    if set(labels) != wanted or len(labels) != 1024:
        raise ValueError("incomplete test labels")
    return np.asarray([labels[value] for value in selected], dtype=np.float64)


def load_original_members(config):
    inference = json.loads(Path(config["model"]["inference_config"]).read_text())
    result = []
    for member in inference["members"]:
        if digest(member["checkpoint"]) != member["sha256"]:
            raise ValueError("original checkpoint hash mismatch")
        checkpoint = torch.load(member["checkpoint"], map_location="cpu", weights_only=False)
        result.append((checkpoint, float(checkpoint["scaler"]["mean"]), float(checkpoint["scaler"]["scale"])))
    return result


def load_adapted_members(root, selection_name):
    selection = json.loads((root / "selection" / selection_name).read_text())
    selected = selection["selected"]
    result = []
    for seed in (11, 23, 37):
        run = root / "training" / selected["name"] / f"seed_{seed}"
        summary = json.loads((run / "summary.json").read_text())
        if digest(run / "best.pt") != summary["best_checkpoint_sha256"]:
            raise ValueError("adapted checkpoint hash mismatch")
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        if checkpoint["model_state_sha256"] != summary["best_model_hash"]:
            raise ValueError("adapted model identity mismatch")
        result.append((checkpoint, float(summary["scaler"]["mean"]), float(summary["scaler"]["scale"])))
    return result


def predict_ensemble(members, codes, device):
    outputs = []
    synthetic = torch.zeros((2, 50), dtype=torch.uint8, device=device)
    with torch.no_grad():
        for checkpoint, mean, scale in members:
            model = FreshSampleCNN()
            model.load_state_dict(checkpoint["model_state"])
            model.to(device=device, dtype=torch.float32).eval()
            smoke = model(codes_to_one_hot(synthetic))
            if not smoke.is_cuda or not torch.isfinite(smoke).all():
                raise RuntimeError("CUDA model smoke failed")
            values = []
            for start in range(0, len(codes), 1024):
                values.append(model(codes_to_one_hot(codes[start:start + 1024])).cpu().numpy())
            outputs.append(np.concatenate(values).astype(np.float64) * scale + mean)
            del model
            torch.cuda.empty_cache()
    result = np.mean(outputs, axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite test predictions")
    return result


def turnover(method_regret, baseline_regret, method_selected, baseline_selected):
    changed = method_selected != baseline_selected
    effect = method_regret - baseline_regret
    return {"comparisons": int(effect.size), "changed": int(np.sum(changed)), "improved_among_changed": int(np.sum(changed & (effect < 0))), "worsened_among_changed": int(np.sum(changed & (effect > 0))), "same_label_among_changed": int(np.sum(changed & (effect == 0))), "mean_selected_mrl_change": float(np.mean(-effect))}


def top_tail(baseline, main, labels, pools):
    top1_flips = 0
    jaccards = []
    flipped = favor_main = favor_baseline = tied = 0
    main_correct = baseline_correct = nontied_pairs = 0
    for pool in pools:
        baseline_order = np.argsort(-baseline[pool], kind="stable")
        main_order = np.argsort(-main[pool], kind="stable")
        top1_flips += int(baseline_order[0] != main_order[0])
        baseline_tail = pool[baseline_order[:16]]
        main_tail = pool[main_order[:16]]
        jaccards.append(len(set(baseline_tail) & set(main_tail)) / len(set(baseline_tail) | set(main_tail)))
        for left in range(16):
            for right in range(left + 1, 16):
                i, j = int(baseline_tail[left]), int(baseline_tail[right])
                baseline_sign = np.sign(baseline[i] - baseline[j])
                main_sign = np.sign(main[i] - main[j])
                target_sign = np.sign(labels[i] - labels[j])
                if target_sign != 0 and baseline_sign != 0:
                    nontied_pairs += 1
                    baseline_correct += int(baseline_sign == target_sign)
                    main_correct += int(main_sign == target_sign)
                if baseline_sign == 0 or main_sign == 0 or baseline_sign == main_sign:
                    continue
                flipped += 1
                if target_sign == 0:
                    tied += 1
                elif target_sign == main_sign:
                    favor_main += 1
                else:
                    favor_baseline += 1
    return {"top1_flip_pools": int(top1_flips), "pools": len(pools), "top16_jaccard_mean": float(np.mean(jaccards)), "baseline_top16_predictor_order_flips": flipped, "flips_target_favor_main": favor_main, "flips_target_favor_baseline": favor_baseline, "flips_target_tied": tied, "baseline_tail_pair_accuracy": float(baseline_correct / nontied_pairs), "main_tail_pair_accuracy": float(main_correct / nontied_pairs), "tail_pair_accuracy_change": float((main_correct - baseline_correct) / nontied_pairs)}


def pointwise(scores, labels):
    error = scores - labels
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(np.mean(error)), "spearman": spearman(scores, labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_hash = digest(config_path)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    prepare = json.loads((root / "prepare_manifest.json").read_text())
    frozen = json.loads((root / "TRAINING_FROZEN.json").read_text())
    amendment_path = config_path.parent / "dev_repeat_amendment_frozen.json"
    dev_repeat = json.loads((root / "DEV_REPEAT_SELECTION_FROZEN.json").read_text())
    if prepare["test_mrl_read"] or frozen["status"] != "PASS" or frozen["config_sha256"] != config_hash or frozen["run_count"] != 72 or not frozen["test_now_permitted_once"] or frozen.get("dev_repeat_amendment_sha256") != digest(amendment_path) or dev_repeat["test_mrl_read"]:
        raise RuntimeError("training/test freeze gate failed")
    target = root / "target"
    target.mkdir(parents=True, exist_ok=False)
    with (target / "ONE_EVALUATION_STARTED.json").open("x") as handle:
        handle.write(json.dumps({"config_sha256": config_hash, "training_freeze_sha256": digest(root / "TRAINING_FROZEN.json"), "evaluation_count": 1}, indent=2) + "\n")

    device, gpu = configure_cuda(20261126, 0.28)
    sequences_all = np.load(root / "data" / "sequences.npy", allow_pickle=False)
    test_indices = np.load(root / "data" / "test_indices.npy", allow_pickle=False)
    sequences = sequences_all[test_indices]
    pools_global = np.load(root / "data" / "test_pools.npy", allow_pickle=False)
    lookup = np.full(len(sequences_all), -1, dtype=np.int64)
    lookup[test_indices] = np.arange(len(test_indices))
    pools = lookup[pools_global]
    proposals_global = np.load(root / "data" / "test_proposals.npy", allow_pickle=False)
    proposals = proposals_global
    codes = torch.from_numpy(encode_codes(sequences)).to(device)
    members = {"original_cnn": load_original_members(config)}
    for method, selection_name in (("main_50k", "main"), ("mse_only_50k", "mse_only"), ("from_scratch_50k", "from_scratch")):
        members[method] = load_adapted_members(root, f"{selection_name}.json")
    selected_main = json.loads((root / "selection" / "main.json").read_text())["selected"]
    nested_dev = json.loads((root / "selection" / "nested_learning_curve_dev.json").read_text())
    for budget in (1000, 5000, 20000):
        row = next(row for row in nested_dev["budgets"] if row["budget"] == budget)
        members[f"main_{budget // 1000}k"] = load_adapted_members(root, f"nested_{budget}.json")
    scores = {}
    for method in ("original_cnn", "main_50k", "mse_only_50k", "from_scratch_50k", "main_1k", "main_5k", "main_20k"):
        scores[method] = predict_ensemble(members[method], codes, device)
        np.save(target / f"{method}_scores.npy", scores[method])
    labels_by_repeat = {repeat: read_test_labels(config["raw"][repeat], [value.decode("ascii") for value in sequences]) for repeat in ("defined_tcell_r1", "defined_tcell_r2")}
    for repeat, labels in labels_by_repeat.items():
        np.save(target / f"{repeat}_labels.npy", labels)

    caps = config["selection"]["caps"]
    methods = config["evaluation"]["methods"]
    regrets, selections = {}, {}
    for repeat, labels in labels_by_repeat.items():
        for method in methods:
            regrets[method, repeat], selections[method, repeat] = [], []
            for cap in caps:
                regret, selected = cap_regret(scores[method], labels, pools, proposals, cap)
                regrets[method, repeat].append(regret)
                selections[method, repeat].append(selected)
            regrets[method, repeat] = np.stack(regrets[method, repeat])
            selections[method, repeat] = np.stack(selections[method, repeat])
            np.save(target / f"{repeat}_{method}_regret.npy", regrets[method, repeat])
            np.save(target / f"{repeat}_{method}_selected.npy", selections[method, repeat])
        regrets["random", repeat] = np.stack([random_regret(labels, pools, proposals, cap)[0] for cap in caps])
        for budget in (1000, 5000, 20000):
            method = f"main_{budget // 1000}k"
            regrets[method, repeat] = cap_regret(scores[method], labels, pools, proposals, 16)[0]

    seed = config["evaluation"]["bootstrap_seed"]
    p_values, p_keys = [], []
    by_repeat_cap = {}
    for repeat in labels_by_repeat:
        by_repeat_cap[repeat] = {}
        for cap_index, cap in enumerate(caps):
            baseline = regrets["original_cnn", repeat][cap_index]
            effect = regrets["main_50k", repeat][cap_index] - baseline
            boot = cluster_bootstrap(effect, seed + config["evaluation"]["repeat_bootstrap_seed_offset"][repeat] + config["evaluation"]["cap_bootstrap_seed_offset"][str(cap)])
            p_values.append(boot["p_two_sided"])
            p_keys.append((repeat, cap))
            row = {"mean_regret": {method: float(regrets[method, repeat][cap_index].mean()) for method in methods + ["random"]}, "main_minus_original": boot, "ablation_minus_original": {}, "main_turnover": turnover(regrets["main_50k", repeat][cap_index], baseline, selections["main_50k", repeat][cap_index], selections["original_cnn", repeat][cap_index])}
            for method in ("mse_only_50k", "from_scratch_50k"):
                row["ablation_minus_original"][method] = cluster_bootstrap(regrets[method, repeat][cap_index] - baseline, seed + config["evaluation"]["repeat_bootstrap_seed_offset"][repeat] + config["evaluation"]["cap_bootstrap_seed_offset"][str(cap)])
            by_repeat_cap[repeat][str(cap)] = row
    adjusted = holm_adjust(p_values)
    for (repeat, cap), value in zip(p_keys, adjusted):
        by_repeat_cap[repeat][str(cap)]["main_minus_original"]["p_holm_10_repeat_cap_tests"] = float(value)

    primary_index = caps.index(16)
    combined = np.mean(np.stack([regrets["main_50k", repeat][primary_index] - regrets["original_cnn", repeat][primary_index] for repeat in labels_by_repeat]), axis=0)
    combined_boot = cluster_bootstrap(combined, seed)
    repeat_effects = {repeat: float(np.mean(regrets["main_50k", repeat][primary_index] - regrets["original_cnn", repeat][primary_index])) for repeat in labels_by_repeat}
    baseline_mean = float(np.mean(np.stack([regrets["original_cnn", repeat][primary_index] for repeat in labels_by_repeat])))
    threshold = max(0.1, 0.05 * baseline_mean)
    success = all(value < 0 for value in repeat_effects.values()) and combined_boot["ci_high"] < 0 and -combined_boot["effect"] >= threshold and prepare["cross_role_exact_h1"] == 0 and prepare["test_sample_exact"] == 0 and prepare["test_sample_h1"] == 0 and gpu["actual_cuda"] and not gpu["cpu_fallback"]
    primary = {"evaluation_count": 1, "repeat_main_minus_original_effects": repeat_effects, "combined_main_minus_original": combined_boot, "combined_original_cap16_regret": baseline_mean, "minimum_required_absolute_reduction": threshold, "absolute_reduction": float(-combined_boot["effect"]), "both_repeats_improve": all(value < 0 for value in repeat_effects.values()), "success": bool(success), "post_evaluation_protocol_changes": 0}

    errors, tail, synchronization = {}, {}, {}
    for repeat, labels in labels_by_repeat.items():
        errors[repeat] = {method: pointwise(scores[method], labels) for method in ("original_cnn", "main_50k", "mse_only_50k", "from_scratch_50k")}
        tail[repeat] = top_tail(scores["original_cnn"], scores["main_50k"], labels, pools)
        synchronization[repeat] = {}
        for method in ("main_50k", "mse_only_50k", "from_scratch_50k"):
            error_change = errors[repeat][method]["mae"] - errors[repeat]["original_cnn"]["mae"]
            regret_change = float(np.mean(regrets[method, repeat][primary_index] - regrets["original_cnn", repeat][primary_index]))
            pool_mae = np.asarray([np.mean(np.abs(scores[method][pool] - labels[pool])) for pool in pools])
            pool_regret = regrets[method, repeat][primary_index].mean(axis=1)
            synchronization[repeat][method] = {"mae_change": error_change, "cap16_regret_change": regret_change, "mae_and_regret_change_same_direction": bool(np.sign(error_change) == np.sign(regret_change)), "pool_mae_regret_spearman": spearman(pool_mae, pool_regret)}
    learning_curve = []
    for budget in (1000, 5000, 20000, 50000):
        method = f"main_{budget // 1000}k"
        dev_row = next(row for row in nested_dev["budgets"] if row["budget"] == budget)
        row = {"budget": budget, "dev_cap16_regret_mean_replicate": dev_row["dev_cap16_regret"]}
        for repeat in labels_by_repeat:
            method_regret = regrets[method, repeat][primary_index] if budget == 50000 else regrets[method, repeat]
            row[f"{repeat}_cap16_regret"] = float(method_regret.mean())
            row[f"{repeat}_minus_original"] = float(np.mean(method_regret - regrets["original_cnn", repeat][primary_index]))
        learning_curve.append(row)
    results = {"analysis_id": config["analysis_id"], "config_sha256": config_hash, "dev_repeat_amendment_sha256": frozen["dev_repeat_amendment_sha256"], "dev_selection_metric": "maximum_of_r1_and_r2_cap16_regret", "training_freeze_sha256": digest(root / "TRAINING_FROZEN.json"), "evaluation_count": 1, "repeat_linked_pools": 16, "proposal_seeds": config["selection"]["proposal_seeds"], "caps": caps, "by_repeat_and_cap": by_repeat_cap, "primary": primary, "pointwise_errors_mrl_units": errors, "error_regret_synchronization": synchronization, "top_tail": tail, "nested_learning_curve": learning_curve, "gpu": gpu, "evidence_boundary": config["evidence_boundary"], "sequence_independent": False, "new_blind_test": False}
    save_json(target / "results.json", results)
    save_json(target / "primary.json", primary)
    save_json(target / "secondary.json", {"pointwise_errors_mrl_units": errors, "error_regret_synchronization": synchronization, "top_tail": tail, "nested_learning_curve": learning_curve})
    score_hashes = {method: digest(target / f"{method}_scores.npy") for method in scores}
    save_json(target / "ONE_EVALUATION_COMPLETED.json", {"config_sha256": config_hash, "evaluation_count": 1, "results_sha256": digest(target / "results.json"), "score_sha256": score_hashes, "post_evaluation_protocol_changes": 0})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
