from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from adaptation import (
    cap_regret, cluster_bootstrap, digest, holm_adjust, primary_success,
    random_regret, save_json, spearman, valid_sequence,
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
                    raise ValueError("invalid selected test-v2 row")
                label = float(row["rl"])
                if not np.isfinite(label):
                    raise ValueError("nonfinite test-v2 MRL")
                labels[value] = label
    if set(labels) != wanted or len(labels) != 1024:
        raise ValueError("incomplete test-v2 labels")
    return np.asarray([labels[value] for value in selected], dtype=np.float64)


def turnover(method_regret, baseline_regret, method_selected, baseline_selected):
    changed = method_selected != baseline_selected
    effect = method_regret - baseline_regret
    return {
        "comparisons": int(effect.size), "changed": int(np.sum(changed)),
        "improved_among_changed": int(np.sum(changed & (effect < 0))),
        "worsened_among_changed": int(np.sum(changed & (effect > 0))),
        "same_label_among_changed": int(np.sum(changed & (effect == 0))),
        "mean_selected_mrl_change": float(np.mean(-effect)),
    }


def top_tail(baseline, listnet, labels, pools):
    top1_flips = 0
    jaccards = []
    flipped = favor_listnet = favor_baseline = tied = 0
    listnet_correct = baseline_correct = nontied_pairs = 0
    for pool in pools:
        baseline_order = np.argsort(-baseline[pool], kind="stable")
        listnet_order = np.argsort(-listnet[pool], kind="stable")
        top1_flips += int(baseline_order[0] != listnet_order[0])
        baseline_tail = pool[baseline_order[:16]]
        listnet_tail = pool[listnet_order[:16]]
        jaccards.append(len(set(baseline_tail) & set(listnet_tail)) / len(set(baseline_tail) | set(listnet_tail)))
        for left in range(16):
            for right in range(left + 1, 16):
                i, j = int(baseline_tail[left]), int(baseline_tail[right])
                baseline_sign = np.sign(baseline[i] - baseline[j])
                listnet_sign = np.sign(listnet[i] - listnet[j])
                target_sign = np.sign(labels[i] - labels[j])
                if target_sign != 0 and baseline_sign != 0:
                    nontied_pairs += 1
                    baseline_correct += int(baseline_sign == target_sign)
                    listnet_correct += int(listnet_sign == target_sign)
                if baseline_sign == 0 or listnet_sign == 0 or baseline_sign == listnet_sign:
                    continue
                flipped += 1
                if target_sign == 0:
                    tied += 1
                elif target_sign == listnet_sign:
                    favor_listnet += 1
                else:
                    favor_baseline += 1
    return {
        "top1_flip_pools": int(top1_flips), "pools": len(pools),
        "top16_jaccard_mean": float(np.mean(jaccards)),
        "baseline_top16_predictor_order_flips": flipped,
        "flips_target_favor_listnet": favor_listnet,
        "flips_target_favor_baseline": favor_baseline,
        "flips_target_tied": tied,
        "baseline_tail_pair_accuracy": float(baseline_correct / nontied_pairs),
        "listnet_tail_pair_accuracy": float(listnet_correct / nontied_pairs),
        "tail_pair_accuracy_change": float((listnet_correct - baseline_correct) / nontied_pairs),
    }


def pointwise(scores, labels):
    error = scores - labels
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)), "spearman": spearman(scores, labels),
    }


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
    predictions_frozen = json.loads((root / "TEST_PREDICTIONS_FROZEN.json").read_text())
    if prepare["test_v2_mrl_read"] or frozen["status"] != "PASS" or frozen["run_count"] != 81 or predictions_frozen["status"] != "PASS" or predictions_frozen["test_v2_mrl_read"] or not predictions_frozen["label_evaluation_now_permitted_once"] or predictions_frozen["config_sha256"] != config_hash:
        raise RuntimeError("model/prediction freeze gate failed")
    target = root / "target_v2"
    target.mkdir(parents=True, exist_ok=False)
    with (target / "ONE_EVALUATION_STARTED.json").open("x") as handle:
        handle.write(json.dumps({"config_sha256": config_hash, "prediction_freeze_sha256": digest(root / "TEST_PREDICTIONS_FROZEN.json"), "evaluation_count": 1}, indent=2) + "\n")

    methods = config["evaluation"]["methods"]
    scores = {}
    for method in methods:
        path = root / "test_predictions" / f"{method}_scores.npy"
        if digest(path) != predictions_frozen["methods"][method]["score_sha256"]:
            raise ValueError("frozen test-v2 score hash mismatch")
        scores[method] = np.load(path, allow_pickle=False)
    sequences_all = np.load(root / "data" / "sequences.npy", allow_pickle=False)
    test_indices = np.load(root / "data" / "test_v2_indices.npy", allow_pickle=False)
    sequences = sequences_all[test_indices]
    pools_global = np.load(root / "data" / "test_v2_pools.npy", allow_pickle=False)
    lookup = np.full(len(sequences_all), -1, dtype=np.int64)
    lookup[test_indices] = np.arange(len(test_indices))
    pools = lookup[pools_global]
    proposals = np.load(root / "data" / "test_v2_proposals.npy", allow_pickle=False)

    labels_by_repeat = {
        repeat: read_test_labels(config["raw"][repeat], [value.decode("ascii") for value in sequences])
        for repeat in ("defined_tcell_r1", "defined_tcell_r2")
    }
    for repeat, labels in labels_by_repeat.items():
        np.save(target / f"{repeat}_labels.npy", labels)
    labels_by_context = dict(labels_by_repeat)
    labels_by_context["mean_replicate"] = (labels_by_repeat["defined_tcell_r1"] + labels_by_repeat["defined_tcell_r2"]) / 2
    np.save(target / "mean_replicate_labels.npy", labels_by_context["mean_replicate"])

    caps = config["selection"]["caps"]
    regrets, selections = {}, {}
    for context, labels in labels_by_context.items():
        for method in methods:
            regrets[method, context], selections[method, context] = [], []
            for cap in caps:
                regret, selected = cap_regret(scores[method], labels, pools, proposals, cap)
                regrets[method, context].append(regret)
                selections[method, context].append(selected)
            regrets[method, context] = np.stack(regrets[method, context])
            selections[method, context] = np.stack(selections[method, context])
            np.save(target / f"{context}_{method}_regret.npy", regrets[method, context])
            np.save(target / f"{context}_{method}_selected.npy", selections[method, context])
        regrets["random", context] = np.stack([random_regret(labels, pools, proposals, cap)[0] for cap in caps])

    seed = config["evaluation"]["bootstrap_seed"]
    context_offsets = {"mean_replicate": 0, "defined_tcell_r1": 10000, "defined_tcell_r2": 20000}
    baseline_offsets = {"original_cnn": 0, "v1_main": 1000}
    p_values, p_keys = [], []
    by_context_cap = {}
    for context in labels_by_context:
        by_context_cap[context] = {}
        for cap_index, cap in enumerate(caps):
            row = {"mean_regret": {method: float(regrets[method, context][cap_index].mean()) for method in methods + ["random"]}, "listnet_comparisons": {}}
            for baseline in ("original_cnn", "v1_main"):
                effect = regrets["listnet", context][cap_index] - regrets[baseline, context][cap_index]
                boot = cluster_bootstrap(effect, seed + context_offsets[context] + baseline_offsets[baseline] + config["evaluation"]["cap_bootstrap_seed_offset"][str(cap)])
                comparison = {
                    "effect": boot,
                    "turnover": turnover(regrets["listnet", context][cap_index], regrets[baseline, context][cap_index], selections["listnet", context][cap_index], selections[baseline, context][cap_index]),
                }
                if context == "mean_replicate":
                    p_values.append(boot["p_two_sided"])
                    p_keys.append((context, cap, baseline))
                row["listnet_comparisons"][baseline] = comparison
            by_context_cap[context][str(cap)] = row
    adjusted = holm_adjust(p_values)
    for (context, cap, baseline), value in zip(p_keys, adjusted):
        by_context_cap[context][str(cap)]["listnet_comparisons"][baseline]["effect"]["p_holm_10_comparison_cap_tests"] = float(value)

    primary_index = caps.index(16)
    primary_rows = {}
    for baseline in ("original_cnn", "v1_main"):
        effect = by_context_cap["mean_replicate"]["16"]["listnet_comparisons"][baseline]["effect"]
        baseline_mean = float(regrets[baseline, "mean_replicate"][primary_index].mean())
        primary_rows[baseline] = {
            "effect": effect, "baseline_cap16_regret": baseline_mean,
            "minimum_required_absolute_reduction": max(0.1, 0.05 * baseline_mean),
            "absolute_reduction": float(-effect["effect"]),
            "repeat_effects": {
                repeat: float(np.mean(regrets["listnet", repeat][primary_index] - regrets[baseline, repeat][primary_index]))
                for repeat in labels_by_repeat
            },
        }
    success = primary_success(
        primary_rows,
        isolation_pass=prepare["cross_role_exact_h1"] == 0 and prepare["test_v2_to_v1_exact_h1"] == 0 and prepare["test_sample_exact"] == 0 and prepare["test_sample_h1"] == 0 and prepare["cross_test_pool_exact_h1"] == 0,
        convergence_pass=frozen["status"] == "PASS" and predictions_frozen["gpu"]["actual_cuda"] and not predictions_frozen["gpu"]["cpu_fallback"],
    )
    primary = {
        "evaluation_count": 1, "estimand": "r1/r2 arithmetic-mean measured-MRL regret",
        "comparisons": primary_rows, **success,
        "repeat_results_are_secondary": True, "post_evaluation_protocol_changes": 0,
    }

    errors, tail, synchronization = {}, {}, {}
    for context, labels in labels_by_context.items():
        errors[context] = {method: pointwise(scores[method], labels) for method in methods}
        tail[context] = {}
        synchronization[context] = {}
        for baseline in ("original_cnn", "v1_main"):
            tail[context][baseline] = top_tail(scores[baseline], scores["listnet"], labels, pools)
            error_change = errors[context]["listnet"]["mae"] - errors[context][baseline]["mae"]
            regret_change = float(np.mean(regrets["listnet", context][primary_index] - regrets[baseline, context][primary_index]))
            pool_mae = np.asarray([np.mean(np.abs(scores["listnet"][pool] - labels[pool])) for pool in pools])
            pool_regret = regrets["listnet", context][primary_index].mean(axis=1)
            synchronization[context][baseline] = {
                "mae_change": error_change, "cap16_regret_change": regret_change,
                "mae_and_regret_change_same_direction": bool(np.sign(error_change) == np.sign(regret_change)),
                "pool_mae_regret_spearman": spearman(pool_mae, pool_regret),
            }
    results = {
        "analysis_id": config["analysis_id"], "config_sha256": config_hash,
        "training_freeze_sha256": digest(root / "TRAINING_FROZEN.json"),
        "prediction_freeze_sha256": digest(root / "TEST_PREDICTIONS_FROZEN.json"),
        "evaluation_count": 1, "repeat_linked_pools": 16,
        "proposal_seeds": config["selection"]["proposal_seeds"], "caps": caps,
        "by_context_and_cap": by_context_cap, "primary": primary,
        "pointwise_errors_mrl_units": errors,
        "error_regret_synchronization": synchronization, "top_tail": tail,
        "evidence_boundary": config["evidence_boundary"],
        "historical_retrospective_boundary_retained": True,
        "sequence_independent": False, "new_prospective_study": False,
    }
    save_json(target / "results.json", results)
    save_json(target / "primary.json", primary)
    save_json(target / "secondary.json", {
        "by_repeat_and_cap": {repeat: by_context_cap[repeat] for repeat in labels_by_repeat},
        "pointwise_errors_mrl_units": errors,
        "error_regret_synchronization": synchronization, "top_tail": tail,
    })
    save_json(target / "ONE_EVALUATION_COMPLETED.json", {
        "config_sha256": config_hash, "evaluation_count": 1,
        "results_sha256": digest(target / "results.json"),
        "post_evaluation_protocol_changes": 0,
    })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
