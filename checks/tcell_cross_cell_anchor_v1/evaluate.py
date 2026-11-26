from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from anchor import (
    ANALYSIS_ID, bootstrap_clusters, cap_regret, digest, fusion_scores,
    holm_adjust, oracle_decomposition, random_regret, save_json, spearman,
    valid_sequence,
)


def sequence(row):
    return row.get("UTR") or row["utr"]


def read_selected_target_labels(path, selected):
    wanted = set(selected)
    labels = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            value = sequence(row)
            if value in wanted and valid_sequence(value):
                if value in labels:
                    raise ValueError("duplicate selected target sequence")
                if not row["rl"]:
                    raise ValueError("missing selected target MRL")
                label = float(row["rl"])
                if not np.isfinite(label):
                    raise ValueError("nonfinite selected target MRL")
                labels[value] = label
    if set(labels) != wanted:
        raise ValueError("selected target labels are incomplete")
    return np.asarray([labels[value] for value in selected], dtype=np.float64)


def selection_turnover(anchor_regret, cnn_regret, anchor_selected, cnn_selected):
    changed = anchor_selected != cnn_selected
    effect = anchor_regret - cnn_regret
    return {
        "total_pool_seed_comparisons": int(effect.size),
        "changed": int(np.sum(changed)),
        "improved_among_changed": int(np.sum(changed & (effect < 0))),
        "worsened_among_changed": int(np.sum(changed & (effect > 0))),
        "unchanged_label_among_changed": int(np.sum(changed & (effect == 0))),
        "mean_selected_mrl_change_anchor_minus_cnn": float(np.mean(-effect)),
    }


def top_tail_metrics(cnn, hepg2, labels, pools):
    top1_flips = 0
    jaccards = []
    flipped_pairs = 0
    tcell_favor_hepg2 = 0
    tcell_favor_cnn = 0
    tcell_tied = 0
    for pool in pools:
        cnn_order = np.argsort(-cnn[pool], kind="stable")
        hepg2_order = np.argsort(-hepg2[pool], kind="stable")
        top1_flips += int(cnn_order[0] != hepg2_order[0])
        cnn_tail = pool[cnn_order[:16]]
        hepg2_tail = set(pool[hepg2_order[:16]].tolist())
        cnn_tail_set = set(cnn_tail.tolist())
        jaccards.append(len(cnn_tail_set & hepg2_tail) / len(cnn_tail_set | hepg2_tail))
        for left in range(16):
            for right in range(left + 1, 16):
                i, j = int(cnn_tail[left]), int(cnn_tail[right])
                cnn_sign = np.sign(cnn[i] - cnn[j])
                hepg2_sign = np.sign(hepg2[i] - hepg2[j])
                if cnn_sign == 0 or hepg2_sign == 0 or cnn_sign == hepg2_sign:
                    continue
                flipped_pairs += 1
                target_sign = np.sign(labels[i] - labels[j])
                if target_sign == 0:
                    tcell_tied += 1
                elif target_sign == hepg2_sign:
                    tcell_favor_hepg2 += 1
                else:
                    tcell_favor_cnn += 1
    return {
        "top1_flip_pools": int(top1_flips), "pools": len(pools),
        "top16_membership_jaccard_mean": float(np.mean(jaccards)),
        "cnn_top16_predictor_order_flipped_pairs": int(flipped_pairs),
        "flipped_pairs_tcell_favor_hepg2": int(tcell_favor_hepg2),
        "flipped_pairs_tcell_favor_cnn": int(tcell_favor_cnn),
        "flipped_pairs_tcell_tied": int(tcell_tied),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_hash = digest(config_path)
    config = json.loads(config_path.read_text())
    if config["analysis_id"] != ANALYSIS_ID:
        raise ValueError("wrong analysis")
    root = Path(config["artifact_root"])
    prepare = json.loads((root / "prepare_manifest.json").read_text())
    if prepare["status"] != "PASS" or prepare["config_sha256"] != config_hash or not prepare["target_evaluation_permitted"]:
        raise RuntimeError("preflight does not permit target evaluation")
    target = root / "target"
    target.mkdir(parents=True, exist_ok=False)
    sentinel = target / "ONE_EVALUATION_STARTED.json"
    with sentinel.open("x") as handle:
        handle.write(json.dumps({"config_sha256": config_hash, "evaluation_count": 1}, indent=2) + "\n")

    sequences_array = np.load(root / "sequences.npy", allow_pickle=False)
    selected_sequences = [value.decode("ascii") for value in sequences_array]
    pools = np.load(root / "pools.npy", allow_pickle=False)
    proposals = np.load(root / "proposals.npy", allow_pickle=False)
    cnn = np.load(root / "cnn_predictions.npy", allow_pickle=False).astype(np.float64)
    hepg2 = np.load(root / "hepg2_labels.npy", allow_pickle=False).astype(np.float64)
    fused, _, _ = fusion_scores(cnn, hepg2, pools)
    scores = {"cnn": cnn, "hepg2_anchor": hepg2, "rank_fusion_50_50": fused}
    caps = config["selection"]["caps"]
    labels_by_repeat = {}
    regrets = {}
    selections = {}
    for repeat in ("defined_tcell_r1", "defined_tcell_r2"):
        labels = read_selected_target_labels(config["raw"][repeat], selected_sequences)
        labels_by_repeat[repeat] = labels
        np.save(target / f"{repeat}_labels.npy", labels)
        for method, values in scores.items():
            regrets[method, repeat] = []
            selections[method, repeat] = []
            for cap in caps:
                regret, chosen = cap_regret(values, labels, pools, proposals, cap)
                regrets[method, repeat].append(regret)
                selections[method, repeat].append(chosen)
            regrets[method, repeat] = np.stack(regrets[method, repeat])
            selections[method, repeat] = np.stack(selections[method, repeat])
            np.save(target / f"{repeat}_{method}_regret.npy", regrets[method, repeat])
            np.save(target / f"{repeat}_{method}_selected.npy", selections[method, repeat])
        regrets["random", repeat] = []
        selections["random", repeat] = []
        for cap in caps:
            regret, chosen = random_regret(labels, pools, proposals, cap)
            regrets["random", repeat].append(regret)
            selections["random", repeat].append(chosen)
        regrets["random", repeat] = np.stack(regrets["random", repeat])
        selections["random", repeat] = np.stack(selections["random", repeat])
        np.save(target / f"{repeat}_random_regret.npy", regrets["random", repeat])

    primary_index = caps.index(config["selection"]["primary_cap"])
    repeat_cap_results = {}
    p_values = []
    p_keys = []
    bootstrap_seed = config["primary_success"]["bootstrap_seed"]
    for repeat_index, repeat in enumerate(("defined_tcell_r1", "defined_tcell_r2")):
        repeat_cap_results[repeat] = {}
        for cap_index, cap in enumerate(caps):
            cnn_regret = regrets["cnn", repeat][cap_index]
            anchor_regret = regrets["hepg2_anchor", repeat][cap_index]
            effect = anchor_regret - cnn_regret
            cap_seed_offset = config["multiplicity"]["cap_bootstrap_seed_offset"][str(cap)]
            repeat_seed_offset = config["multiplicity"]["repeat_bootstrap_seed_offset"][repeat]
            effect_boot = bootstrap_clusters(effect.mean(axis=1), bootstrap_seed + repeat_seed_offset + cap_seed_offset)
            p_values.append(effect_boot["p_two_sided"])
            p_keys.append((repeat, cap))
            method_regret = {method: float(np.mean(regrets[method, repeat][cap_index])) for method in ("cnn", "hepg2_anchor", "rank_fusion_50_50", "random")}
            oracle = {}
            for method in ("cnn", "hepg2_anchor", "rank_fusion_50_50"):
                opportunity, headroom = oracle_decomposition(
                    labels_by_repeat[repeat], pools, proposals,
                    selections[method, repeat][cap_index], cap,
                )
                oracle[method] = {
                    "oracle_opportunity_loss_mean": float(opportunity.mean()),
                    "proposal_headroom_mean": float(headroom.mean()),
                }
            repeat_cap_results[repeat][str(cap)] = {
                "mean_regret": method_regret,
                "anchor_minus_cnn": effect_boot,
                "fusion_minus_cnn_mean": float(np.mean(regrets["rank_fusion_50_50", repeat][cap_index] - cnn_regret)),
                "turnover_anchor_vs_cnn": selection_turnover(
                    anchor_regret, cnn_regret,
                    selections["hepg2_anchor", repeat][cap_index],
                    selections["cnn", repeat][cap_index],
                ),
                "oracle": oracle,
            }
    adjusted = holm_adjust(p_values)
    for (repeat, cap), p_adjusted in zip(p_keys, adjusted):
        repeat_cap_results[repeat][str(cap)]["anchor_minus_cnn"]["p_holm_10_repeat_cap_tests"] = float(p_adjusted)

    combined_anchor_effect = np.mean(np.stack([
        regrets["hepg2_anchor", "defined_tcell_r1"][primary_index] - regrets["cnn", "defined_tcell_r1"][primary_index],
        regrets["hepg2_anchor", "defined_tcell_r2"][primary_index] - regrets["cnn", "defined_tcell_r2"][primary_index],
    ]), axis=0)
    combined_boot = bootstrap_clusters(combined_anchor_effect.mean(axis=1), bootstrap_seed)
    repeat_effects = {
        repeat: float(np.mean(regrets["hepg2_anchor", repeat][primary_index] - regrets["cnn", repeat][primary_index]))
        for repeat in ("defined_tcell_r1", "defined_tcell_r2")
    }
    cnn_baseline = float(np.mean(np.stack([
        regrets["cnn", "defined_tcell_r1"][primary_index],
        regrets["cnn", "defined_tcell_r2"][primary_index],
    ])))
    threshold = max(0.1, 0.05 * cnn_baseline)
    success = (
        all(value < 0 for value in repeat_effects.values())
        and combined_boot["ci_high"] < 0
        and -combined_boot["effect"] >= threshold
        and prepare["isolation"] == {"within_and_cross_pool_exact": 0, "within_and_cross_pool_h1": 0, "sample_exact": 0, "sample_h1": 0}
        and prepare["inference"]["actual_cuda"] and not prepare["inference"]["cpu_fallback"]
    )
    primary = {
        "analysis_id": ANALYSIS_ID, "evaluation_count": 1, "primary_cap": 16,
        "repeat_anchor_minus_cnn_effects": repeat_effects,
        "combined_anchor_minus_cnn": combined_boot,
        "combined_cnn_baseline_regret": cnn_baseline,
        "minimum_required_absolute_reduction": threshold,
        "absolute_reduction": float(-combined_boot["effect"]),
        "both_repeats_improve": all(value < 0 for value in repeat_effects.values()),
        "success": bool(success),
        "evidence_boundary": config["evidence_boundary"],
        "post_evaluation_protocol_changes": 0,
    }
    fusion_combined = np.mean(np.stack([
        regrets["rank_fusion_50_50", repeat][primary_index] - regrets["cnn", repeat][primary_index]
        for repeat in ("defined_tcell_r1", "defined_tcell_r2")
    ]), axis=0)
    associations = {}
    tail = {}
    for repeat, labels in labels_by_repeat.items():
        associations[repeat] = {}
        for method in ("cnn", "hepg2_anchor"):
            within = [spearman(scores[method][pool], labels[pool]) for pool in pools]
            valid = [value for value in within if value is not None]
            associations[repeat][method] = {
                "overall_spearman": spearman(scores[method], labels),
                "within_pool_median": float(np.median(valid)),
                "within_pool_min": float(np.min(valid)), "within_pool_max": float(np.max(valid)),
                "valid_pools": len(valid),
            }
        tail[repeat] = top_tail_metrics(cnn, hepg2, labels, pools)
    secondary = {
        "spearman": associations, "top_tail": tail,
        "fusion_cap16_combined_minus_cnn": bootstrap_clusters(fusion_combined.mean(axis=1), bootstrap_seed),
        "fusion_weight_fixed_before_results": 0.5,
    }
    results = {
        "analysis_id": ANALYSIS_ID, "config_sha256": config_hash,
        "evaluation_count": 1, "repeat_linked_pools": len(pools),
        "proposal_seeds": config["selection"]["proposal_seeds"], "caps": caps,
        "by_repeat_and_cap": repeat_cap_results, "primary": primary, "secondary": secondary,
        "evidence_boundary": config["evidence_boundary"],
        "sequence_independent": False, "new_blind_test": False,
    }
    save_json(target / "results.json", results)
    save_json(target / "primary.json", primary)
    save_json(target / "secondary.json", secondary)
    save_json(target / "ONE_EVALUATION_COMPLETED.json", {"config_sha256": config_hash, "evaluation_count": 1, "results_sha256": digest(target / "results.json")})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
