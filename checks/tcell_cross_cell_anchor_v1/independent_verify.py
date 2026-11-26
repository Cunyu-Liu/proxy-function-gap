from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def independent_rank(values):
    values = np.asarray(values, dtype=float)
    result = np.empty(len(values), dtype=float)
    for value in np.unique(values):
        better = int(np.sum(values > value))
        tied = int(np.sum(values == value))
        result[values == value] = 1.0 - (better + (tied - 1) / 2.0) / (len(values) - 1)
    return result


def independent_spearman(left, right):
    a = independent_rank(left)
    b = independent_rank(right)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def independent_bootstrap(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    draws = values[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    p = min(1.0, 2.0 * min((1 + np.sum(draws <= 0)) / 10001, (1 + np.sum(draws >= 0)) / 10001))
    return {"effect": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "p_two_sided": float(p)}


def independent_holm(values):
    values = np.asarray(values)
    indexed = sorted(enumerate(values), key=lambda pair: (pair[1], pair[0]))
    output = np.empty(len(values))
    previous = 0.0
    for rank, (index, value) in enumerate(indexed):
        previous = max(previous, min(1.0, (len(values) - rank) * value))
        output[index] = previous
    return output


def independent_read_labels(path, selected):
    selected_set = set(selected)
    result = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("UTR") or row["utr"]
            if key in selected_set:
                result[key] = float(row["rl"])
    assert set(result) == selected_set
    return np.asarray([result[key] for key in selected])


def independent_select(score, labels, pools, proposals, cap):
    chosen = np.zeros((len(pools), proposals.shape[1]), dtype=np.int64)
    regret = np.zeros_like(chosen, dtype=float)
    for pool_index, pool in enumerate(pools):
        for seed_index in range(proposals.shape[1]):
            prefix = proposals[pool_index, seed_index, :cap]
            best = 0
            for position in range(1, cap):
                if score[pool[prefix[position]]] > score[pool[prefix[best]]]:
                    best = position
            chosen[pool_index, seed_index] = best
            regret[pool_index, seed_index] = np.max(labels[pool]) - labels[pool[prefix[best]]]
    return regret, chosen


def independent_top_tail(cnn, hepg2, labels, pools):
    top1 = 0
    jaccard = []
    flips = favor_hepg2 = favor_cnn = tied = 0
    for pool in pools:
        cnn_order = sorted(pool, key=lambda index: (-cnn[index], int(index)))
        hepg2_order = sorted(pool, key=lambda index: (-hepg2[index], int(index)))
        top1 += int(cnn_order[0] != hepg2_order[0])
        cnn_tail = cnn_order[:16]
        a = set(cnn_tail)
        b = set(hepg2_order[:16])
        jaccard.append(len(a & b) / len(a | b))
        for left in range(16):
            for right in range(left + 1, 16):
                i, j = cnn_tail[left], cnn_tail[right]
                cs = np.sign(cnn[i] - cnn[j])
                hs = np.sign(hepg2[i] - hepg2[j])
                if cs == 0 or hs == 0 or cs == hs:
                    continue
                flips += 1
                ts = np.sign(labels[i] - labels[j])
                if ts == 0:
                    tied += 1
                elif ts == hs:
                    favor_hepg2 += 1
                else:
                    favor_cnn += 1
    return {
        "top1_flip_pools": top1, "top16_membership_jaccard_mean": float(np.mean(jaccard)),
        "cnn_top16_predictor_order_flipped_pairs": flips,
        "flipped_pairs_tcell_favor_hepg2": favor_hepg2,
        "flipped_pairs_tcell_favor_cnn": favor_cnn,
        "flipped_pairs_tcell_tied": tied,
    }


def close(actual, expected, tolerance=1e-12):
    assert abs(float(actual) - float(expected)) <= tolerance, (actual, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["artifact_root"])
    target = root / "target"
    results = json.loads((target / "results.json").read_text())
    sequences = [value.decode("ascii") for value in np.load(root / "sequences.npy", allow_pickle=False)]
    pools = np.load(root / "pools.npy", allow_pickle=False)
    saved_proposals = np.load(root / "proposals.npy", allow_pickle=False)
    caps = config["selection"]["caps"]
    seeds = config["selection"]["proposal_seeds"]
    expected = np.empty_like(saved_proposals)
    for pool in range(16):
        for seed_index, seed in enumerate(seeds):
            rng = random.Random(f"tcell_cross_cell_anchor_v1|{pool}|{seed}")
            expected[pool, seed_index] = [min(63, int(rng.random() * 64)) for _ in range(16)]
    assert np.array_equal(expected, saved_proposals)
    proposals = saved_proposals
    cnn = np.load(root / "cnn_predictions.npy", allow_pickle=False)
    hepg2 = np.load(root / "hepg2_labels.npy", allow_pickle=False)
    fusion = np.zeros(len(sequences))
    for pool in pools:
        fusion[pool] = 0.5 * independent_rank(cnn[pool]) + 0.5 * independent_rank(hepg2[pool])
    scores = {"cnn": cnn, "hepg2_anchor": hepg2, "rank_fusion_50_50": fusion}
    repeats = ("defined_tcell_r1", "defined_tcell_r2")
    raw_p = []
    checks = 0
    max_difference = 0.0
    for repeat in repeats:
        labels = independent_read_labels(config["raw"][repeat], sequences)
        assert np.array_equal(labels, np.load(target / f"{repeat}_labels.npy", allow_pickle=False))
        for method, score in scores.items():
            for cap_index, cap in enumerate(caps):
                regret, selected = independent_select(score, labels, pools, proposals, cap)
                assert np.array_equal(regret, np.load(target / f"{repeat}_{method}_regret.npy", allow_pickle=False)[cap_index])
                assert np.array_equal(selected, np.load(target / f"{repeat}_{method}_selected.npy", allow_pickle=False)[cap_index])
                row = results["by_repeat_and_cap"][repeat][str(cap)]
                close(regret.mean(), row["mean_regret"][method])
                checks += 1
        for cap_index, cap in enumerate(caps):
            random_expected = np.empty((16, 5))
            for pool_index, pool in enumerate(pools):
                for seed_index in range(5):
                    first = proposals[pool_index, seed_index, 0]
                    random_expected[pool_index, seed_index] = labels[pool].max() - labels[pool[first]]
            saved_random = np.load(target / f"{repeat}_random_regret.npy", allow_pickle=False)[cap_index]
            assert np.array_equal(random_expected, saved_random)
            close(random_expected.mean(), results["by_repeat_and_cap"][repeat][str(cap)]["mean_regret"]["random"])
            checks += 1
        for cap_index, cap in enumerate(caps):
            row = results["by_repeat_and_cap"][repeat][str(cap)]
            anchor = np.load(target / f"{repeat}_hepg2_anchor_regret.npy", allow_pickle=False)[cap_index]
            baseline = np.load(target / f"{repeat}_cnn_regret.npy", allow_pickle=False)[cap_index]
            seed = 20261126 + config["multiplicity"]["repeat_bootstrap_seed_offset"][repeat] + config["multiplicity"]["cap_bootstrap_seed_offset"][str(cap)]
            boot = independent_bootstrap((anchor - baseline).mean(axis=1), seed)
            for key, value in boot.items():
                close(value, row["anchor_minus_cnn"][key])
                max_difference = max(max_difference, abs(value - row["anchor_minus_cnn"][key]))
            raw_p.append(boot["p_two_sided"])
            checks += len(boot)
            selected_anchor = np.load(target / f"{repeat}_hepg2_anchor_selected.npy", allow_pickle=False)[cap_index]
            selected_cnn = np.load(target / f"{repeat}_cnn_selected.npy", allow_pickle=False)[cap_index]
            changed = selected_anchor != selected_cnn
            effect = anchor - baseline
            turnover = row["turnover_anchor_vs_cnn"]
            assert turnover["changed"] == int(np.sum(changed))
            assert turnover["improved_among_changed"] == int(np.sum(changed & (effect < 0)))
            assert turnover["worsened_among_changed"] == int(np.sum(changed & (effect > 0)))
            assert turnover["unchanged_label_among_changed"] == int(np.sum(changed & (effect == 0)))
            close(turnover["mean_selected_mrl_change_anchor_minus_cnn"], -effect.mean())
            checks += 5
            for method in scores:
                selected = np.load(target / f"{repeat}_{method}_selected.npy", allow_pickle=False)[cap_index]
                opportunity = np.empty((16, 5))
                headroom = np.empty((16, 5))
                for pool_index, pool in enumerate(pools):
                    for seed_index in range(5):
                        prefix = proposals[pool_index, seed_index, :cap]
                        proposed = labels[pool[prefix]]
                        best = proposed.max()
                        opportunity[pool_index, seed_index] = best - proposed[selected[pool_index, seed_index]]
                        headroom[pool_index, seed_index] = labels[pool].max() - best
                oracle = row["oracle"][method]
                close(opportunity.mean(), oracle["oracle_opportunity_loss_mean"])
                close(headroom.mean(), oracle["proposal_headroom_mean"])
                checks += 2
        for method in ("cnn", "hepg2_anchor"):
            association = results["secondary"]["spearman"][repeat][method]
            close(independent_spearman(scores[method], labels), association["overall_spearman"])
            within = [independent_spearman(scores[method][pool], labels[pool]) for pool in pools]
            within = [value for value in within if value is not None]
            close(np.median(within), association["within_pool_median"])
            close(np.min(within), association["within_pool_min"])
            close(np.max(within), association["within_pool_max"])
            checks += 4
        expected_tail = independent_top_tail(cnn, hepg2, labels, pools)
        for key, value in expected_tail.items():
            close(value, results["secondary"]["top_tail"][repeat][key])
            checks += 1
    adjusted = independent_holm(raw_p)
    for (repeat, cap), value in zip(((repeat, cap) for repeat in repeats for cap in caps), adjusted):
        close(value, results["by_repeat_and_cap"][repeat][str(cap)]["anchor_minus_cnn"]["p_holm_10_repeat_cap_tests"])
        checks += 1
    primary_index = caps.index(16)
    combined = np.mean(np.stack([
        np.load(target / f"{repeat}_hepg2_anchor_regret.npy", allow_pickle=False)[primary_index]
        - np.load(target / f"{repeat}_cnn_regret.npy", allow_pickle=False)[primary_index]
        for repeat in repeats
    ]), axis=0)
    boot = independent_bootstrap(combined.mean(axis=1), 20261126)
    for key, value in boot.items():
        close(value, results["primary"]["combined_anchor_minus_cnn"][key])
        checks += 1
    fusion_combined = np.mean(np.stack([
        np.load(target / f"{repeat}_rank_fusion_50_50_regret.npy", allow_pickle=False)[primary_index]
        - np.load(target / f"{repeat}_cnn_regret.npy", allow_pickle=False)[primary_index]
        for repeat in repeats
    ]), axis=0)
    fusion_boot = independent_bootstrap(fusion_combined.mean(axis=1), 20261126)
    for key, value in fusion_boot.items():
        close(value, results["secondary"]["fusion_cap16_combined_minus_cnn"][key])
        checks += 1
    baseline = np.mean(np.stack([np.load(target / f"{repeat}_cnn_regret.npy", allow_pickle=False)[primary_index] for repeat in repeats]))
    close(baseline, results["primary"]["combined_cnn_baseline_regret"])
    close(max(0.1, 0.05 * baseline), results["primary"]["minimum_required_absolute_reduction"])
    expected_success = all(results["primary"]["repeat_anchor_minus_cnn_effects"][repeat] < 0 for repeat in repeats) and boot["ci_high"] < 0 and -boot["effect"] >= max(0.1, 0.05 * baseline)
    assert results["primary"]["success"] == expected_success
    report = {
        "status": "PASS", "round": 2, "implementation_imported": False,
        "raw_target_labels_independently_reread": True,
        "all_method_regret_and_selection_arrays_exact": True,
        "checks": checks, "max_absolute_numeric_difference": max_difference,
        "primary_success_independently_reproduced": expected_success,
        "target_evaluation_count": 1,
    }
    save_json(root / "verification" / "round2_independent.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
