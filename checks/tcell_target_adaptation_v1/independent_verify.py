from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def regret(scores, labels, pools, proposals, cap):
    prefix = proposals[:, :, :cap]
    proposed_scores = np.take_along_axis(scores[pools][:, None, :], prefix, axis=2)
    proposed_labels = np.take_along_axis(labels[pools][:, None, :], prefix, axis=2)
    chosen = np.argmax(proposed_scores, axis=2)
    selected = np.take_along_axis(proposed_labels, chosen[..., None], axis=2)[..., 0]
    return labels[pools].max(axis=1)[:, None] - selected


def boot(values, seed):
    clusters = np.asarray(values, dtype=np.float64).mean(axis=1)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = clusters[rng.integers(0, len(clusters), size=(10000, len(clusters)))].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    p = min(1.0, 2 * min((np.sum(draws >= 0) + 1) / 10001, (np.sum(draws <= 0) + 1) / 10001))
    return {"effect": float(clusters.mean()), "ci_low": float(low), "ci_high": float(high), "p_two_sided": float(p)}


def holm(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        result[index] = min(1.0, running)
    return result


def proposals(role, pools, seeds):
    result = np.empty((len(pools), len(seeds), 16), dtype=np.int64)
    for pool in range(len(pools)):
        for index, seed in enumerate(seeds):
            rng = random.Random(f"tcell_target_adaptation_v1|{role}|{pool}|{seed}")
            result[pool, index] = [int(rng.random() * 64) for _ in range(16)]
    return result


def pooled_indices(indices, sequences, role, budget):
    keyed = sorted((hashlib.sha256(f"tcell_target_adaptation_v1|pools|{role}|{budget}|{sequences[index].decode('ascii')}".encode()).digest(), int(index)) for index in indices)
    rows = []
    for start in range(0, len(keyed), 64):
        row = [index for _, index in keyed[start:start + 64]]
        row.extend([-1] * (64 - len(row)))
        rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def read_labels(path, wanted):
    result = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("UTR") or row["utr"]
            if value in wanted:
                result[value] = float(row["rl"])
    if set(result) != wanted:
        raise ValueError("independent test labels incomplete")
    return result


def rankdata(values):
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return result


def corr(left, right):
    x, y = rankdata(np.asarray(left)), rankdata(np.asarray(right))
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["artifact_root"])
    saved = json.loads((root / "target" / "results.json").read_text())
    maximum = 0.0
    comparisons = 0
    def compare(actual, expected):
        nonlocal maximum, comparisons
        if actual is None or expected is None:
            assert actual is expected
        else:
            maximum = max(maximum, abs(float(actual) - float(expected)))
            comparisons += 1
            assert abs(float(actual) - float(expected)) < 1e-10, (actual, expected)

    runs = list(sorted((root / "training").glob("*/seed_*/summary.json")))
    assert len(runs) == 72
    for path in runs:
        summary = json.loads(path.read_text())
        history = [json.loads(line) for line in (path.parent / "history.jsonl").read_text().splitlines()]
        assert len(history) == summary["total_epochs"] == summary["history_records"]
        assert summary["scientific_max_epoch"] is None and summary["converged_dev_plateau"] and summary["gpu"]["actual_cuda"] and not summary["gpu"]["cpu_fallback"]
        assert min(history, key=lambda row: row["dev"]["loss"])["epoch"] == summary["best_epoch"]
        assert sha(path.parent / "best.pt") == summary["best_checkpoint_sha256"]
        assert sha(path.parent / "history.jsonl") == summary["history_sha256"]

    sequences = np.load(root / "data" / "sequences.npy", allow_pickle=False)
    split = np.load(root / "data" / "split.npy", allow_pickle=False)
    test_indices = np.load(root / "data" / "test_indices.npy", allow_pickle=False)
    dev_indices = np.load(root / "data" / "dev_indices.npy", allow_pickle=False)
    assert np.sum(split == 0) == 50000 and np.sum(split == 1) == 10000 and np.sum(split == 2) == 1024
    strings = [value.decode("ascii") for value in sequences]
    lookup = {value: index for index, value in enumerate(strings)}
    halves = {}
    for index, value in enumerate(strings):
        halves.setdefault((0, value[:25]), []).append(index)
        halves.setdefault((1, value[25:]), []).append(index)
    cross_role = 0
    for group in halves.values():
        for offset, left in enumerate(group):
            for right in group[offset + 1:]:
                if sum(a != b for a, b in zip(strings[left], strings[right])) == 1 and split[left] != split[right]:
                    cross_role += 1
    assert cross_role == 0
    sample_exact = set()
    sample_halves = {}
    for path in config["sample_roles"]:
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                value = row["utr50"]
                sample_exact.add(value)
                for side, half in ((0, value[:25]), (1, value[25:])):
                    sample_halves.setdefault((side, half), []).append(value)
    sample_test_matches = 0
    for index in test_indices:
        value = strings[index]
        matched = value in sample_exact
        if not matched:
            for side, half in ((0, value[:25]), (1, value[25:])):
                if any(sum(a != b for a, b in zip(value, other)) == 1 for other in sample_halves.get((side, half), [])):
                    matched = True
                    break
        sample_test_matches += int(matched)
    assert sample_test_matches == 0
    for smaller, larger in ((1000, 5000), (5000, 20000), (20000, 50000)):
        small = set(np.load(root / "data" / f"train_{smaller}_indices.npy", allow_pickle=False).tolist())
        large = set(np.load(root / "data" / f"train_{larger}_indices.npy", allow_pickle=False).tolist())
        assert small < large

    labels_mean = np.load(root / "data" / "train_dev_mean_labels.npy", allow_pickle=False)
    dev_global = pooled_indices(dev_indices, sequences, "dev", 10000)
    dev_global = dev_global[(dev_global >= 0).all(axis=1)]
    dev_lookup = np.full(len(sequences), -1, dtype=np.int64)
    dev_lookup[dev_indices] = np.arange(len(dev_indices))
    dev_pools = dev_lookup[dev_global]
    dev_prop = proposals("dev", dev_pools, config["selection"]["proposal_seeds"])
    for variant in ("main", "mse_only", "from_scratch"):
        selection = json.loads((root / "selection" / f"{variant}.json").read_text())
        for candidate in selection["candidates"]:
            predictions = [np.load(root / "training" / candidate["name"] / f"seed_{seed}" / "dev_predictions.npy", allow_pickle=False) for seed in (11, 23, 37)]
            ensemble = np.mean(predictions, axis=0)
            value = float(regret(ensemble, labels_mean[dev_indices], dev_pools, dev_prop, 16).mean())
            compare(value, candidate["dev_cap16_regret"])
            repeat_values = {}
            for repeat in ("defined_tcell_r1", "defined_tcell_r2"):
                repeat_labels = np.load(root / "data" / f"dev_{repeat}_labels.npy", allow_pickle=False)
                repeat_values[repeat] = float(regret(ensemble, repeat_labels, dev_pools, dev_prop, 16).mean())
                compare(repeat_values[repeat], candidate["dev_repeat_cap16_regret"][repeat])
            compare(max(repeat_values.values()), candidate["dev_cap16_regret_worst_repeat"])
        reordered = sorted(selection["candidates"], key=lambda row: (round(row["dev_cap16_regret_worst_repeat"] / 1e-12) * 1e-12, round(row["dev_cap16_regret"] / 1e-12) * 1e-12, row["dev_combined_loss_mean"], row["learning_rate"], row["lambda"]))
        assert reordered[0]["name"] == selection["selected"]["name"]

    pools_global = pooled_indices(test_indices, sequences, "test", 1024)
    assert np.array_equal(pools_global, np.load(root / "data" / "test_pools.npy", allow_pickle=False))
    test_lookup = np.full(len(sequences), -1, dtype=np.int64)
    test_lookup[test_indices] = np.arange(1024)
    pools = test_lookup[pools_global]
    prop = proposals("test", pools, config["selection"]["proposal_seeds"])
    assert np.array_equal(prop, np.load(root / "data" / "test_proposals.npy", allow_pickle=False))
    wanted = {strings[index] for index in test_indices}
    labels_by_repeat = {}
    for repeat in ("defined_tcell_r1", "defined_tcell_r2"):
        labels = read_labels(config["raw"][repeat], wanted)
        labels_by_repeat[repeat] = np.asarray([labels[value] for value in np.asarray(strings)[test_indices]])
    scores = {method: np.load(root / "target" / f"{method}_scores.npy", allow_pickle=False) for method in ("original_cnn", "main_50k", "mse_only_50k", "from_scratch_50k", "main_1k", "main_5k", "main_20k")}
    caps = config["selection"]["caps"]
    recomputed_regrets = {}
    p_values, p_keys = [], []
    for repeat, labels in labels_by_repeat.items():
        for method in config["evaluation"]["methods"]:
            for cap_index, cap in enumerate(caps):
                values = regret(scores[method], labels, pools, prop, cap)
                recomputed_regrets[method, repeat, cap] = values
                compare(values.mean(), saved["by_repeat_and_cap"][repeat][str(cap)]["mean_regret"][method])
                if method == "main_50k":
                    baseline = recomputed_regrets["original_cnn", repeat, cap]
                    effect = boot(values - baseline, config["evaluation"]["bootstrap_seed"] + config["evaluation"]["repeat_bootstrap_seed_offset"][repeat] + config["evaluation"]["cap_bootstrap_seed_offset"][str(cap)])
                    expected = saved["by_repeat_and_cap"][repeat][str(cap)]["main_minus_original"]
                    for key in ("effect", "ci_low", "ci_high", "p_two_sided"):
                        compare(effect[key], expected[key])
                    p_values.append(effect["p_two_sided"])
                    p_keys.append((repeat, cap))
        for cap_index, cap in enumerate(caps):
            random_values = labels[pools].max(axis=1)[:, None] - labels[pools][np.arange(len(pools))[:, None], prop[:, :, 0]]
            compare(random_values.mean(), saved["by_repeat_and_cap"][repeat][str(cap)]["mean_regret"]["random"])
        error = scores["main_50k"] - labels
        compare(np.mean(np.abs(error)), saved["pointwise_errors_mrl_units"][repeat]["main_50k"]["mae"])
        compare(np.sqrt(np.mean(error ** 2)), saved["pointwise_errors_mrl_units"][repeat]["main_50k"]["rmse"])
        compare(np.mean(error), saved["pointwise_errors_mrl_units"][repeat]["main_50k"]["bias"])
        compare(corr(scores["main_50k"], labels), saved["pointwise_errors_mrl_units"][repeat]["main_50k"]["spearman"])
    adjusted = holm(p_values)
    for (repeat, cap), value in zip(p_keys, adjusted):
        compare(value, saved["by_repeat_and_cap"][repeat][str(cap)]["main_minus_original"]["p_holm_10_repeat_cap_tests"])
    combined = np.mean(np.stack([recomputed_regrets["main_50k", repeat, 16] - recomputed_regrets["original_cnn", repeat, 16] for repeat in labels_by_repeat]), axis=0)
    primary = boot(combined, config["evaluation"]["bootstrap_seed"])
    for key in ("effect", "ci_low", "ci_high", "p_two_sided"):
        compare(primary[key], saved["primary"]["combined_main_minus_original"][key])
    for budget in (1000, 5000, 20000, 50000):
        method = f"main_{budget // 1000}k"
        for repeat, labels in labels_by_repeat.items():
            values = regret(scores[method], labels, pools, prop, 16)
            row = next(row for row in saved["nested_learning_curve"] if row["budget"] == budget)
            compare(values.mean(), row[f"{repeat}_cap16_regret"])
    success = all(saved["primary"]["repeat_main_minus_original_effects"][repeat] < 0 for repeat in labels_by_repeat) and primary["ci_high"] < 0 and -primary["effect"] >= saved["primary"]["minimum_required_absolute_reduction"]
    assert success == saved["primary"]["success"]
    report = {"status": "PASS", "round": 2, "independent_import_of_implementation_modules": False, "training_runs": len(runs), "comparisons": comparisons, "maximum_absolute_difference": maximum, "cross_role_exact_h1_edges": cross_role, "nested_subsets": True, "dev_selection_replay": True, "test_regret_bootstrap_holm_replay": True, "primary_success": success, "target_evaluation_count": 1}
    (root / "verification" / "round2_independent.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
