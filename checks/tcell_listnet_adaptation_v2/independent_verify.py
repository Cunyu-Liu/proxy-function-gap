from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def hamming1(left, right):
    return sum(a != b for a, b in zip(left, right)) == 1


class Index:
    def __init__(self):
        self.exact = set()
        self.halves = {}

    def add(self, value):
        if value in self.exact:
            return
        self.exact.add(value)
        for side, half in ((0, value[:25]), (1, value[25:])):
            self.halves.setdefault((side, half), []).append(value)

    def match(self, value):
        if value in self.exact:
            return True
        return any(hamming1(value, other) for side, half in ((0, value[:25]), (1, value[25:])) for other in self.halves.get((side, half), ()))


def components_of(sequences):
    parent = list(range(len(sequences)))
    size = [1] * len(sequences)
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]
    halves = {}
    for index, value in enumerate(sequences):
        halves.setdefault((0, value[:25]), []).append(index)
        halves.setdefault((1, value[25:]), []).append(index)
    edges = 0
    for group in halves.values():
        for offset, left in enumerate(group):
            for right in group[offset + 1:]:
                if hamming1(sequences[left], sequences[right]):
                    union(left, right)
                    edges += 1
    result = {}
    for index in range(len(sequences)):
        result.setdefault(find(index), []).append(index)
    return list(result.values()), edges


def allocate(components, keys, budget):
    chosen = []
    count = 0
    for index in sorted(range(len(components)), key=lambda item: keys[item]):
        if len(components[index]) <= budget - count:
            chosen.append(index)
            count += len(components[index])
            if count == budget:
                return chosen
    raise RuntimeError("independent allocation failed")


def pack(components, keys):
    remaining = list(range(len(components)))
    rows = []
    for _ in range(16):
        chosen_local = allocate([components[index] for index in remaining], [keys[index] for index in remaining], 64)
        chosen = {remaining[index] for index in chosen_local}
        rows.append([member for component in sorted(chosen, key=lambda index: keys[index]) for member in components[component]])
        remaining = [index for index in remaining if index not in chosen]
    return np.asarray(rows, dtype=np.int64)


def regrets(scores, labels, pools, proposals, cap):
    prefix = proposals[:, :, :cap]
    proposed_scores = np.take_along_axis(scores[pools][:, None, :], prefix, axis=2)
    selected = np.argmax(proposed_scores, axis=2)
    proposed_labels = np.take_along_axis(labels[pools][:, None, :], prefix, axis=2)
    chosen = np.take_along_axis(proposed_labels, selected[..., None], axis=2)[..., 0]
    return labels[pools].max(axis=1)[:, None] - chosen, selected


def bootstrap(values, seed):
    clusters = np.asarray(values, dtype=np.float64).mean(axis=1)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(clusters), size=(10000, len(clusters)))
    draws = clusters[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((np.sum(draws >= 0) + 1) / 10001, (np.sum(draws <= 0) + 1) / 10001)
    return {"effect": float(clusters.mean()), "ci_low": float(low), "ci_high": float(high), "p_two_sided": float(min(1, p))}


def holm(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        result[index] = min(1.0, running)
    return result


class FreshSampleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(4, 32, 5), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(32, 64, 5), nn.ReLU())
        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, inputs):
        return self.regressor(self.features(inputs).mean(dim=2)).squeeze(1)


def encode(sequences):
    mapping = np.full(256, 255, dtype=np.uint8)
    for index, base in enumerate("ACGT"):
        mapping[ord(base)] = index
    return mapping[sequences.view(np.uint8).reshape(len(sequences), 50)]


def load_adapted(root, selection_path):
    selected = json.loads((root / selection_path).read_text())["selected"]
    members = []
    for seed in (11, 23, 37):
        run = root / "training" / selected["name"] / f"seed_{seed}"
        summary = json.loads((run / "summary.json").read_text())
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        assert digest(run / "best.pt") == summary["best_checkpoint_sha256"]
        members.append((checkpoint, summary["scaler"]["mean"], summary["scaler"]["scale"]))
    return members


def predict(members, codes):
    outputs = []
    with torch.no_grad():
        for checkpoint, mean, scale in members:
            model = FreshSampleCNN()
            model.load_state_dict(checkpoint["model_state"])
            model.cuda().eval()
            values = model(torch.nn.functional.one_hot(codes.long(), 4).permute(0, 2, 1).float()).cpu().numpy()
            outputs.append(values.astype(np.float64) * scale + mean)
            del model
            torch.cuda.empty_cache()
    return np.mean(outputs, axis=0)


def read_labels(path, selected):
    wanted = set(selected)
    result = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get("UTR") or row["utr"]
            if value in wanted:
                result[value] = float(row["rl"])
    assert set(result) == wanted
    return np.asarray([result[value] for value in selected])


def tail_metrics(baseline, listnet, labels, pools):
    top1 = flipped = favor_listnet = favor_baseline = tied = 0
    baseline_correct = listnet_correct = nontied = 0
    jaccards = []
    for pool in pools:
        left_order = np.argsort(-baseline[pool], kind="stable")
        right_order = np.argsort(-listnet[pool], kind="stable")
        top1 += int(left_order[0] != right_order[0])
        left_tail = pool[left_order[:16]]
        right_tail = pool[right_order[:16]]
        jaccards.append(len(set(left_tail) & set(right_tail)) / len(set(left_tail) | set(right_tail)))
        for left in range(16):
            for right in range(left + 1, 16):
                i, j = int(left_tail[left]), int(left_tail[right])
                a, b, y = np.sign(baseline[i] - baseline[j]), np.sign(listnet[i] - listnet[j]), np.sign(labels[i] - labels[j])
                if y != 0 and a != 0:
                    nontied += 1
                    baseline_correct += int(a == y)
                    listnet_correct += int(b == y)
                if a == 0 or b == 0 or a == b:
                    continue
                flipped += 1
                if y == 0:
                    tied += 1
                elif b == y:
                    favor_listnet += 1
                else:
                    favor_baseline += 1
    return {
        "top1_flip_pools": top1, "pools": len(pools), "top16_jaccard_mean": float(np.mean(jaccards)),
        "baseline_top16_predictor_order_flips": flipped, "flips_target_favor_listnet": favor_listnet,
        "flips_target_favor_baseline": favor_baseline, "flips_target_tied": tied,
        "baseline_tail_pair_accuracy": baseline_correct / nontied,
        "listnet_tail_pair_accuracy": listnet_correct / nontied,
        "tail_pair_accuracy_change": (listnet_correct - baseline_correct) / nontied,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    v1 = Path(config["v1_artifact_root"])
    prepare = json.loads((root / "prepare_manifest.json").read_text())
    results = json.loads((root / "target_v2/results.json").read_text())
    completed = json.loads((root / "target_v2/ONE_EVALUATION_COMPLETED.json").read_text())
    assert completed["evaluation_count"] == 1 and completed["results_sha256"] == digest(root / "target_v2/results.json")
    checks = [0]
    maximum = [0.0]
    def check(actual, expected, tolerance=0.0):
        difference = abs(float(actual) - float(expected))
        maximum[0] = max(maximum[0], difference)
        assert difference <= tolerance, (actual, expected, difference)
        checks[0] += 1

    sequences = np.load(root / "data/sequences.npy", allow_pickle=False)
    sequence_strings = [value.decode("ascii") for value in sequences]
    v1_split = np.load(root / "data/v1_split.npy", allow_pickle=False)
    components, edges = components_of(sequence_strings)
    check(len(components), prepare["components"])
    check(edges, prepare["h1_edges"])
    sample = Index()
    for path in config["sample_roles"]:
        with gzip.open(path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample.add(row["utr50"])
    eligible = []
    keys = []
    for members in components:
        roles = {int(v1_split[index]) for index in members if v1_split[index] >= 0}
        assert len(roles) <= 1
        values = sorted(sequence_strings[index] for index in members)
        identity = hashlib.sha256("".join(values).encode("ascii")).hexdigest()
        keys.append((hashlib.sha256(f"tcell_listnet_adaptation_v2|component|{identity}".encode()).hexdigest(), values[0]))
        eligible.append(not roles and not any(sample.match(sequence_strings[index]) for index in members))
    eligible_indices = [index for index in range(len(components)) if eligible[index]]
    chosen = allocate([components[index] for index in eligible_indices], [keys[index] for index in eligible_indices], 1024)
    selected_components = [eligible_indices[index] for index in chosen]
    expected_pools = pack([components[index] for index in selected_components], [keys[index] for index in selected_components])
    pools_global = np.load(root / "data/test_v2_pools.npy", allow_pickle=False)
    assert np.array_equal(expected_pools, pools_global)
    checks[0] += 1
    test_indices = np.load(root / "data/test_v2_indices.npy", allow_pickle=False)
    assert np.array_equal(test_indices, np.asarray(sorted(pools_global.ravel())))
    checks[0] += 1

    train_indices = np.load(root / "data/train_50000_indices.npy", allow_pickle=False)
    dev_indices = np.load(root / "data/dev_indices.npy", allow_pickle=False)
    labels = np.load(root / "data/train_dev_mean_labels.npy", allow_pickle=False)
    assert np.array_equal(train_indices, np.load(v1 / "data/train_indices.npy", allow_pickle=False))
    assert np.array_equal(dev_indices, np.load(v1 / "data/dev_indices.npy", allow_pickle=False))
    checks[0] += 2
    dev_pools_global = np.load(root / "data/dev_pools.npy", allow_pickle=False)
    dev_pools_global = dev_pools_global[(dev_pools_global >= 0).all(axis=1)]
    lookup = np.full(len(sequences), -1, dtype=np.int64)
    lookup[dev_indices] = np.arange(len(dev_indices))
    dev_pools = lookup[dev_pools_global]
    dev_proposals = np.load(root / "data/dev_proposals.npy", allow_pickle=False)
    candidate_rows = json.loads((root / "selection/listnet.json").read_text())["candidates"]
    dev_max = 0.0
    total_epochs = 0
    run_count = 0
    for candidate in candidate_rows:
        predictions = []
        for seed in (11, 23, 37):
            run = root / "training" / candidate["name"] / f"seed_{seed}"
            summary = json.loads((run / "summary.json").read_text())
            assert summary["converged_dev_plateau"] and summary["scientific_max_epoch"] is None and summary["gpu"]["actual_cuda"] and not summary["test_v2_mrl_read"]
            assert digest(run / "best.pt") == summary["best_checkpoint_sha256"]
            assert digest(run / "dev_predictions.npy") == summary["dev_predictions_sha256"]
            history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
            assert len(history) == summary["total_epochs"] == summary["history_records"]
            assert all(np.isfinite([row["train"]["loss"], row["dev"]["loss"], row["train"]["listnet"], row["dev"]["listnet"]]).all() for row in history)
            total_epochs += summary["total_epochs"]
            run_count += 1
            predictions.append(np.load(run / "dev_predictions.npy", allow_pickle=False))
        regret, _ = regrets(np.mean(predictions, axis=0), labels[dev_indices], dev_pools, dev_proposals, 16)
        difference = abs(float(regret.mean()) - candidate["dev_cap16_regret_mean_replicate"])
        dev_max = max(dev_max, difference)
        check(regret.mean(), candidate["dev_cap16_regret_mean_replicate"], 1e-12)
    check(run_count, 81)
    expected_selected = sorted(candidate_rows, key=lambda row: (round(row["dev_cap16_regret_mean_replicate"] / 1e-12) * 1e-12, row["dev_combined_loss_mean"], row["learning_rate"], row["temperature"], row["mse_weight"]))[0]
    assert expected_selected["name"] == json.loads((root / "selection/listnet.json").read_text())["selected"]["name"]
    checks[0] += 1

    prediction_freeze = json.loads((root / "TEST_PREDICTIONS_FROZEN.json").read_text())
    assert prediction_freeze["status"] == "PASS" and not prediction_freeze["test_v2_mrl_read"]
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    codes = torch.from_numpy(encode(sequences[test_indices])).cuda()
    inference = json.loads(Path(config["model"]["inference_config"]).read_text())
    original = []
    for member in inference["members"]:
        checkpoint = torch.load(member["checkpoint"], map_location="cpu", weights_only=False)
        original.append((checkpoint, checkpoint["scaler"]["mean"], checkpoint["scaler"]["scale"]))
    members = {
        "original_cnn": original,
        "v1_main": load_adapted(v1, Path("selection/main.json")),
        "v1_mse_only": load_adapted(v1, Path("selection/mse_only.json")),
        "listnet": load_adapted(root, Path("selection/listnet.json")),
    }
    scores = {}
    prediction_max = 0.0
    for method, method_members in members.items():
        scores[method] = predict(method_members, codes)
        frozen_scores = np.load(root / "test_predictions" / f"{method}_scores.npy", allow_pickle=False)
        difference = float(np.max(np.abs(scores[method] - frozen_scores)))
        prediction_max = max(prediction_max, difference)
        assert np.array_equal(scores[method], frozen_scores)
        checks[0] += 1

    selected_strings = [sequence_strings[index] for index in test_indices]
    labels_by_context = {repeat: read_labels(config["raw"][repeat], selected_strings) for repeat in ("defined_tcell_r1", "defined_tcell_r2")}
    labels_by_context["mean_replicate"] = (labels_by_context["defined_tcell_r1"] + labels_by_context["defined_tcell_r2"]) / 2
    lookup[test_indices] = np.arange(len(test_indices))
    pools = lookup[pools_global]
    proposals = np.load(root / "data/test_v2_proposals.npy", allow_pickle=False)
    caps = config["selection"]["caps"]
    all_regrets = {}
    all_selected = {}
    for context, context_labels in labels_by_context.items():
        for method in config["evaluation"]["methods"]:
            for cap in caps:
                regret, selected = regrets(scores[method], context_labels, pools, proposals, cap)
                all_regrets[method, context, cap] = regret
                all_selected[method, context, cap] = selected
                saved_regret = np.load(root / "target_v2" / f"{context}_{method}_regret.npy", allow_pickle=False)[caps.index(cap)]
                saved_selected = np.load(root / "target_v2" / f"{context}_{method}_selected.npy", allow_pickle=False)[caps.index(cap)]
                assert np.array_equal(regret, saved_regret) and np.array_equal(selected, saved_selected)
                checks[0] += 2
                check(results["by_context_and_cap"][context][str(cap)]["mean_regret"][method], regret.mean(), 1e-12)

    context_offsets = {"mean_replicate": 0, "defined_tcell_r1": 10000, "defined_tcell_r2": 20000}
    baseline_offsets = {"original_cnn": 0, "v1_main": 1000}
    p_values, p_keys = [], []
    bootstrap_max = 0.0
    for context in labels_by_context:
        for cap in caps:
            for baseline in ("original_cnn", "v1_main"):
                effect = all_regrets["listnet", context, cap] - all_regrets[baseline, context, cap]
                expected = bootstrap(effect, 20261126 + context_offsets[context] + baseline_offsets[baseline] + config["evaluation"]["cap_bootstrap_seed_offset"][str(cap)])
                actual = results["by_context_and_cap"][context][str(cap)]["listnet_comparisons"][baseline]["effect"]
                for key, value in expected.items():
                    difference = abs(actual[key] - value)
                    bootstrap_max = max(bootstrap_max, difference)
                    check(actual[key], value, 1e-12)
                if context == "mean_replicate":
                    p_values.append(expected["p_two_sided"])
                    p_keys.append((cap, baseline))
    adjusted = holm(p_values)
    for (cap, baseline), value in zip(p_keys, adjusted):
        actual = results["by_context_and_cap"]["mean_replicate"][str(cap)]["listnet_comparisons"][baseline]["effect"]["p_holm_10_comparison_cap_tests"]
        check(actual, value, 1e-15)

    for baseline in ("original_cnn", "v1_main"):
        primary = results["primary"]["comparisons"][baseline]
        baseline_mean = float(all_regrets[baseline, "mean_replicate", 16].mean())
        threshold = max(0.1, 0.05 * baseline_mean)
        check(primary["baseline_cap16_regret"], baseline_mean, 1e-12)
        check(primary["minimum_required_absolute_reduction"], threshold, 1e-12)
        expected_pass = primary["effect"]["ci_high"] < 0 and -primary["effect"]["effect"] >= threshold
        assert primary["comparisons"][baseline] if "comparisons" in primary else True
        assert results["primary"]["comparisons"][baseline]["effect"]["effect"] == results["by_context_and_cap"]["mean_replicate"]["16"]["listnet_comparisons"][baseline]["effect"]["effect"]
        assert (results["primary"]["success"] and expected_pass) or not results["primary"]["success"]
        checks[0] += 2
        for repeat in ("defined_tcell_r1", "defined_tcell_r2"):
            expected_effect = float(np.mean(all_regrets["listnet", repeat, 16] - all_regrets[baseline, repeat, 16]))
            check(primary["repeat_effects"][repeat], expected_effect, 1e-12)
        for context, context_labels in labels_by_context.items():
            expected_tail = tail_metrics(scores[baseline], scores["listnet"], context_labels, pools)
            actual_tail = results["top_tail"][context][baseline]
            for key, value in expected_tail.items():
                check(actual_tail[key], value, 1e-12)
            for method in config["evaluation"]["methods"]:
                error = scores[method] - context_labels
                actual_mae = results["pointwise_errors_mrl_units"][context][method]["mae"]
                check(actual_mae, np.mean(np.abs(error)), 1e-12)

    expected_success = all(results["primary"]["comparisons"][baseline]["effect"]["ci_high"] < 0 and -results["primary"]["comparisons"][baseline]["effect"]["effect"] >= results["primary"]["comparisons"][baseline]["minimum_required_absolute_reduction"] for baseline in ("original_cnn", "v1_main"))
    assert results["primary"]["success"] == expected_success
    checks[0] += 1
    report = {
        "status": "PASS", "analysis_id": config["analysis_id"], "checks": checks[0],
        "maximum_absolute_difference": maximum[0], "dev_selection_maximum_absolute_difference": dev_max,
        "prediction_maximum_absolute_difference": prediction_max,
        "test_pool_exact_reconstruction": True, "test_v2_to_v1_and_sample_exact_h1": 0,
        "training_runs": run_count, "total_epochs": total_epochs,
        "dev_candidates": len(candidate_rows), "evaluation_count": 1,
        "all_context_method_cap_regret_and_selection_recomputed": True,
        "bootstrap_holm_primary_tail_mae_recomputed": True,
        "implementation_module_imported": False,
    }
    path = root / "verification/round2_independent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
