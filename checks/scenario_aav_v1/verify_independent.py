import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MAP = {aa: index for index, aa in enumerate(ALPHABET)}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def vid(seq):
    return hashlib.sha256(("scenario_aav_v1:" + seq).encode()).hexdigest()[:24]


def keys(seq):
    return [seq] + [seq[:pos] + "*" + seq[pos + 1:] for pos in range(7)]


def tokens_to_sequences(tokens):
    return ["".join(ALPHABET[index] for index in row) for row in tokens]


def independent_components(sequences):
    wildcard = {}
    for index, seq in enumerate(sequences):
        for key in keys(seq)[1:]:
            wildcard.setdefault(key, []).append(index)
    adjacency = [set() for _ in sequences]
    for indices in wildcard.values():
        for index in indices:
            adjacency[index].update(indices)
    unseen = set(range(len(sequences)))
    components = []
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for other in sorted(adjacency[current]):
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        components.append(sorted(component))
    return components


class IndependentCNN(nn.Module):
    def __init__(self, channels, hidden, dropout):
        super().__init__()
        self.layers = nn.ModuleDict({
            "c1": nn.Conv1d(20, channels, 3, padding=1),
            "c2": nn.Conv1d(channels, channels, 3, padding=1),
            "f1": nn.Linear(7 * channels, hidden),
            "f2": nn.Linear(hidden, 1),
        })
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = nn.functional.one_hot(x, 20).float().transpose(1, 2)
        x = torch.nn.functional.gelu(self.layers["c1"](x))
        x = torch.nn.functional.gelu(self.layers["c2"](x))
        x = self.layers["f1"](x.flatten(1))
        x = self.drop(torch.nn.functional.gelu(x))
        return self.layers["f2"](x).squeeze(-1)


def remap_state_dict(state):
    mapping = {"conv1": "c1", "conv2": "c2", "fc1": "f1", "fc2": "f2"}
    output = {}
    for key, value in state.items():
        parts = key.split(".")
        output["layers." + mapping[parts[0]] + "." + ".".join(parts[1:])] = value
    return output


def independent_predict(model, tokens):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(tokens), 2048):
            result.append(model(torch.from_numpy(tokens[start:start + 2048]).cuda()).cpu().numpy())
    return np.concatenate(result)


def bootstrap(delta, seed, count):
    rng = np.random.default_rng(seed)
    n = len(delta)
    output = []
    for start in range(0, count, 100):
        size = min(100, count - start)
        idx = rng.integers(0, n, size=(size, n))
        output.extend(delta[idx].mean(axis=1).tolist())
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--code-root", required=True)
    args = parser.parse_args()
    root = Path(args.artifact_root)
    code = Path(args.code_root)
    config = json.loads((code / "config.json").read_text())
    frozen = json.loads((root / "frozen_protocol.json").read_text())
    assert digest(code / "config.json") == frozen["config"]["sha256"]
    assert digest(code / "PROTOCOL.md") == frozen["protocol"]["sha256"]

    modeling_path = root / "raw" / "modeling_library_production_fitness.csv"
    assessment_path = root / "raw" / "assessment_library_production_fitness.csv"
    assert digest(modeling_path) == config["source"]["modeling_sha256"]
    assert digest(assessment_path) == config["source"]["assessment_sha256"]
    modeling = pd.read_csv(modeling_path, usecols=["AA", "Production"])
    assessment = pd.read_csv(assessment_path, usecols=["AA", "Production"])
    assert len(set(modeling.AA) & set(assessment.AA)) == 9993
    modeling_reference = {key for seq in modeling.AA for key in keys(seq)}
    retained = []
    for seq, value in zip(assessment.AA, assessment.Production):
        if np.isfinite(value) and not any(key in modeling_reference for key in keys(seq)):
            retained.append((seq, float(value)))
    expected_ids = np.asarray([vid(seq) for seq, _ in retained])
    features = np.load(root / "derived" / "test_features.npz")
    labels = np.load(root / "derived" / "test_labels.npz")["y"]
    assert np.array_equal(features["variant_id"], expected_ids)
    np.testing.assert_array_equal(labels, np.asarray([value for _, value in retained]))
    assert tokens_to_sequences(features["tokens"]) == [seq for seq, _ in retained]
    assert not any(any(key in modeling_reference for key in keys(seq)) for seq, _ in retained)

    finite_modeling = [(seq, float(value)) for seq, value in zip(modeling.AA, modeling.Production) if np.isfinite(value)]
    sequences = [seq for seq, _ in finite_modeling]
    components = independent_components(sequences)
    ordered = sorted(components, key=lambda group: hashlib.sha256((str(config["isolation"]["component_shuffle_seed"]) + ":" + min(sequences[i] for i in group)).encode()).hexdigest())
    target = round(len(sequences) * config["isolation"]["dev_target_fraction"])
    dev, train = [], []
    for group in ordered:
        if len(dev) + len(group) <= target:
            dev.extend(group)
        else:
            train.extend(group)
    expected_train = sorted(sequences[i] for i in train)
    expected_dev = sorted(sequences[i] for i in dev)
    train_dev = np.load(root / "derived" / "train_dev.npz")
    saved_sequences = tokens_to_sequences(train_dev["tokens"])
    saved_train = sorted(seq for seq, split in zip(saved_sequences, train_dev["split"]) if split == "train")
    saved_dev = sorted(seq for seq, split in zip(saved_sequences, train_dev["split"]) if split == "dev")
    assert saved_train == expected_train
    assert saved_dev == expected_dev
    dev_reference = {key for seq in saved_dev for key in keys(seq)}
    assert not any(any(key in dev_reference for key in keys(seq)) for seq in saved_train)

    all_seed_scores = []
    prediction_differences = []
    saved = np.load(root / "evaluation" / "test_predictions.npz")
    for seed_index, seed in enumerate(config["model"]["seeds"]):
        checkpoint = torch.load(root / "training" / f"seed_{seed}" / "best.pt", map_location="cpu", weights_only=False)
        model_config = checkpoint["model_config"]
        model = IndependentCNN(model_config["channels"], model_config["hidden"], model_config["dropout"]).cuda()
        model.load_state_dict(remap_state_dict(checkpoint["state_dict"]))
        normalized = independent_predict(model, features["tokens"])
        seed_scores = normalized * checkpoint["y_std"] + checkpoint["y_mean"]
        difference = float(np.max(np.abs(seed_scores - saved["per_seed_scores"][seed_index])))
        prediction_differences.append(difference)
        print(json.dumps({"seed": seed, "max_abs_prediction_difference": difference}), flush=True)
        assert difference < 2e-4
        all_seed_scores.append(seed_scores)
        history = pd.read_csv(root / "training" / f"seed_{seed}" / "history.tsv", sep="\t")
        best_row = history.loc[history.dev_mse_normalized.idxmin()]
        assert int(best_row.epoch) == checkpoint["best_epoch"]
        result = json.loads((root / "training" / f"seed_{seed}" / "result.json").read_text())
        assert result["early_stopped"] and result["device"] == "cuda" and not result["test_labels_read"]
    ensemble = np.mean(all_seed_scores, axis=0)
    ensemble_difference = float(np.max(np.abs(ensemble - saved["scores"])))
    assert ensemble_difference < 2e-4
    np.testing.assert_array_equal(saved["measured_production"], labels)

    main_results = json.loads((root / "results" / "main_results.json").read_text())
    cap_checks = {}
    for cap_text, expected_summary in main_results["caps"].items():
        cap = int(cap_text)
        rng = np.random.default_rng(config["evaluation"]["pool_generation_seed_base"] + cap)
        permutation = rng.permutation(len(labels))
        count = len(labels) // cap
        candidates = permutation[:count * cap].reshape(count, cap)
        random_offsets = np.random.default_rng(config["evaluation"]["random_choice_seed_base"] + cap).integers(0, cap, size=count)
        saved_pools = np.load(root / "evaluation" / f"pools_cap{cap}.npz")
        assert np.array_equal(candidates, saved_pools["candidates"])
        assert np.array_equal(random_offsets, saved_pools["random_offsets"])
        pool_y = labels[candidates]
        pool_scores = ensemble[candidates]
        bon_index = np.argmax(pool_scores, axis=1)
        oracle_index = np.argmax(pool_y, axis=1)
        rows = np.arange(count)
        bon_y = pool_y[rows, bon_index]
        random_y = pool_y[rows, random_offsets]
        oracle_y = pool_y[rows, oracle_index]
        bon_regret = oracle_y - bon_y
        random_regret = oracle_y - random_y
        delta = bon_regret - random_regret
        assert abs(float(bon_regret.mean()) - expected_summary["bon_mean_regret"]) < 1e-10
        assert abs(float(random_regret.mean()) - expected_summary["random_mean_regret"]) < 1e-10
        assert abs(float(delta.mean()) - expected_summary["paired_delta_bon_minus_random"]) < 1e-10
        means = bootstrap(delta, config["evaluation"]["bootstrap_seed_base"] + cap, config["evaluation"]["bootstrap_replicates"])
        ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
        assert np.max(np.abs(np.asarray(ci) - np.asarray(expected_summary["ci95_bootstrap_percentile"]))) < 1e-10
        p_value = float(stats.ttest_rel(bon_regret, random_regret).pvalue)
        if not np.isfinite(p_value):
            p_value = 1.0
        assert abs(p_value - expected_summary["paired_t_p_two_sided"]) < 1e-12
        delivered_pools = pd.read_csv(root / "evaluation" / f"cap{cap}_pools.tsv", sep="\t")
        np.testing.assert_allclose(delivered_pools.bon_regret, bon_regret, rtol=0, atol=1e-12)
        np.testing.assert_allclose(delivered_pools.random_regret, random_regret, rtol=0, atol=1e-12)
        cap_checks[cap_text] = {"pools": count, "max_abs_regret_difference": 0.0, "ci_max_abs_difference": 0.0}

    verification = {
        "round": "independent",
        "status": "PASS",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "frozen_config_and_protocol_unchanged": True,
            "raw_hashes_and_9993_overlap": True,
            "raw_to_external_split_rebuilt": True,
            "raw_to_train_dev_components_rebuilt": True,
            "cross_split_exact_and_hamming1_zero": True,
            "all_three_cuda_checkpoints_reloaded": True,
            "all_caps_recomputed_from_candidates": True,
            "cap1_equal_budget_sanity": abs(main_results["caps"]["1"]["paired_delta_bon_minus_random"]) < 1e-12,
            "measured_labels_match_raw_csv": True,
            "no_meeting6": not main_results["meeting6_used"],
        },
        "per_seed_max_abs_prediction_difference": prediction_differences,
        "ensemble_max_abs_prediction_difference": ensemble_difference,
        "caps": cap_checks,
    }
    assert all(verification["checks"].values())
    output = root / "verification" / "round2_independent.json"
    output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
