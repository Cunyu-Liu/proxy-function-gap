import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(ALPHABET)}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def variant_id(sequence):
    return hashlib.sha256(("scenario_aav_v1:" + sequence).encode()).hexdigest()[:24]


def encode(sequences):
    tokens = np.asarray([[AA_TO_INT[aa] for aa in seq] for seq in sequences], dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] != 7:
        raise ValueError("All variants must be 7-mers")
    return tokens


def neighbor_keys(sequence):
    return [sequence] + [sequence[:i] + "*" + sequence[i + 1:] for i in range(7)]


def hamming_leq_one(left, right):
    return sum(a != b for a, b in zip(left, right)) <= 1


def assert_cross_isolation(left, right):
    right_keys = {key for seq in right for key in neighbor_keys(seq)}
    violations = [seq for seq in left if any(key in right_keys for key in neighbor_keys(seq))]
    if violations:
        raise AssertionError(f"Cross-split Hamming<=1 violations: {len(violations)}")


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def hamming_components(sequences):
    uf = UnionFind(len(sequences))
    wildcard = {}
    for index, sequence in enumerate(sequences):
        for key in neighbor_keys(sequence)[1:]:
            wildcard.setdefault(key, []).append(index)
    for indices in wildcard.values():
        for other in indices[1:]:
            uf.union(indices[0], other)
    components = {}
    for index in range(len(sequences)):
        components.setdefault(uf.find(index), []).append(index)
    return list(components.values())


def prepare(config_path, artifact_root):
    artifact_root = Path(artifact_root)
    config = json.loads(Path(config_path).read_text())
    raw = artifact_root / "raw"
    modeling_path = raw / "modeling_library_production_fitness.csv"
    assessment_path = raw / "assessment_library_production_fitness.csv"
    observed_hashes = {
        "modeling": sha256_file(modeling_path),
        "assessment": sha256_file(assessment_path),
    }
    if observed_hashes["modeling"] != config["source"]["modeling_sha256"]:
        raise AssertionError("Modeling raw SHA-256 mismatch")
    if observed_hashes["assessment"] != config["source"]["assessment_sha256"]:
        raise AssertionError("Assessment raw SHA-256 mismatch")
    license_path = raw / "UPSTREAM_LICENSE"
    license_text = license_path.read_text()
    if "bsd 3-clause" not in license_text.lower():
        raise AssertionError("Expected BSD-3-Clause upstream code license")

    usecols = ["AA", "Production"]
    modeling = pd.read_csv(modeling_path, usecols=usecols)
    assessment = pd.read_csv(assessment_path, usecols=usecols)
    if list(modeling.columns) != usecols or list(assessment.columns) != usecols:
        raise AssertionError("Unexpected source columns")
    for frame in (modeling, assessment):
        if not frame["AA"].is_unique:
            raise AssertionError("AA rows are not unique")
        if not frame["AA"].map(lambda x: len(x) == 7 and all(a in AA_TO_INT for a in x)).all():
            raise AssertionError("Invalid AA token")

    modeling_aa = modeling["AA"].tolist()
    assessment_aa = assessment["AA"].tolist()
    exact_overlap = sorted(set(modeling_aa) & set(assessment_aa))
    if len(exact_overlap) != config["source"]["expected_raw_exact_overlap"]:
        raise AssertionError(f"Unexpected exact overlap {len(exact_overlap)}")

    modeling_keys = {key for seq in modeling_aa for key in neighbor_keys(seq)}
    external_near = np.asarray([
        any(key in modeling_keys for key in neighbor_keys(seq)) for seq in assessment_aa
    ])
    exact_mask = assessment["AA"].isin(set(modeling_aa)).to_numpy()
    modeling_finite = np.isfinite(modeling["Production"].to_numpy(dtype=np.float64))
    assessment_finite = np.isfinite(assessment["Production"].to_numpy(dtype=np.float64))
    keep_external = assessment_finite & ~external_near

    eligible_modeling = modeling.loc[modeling_finite].reset_index(drop=True)
    eligible_sequences = eligible_modeling["AA"].tolist()
    components = hamming_components(eligible_sequences)
    ordered_components = sorted(
        components,
        key=lambda idxs: hashlib.sha256(
            (str(config["isolation"]["component_shuffle_seed"]) + ":" + min(eligible_sequences[i] for i in idxs)).encode()
        ).hexdigest(),
    )
    target = int(round(len(eligible_sequences) * config["isolation"]["dev_target_fraction"]))
    dev_indices, train_indices = [], []
    for indices in ordered_components:
        if len(dev_indices) + len(indices) <= target:
            dev_indices.extend(indices)
        else:
            train_indices.extend(indices)
    if not train_indices or not dev_indices:
        raise AssertionError("Empty train or dev")
    train_sequences = [eligible_sequences[i] for i in train_indices]
    dev_sequences = [eligible_sequences[i] for i in dev_indices]
    assert_cross_isolation(train_sequences, dev_sequences)

    test = assessment.loc[keep_external].reset_index(drop=True)
    test_sequences = test["AA"].tolist()
    assert_cross_isolation(test_sequences, modeling_aa)
    test_components = hamming_components(test_sequences)
    component_ids = {}
    for indices in components:
        cid = hashlib.sha256(min(eligible_sequences[i] for i in indices).encode()).hexdigest()[:24]
        for i in indices:
            component_ids[eligible_sequences[i]] = cid
    test_component_ids = np.empty(len(test_sequences), dtype="U24")
    for indices in test_components:
        cid = hashlib.sha256(min(test_sequences[i] for i in indices).encode()).hexdigest()[:24]
        for i in indices:
            test_component_ids[i] = cid

    derived = artifact_root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    train_dev_sequences = train_sequences + dev_sequences
    train_dev_y = eligible_modeling["Production"].to_numpy(dtype=np.float64)[train_indices + dev_indices]
    train_dev_split = np.asarray(["train"] * len(train_sequences) + ["dev"] * len(dev_sequences))
    np.savez_compressed(
        derived / "train_dev.npz",
        tokens=encode(train_dev_sequences),
        y=train_dev_y,
        split=train_dev_split,
        variant_id=np.asarray([variant_id(seq) for seq in train_dev_sequences]),
        component_id=np.asarray([component_ids[seq] for seq in train_dev_sequences]),
    )
    np.savez_compressed(
        derived / "test_features.npz",
        tokens=encode(test_sequences),
        variant_id=np.asarray([variant_id(seq) for seq in test_sequences]),
        component_id=test_component_ids,
    )
    np.savez_compressed(derived / "test_labels.npz", y=test["Production"].to_numpy(dtype=np.float64))

    component_sizes = [len(c) for c in components]
    test_component_sizes = [len(c) for c in test_components]
    audit = {
        "schema_version": "1.0",
        "source_commit": config["source"]["commit"],
        "source_urls": {
            "modeling": "https://raw.githubusercontent.com/vector-engineering/fit4function/" + config["source"]["commit"] + "/data/modeling_library_production_fitness.csv",
            "assessment": "https://raw.githubusercontent.com/vector-engineering/fit4function/" + config["source"]["commit"] + "/data/assessment_library_production_fitness.csv",
        },
        "raw_sha256": observed_hashes,
        "upstream_license": {"path": str(license_path), "sha256": sha256_file(license_path), "code_license": "BSD-3-Clause", "data_redistribution_license_found": False, "local_raw_redistribution": False},
        "raw": {
            "modeling_rows": len(modeling),
            "assessment_rows": len(assessment),
            "modeling_finite": int(modeling_finite.sum()),
            "assessment_finite": int(assessment_finite.sum()),
            "modeling_zero": int((modeling["Production"] == 0).sum()),
            "assessment_zero": int((assessment["Production"] == 0).sum()),
            "exact_overlap": len(exact_overlap),
            "calibration_label_used_for_exclusion": False,
        },
        "external_exclusion": {
            "hamming_radius": 1,
            "exact": int(exact_mask.sum()),
            "hamming1_nonexact": int((external_near & ~exact_mask).sum()),
            "total_near": int(external_near.sum()),
            "nonfinite": int((~assessment_finite).sum()),
            "retained": len(test),
            "retained_zero": int((test["Production"] == 0).sum()),
        },
        "split": {
            "train": len(train_sequences),
            "dev": len(dev_sequences),
            "test": len(test_sequences),
            "development_components": len(components),
            "development_largest_component": max(component_sizes),
            "test_components": len(test_components),
            "test_largest_component": max(test_component_sizes),
            "cross_split_hamming_leq1": 0,
            "labels_used_for_assignment": False,
        },
        "endpoint": config["endpoint"],
        "retrospective": True,
        "meeting6_used": False,
    }
    write_json(artifact_root / "results" / "data_audit.json", audit)
    round1 = {
        "round": "raw",
        "status": "PASS",
        "checks": {
            "raw_hashes": True,
            "source_shape": len(modeling) == 74464 and len(assessment) == 74467,
            "raw_exact_overlap_9993": len(exact_overlap) == 9993,
            "license_recorded": True,
            "zero_labels_retained": int((test["Production"] == 0).sum()) > 0,
            "external_exact_and_hamming1_isolated": True,
            "train_dev_component_isolated": True,
            "meeting6_absent": True,
        },
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    if not all(round1["checks"].values()):
        raise AssertionError(f"Round 1 failed: {round1}")
    write_json(artifact_root / "verification" / "round1_raw.json", round1)
    print(json.dumps(audit, indent=2))


class SequenceCNN(nn.Module):
    def __init__(self, channels=64, hidden=128, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(20, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(channels * 7, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, tokens):
        x = nn.functional.one_hot(tokens, num_classes=20).float().transpose(1, 2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        return self.fc2(self.dropout(x)).squeeze(-1)


def predict(model, tokens, batch_size=2048, device="cuda"):
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = torch.as_tensor(tokens[start:start + batch_size], dtype=torch.long, device=device)
            output.append(model(batch).float().cpu().numpy())
    return np.concatenate(output)


def train_one(seed, data, config, output_dir, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    split = data["split"]
    train_idx = np.flatnonzero(split == "train")
    dev_idx = np.flatnonzero(split == "dev")
    tokens, y = data["tokens"], data["y"]
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std())
    if not math.isfinite(y_std) or y_std <= 0:
        raise AssertionError("Invalid train normalization")
    normalized = (y - y_mean) / y_std
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tokens[train_idx]), torch.from_numpy(normalized[train_idx]).float()),
        batch_size=config["model"]["batch_size"], shuffle=True, generator=generator,
    )
    dev_tokens = torch.from_numpy(tokens[dev_idx])
    dev_y = torch.from_numpy(normalized[dev_idx]).float()
    model_config = config["model"]
    model = SequenceCNN(model_config["channels"], model_config["hidden"], model_config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["learning_rate"], weight_decay=model_config["weight_decay"])
    loss_fn = nn.MSELoss()
    best = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, model_config["max_epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch_tokens, batch_y in loader:
            batch_tokens, batch_y = batch_tokens.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_tokens), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(batch_tokens)
        train_loss /= len(train_idx)
        model.eval()
        with torch.no_grad():
            dev_loss = float(loss_fn(model(dev_tokens.to(device)), dev_y.to(device)).cpu())
        history.append({"epoch": epoch, "train_mse_normalized": train_loss, "dev_mse_normalized": dev_loss})
        if dev_loss < best - model_config["early_stopping_min_delta"]:
            best = dev_loss
            best_epoch = epoch
            stale = 0
            torch.save({
                "seed": seed,
                "best_epoch": best_epoch,
                "dev_mse_normalized": best,
                "y_mean": y_mean,
                "y_std": y_std,
                "model_config": model_config,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }, checkpoint_path)
        else:
            stale += 1
        if stale >= model_config["early_stopping_patience"]:
            break
    early_stopped = stale >= model_config["early_stopping_patience"]
    pd.DataFrame(history).to_csv(output_dir / "history.tsv", sep="\t", index=False)
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "last_epoch": len(history),
        "best_dev_mse_normalized": best,
        "early_stopped": early_stopped,
        "converged_by_frozen_rule": early_stopped,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "train_label_normalization": {"mean": y_mean, "std": y_std},
        "test_labels_read": False,
    }
    write_json(output_dir / "result.json", result)
    if not early_stopped:
        raise RuntimeError(f"Seed {seed} reached epoch limit without convergence")
    return result


def train(config_path, artifact_root):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    artifact_root = Path(artifact_root)
    config = json.loads(Path(config_path).read_text())
    data = np.load(artifact_root / "derived" / "train_dev.npz")
    training_root = artifact_root / "training"
    results = [train_one(seed, data, config, training_root / f"seed_{seed}", "cuda") for seed in config["model"]["seeds"]]
    write_json(training_root / "training_summary.json", {"all_converged": all(r["early_stopped"] for r in results), "runs": results})
    print(json.dumps(results, indent=2))


def paired_summary(bon_regret, random_regret, seed, replicates):
    delta = bon_regret - random_regret
    n = len(delta)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 100):
        count = min(100, replicates - start)
        indices = rng.integers(0, n, size=(count, n))
        means[start:start + count] = delta[indices].mean(axis=1)
    t_result = stats.ttest_rel(bon_regret, random_regret)
    return {
        "pools": n,
        "bon_mean_regret": float(bon_regret.mean()),
        "random_mean_regret": float(random_regret.mean()),
        "paired_delta_bon_minus_random": float(delta.mean()),
        "random_minus_bon_advantage": float(-delta.mean()),
        "delta_median": float(np.median(delta)),
        "ci95_bootstrap_percentile": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "paired_t_p_two_sided": float(t_result.pvalue) if np.isfinite(t_result.pvalue) else 1.0,
    }


def select_from_pools(y, scores, candidates, random_offsets):
    pool_scores = scores[candidates]
    pool_y = y[candidates]
    bon_local = np.argmax(pool_scores, axis=1)
    oracle_local = np.argmax(pool_y, axis=1)
    random_local = random_offsets
    rows = np.arange(len(candidates))
    oracle_y = pool_y[rows, oracle_local]
    bon_y = pool_y[rows, bon_local]
    random_y = pool_y[rows, random_local]
    return {
        "bon_local": bon_local,
        "random_local": random_local,
        "oracle_local": oracle_local,
        "bon_y": bon_y,
        "random_y": random_y,
        "oracle_y": oracle_y,
        "bon_regret": oracle_y - bon_y,
        "random_regret": oracle_y - random_y,
        "bon_score": pool_scores[rows, bon_local],
    }


def evaluate_cap(cap, y, scores, ids, config, evaluation_root):
    generation_seed = config["evaluation"]["pool_generation_seed_base"] + cap
    choice_seed = config["evaluation"]["random_choice_seed_base"] + cap
    rng = np.random.default_rng(generation_seed)
    permutation = rng.permutation(len(y))
    pool_count = len(y) // cap
    candidates = permutation[:pool_count * cap].reshape(pool_count, cap)
    choice_rng = np.random.default_rng(choice_seed)
    random_offsets = choice_rng.integers(0, cap, size=pool_count)
    selected = select_from_pools(y, scores, candidates, random_offsets)
    summary = paired_summary(
        selected["bon_regret"], selected["random_regret"],
        config["evaluation"]["bootstrap_seed_base"] + cap,
        config["evaluation"]["bootstrap_replicates"],
    )
    summary.update({
        "cap": cap,
        "role": "primary" if cap == config["evaluation"]["primary_cap"] else ("sanity" if cap == config["evaluation"]["sanity_cap"] else "secondary"),
        "bon_mean_selected_production": float(selected["bon_y"].mean()),
        "random_mean_selected_production": float(selected["random_y"].mean()),
        "oracle_mean_production": float(selected["oracle_y"].mean()),
        "candidate_generation_seed": generation_seed,
        "random_choice_seed": choice_seed,
        "regret_unit": "raw Production enrichment ratio",
    })
    evaluation_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(evaluation_root / f"pools_cap{cap}.npz", candidates=candidates, random_offsets=random_offsets)
    rows = []
    for pool_index in range(pool_count):
        candidates_hash = hashlib.sha256("|".join(ids[candidates[pool_index]]).encode()).hexdigest()
        rows.append({
            "pool_id": f"cap{cap}_pool{pool_index:06d}",
            "candidates_sha256": candidates_hash,
            "bon_selected_id": ids[candidates[pool_index, selected["bon_local"][pool_index]]],
            "random_selected_id": ids[candidates[pool_index, selected["random_local"][pool_index]]],
            "oracle_selected_id": ids[candidates[pool_index, selected["oracle_local"][pool_index]]],
            "bon_proxy_score": selected["bon_score"][pool_index],
            "bon_measured_production": selected["bon_y"][pool_index],
            "random_measured_production": selected["random_y"][pool_index],
            "oracle_measured_production": selected["oracle_y"][pool_index],
            "bon_regret": selected["bon_regret"][pool_index],
            "random_regret": selected["random_regret"][pool_index],
            "paired_delta": selected["bon_regret"][pool_index] - selected["random_regret"][pool_index],
        })
    pd.DataFrame(rows).to_csv(evaluation_root / f"cap{cap}_pools.tsv", sep="\t", index=False)
    write_json(evaluation_root / f"cap{cap}_summary.json", summary)
    return summary, candidates, random_offsets


def holm_adjust(items):
    ordered = sorted(items, key=lambda item: item["paired_t_p_two_sided"])
    running = 0.0
    m = len(ordered)
    for rank, item in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * item["paired_t_p_two_sided"]))
        item["holm_adjusted_p_secondary"] = running
    return items


def evaluate(config_path, artifact_root):
    artifact_root = Path(artifact_root)
    config = json.loads(Path(config_path).read_text())
    device = "cuda"
    features = np.load(artifact_root / "derived" / "test_features.npz")
    labels = np.load(artifact_root / "derived" / "test_labels.npz")["y"]
    tokens = features["tokens"]
    ids = features["variant_id"]
    predictions = []
    per_seed_metadata = []
    for seed in config["model"]["seeds"]:
        checkpoint_path = artifact_root / "training" / f"seed_{seed}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = checkpoint["model_config"]
        model = SequenceCNN(model_config["channels"], model_config["hidden"], model_config["dropout"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        raw_normalized = predict(model, tokens, device=device)
        predictions.append(raw_normalized * checkpoint["y_std"] + checkpoint["y_mean"])
        per_seed_metadata.append({"seed": seed, "checkpoint_sha256": sha256_file(checkpoint_path), "best_epoch": checkpoint["best_epoch"]})
    predictions = np.asarray(predictions)
    scores = predictions.mean(axis=0)
    if not np.isfinite(scores).all():
        raise AssertionError("Nonfinite test predictions")
    evaluation_root = artifact_root / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(evaluation_root / "test_predictions.npz", variant_id=ids, component_id=features["component_id"], scores=scores, per_seed_scores=predictions, measured_production=labels)

    caps = sorted(set(config["evaluation"]["secondary_caps"] + [config["evaluation"]["primary_cap"], config["evaluation"]["sanity_cap"]]))
    cap_results = {}
    primary_seed_sensitivity = []
    for cap in caps:
        summary, candidates, random_offsets = evaluate_cap(cap, labels, scores, ids, config, evaluation_root)
        cap_results[str(cap)] = summary
        if cap == config["evaluation"]["primary_cap"]:
            for seed_index, seed in enumerate(config["model"]["seeds"]):
                individual = select_from_pools(labels, predictions[seed_index], candidates, random_offsets)
                primary_seed_sensitivity.append({
                    "seed": seed,
                    "bon_mean_regret": float(individual["bon_regret"].mean()),
                    "random_mean_regret": float(individual["random_regret"].mean()),
                    "paired_delta_bon_minus_random": float((individual["bon_regret"] - individual["random_regret"]).mean()),
                })
    secondary = holm_adjust([cap_results[str(cap)] for cap in config["evaluation"]["secondary_caps"]])
    sanity_delta = cap_results[str(config["evaluation"]["sanity_cap"])]["paired_delta_bon_minus_random"]
    if abs(sanity_delta) > 1e-12:
        raise AssertionError("cap1 BoN and RandomN are not identical")

    order = np.argsort(-scores, kind="stable")
    top_tail = []
    for quantile in config["evaluation"]["top_tail_quantiles"]:
        count = max(1, int(math.ceil(len(labels) * quantile)))
        proxy_top = order[:count]
        measured_top = np.argsort(-labels, kind="stable")[:count]
        top_tail.append({
            "quantile": quantile,
            "count": count,
            "proxy_top_measured_mean": float(labels[proxy_top].mean()),
            "proxy_top_measured_median": float(np.median(labels[proxy_top])),
            "actual_top_measured_mean": float(labels[measured_top].mean()),
            "all_pool_measured_mean": float(labels.mean()),
            "proxy_top_mean_enrichment_over_all": float(labels[proxy_top].mean() / labels.mean()) if labels.mean() != 0 else None,
            "actual_top_tail_recall": float(len(set(proxy_top) & set(measured_top)) / count),
        })
    result = {
        "schema_version": "1.0",
        "experiment": config["experiment"],
        "endpoint_semantics": config["endpoint"]["semantics"],
        "retrospective_public_data": True,
        "new_blinded_validation": False,
        "meeting6_used": False,
        "primary_cap": config["evaluation"]["primary_cap"],
        "caps": cap_results,
        "primary_per_seed_sensitivity": primary_seed_sensitivity,
        "secondary_holm": [{"cap": item["cap"], "holm_adjusted_p": item["holm_adjusted_p_secondary"]} for item in secondary],
        "test_correlation": {
            "spearman": float(stats.spearmanr(scores, labels).statistic),
            "pearson": float(stats.pearsonr(scores, labels).statistic),
        },
        "top_tail": top_tail,
        "models": per_seed_metadata,
        "interpretation_limits": [
            "Retrospective archived-library reranking only",
            "Production is not tropism, infectivity, transgene expression, or clinical efficacy",
            "No de novo design or prospective wet-lab claim",
        ],
    }
    write_json(artifact_root / "results" / "main_results.json", result)
    print(json.dumps(result, indent=2))


def freeze(config_path, artifact_root):
    config_path = Path(config_path)
    artifact_root = Path(artifact_root)
    protocol_path = config_path.with_name("PROTOCOL.md")
    frozen = {
        "experiment": "scenario_aav_v1",
        "frozen_before_training_or_test_evaluation": True,
        "config": {"sha256": sha256_file(config_path), "bytes": config_path.stat().st_size},
        "protocol": {"sha256": sha256_file(protocol_path), "bytes": protocol_path.stat().st_size},
        "result_driven_reconfiguration_allowed": False,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(artifact_root / "frozen_protocol.json", frozen)
    print(json.dumps(frozen, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "prepare", "train", "evaluate"])
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    globals()[args.command](args.config, args.artifact_root)


if __name__ == "__main__":
    main()
