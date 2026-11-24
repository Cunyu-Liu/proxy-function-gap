import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .inference import inverse_target_transform, predict_array
from .outer_learner import load_outer_pool
from .pool_manifest import load_pool_manifest
from .production_selection import encode_utr50
from .trainer.checkpoint import atomic_json_write, load_checkpoint
from .trainer.cuda import configure_cuda
from .trainer.data import sha256_file
from .trainer.model import FreshSampleCNN, model_definition_hash, model_hash


METHODS = ("bon", "lcb", "distance", "com", "diversity_lcb")
CAPS = (1, 2, 4, 8, 16)
PAIRED_SEEDS = (101, 202, 303, 404, 505)
TOP_K = (1, 2, 4, 8, 16)


def _evidence(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def _unique(values):
    return list(dict.fromkeys(values))


def hamming_distance(left, right):
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal lengths")
    return sum(a != b for a, b in zip(left, right))


def rank_candidates(method, candidates, mean, std, distance, sequences, lambda_=0.0, gamma=0.0):
    candidates = _unique(candidates)
    if not candidates:
        raise ValueError("empty candidate ranking")
    if method == "bon":
        score = mean
    elif method == "lcb":
        score = mean - lambda_ * std
    elif method == "distance":
        score = mean - lambda_ * distance
    elif method == "com":
        score = mean
    elif method == "diversity_lcb":
        score = mean - lambda_ * std
    else:
        raise ValueError(f"unknown Paper2 baseline: {method}")
    if method != "diversity_lcb":
        return sorted(candidates, key=lambda index: (-float(score[index]), index))
    remaining = set(candidates)
    selected = []
    while remaining:
        def utility(index):
            if not selected:
                diversity = 0.0
            else:
                diversity = min(hamming_distance(sequences[index], sequences[other]) / len(sequences[index]) for other in selected)
            return float(score[index] + gamma * diversity)
        chosen = max(remaining, key=lambda index: (utility(index), -index))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def permutation_seed(pool_id, method, paired_seed, permutation_id):
    value = f"{pool_id}|{method}|{paired_seed}|{permutation_id}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def permute_features(method, mean, std, distance, pool_id, paired_seed, permutation_id, candidates=None):
    if permutation_id == 0:
        return mean, std, distance
    rng = np.random.default_rng(permutation_seed(pool_id, method, paired_seed, permutation_id))
    mean, std, distance = np.asarray(mean).copy(), np.asarray(std).copy(), np.asarray(distance).copy()
    candidates = np.arange(len(mean)) if candidates is None else np.asarray(candidates)
    if method in ("bon", "com"):
        values = mean[candidates].copy()
        rng.shuffle(values)
        mean[candidates] = values
    elif method in ("lcb", "diversity_lcb"):
        values = std[candidates].copy()
        rng.shuffle(values)
        std[candidates] = values
    elif method == "distance":
        values = distance[candidates].copy()
        rng.shuffle(values)
        distance[candidates] = values
    return mean, std, distance


class CUDABaselineScorer:
    def __init__(self, checkpoints, com=False, gpu_index=0, memory_fraction=0.08, batch_size=1024):
        if len(checkpoints) != 3:
            raise ValueError("exactly three checkpoints are required")
        self.com = com
        self.models = []
        self.scalers = []
        self.gpu = []
        self.item_forwards = 0
        self.batch_size = batch_size
        self.device = None
        self.dtype = None
        for index, item in enumerate(checkpoints):
            path = Path(item["path"])
            if sha256_file(path) != item["sha256"]:
                raise ValueError("baseline checkpoint hash mismatch")
            device, dtype, gpu = configure_cuda(gpu_index, memory_fraction, "float32", 1100 + index + (100 if com else 0))
            checkpoint = load_checkpoint(path, device)
            model = FreshSampleCNN()
            if checkpoint["model_definition_sha256"] != model_definition_hash(model):
                raise ValueError("baseline model architecture mismatch")
            model.load_state_dict(checkpoint["model_state"])
            model.to(device=device, dtype=dtype)
            model.eval()
            if item.get("model_hash") and model_hash(model) != item["model_hash"]:
                raise ValueError("baseline model hash mismatch")
            self.models.append(model)
            self.scalers.append(checkpoint["scaler"])
            self.gpu.append(gpu)
            self.device = device
            self.dtype = dtype

    def predict(self, sequences):
        values = np.concatenate([encode_utr50(sequence) for sequence in sequences], axis=0)
        members = []
        for model, scaler in zip(self.models, self.scalers):
            normalized = predict_array(model, values, self.batch_size, self.device, self.dtype)
            members.append(inverse_target_transform(normalized, scaler))
            self.item_forwards += len(values)
        stacked = np.stack(members)
        return stacked.mean(axis=0), stacked.std(axis=0, ddof=1), stacked


def _checkpoints_from_config(value, com=False):
    key = "com_members" if com else "cnn_members"
    result = []
    for item in value[key]:
        result.append({"path": item["checkpoint"], "sha256": item["sha256"], "model_hash": item.get("model_hash")})
    return result


def load_role_pool(config, role, index):
    pool_id = f"{role}_{index:03d}"
    if role.startswith("outer_"):
        return load_outer_pool(config["source_pool_manifest"], pool_id)
    return load_pool_manifest(Path(config["pool_root"]) / pool_id / "pool_manifest.json")


def run_pool_baselines(pool, config, output, permutations=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    standard = CUDABaselineScorer(_checkpoints_from_config(config), com=False, memory_fraction=config.get("memory_fraction", 0.08))
    com = CUDABaselineScorer(_checkpoints_from_config(config, com=True), com=True, memory_fraction=config.get("memory_fraction", 0.08))
    sequences = list(pool.utr50)
    distance = 1.0 - np.asarray(pool.support, dtype=np.float64)
    encoded = {index: sequences[index] for index in range(len(sequences))}
    records = []
    method_settings = config["methods"]
    caps = config.get("caps", CAPS)
    seeds = config.get("paired_seeds", PAIRED_SEEDS)
    for cap in caps:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            proposals = rng.choice(len(sequences), size=cap, replace=True, p=np.asarray(pool.proposal_probs, dtype=np.float64)).tolist()
            proposal_sequences = [sequences[index] for index in proposals]
            cnn_mean_values, cnn_std_values, _ = standard.predict(proposal_sequences)
            com_mean_values, com_std_values, _ = com.predict(proposal_sequences)
            cnn_mean = np.zeros(len(sequences))
            cnn_std = np.zeros(len(sequences))
            com_mean = np.zeros(len(sequences))
            com_std = np.zeros(len(sequences))
            for position, index in enumerate(proposals):
                cnn_mean[index] = cnn_mean_values[position]
                cnn_std[index] = cnn_std_values[position]
                com_mean[index] = com_mean_values[position]
                com_std[index] = com_std_values[position]
            for method, settings in method_settings.items():
                if method not in METHODS:
                    raise ValueError(f"unknown method in config: {method}")
                combinations = settings if isinstance(settings, list) else [settings]
                for setting in combinations:
                    lambda_ = float(setting.get("lambda", 0.0))
                    gamma = float(setting.get("gamma", 0.0))
                    for permutation_id in range(permutations + 1):
                        if method in ("bon", "lcb", "distance", "diversity_lcb"):
                            mean, std = cnn_mean.copy(), cnn_std.copy()
                        else:
                            mean, std = com_mean.copy(), com_std.copy()
                        mean, std, current_distance = permute_features(method, mean, std, distance.copy(), pool.pool_id, seed, permutation_id, candidates=_unique(proposals))
                        ranking = rank_candidates(method, proposals, mean, std, current_distance, encoded, lambda_, gamma)
                        top = [ranking[min(k, len(ranking)) - 1] for k in TOP_K]
                        top_indices = [ranking[:min(k, len(ranking))] for k in TOP_K]
                        selected = ranking[0]
                        risk = float(std[selected] if method != "distance" else current_distance[selected])
                        records.append({
                            "pool_id": pool.pool_id,
                            "pool_role": pool.pool_role,
                            "method": method,
                            "lambda": lambda_,
                            "gamma": gamma,
                            "proposal_cap": cap,
                            "seed": seed,
                            "permutation_id": permutation_id,
                            "selected_index": selected,
                            "selected_id": pool.sequence_ids[selected],
                            "top_indices": {str(k): indices for k, indices in zip(TOP_K, top_indices)},
                            "top_ids": {str(k): [pool.sequence_ids[index] for index in indices] for k, indices in zip(TOP_K, top_indices)},
                            "top_representatives": top,
                            "risk": risk,
                            "proposals": proposals,
                            "unique_proposals": len(set(proposals)),
                            "cnn_member_item_forwards": 3 * cap,
                            "training_forward_cost_unamortized": None,
                            "abstain": False,
                            "regret": None,
                        })
    payload = {
        "schema_version": "1.0",
        "pool_id": pool.pool_id,
        "pool_role": pool.pool_role,
        "records": records,
        "actual_cuda_inference": True,
        "standard_gpu": standard.gpu,
        "com_gpu": com.gpu,
        "actual_standard_member_forwards_shared_across_methods": standard.item_forwards,
        "actual_com_member_forwards_shared_across_methods": com.item_forwards,
        "outer_labels_read": False,
        "outer_metrics_computed": False,
        "evidence_status": "retrospective_previously_viewed_outer" if pool.pool_role.startswith("outer_") else "development_or_calibration_not_scientific_result",
        "com_identity": "COM-style conservative objective adaptation; not official COM UTR-ResNet or official weights",
    }
    atomic_json_write(payload, output)
    return {"pool_id": pool.pool_id, "records": len(records), "actual_cuda_inference": True}


def evaluate_pool(pool, selection_path, output, label_path=None):
    selection_path = Path(selection_path)
    payload = json.loads(selection_path.read_text())
    if pool.pool_role.startswith("outer_"):
        if label_path is None:
            raise ValueError("outer evaluation requires the Evaluator label vault")
        import csv
        import gzip
        labels = {}
        with gzip.open(label_path, "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("measured_rl"):
                    labels.setdefault(row["sequence_id"], []).append(float(row["measured_rl"]))
        labels = {key: float(np.median(values)) for key, values in labels.items()}
        measured = np.asarray([labels[value] for value in pool.sequence_ids])
    else:
        measured = np.asarray(pool.measured)
    best = float(np.max(measured))
    records = []
    for record in payload["records"]:
        selected_values = measured[record["selected_index"]]
        top_metrics = {}
        for k, indices in record["top_indices"].items():
            values = measured[indices]
            top_metrics[k] = {
                "regret_at_k": float(best - np.max(values)),
                "max_measured_rl": float(np.max(values)),
                "mean_measured_rl": float(np.mean(values)),
                "min_measured_rl": float(np.min(values)),
                "unique_sequences": len(set(indices)),
            }
        records.append({**record, "selected_measured_rl": float(selected_values), "regret": float(best - selected_values), "top_k": top_metrics})
    result = {**payload, "records": records, "outer_metrics_computed": pool.pool_role.startswith("outer_"), "evaluator_labels_used": pool.pool_role.startswith("outer_"), "labels_returned_to_learner": False, "selection_sha256": sha256_file(selection_path)}
    atomic_json_write(result, output)
    return {"pool_id": pool.pool_id, "records": len(records), "outer_metrics_computed": result["outer_metrics_computed"]}


def _pool_means(records, value):
    groups = defaultdict(list)
    for record in records:
        groups[record["pool_id"]].append(value(record))
    return {pool: float(np.mean(values)) for pool, values in groups.items()}


def freeze_dev_config(evaluation_paths, output, source_config):
    records = []
    for path in evaluation_paths:
        payload = json.loads(Path(path).read_text())
        if payload["pool_role"] != "dev" or payload["outer_metrics_computed"]:
            raise PermissionError("only dev evaluations can select hyperparameters")
        records.extend(row for row in payload["records"] if row["permutation_id"] == 0 and row["proposal_cap"] == 16)
    configurations = []
    for method in METHODS:
        settings = sorted({(row["lambda"], row["gamma"]) for row in records if row["method"] == method})
        for lambda_, gamma in settings:
            current = [row for row in records if row["method"] == method and row["lambda"] == lambda_ and row["gamma"] == gamma]
            top1 = _pool_means(current, lambda row: row["regret"])
            top4 = _pool_means(current, lambda row: row["top_k"]["4"]["regret_at_k"])
            configurations.append({"method": method, "lambda": lambda_, "gamma": gamma, "mean_regret": float(np.mean(list(top1.values()))), "mean_regret_at_4": float(np.mean(list(top4.values()))), "pools": len(top1)})
    selected = {}
    for method in ("bon", "com"):
        selected[method] = next(row for row in configurations if row["method"] == method)
    for method in ("lcb", "distance"):
        selected[method] = min((row for row in configurations if row["method"] == method), key=lambda row: (row["mean_regret"], row["lambda"]))
    selected["diversity_lcb"] = min((row for row in configurations if row["method"] == "diversity_lcb"), key=lambda row: (row["mean_regret_at_4"], row["lambda"], row["gamma"]))
    result = {
        "schema_version": "1.0",
        "selection_rule": "Dev cap16 only: LCB/distance minimize top1 mean regret; diversity-LCB minimizes regret@4; ties use smaller lambda/gamma. Calibration and outer are not used.",
        "selected": selected,
        "all_configurations": configurations,
        "evaluation_sources": [_evidence(path) for path in evaluation_paths],
        "source_config_sha256": sha256_file(source_config),
        "calibration_used_for_selection": False,
        "outer_used_for_selection": False,
    }
    atomic_json_write(result, output)
    return result


def frozen_methods(frozen):
    return {method: {"lambda": row["lambda"], "gamma": row["gamma"]} for method, row in frozen["selected"].items()}


def _bootstrap_interval(values, seed=20261124, rounds=10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = np.asarray([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(rounds)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize_evaluations(evaluation_paths, output, permutations=32):
    records = []
    sources = []
    for path in evaluation_paths:
        payload = json.loads(Path(path).read_text())
        records.extend(payload["records"])
        sources.append(_evidence(path))
    roles = sorted({row["pool_role"] for row in records})
    actual = [row for row in records if row["permutation_id"] == 0]
    configurations = []
    for method in METHODS:
        for cap in CAPS:
            current = [row for row in actual if row["method"] == method and row["proposal_cap"] == cap]
            if not current:
                continue
            per_pool = _pool_means(current, lambda row: row["regret"])
            configurations.append({"method": method, "proposal_cap": cap, "pools": len(per_pool), "mean_regret": float(np.mean(list(per_pool.values()))), "pool_standard_error": float(np.std(list(per_pool.values()), ddof=1) / math.sqrt(len(per_pool))), "mean_cnn_member_item_forwards": float(np.mean([row["cnn_member_item_forwards"] for row in current])), "coverage": 1.0})
    comparisons = []
    for method in METHODS:
        if method in ("bon", "lcb"):
            continue
        current = [row for row in actual if row["method"] == method and row["proposal_cap"] == 16]
        for reference in ("bon", "lcb"):
            reference_rows = [row for row in actual if row["method"] == reference and row["proposal_cap"] == 16]
            for metric in ("regret", "regret_at_4"):
                value = (lambda row: row["regret"]) if metric == "regret" else (lambda row: row["top_k"]["4"]["regret_at_k"])
                left = _pool_means(current, value)
                right = _pool_means(reference_rows, value)
                common = sorted(set(left) & set(right))
                differences = [left[pool] - right[pool] for pool in common]
                comparisons.append({"method": method, "reference": reference, "metric": metric, "mean_difference": float(np.mean(differences)), "pool_cluster_bootstrap_95_percent_interval": _bootstrap_interval(differences), "pools": len(common), "direction": "negative favors method"})
    top_k = []
    for method in METHODS:
        current = [row for row in actual if row["method"] == method and row["proposal_cap"] == 16]
        for k in TOP_K:
            key = str(k)
            top_k.append({"method": method, "k": k, "regret_at_k": float(np.mean([row["top_k"][key]["regret_at_k"] for row in current])), "max_measured_rl": float(np.mean([row["top_k"][key]["max_measured_rl"] for row in current])), "mean_measured_rl": float(np.mean([row["top_k"][key]["mean_measured_rl"] for row in current]))})
    risk_coverage = []
    for method in METHODS:
        current = [row for row in actual if row["method"] == method and row["proposal_cap"] == 16]
        ordered = sorted(current, key=lambda row: row["risk"])
        for coverage in np.linspace(0.1, 1.0, 10):
            count = max(1, int(math.ceil(coverage * len(ordered))))
            accepted = ordered[:count]
            risk_coverage.append({"method": method, "coverage": count / len(ordered), "accepted": count, "mean_regret": float(np.mean([row["regret"] for row in accepted])), "risk_definition": "train-distance for distance; ensemble standard deviation otherwise", "threshold_source": "retrospective empirical coverage; not a deployment threshold"})
    negative_controls = []
    for method in METHODS:
        actual_rows = [row for row in actual if row["method"] == method and row["proposal_cap"] == 16]
        control_rows = [row for row in records if row["method"] == method and row["proposal_cap"] == 16 and row["permutation_id"] > 0]
        if not control_rows:
            continue
        observed = float(np.mean([row["regret"] for row in actual_rows]))
        null_by_permutation = []
        for permutation_id in range(1, permutations + 1):
            null_by_permutation.append(float(np.mean([row["regret"] for row in control_rows if row["permutation_id"] == permutation_id])))
        negative_controls.append({"method": method, "observed_mean_regret": observed, "permutation_mean_regret": float(np.mean(null_by_permutation)), "permutation_95_percent_interval": [float(np.percentile(null_by_permutation, 2.5)), float(np.percentile(null_by_permutation, 97.5))], "permutations": len(null_by_permutation), "permutation_target": {"bon": "ensemble mean", "com": "COM ensemble mean", "lcb": "ensemble std", "distance": "train distance", "diversity_lcb": "ensemble std"}[method]})
    result = {"schema_version": "1.0", "pool_roles": roles, "pools": len(sources), "evidence_status": "retrospective_sensitivity_previously_viewed_outer_not_new_blind_test" if any(role.startswith("outer_") for role in roles) else "development_not_scientific_result", "configurations": configurations, "paired_comparisons": comparisons, "top_k": top_k, "risk_coverage": risk_coverage, "permutation_negative_controls": negative_controls, "sources": sources, "labels_returned_to_learner": False}
    atomic_json_write(result, output)
    return result
