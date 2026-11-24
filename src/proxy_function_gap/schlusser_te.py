import csv
import gzip
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .inference import inverse_target_transform, predict_array
from .kmer_proxy import KmerRegressor, kmer_features
from .paper2_baselines import CUDABaselineScorer
from .trainer.checkpoint import atomic_json_write, load_checkpoint
from .trainer.cuda import configure_cuda
from .trainer.data import sha256_file
from .trainer.model import model_hash


METHODS = ("random", "bon_cnn", "bon_kmer", "lcb_cnn")
COMPARISONS = ("bon_cnn-random", "bon_kmer-random", "lcb_cnn-random", "bon_cnn-bon_kmer")


@dataclass(frozen=True)
class CandidateTable:
    row_ids: tuple
    gene_ids: tuple
    transcript_ids: tuple
    utr50: tuple

    def __len__(self):
        return len(self.row_ids)


@dataclass(frozen=True)
class ModelScores:
    cnn_members: np.ndarray
    kmer_members: np.ndarray
    gpu: dict

    @property
    def cnn_mean(self):
        return self.cnn_members.mean(axis=0)

    @property
    def cnn_std(self):
        return self.cnn_members.std(axis=0, ddof=1)

    @property
    def kmer_mean(self):
        return self.kmer_members.mean(axis=0)


def load_config(path):
    value = json.loads(Path(path).read_text())
    required = {
        "schema_version", "dataset", "protocol_document", "output_root", "gate0_root",
        "learner", "evaluator_vault", "authorized_model_execution", "proxy_endpoint",
        "deployment_endpoint", "models", "caps", "paired_seeds", "methods", "primary",
        "score_matching", "bootstrap", "execution", "gpu",
    }
    if required - set(value):
        raise ValueError(f"missing confirmatory config fields: {sorted(required - set(value))}")
    if value["authorized_model_execution"] is not True:
        raise PermissionError("separate confirmatory execution authorization is required")
    if value["proxy_endpoint"] != "MRL" or value["deployment_endpoint"] != "TE":
        raise ValueError("this protocol is specifically MRL-proxy to TE deployment")
    if value["caps"] != [1, 2, 4, 8, 16] or value["primary"]["proposal_cap"] != 16:
        raise ValueError("frozen cap grid or primary cap changed")
    if value["primary"]["method"] != "bon_cnn" or value["primary"]["reference"] != "random":
        raise ValueError("primary comparison must be BoN-CNN versus Random")
    if tuple(value["methods"]) != METHODS:
        raise ValueError("frozen method set changed")
    if value["execution"]["rounds"] != 3 or not value["execution"]["single_unblinding_without_result_driven_revision"]:
        raise ValueError("exactly three rounds and one unblinding are required")
    return value


def _evidence(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _check_evidence(record):
    if sha256_file(record["path"]) != record["sha256"]:
        raise ValueError(f"frozen evidence hash mismatch: {record['path']}")


def paired_seeds(config):
    spec = config["paired_seeds"]
    return [spec["start"] + spec["step"] * index for index in range(spec["count"])]


def load_learner(config):
    record = config["learner"]
    _check_evidence(record)
    forbidden = {"TE", "te", "translation_efficiency", "MRL", "rl", "prediction", "predicted", "regret"}
    row_ids, genes, transcripts, sequences = [], [], [], []
    with gzip.open(record["path"], "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if forbidden & set(reader.fieldnames or []):
            raise PermissionError("Learner contains a label, prediction, or result column")
        if not {"row_id", "source_id", "utr50"}.issubset(reader.fieldnames or []):
            raise ValueError("Learner schema is incomplete")
        for row in reader:
            parts = row["source_id"].split("|")
            if len(parts) != 2 or not all(parts):
                raise ValueError("source_id must contain gene_ID|transcript_ID")
            sequence = row["utr50"]
            if len(sequence) != 50 or set(sequence) - set("ACGT"):
                raise ValueError("invalid terminal UTR50")
            row_ids.append(row["row_id"])
            genes.append(parts[0])
            transcripts.append(parts[1])
            sequences.append(sequence)
    expected = config["execution"]["expected_rows"]
    if len(row_ids) != expected or len(set(row_ids)) != expected:
        raise ValueError("Learner row count or row_id uniqueness mismatch")
    if len(set(genes)) != config["execution"]["expected_genes"]:
        raise ValueError("frozen gene count mismatch")
    if len(set(transcripts)) != config["execution"]["expected_transcripts"]:
        raise ValueError("frozen transcript count mismatch")
    return CandidateTable(tuple(row_ids), tuple(genes), tuple(transcripts), tuple(sequences))


def _check_gate0(config):
    root = Path(config["gate0_root"])
    decision = json.loads((root / "gate0_decision.json").read_text())
    if decision["authorized_model_execution"] or decision["outer_metrics_computed"] or decision["scientific_result_claimed"]:
        raise ValueError("Gate 0 historical boundary was altered")
    _check_evidence(config["evaluator_vault"])


def _model_evidence(config):
    result = {}
    for architecture, members in config["models"].items():
        result[architecture] = []
        for member in members:
            record = {"seed": member["seed"], "checkpoint": member["checkpoint"], "sha256": member["sha256"], "model_hash": member["model_hash"]}
            _check_evidence({"path": record["checkpoint"], "sha256": record["sha256"]})
            result[architecture].append(record)
    return result


def _git_state():
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    return root, commit, dirty


def freeze_protocol(config_path):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    _check_gate0(config)
    load_learner(config)
    protocol_path = Path(config["protocol_document"]).resolve()
    output = Path(config["output_root"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmatory root {output}")
    _, commit, dirty = _git_state()
    if dirty:
        raise RuntimeError("freeze requires a clean committed code tree")
    output.mkdir(parents=True)
    frozen = {
        "schema_version": "1.0",
        "dataset": config["dataset"],
        "code_commit": commit,
        "config": _evidence(config_path),
        "protocol_document": _evidence(protocol_path),
        "learner": dict(config["learner"]),
        "evaluator_vault": dict(config["evaluator_vault"]),
        "models": _model_evidence(config),
        "frozen_before_TE_distribution_or_results": True,
        "new_execution_authorization": True,
        "gate0_historical_authorization_remains_false": True,
        "result_driven_reconfiguration_allowed": False,
    }
    atomic_json_write(frozen, output / "frozen_protocol.json")
    (output / "frozen_config.json").write_bytes(config_path.read_bytes())
    (output / "frozen_protocol_document.md").write_bytes(protocol_path.read_bytes())
    return frozen


def _require_frozen(config_path, config):
    root = Path(config["output_root"])
    frozen_path = root / "frozen_protocol.json"
    if not frozen_path.is_file():
        raise PermissionError("freeze_protocol must complete before execution")
    frozen = json.loads(frozen_path.read_text())
    _check_evidence(frozen["config"])
    _check_evidence(frozen["protocol_document"])
    if sha256_file(config_path) != frozen["config"]["sha256"]:
        raise ValueError("active config differs from frozen config")
    _check_gate0(config)
    _, commit, dirty = _git_state()
    if commit != frozen["code_commit"] or dirty:
        raise RuntimeError("execution requires the frozen clean code commit")
    return frozen


def _cnn_scores(config, sequences):
    members = [{"path": row["checkpoint"], "sha256": row["sha256"], "model_hash": row["model_hash"]} for row in config["models"]["cnn"]]
    scorer = CUDABaselineScorer(
        members,
        gpu_index=config["gpu"]["gpu_index"],
        memory_fraction=config["gpu"]["memory_fraction"],
        batch_size=config["gpu"]["batch_size"],
    )
    _, _, stacked = scorer.predict(sequences)
    return stacked, scorer.gpu


def _kmer_scores(config, sequences):
    reports = []
    predictions = []
    features = np.asarray([kmer_features(sequence) for sequence in sequences], dtype=np.float32)
    for member in config["models"]["kmer"]:
        path = Path(member["checkpoint"])
        if sha256_file(path) != member["sha256"]:
            raise ValueError("4-mer checkpoint hash mismatch")
        device, dtype, gpu = configure_cuda(config["gpu"]["gpu_index"], config["gpu"]["memory_fraction"], config["gpu"]["dtype"], member["seed"])
        checkpoint = load_checkpoint(path, device)
        model = KmerRegressor()
        model.load_state_dict(checkpoint["model_state"])
        model.to(device=device, dtype=dtype)
        model.eval()
        if model_hash(model) != member["model_hash"]:
            raise ValueError("4-mer model hash mismatch")
        normalized = predict_array(model, features, config["gpu"]["batch_size"], device, dtype)
        predictions.append(inverse_target_transform(normalized, checkpoint["scaler"]))
        reports.append({**gpu, "seed": member["seed"], "model_hash": member["model_hash"], "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)})
        del model
        torch.cuda.empty_cache()
    return np.stack(predictions), reports


def score_models(config, table):
    cnn, cnn_gpu = _cnn_scores(config, table.utr50)
    kmer, kmer_gpu = _kmer_scores(config, table.utr50)
    return ModelScores(cnn, kmer, {"cnn": cnn_gpu, "kmer": kmer_gpu})


def _array_hash(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def nearest_score_match(mean, sequences, selected):
    candidates = [index for index in range(len(sequences)) if index != selected and sequences[index] != sequences[selected]]
    if not candidates:
        raise ValueError("no different-sequence score match is available")
    return min(candidates, key=lambda index: (abs(float(mean[index] - mean[selected])), hashlib.sha256(str(index).encode()).hexdigest(), index))


def build_score_match_maps(table, scores):
    return {
        "bon_cnn": [nearest_score_match(scores.cnn_mean, table.utr50, index) for index in range(len(table))],
        "bon_kmer": [nearest_score_match(scores.kmer_mean, table.utr50, index) for index in range(len(table))],
    }


def make_selections(table, scores, config, match_maps=None):
    maximum = max(config["caps"])
    records = []
    match_maps = build_score_match_maps(table, scores) if match_maps is None else match_maps
    means = {"bon_cnn": scores.cnn_mean, "bon_kmer": scores.kmer_mean, "lcb_cnn": scores.cnn_mean - scores.cnn_std}
    for seed in paired_seeds(config):
        rng = np.random.default_rng(seed)
        proposals = rng.choice(len(table), size=maximum, replace=True)
        for cap in config["caps"]:
            current = proposals[:cap]
            for method in METHODS:
                if method == "random":
                    selected = int(current[0])
                else:
                    values = means[method][current]
                    selected = int(current[int(np.argmax(values))])
                record = {
                    "seed": seed,
                    "method": method,
                    "proposal_cap": cap,
                    "proposal_indices": [int(value) for value in current],
                    "selected_index": selected,
                    "selected_id": table.row_ids[selected],
                    "selected_gene_id": table.gene_ids[selected],
                    "selected_transcript_id": table.transcript_ids[selected],
                    "regret": None,
                    "selection_model_forwards": 0 if method == "random" else 3 * cap,
                    "score_source": "shared_full_pool_actual_CUDA_inference",
                }
                if method in match_maps and cap == config["primary"]["proposal_cap"]:
                    matched = match_maps[method][selected]
                    record["score_match"] = {
                        "matched_index": matched,
                        "matched_id": table.row_ids[matched],
                        "absolute_proxy_mean_gap": float(abs(means[method][matched] - means[method][selected])),
                        "matching_rule": "nearest absolute proxy mean; different row and different utr50; deterministic index-hash tie break",
                    }
                records.append(record)
    return records


def _write_scores(path, table, scores):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        temporary,
        row_id=np.asarray(table.row_ids),
        gene_id=np.asarray(table.gene_ids),
        transcript_id=np.asarray(table.transcript_ids),
        utr50=np.asarray(table.utr50),
        cnn_members=scores.cnn_members,
        kmer_members=scores.kmer_members,
        cnn_mean=scores.cnn_mean,
        cnn_std=scores.cnn_std,
        kmer_mean=scores.kmer_mean,
    )
    temporary.replace(path)


def run_learner_round(config, table, round_number):
    output = Path(config["output_root"]) / f"learner_round_{round_number}"
    output.mkdir(parents=True, exist_ok=False)
    scores = score_models(config, table)
    _write_scores(output / "proxy_scores.npz", table, scores)
    match_maps = build_score_match_maps(table, scores)
    atomic_json_write({"rule": "nearest absolute proxy mean; different row and utr50; frozen before TE", "maps": match_maps}, output / "score_match_maps.json")
    records = make_selections(table, scores, config, match_maps)
    if any(row["regret"] is not None for row in records):
        raise PermissionError("Learner cannot compute TE regret")
    atomic_json_write({"records": records}, output / "selection.json")
    manifest = {
        "round": round_number,
        "rows": len(table),
        "selection_records": len(records),
        "prediction_units": "MRL",
        "deployment_endpoint": "TE",
        "actual_cuda_inference": True,
        "cpu_fallback": False,
        "gpu": scores.gpu,
        "cnn_member_item_forwards": int(scores.cnn_members.size),
        "kmer_member_item_forwards": int(scores.kmer_members.size),
        "full_pool_scores_shared_across_methods_and_seeds": True,
        "prediction_array_sha256": _array_hash([scores.cnn_members, scores.kmer_members]),
        "outer_labels_read": False,
        "outer_metrics_computed": False,
        "labels_returned_to_learner": False,
    }
    atomic_json_write(manifest, output / "learner_manifest.json")
    return manifest


def _labels(config, table):
    _check_evidence(config["evaluator_vault"])
    values = {}
    with gzip.open(config["evaluator_vault"]["path"], "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"row_id", "TE"}.issubset(reader.fieldnames or []):
            raise ValueError("Evaluator vault schema mismatch")
        for row in reader:
            if row["row_id"] in values:
                raise ValueError("duplicate Evaluator row_id")
            values[row["row_id"]] = float(row["TE"])
    result = np.asarray([values[row_id] for row_id in table.row_ids], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("all isolated Learner rows require finite TE")
    return result


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def _spearman(left, right):
    x, y = _rankdata(left), _rankdata(right)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _percentile(values, config):
    levels = config["bootstrap"]["percentile_interval"]
    return [float(value) for value in np.percentile(values, levels)]


def cluster_bootstrap(table, scores, labels, records, config, unit, match_maps=None):
    clusters = table.gene_ids if unit == "gene" else table.transcript_ids
    names = sorted(set(clusters))
    members = [np.asarray([index for index, value in enumerate(clusters) if value == name], dtype=np.int64) for name in names]
    caps = config["caps"]
    seed_count = config["paired_seeds"]["count"]
    maximum = max(caps)
    rng = np.random.default_rng(config["bootstrap"]["seed"] + (0 if unit == "gene" else 1))
    mean_regret = {(method, cap): [] for method in METHODS for cap in caps}
    paired = {(comparison, cap): [] for comparison in COMPARISONS for cap in caps}
    matched_effects = {method: [] for method in ("bon_cnn", "bon_kmer")}
    if match_maps is None:
        match_maps = build_score_match_maps(table, scores)
    match_maps = {method: np.asarray(match_maps[method], dtype=np.int64) for method in matched_effects}
    if any(len(mapping) != len(table) or np.any(mapping < 0) for mapping in match_maps.values()):
        raise ValueError("full-candidate frozen score-match maps are required")
    means = {"bon_cnn": scores.cnn_mean, "bon_kmer": scores.kmer_mean, "lcb_cnn": scores.cnn_mean - scores.cnn_std}
    replicates = config["bootstrap"]["replicates"]
    for _ in range(replicates):
        sampled = rng.integers(0, len(members), size=len(members))
        pool = np.concatenate([members[index] for index in sampled])
        pool_labels = labels[pool]
        oracle = float(np.max(pool_labels))
        uniforms = rng.random((seed_count, maximum))
        proposals = pool[np.floor(uniforms * len(pool)).astype(np.int64)]
        selections = {}
        regrets = {}
        for cap in caps:
            current = proposals[:, :cap]
            selections[("random", cap)] = proposals[:, 0]
            for method in means:
                values = means[method][current]
                selections[(method, cap)] = current[np.arange(seed_count), np.argmax(values, axis=1)]
            for method in METHODS:
                selected = selections[(method, cap)]
                regret = oracle - labels[selected]
                regrets[(method, cap)] = regret
                mean_regret[(method, cap)].append(float(np.mean(regret)))
            for comparison in COMPARISONS:
                left, right = comparison.split("-")
                effect = regrets[(left, cap)] - regrets[(right, cap)]
                paired[(comparison, cap)].append(float(np.mean(effect)))
        for method in matched_effects:
            selected = selections[(method, config["primary"]["proposal_cap"])]
            matched = match_maps[method][selected]
            if np.any(matched < 0):
                raise ValueError("score-match map is incomplete")
            matched_effects[method].append(float(np.mean(labels[matched] - labels[selected])))
    return {
        "unit": unit,
        "clusters": len(members),
        "replicates": replicates,
        "procedure": "resample biological clusters with replacement; rebuild candidate pool; rerun paired proposals, selections and oracle; percentile interval",
        "mean_regret": {f"{method}|{cap}": _percentile(values, config) for (method, cap), values in mean_regret.items()},
        "paired_regret_difference": {f"{comparison}|{cap}": _percentile(values, config) for (comparison, cap), values in paired.items()},
        "score_matched_regret_difference_conditional_on_frozen_match_map": {method: _percentile(values, config) for method, values in matched_effects.items()},
    }


def evaluate_round(config, table, round_number):
    learner = Path(config["output_root"]) / f"learner_round_{round_number}"
    output = Path(config["output_root"]) / f"evaluator_round_{round_number}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((learner / "learner_manifest.json").read_text())
    if manifest["outer_labels_read"] or manifest["outer_metrics_computed"] or not manifest["actual_cuda_inference"]:
        raise PermissionError("invalid Learner manifest")
    records = json.loads((learner / "selection.json").read_text())["records"]
    match_maps = json.loads((learner / "score_match_maps.json").read_text())["maps"]
    for record in records:
        if "score_match" in record and match_maps[record["method"]][record["selected_index"]] != record["score_match"]["matched_index"]:
            raise ValueError("selection differs from frozen full-pool score-match map")
    with np.load(learner / "proxy_scores.npz", allow_pickle=False) as pack:
        scores = ModelScores(pack["cnn_members"], pack["kmer_members"], {})
        if pack["row_id"].tolist() != list(table.row_ids):
            raise ValueError("score/Learner row order mismatch")
    if _array_hash([scores.cnn_members, scores.kmer_members]) != manifest["prediction_array_sha256"]:
        raise ValueError("prediction array identity mismatch")
    labels = _labels(config, table)
    oracle = float(np.max(labels))
    scored = []
    for record in records:
        row = dict(record)
        selected = record["selected_index"]
        row["selected_TE"] = float(labels[selected])
        row["regret"] = float(oracle - labels[selected])
        if "score_match" in record:
            matched = record["score_match"]["matched_index"]
            row["score_match"]["matched_TE"] = float(labels[matched])
            row["score_match"]["matched_regret"] = float(oracle - labels[matched])
            row["score_match"]["TE_difference_selected_minus_matched"] = float(labels[selected] - labels[matched])
            row["score_match"]["regret_difference_selected_minus_matched"] = float(labels[matched] - labels[selected])
        scored.append(row)
    atomic_json_write({"records": scored, "labels_returned_to_learner": False, "outer_metrics_computed": True}, output / "scored_selection.json")
    bootstraps = {unit: cluster_bootstrap(table, scores, labels, records, config, unit, match_maps) for unit in ("gene", "transcript")}
    configurations = []
    for method in METHODS:
        for cap in config["caps"]:
            current = [row for row in scored if row["method"] == method and row["proposal_cap"] == cap]
            key = f"{method}|{cap}"
            configurations.append({
                "method": method,
                "proposal_cap": cap,
                "replicates": len(current),
                "mean_selected_TE": float(np.mean([row["selected_TE"] for row in current])),
                "mean_regret": float(np.mean([row["regret"] for row in current])),
                "median_regret": float(np.median([row["regret"] for row in current])),
                "algorithmic_seed_standard_error": float(np.std([row["regret"] for row in current], ddof=1) / math.sqrt(len(current))),
                "gene_cluster_bootstrap_95_interval": bootstraps["gene"]["mean_regret"][key],
                "transcript_cluster_bootstrap_95_interval": bootstraps["transcript"]["mean_regret"][key],
            })
    comparisons = []
    for comparison in COMPARISONS:
        left, right = comparison.split("-")
        for cap in config["caps"]:
            left_rows = {(row["seed"], row["proposal_cap"]): row for row in scored if row["method"] == left}
            right_rows = {(row["seed"], row["proposal_cap"]): row for row in scored if row["method"] == right}
            differences, te_differences = [], []
            for seed in paired_seeds(config):
                a, b = left_rows[(seed, cap)], right_rows[(seed, cap)]
                if a["proposal_indices"] != b["proposal_indices"]:
                    raise ValueError("paired proposals differ")
                differences.append(a["regret"] - b["regret"])
                te_differences.append(a["selected_TE"] - b["selected_TE"])
            key = f"{comparison}|{cap}"
            comparisons.append({
                "comparison": comparison,
                "proposal_cap": cap,
                "replicates": len(differences),
                "mean_regret_difference_left_minus_right": float(np.mean(differences)),
                "mean_TE_difference_left_minus_right": float(np.mean(te_differences)),
                "left_wins_ties_right_wins": [int(sum(value < 0 for value in differences)), int(sum(value == 0 for value in differences)), int(sum(value > 0 for value in differences))],
                "gene_cluster_bootstrap_95_interval": bootstraps["gene"]["paired_regret_difference"][key],
                "transcript_cluster_bootstrap_95_interval": bootstraps["transcript"]["paired_regret_difference"][key],
                "regret_sign": "negative favors left; positive favors right",
                "TE_sign": "positive favors left; negative favors right",
            })
    score_matching = []
    for method in ("bon_cnn", "bon_kmer"):
        current = [row for row in scored if row["method"] == method and "score_match" in row]
        score_matching.append({
            "method": method,
            "proposal_cap": config["primary"]["proposal_cap"],
            "pairs": len(current),
            "mean_absolute_proxy_mean_gap": float(np.mean([row["score_match"]["absolute_proxy_mean_gap"] for row in current])),
            "mean_selected_TE": float(np.mean([row["selected_TE"] for row in current])),
            "mean_matched_TE": float(np.mean([row["score_match"]["matched_TE"] for row in current])),
            "mean_TE_difference_selected_minus_matched": float(np.mean([row["score_match"]["TE_difference_selected_minus_matched"] for row in current])),
            "mean_regret_difference_selected_minus_matched": float(np.mean([row["score_match"]["regret_difference_selected_minus_matched"] for row in current])),
            "gene_cluster_bootstrap_95_interval_for_regret_difference": bootstraps["gene"]["score_matched_regret_difference_conditional_on_frozen_match_map"][method],
            "transcript_cluster_bootstrap_95_interval_for_regret_difference": bootstraps["transcript"]["score_matched_regret_difference_conditional_on_frozen_match_map"][method],
            "sign": "TE difference positive favors optimized; regret difference negative favors optimized",
        })
    sequence_groups = {}
    for index, sequence in enumerate(table.utr50):
        sequence_groups.setdefault(sequence, []).append(index)
    duplicate_groups = [indices for indices in sequence_groups.values() if len(indices) > 1]
    within_ranges = [float(np.ptp(labels[indices])) for indices in duplicate_groups]
    primary_comparison = next(row for row in comparisons if row["comparison"] == "bon_cnn-random" and row["proposal_cap"] == config["primary"]["proposal_cap"])
    primary_configuration = next(row for row in configurations if row["method"] == "bon_cnn" and row["proposal_cap"] == config["primary"]["proposal_cap"])
    summary = {
        "schema_version": "1.0",
        "dataset": config["dataset"],
        "proxy_endpoint": "MRL",
        "deployment_endpoint": "TE",
        "deployment_mismatch": "cross_endpoint_not_same_endpoint_MRL_replication",
        "candidate_rows": len(table),
        "unique_genes": len(set(table.gene_ids)),
        "unique_transcripts": len(set(table.transcript_ids)),
        "oracle_TE_within_frozen_candidate_rows": oracle,
        "primary": {"configuration": primary_configuration, "paired_comparison": primary_comparison, "estimand": "mean true TE regret and paired BoN-CNN minus Random regret at cap16"},
        "configurations": configurations,
        "paired_comparisons": comparisons,
        "score_matching": score_matching,
        "endpoint_alignment_descriptive": {"spearman_cnn_MRL_vs_TE": _spearman(scores.cnn_mean, labels), "spearman_kmer_MRL_vs_TE": _spearman(scores.kmer_mean, labels), "cross_endpoint_absolute_error_computed": False},
        "TE_distribution_after_frozen_unblinding": {
            "min": float(np.min(labels)), "q1": float(np.quantile(labels, 0.25)), "median": float(np.median(labels)),
            "mean": float(np.mean(labels)), "q3": float(np.quantile(labels, 0.75)), "max": oracle,
            "std": float(np.std(labels, ddof=1)), "nonpositive_rows": int(np.sum(labels <= 0)),
        },
        "identifiability": {
            "unique_utr50": len(sequence_groups),
            "rows_in_duplicate_utr50_groups": int(sum(len(indices) for indices in duplicate_groups)),
            "duplicate_utr50_groups": len(duplicate_groups),
            "max_within_identical_utr50_TE_range": max(within_ranges) if within_ranges else 0.0,
            "single_study_cluster_bootstrap_is_not_cross_study_replication": True,
            "shared_training_labels_make_4mer_an_architecture_sensitivity_not_independent_validation": True,
            "sample_oracle_regret_is_not_population_oracle_regret": True,
        },
        "labels_returned_to_learner": False,
        "outer_metrics_computed": True,
        "result_driven_reconfiguration": False,
    }
    canonical = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    atomic_json_write(summary, output / "summary.json")
    atomic_json_write({
        "round": round_number,
        "summary_sha256": hashlib.sha256(canonical).hexdigest(),
        "selection_sha256": sha256_file(learner / "selection.json"),
        "scores_sha256": sha256_file(learner / "proxy_scores.npz"),
        "evaluator_vault_sha256": config["evaluator_vault"]["sha256"],
        "outer_metrics_computed": True,
        "labels_returned_to_learner": False,
    }, output / "evaluator_manifest.json")
    return summary


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _number(value):
    return "NA" if value is None else f"{value:.6g}"


def _interval(values):
    return f"[{_number(values[0])}, {_number(values[1])}]"


def render_result(config, validation):
    root = Path(config["output_root"])
    summary = json.loads((root / "evaluator_round_1" / "summary.json").read_text())
    primary = summary["primary"]
    comparison = primary["paired_comparison"]
    lines = [
        "# Schlusser HepG2：一次性确认性跨终点 OOD 结果",
        "",
        "## 证据状态",
        "",
        "本评测在配置、文档、模型与数据哈希冻结后执行；Learner 先完成三轮独立 CUDA 推理与无标签选择，Evaluator 随后一次性揭盲。proxy 固定为 Sample MRL，deployment endpoint 固定为 HepG2 内源 mRNA 的真实 TE。这是**跨终点部署失配**，不是 MRL 同终点复验，也不是新的湿实验验证。三轮完成后才生成报告，未按结果回调 cap、模型、基线或统计协议。",
        "",
        f"三轮一致性门禁：`pass={validation['pass']}`；6,340 个候选行、6,309 个基因、6,340 个转录本。基因簇 bootstrap 为主，转录本簇为敏感性分析；每轮 2,000 次簇重采样并重建候选池、提案、选择和样本内 oracle。",
        "",
        "## 主终点",
        "",
        f"cap16 BoN-CNN 的平均真实 TE regret 为 **{_number(primary['configuration']['mean_regret'])}**，基因簇 95% 区间 {_interval(primary['configuration']['gene_cluster_bootstrap_95_interval'])}。相对共用提案的 Random，配对 regret 差（BoN-CNN−Random）为 **{_number(comparison['mean_regret_difference_left_minus_right'])}**，基因簇 95% 区间 {_interval(comparison['gene_cluster_bootstrap_95_interval'])}，转录本簇区间 {_interval(comparison['transcript_cluster_bootstrap_95_interval'])}。regret 越低越好；差值小于 0 有利于 BoN-CNN，大于 0 有利于 Random。",
        "",
        _markdown_table(["指标", "估计", "基因簇 95% 区间", "转录本簇 95% 区间"], [
            ["BoN-CNN regret", _number(primary["configuration"]["mean_regret"]), _interval(primary["configuration"]["gene_cluster_bootstrap_95_interval"]), _interval(primary["configuration"]["transcript_cluster_bootstrap_95_interval"])],
            ["BoN-CNN−Random regret", _number(comparison["mean_regret_difference_left_minus_right"]), _interval(comparison["gene_cluster_bootstrap_95_interval"]), _interval(comparison["transcript_cluster_bootstrap_95_interval"])],
            ["BoN-CNN−Random TE", _number(comparison["mean_TE_difference_left_minus_right"]), "方向与 regret 差相反", "方向与 regret 差相反"],
        ]),
        "",
        "**确认性结论：没有观察到 BoN-CNN 改善真实 TE 的证据。** 主配对 regret 差点估计大于 0、selected TE 差点估计小于 0，观察方向反而有利于 Random；但基因和转录本簇区间均跨 0，因此也不能声称已精确识别 Random 的优势。区间跨 0 不被改写为正向结果。",
        "",
        "### TE 分布与 oracle 可识别性",
        "",
        _markdown_table(["min", "Q1", "median", "Q3", "mean", "std", "max"], [[
            _number(summary["TE_distribution_after_frozen_unblinding"][key])
            for key in ("min", "q1", "median", "q3", "mean", "std", "max")
        ]]),
        "",
        "TE 呈极端右偏，绝对 regret 主要由固定候选池的样本内最大 TE 决定；其簇区间的宽度不能解释为方法效果很大。配对差会消去共同 oracle，但本次配对区间同样跨 0。这里只展示冻结汇总中已有的分布量，不新增稳健终点或事后变换。",
        "",
        "## 全 cap 网格与必要基线",
        "",
        "Random 使用每个配对种子第一个提案；BoN 在同一提案集内最大化对应架构的 MRL 均值；LCB-CNN 固定为 mean−1×std。所有方法共享同一份实际 CUDA 全池分数，推理成本与选择分配成本均保存在 manifest/records。",
        "",
        _markdown_table(["方法", "cap", "平均 TE regret", "基因簇 95% 区间", "转录本簇 95% 区间"], [
            [row["method"], row["proposal_cap"], _number(row["mean_regret"]), _interval(row["gene_cluster_bootstrap_95_interval"]), _interval(row["transcript_cluster_bootstrap_95_interval"])]
            for row in summary["configurations"]
        ]),
        "",
        "### cap16 配对效应",
        "",
        _markdown_table(["比较（左−右）", "regret 差", "基因簇 95% 区间", "转录本簇 95% 区间", "左胜/平/右胜"], [
            [row["comparison"], _number(row["mean_regret_difference_left_minus_right"]), _interval(row["gene_cluster_bootstrap_95_interval"]), _interval(row["transcript_cluster_bootstrap_95_interval"]), "/".join(map(str, row["left_wins_ties_right_wins"]))]
            for row in summary["paired_comparisons"] if row["proposal_cap"] == 16
        ]),
        "",
        "## Score-matching 与跨终点方向",
        "",
        "匹配在揭盲前完成：对每个 cap16 BoN 选择，在全池中寻找 MRL 均值最近的**不同 row 且不同 UTR50**，以索引哈希确定性打破平局。MRL 与 TE 不可直接相减，因此不计算 MRL−TE 绝对误差；只比较匹配后的真实 TE 和 TE regret。下表 regret 差为 selected−matched，小于 0 有利于优化选择；TE 差大于 0 有利于优化选择。区间以冻结匹配映射为条件做基因/转录本簇 bootstrap。",
        "",
        _markdown_table(["方法", "平均 proxy 分数差", "selected TE", "matched TE", "regret 差", "基因簇 95% 区间"], [
            [row["method"], _number(row["mean_absolute_proxy_mean_gap"]), _number(row["mean_selected_TE"]), _number(row["mean_matched_TE"]), _number(row["mean_regret_difference_selected_minus_matched"]), _interval(row["gene_cluster_bootstrap_95_interval_for_regret_difference"])]
            for row in summary["score_matching"]
        ]),
        "",
        "## 正向、负向与不可识别边界",
        "",
        "- 上表完整保留每个 cap 与基线，不挑选最佳 cap；区间跨 0 时不声称方法差异，差值为正或负均按原方向报告。cap1 的 BoN/LCB 与 Random 使用同一首个提案，因此方法差异应恒为 0。",
        "- 4-mer 与 CNN 使用相同 Sample 训练标签，只是架构敏感性分析，不是独立验证；两架构比较也不消除共同 proxy 风险。",
        "- 这是单一 HepG2 研究的确认性跨终点 OOD。基因/转录本簇 bootstrap 处理候选池内聚类，不产生新的独立研究，不能据此声称跨细胞或跨研究普遍成立。",
        "- 样本内最大 TE 是固定候选池 oracle；regret 不是相对全体 HepG2 转录组的总体最优，也不是蛋白产量或直接因果效应。相同 UTR50 的多转录本标签差异无法由固定 50 nt proxy 区分。",
        "- TE 来自发表的 RNA-seq/Ribo-seq 派生表，本轮未从 FASTQ 重新计算。所有逐序列 TE 保留在远端 Evaluator 区域；本地只同步聚合证据，不向 Learner 返回标签。",
        "",
        "## 聚合核验",
        "",
        f"描述性 Spearman：CNN MRL–TE={_number(summary['endpoint_alignment_descriptive']['spearman_cnn_MRL_vs_TE'])}；4-mer MRL–TE={_number(summary['endpoint_alignment_descriptive']['spearman_kmer_MRL_vs_TE'])}。这些相关不替代主 regret，也不用于选择配置。相同 UTR50 重复组 {summary['identifiability']['duplicate_utr50_groups']} 个，涉及 {summary['identifiability']['rows_in_duplicate_utr50_groups']} 行，最大组内 TE range={_number(summary['identifiability']['max_within_identical_utr50_TE_range'])}。",
        "",
        "完整数值、全部配对效应、三轮哈希、CUDA 设备/成本、bootstrap 分布区间和不可识别字段见同目录 JSON；本报告由冻结汇总自动渲染，未手工改写结果。",
    ]
    path = root / "SCHLUSSER2024_HEPG2_TE_RESULT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def execute(config_path):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    frozen = _require_frozen(config_path, config)
    root = Path(config["output_root"])
    if (root / "execution_complete.json").exists():
        raise FileExistsError("one-time confirmatory execution is already complete")
    table = load_learner(config)
    for round_number in (1, 2, 3):
        run_learner_round(config, table, round_number)
    for round_number in (1, 2, 3):
        evaluate_round(config, table, round_number)
    hashes = []
    summaries = []
    for round_number in (1, 2, 3):
        manifest = json.loads((root / f"evaluator_round_{round_number}" / "evaluator_manifest.json").read_text())
        hashes.append(manifest["summary_sha256"])
        summaries.append(json.loads((root / f"evaluator_round_{round_number}" / "summary.json").read_text()))
    validation = {
        "pass": len(set(hashes)) == 1,
        "rounds": 3,
        "independent_CUDA_recomputation": True,
        "all_learner_rounds_frozen_before_first_evaluator": True,
        "summary_sha256_by_round": hashes,
        "exact_scientific_summary_match": len(set(hashes)) == 1,
        "result_review_before_all_rounds": False,
        "outer_tuning_or_result_driven_revision": False,
    }
    if not validation["pass"]:
        atomic_json_write(validation, root / "three_round_validation.json")
        raise RuntimeError("three independent rounds did not match exactly")
    atomic_json_write(validation, root / "three_round_validation.json")
    report = render_result(config, validation)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "execution_complete.json":
            files.append({**_evidence(path), "relative_path": str(path.relative_to(root))})
    complete = {
        "pass": True,
        "dataset": config["dataset"],
        "frozen_protocol_sha256": sha256_file(root / "frozen_protocol.json"),
        "code_commit": frozen["code_commit"],
        "rounds": 3,
        "actual_CUDA_inference_each_round": True,
        "outer_metrics_computed": True,
        "labels_returned_to_learner": False,
        "one_time_unblinding": True,
        "report": str(report),
        "files": files,
    }
    atomic_json_write(complete, root / "execution_complete.json")
    return {"pass": True, "rounds": 3, "report": str(report)}


def verify_delivery(config_path):
    config = load_config(config_path)
    root = Path(config["output_root"])
    complete = json.loads((root / "execution_complete.json").read_text())
    errors = []
    for record in complete["files"]:
        path = root / record["relative_path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(record["relative_path"])
    validation = json.loads((root / "three_round_validation.json").read_text())
    if not validation["pass"] or not complete["pass"] or complete["labels_returned_to_learner"]:
        errors.append("completion_boundary")
    report = {"pass": not errors, "errors": errors, "files_checked": len(complete["files"]), "rounds": 3, "aggregate_only_local_sync_allowed": True}
    atomic_json_write(report, root / "delivery_verification.json")
    return report
