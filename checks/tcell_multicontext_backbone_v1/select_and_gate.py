from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backbone import FreshSampleCNN, cap_regret, cluster_bootstrap, codes_to_one_hot, digest, make_proposals, save_json


def candidate_names(config, variant):
    if variant == "extended_mse_only":
        return ["extended_mse_only"]
    return [f"{variant}_lambda_{value:g}" for value in config["model"]["lambda_grid"]]


def load_ensemble(root, name, config):
    models = []
    summaries = []
    for seed in config["model"]["seeds"]:
        run = root / "training" / name / f"seed_{seed}"
        summary = json.loads((run / "summary.json").read_text())
        if summary["status"] != "completed" or not summary["converged_dev_plateau"] or not summary["finite_losses"] or not summary["gpu"]["actual_cuda"] or summary["gpu"]["cpu_fallback"] or summary["best_epoch"] < 1:
            raise RuntimeError(f"training gate failed: {name} seed {seed}")
        if digest(run / "best.pt") != summary["best_checkpoint_sha256"] or digest(run / "history.jsonl") != summary["history_sha256"]:
            raise ValueError("training artifact hash mismatch")
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        if checkpoint["model_state_sha256"] != summary["best_model_hash"] or checkpoint["epoch"] != summary["best_epoch"]:
            raise ValueError("best checkpoint identity mismatch")
        model = FreshSampleCNN()
        model.load_state_dict(checkpoint["model_state"])
        model.to("cuda:0").eval()
        models.append(model)
        summaries.append(summary)
    return models, summaries


def predict(models, sequences, batch_size=4096):
    codes = np.empty((len(sequences), 50), dtype=np.uint8)
    mapping = np.full(256, 255, dtype=np.uint8)
    for index, base in enumerate("ACGT"):
        mapping[ord(base)] = index
    for row, sequence in enumerate(sequences):
        value = sequence.decode("ascii") if isinstance(sequence, (bytes, np.bytes_)) else str(sequence)
        codes[row] = mapping[np.frombuffer(value.encode("ascii"), dtype=np.uint8)]
    members = []
    with torch.no_grad():
        for model in models:
            values = []
            for start in range(0, len(codes), batch_size):
                tensor = torch.from_numpy(codes[start:start + batch_size]).cuda()
                values.append(model(codes_to_one_hot(tensor)).cpu().numpy())
            members.append(np.concatenate(values).astype(np.float64))
    result = np.mean(members, axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite auxiliary inference")
    return result


def sample_selection_score(root, name, config, summaries):
    scores = []
    dev_losses = []
    for seed, summary in zip(config["model"]["seeds"], summaries):
        predictions = np.load(root / "training" / name / f"seed_{seed}" / "dev_predictions.npy", allow_pickle=False)
        scores.append(predictions)
        history = [json.loads(line) for line in (root / "training" / name / f"seed_{seed}" / "history.jsonl").read_text().splitlines()]
        dev_losses.append(next(row["dev"]["loss"] for row in history if row["epoch"] == summary["best_epoch"]))
    ensemble = np.mean(scores, axis=0)
    variant = summaries[0]["variant"]
    prefix = "original" if variant == "original_mse_tail_rank" else "extended"
    with np.load(root / "data" / f"{prefix}_dev.npz", allow_pickle=False) as pack:
        labels = pack["labels"]
        pools = pack["selection_pools"]
        contexts = pack["selection_pool_contexts"]
        names = pack["context_names"].tolist()
    scalers = summaries[0]["scalers"]
    values = []
    by_context = {}
    for context_id, context in enumerate(names):
        selected = contexts == context_id
        context_pools = pools[selected]
        proposals = make_proposals(len(context_pools), 64, config["auxiliary"]["proposal_seeds"], namespace=f"{context}|dev")
        regret, _ = cap_regret(ensemble, labels, context_pools, proposals)
        normalized = float(regret.mean() / scalers[str(context_id)]["scale"])
        values.append(normalized)
        by_context[context] = {"normalized_cap16_regret": normalized, "pools": len(context_pools)}
    return values, float(np.mean(dev_losses)), by_context


def source_dev_score(root, name, config, models, summaries):
    (root / "selection" / name).mkdir(parents=True, exist_ok=True)
    values = []
    details = {}
    for source, destination in (("n50_hek293t_r1", "defined_hepg2_r1"), ("defined_hepg2_r1", "n50_hek293t_r1")):
        base = root / "auxiliary" / source / "dev"
        sequences = np.load(base / "sequences.npy", allow_pickle=False)
        labels = np.load(base / "labels.npy", allow_pickle=False)
        pools = np.load(base / "pools.npy", allow_pickle=False)
        proposals = np.load(base / "proposals.npy", allow_pickle=False)
        scores = predict(models, sequences)
        np.save(root / "selection" / name / f"{source}_dev_predictions.npy", scores)
        regret, _ = cap_regret(scores, labels, pools, proposals)
        train = np.load(root / "auxiliary" / source / "train" / "labels.npy", allow_pickle=False)
        normalized = float(regret.mean() / train.std(ddof=0))
        values.append(normalized)
        details[source] = {"normalized_cap16_regret": normalized, "raw_cap16_regret": float(regret.mean()), "pools": len(pools)}
    return values, details


def select_variant(root, variant, config):
    candidates = []
    for name in candidate_names(config, variant):
        models, summaries = load_ensemble(root, name, config)
        sample_values, dev_loss, sample_details = sample_selection_score(root, name, config, summaries)
        source_values, source_details = source_dev_score(root, name, config, models, summaries)
        metric = float(np.mean(sample_values + source_values))
        candidates.append({"name": name, "lambda": None if variant == "extended_mse_only" else float(summaries[0]["lambda"]), "selection_metric": metric, "dev_combined_loss_mean": dev_loss, "sample_dev": sample_details, "source_inner_dev": source_details, "seed_best_epochs": [row["best_epoch"] for row in summaries], "seed_model_hashes": [row["best_model_hash"] for row in summaries]})
        del models
        torch.cuda.empty_cache()
    candidates.sort(key=lambda row: (round(row["selection_metric"] / 1e-12) * 1e-12, row["dev_combined_loss_mean"], row["lambda"] if row["lambda"] is not None else 0.0))
    selected = candidates[0]
    save_json(root / "selection" / f"{variant}.json", {"variant": variant, "selected": selected, "candidates": candidates, "destination_dev_used": False, "destination_test_used": False, "target_mrl_read": False})
    return selected


def destination_gate(root, selected, config):
    models, _ = load_ensemble(root, selected["name"], config)
    started = root / "auxiliary" / "DESTINATION_TEST_STARTED.json"
    with started.open("x") as handle:
        handle.write(json.dumps({"selected_main": selected, "evaluation_count_per_direction": 1}, indent=2) + "\n")
    results = {}
    for source, destination in (("n50_hek293t_r1", "defined_hepg2_r1"), ("defined_hepg2_r1", "n50_hek293t_r1")):
        base = root / "auxiliary" / destination / "test"
        sequences = np.load(base / "sequences.npy", allow_pickle=False)
        labels = np.load(base / "labels.npy", allow_pickle=False)
        baseline = np.load(base / "baseline.npy", allow_pickle=False)
        pools = np.load(base / "pools.npy", allow_pickle=False)
        proposals = np.load(base / "proposals.npy", allow_pickle=False)
        scores = predict(models, sequences)
        np.save(root / "auxiliary" / f"{destination}_test_main_predictions.npy", scores)
        main_regret, _ = cap_regret(scores, labels, pools, proposals)
        baseline_regret, _ = cap_regret(baseline, labels, pools, proposals)
        effect = main_regret - baseline_regret
        bootstrap = cluster_bootstrap(effect, config["auxiliary"]["gate"]["bootstrap_seed"], config["auxiliary"]["gate"]["bootstrap_replicates"])
        results[f"{source}_to_{destination}"] = {"destination": destination, "main_cap16_regret": float(main_regret.mean()), "baseline_cap16_regret": float(baseline_regret.mean()), "effect": bootstrap, "point_pass": bootstrap["effect"] < 0, "ci_pass": bootstrap["ci_high"] < 0, "evaluation_count": 1}
    passed = all(row["point_pass"] and row["ci_pass"] for row in results.values())
    report = {"analysis_id": config["analysis_id"], "selected_main": selected, "directions": results, "both_directions_pass": passed, "target_evaluation_permitted": passed, "target_mrl_read": False, "target_evaluation_count": 0}
    save_json(root / "auxiliary_gate.json", report)
    save_json(root / "auxiliary" / "DESTINATION_TEST_COMPLETED.json", {"evaluation_count_per_direction": 1, "gate_passed": passed, "results_sha256": digest(root / "auxiliary_gate.json")})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.use_deterministic_algorithms(True)
    root = Path(config["artifact_root"])
    selections = {variant: select_variant(root, variant, config) for variant in ("main", "original_mse_tail_rank")}
    save_json(root / "selection" / "extended_mse_only.json", {"selected": {"name": "extended_mse_only"}, "lambda_selection": "not_applicable_mse_only", "target_mrl_read": False})
    gate = destination_gate(root, selections["main"], config)
    print(json.dumps({"selections": selections, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
