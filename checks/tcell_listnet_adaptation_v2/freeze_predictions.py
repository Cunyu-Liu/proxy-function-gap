from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adaptation import FreshSampleCNN, codes_to_one_hot, configure_cuda, digest, encode_codes, save_json


def load_original_members(config):
    inference = json.loads(Path(config["model"]["inference_config"]).read_text())
    result = []
    for member in inference["members"]:
        if digest(member["checkpoint"]) != member["sha256"]:
            raise ValueError("original checkpoint hash mismatch")
        checkpoint = torch.load(member["checkpoint"], map_location="cpu", weights_only=False)
        result.append((checkpoint, float(checkpoint["scaler"]["mean"]), float(checkpoint["scaler"]["scale"]), member["sha256"]))
    return result


def load_adapted_members(root, selection_path):
    selection = json.loads((root / selection_path).read_text())
    selected = selection["selected"]
    result = []
    for seed in (11, 23, 37):
        run = root / "training" / selected["name"] / f"seed_{seed}"
        summary = json.loads((run / "summary.json").read_text())
        if summary["status"] != "completed" or not summary["converged_dev_plateau"] or digest(run / "best.pt") != summary["best_checkpoint_sha256"]:
            raise ValueError("adapted checkpoint freeze mismatch")
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        if checkpoint["model_state_sha256"] != summary["best_model_hash"]:
            raise ValueError("adapted model identity mismatch")
        result.append((checkpoint, float(summary["scaler"]["mean"]), float(summary["scaler"]["scale"]), summary["best_checkpoint_sha256"]))
    return result


def predict_ensemble(members, codes, device):
    outputs = []
    synthetic = torch.zeros((2, 50), dtype=torch.uint8, device=device)
    with torch.no_grad():
        for checkpoint, mean, scale, _ in members:
            model = FreshSampleCNN()
            model.load_state_dict(checkpoint["model_state"])
            model.to(device=device, dtype=torch.float32).eval()
            smoke = model(codes_to_one_hot(synthetic))
            if not smoke.is_cuda or not torch.isfinite(smoke).all():
                raise RuntimeError("CUDA model smoke failed")
            values = []
            for start in range(0, len(codes), 1024):
                values.append(model(codes_to_one_hot(codes[start:start + 1024])).cpu().numpy())
            outputs.append(np.concatenate(values).astype(np.float64) * scale + mean)
            del model
            torch.cuda.empty_cache()
    result = np.mean(outputs, axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite test-v2 predictions")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_hash = digest(config_path)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    v1 = Path(config["v1_artifact_root"])
    prepare = json.loads((root / "prepare_manifest.json").read_text())
    frozen = json.loads((root / "TRAINING_FROZEN.json").read_text())
    if prepare["test_v2_mrl_read"] or frozen["status"] != "PASS" or frozen["config_sha256"] != config_hash or frozen["run_count"] != 81 or not frozen["test_prediction_freeze_now_permitted"]:
        raise RuntimeError("training/test-v2 freeze gate failed")
    output = root / "test_predictions"
    output.mkdir(parents=True, exist_ok=False)
    device, gpu = configure_cuda(20261126, 0.28)
    sequences_all = np.load(root / "data" / "sequences.npy", allow_pickle=False)
    test_indices = np.load(root / "data" / "test_v2_indices.npy", allow_pickle=False)
    sequences = sequences_all[test_indices]
    codes = torch.from_numpy(encode_codes(sequences)).to(device)
    members = {
        "original_cnn": load_original_members(config),
        "v1_main": load_adapted_members(v1, Path("selection/main.json")),
        "v1_mse_only": load_adapted_members(v1, Path("selection/mse_only.json")),
        "listnet": load_adapted_members(root, Path("selection/listnet.json")),
    }
    scores = {}
    provenance = {}
    for method, method_members in members.items():
        scores[method] = predict_ensemble(method_members, codes, device)
        np.save(output / f"{method}_scores.npy", scores[method])
        provenance[method] = {
            "seed_checkpoint_sha256": [member[3] for member in method_members],
            "score_sha256": digest(output / f"{method}_scores.npy"),
            "three_seed_equal_weight": True,
        }
    save_json(root / "TEST_PREDICTIONS_FROZEN.json", {
        "status": "PASS", "analysis_id": config["analysis_id"], "config_sha256": config_hash,
        "training_freeze_sha256": digest(root / "TRAINING_FROZEN.json"),
        "v1_training_freeze_sha256": digest(v1 / "TRAINING_FROZEN.json"),
        "test_indices_sha256": digest(root / "data" / "test_v2_indices.npy"),
        "test_pools_sha256": digest(root / "data" / "test_v2_pools.npy"),
        "test_proposals_sha256": digest(root / "data" / "test_v2_proposals.npy"),
        "methods": provenance, "gpu": gpu,
        "test_v2_mrl_read": False, "test_mrl_read": False,
        "target_evaluation_count": 0, "label_evaluation_now_permitted_once": True,
        "post_freeze_temperature_weight_budget_changes": 0,
    })
    print(json.dumps({"status": "PASS", "methods": provenance, "test_v2_mrl_read": False}, indent=2))


if __name__ == "__main__":
    main()
