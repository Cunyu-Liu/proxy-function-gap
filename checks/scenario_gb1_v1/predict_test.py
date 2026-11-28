#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gb1_common import load_unlabeled_csv, parse_mutant, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    data, frozen, output = Path(args.data), Path(args.frozen), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Prediction output is not empty; refusing to overwrite frozen test scores")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((frozen / "frozen_config.json").read_text())
    unlabeled_path = data / "roles/test_unlabeled.csv"
    if sha256_file(unlabeled_path) != config["test_unlabeled_sha256"]:
        raise RuntimeError("Unlabeled test manifest changed after freeze")
    mutants = load_unlabeled_csv(unlabeled_path)
    vocabulary = [parse_mutant(token)[0] for token in config["feature_vocabulary"][1:]]
    feature_map = {token: i + 1 for i, token in enumerate(vocabulary)}
    features = np.zeros((len(mutants), len(feature_map) + 1), dtype=np.float64)
    features[:, 0] = 1.0
    for row, mutant in enumerate(mutants):
        for wild, _, replacement in parse_mutant(str(mutant)):
            token = (wild, 0, replacement)
            if token not in feature_map:
                raise RuntimeError(f"Unknown test substitution: {token}")
            features[row, feature_map[token]] += 1.0
    features_cuda = torch.from_numpy(features).cuda()
    score_rows = []
    model_hashes = {}
    for seed in config["seeds"]:
        model_path = frozen / f"additive_ridge_seed_{seed}.npy"
        if sha256_file(model_path) != config["models"][str(seed)]:
            raise RuntimeError("Frozen model hash mismatch")
        weights_np = np.load(model_path)
        if weights_np.shape != (features.shape[1],) or not np.isfinite(weights_np).all():
            raise RuntimeError("Invalid frozen weights")
        weights = torch.from_numpy(weights_np).cuda()
        scores = (features_cuda @ weights).cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError("Nonfinite test predictions")
        score_rows.append(scores)
        model_hashes[str(seed)] = sha256_file(model_path)
    scores = np.stack(score_rows)
    prediction_path = output / "test_predictions.npz"
    np.savez(prediction_path, mutants=mutants, seeds=np.asarray(config["seeds"]), scores=scores)
    manifest = {
        "schema_version": config["schema_version"],
        "frozen_config_sha256": sha256_file(frozen / "frozen_config.json"),
        "test_unlabeled_sha256": sha256_file(unlabeled_path),
        "prediction_sha256": sha256_file(prediction_path),
        "model_sha256": model_hashes,
        "n_variants": len(mutants),
        "n_seeds": len(config["seeds"]),
        "device": torch.cuda.get_device_name(0),
        "test_labels_used": False,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
