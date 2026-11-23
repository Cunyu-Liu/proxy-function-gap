import json
from pathlib import Path

import numpy as np
import torch

from .inference import EnsembleInferenceConfig, inverse_target_transform, predict_array
from .selection_eval import run_budget_grid, write_selection_records
from .small_model import build_classifier, predict_model
from .trainer.checkpoint import load_checkpoint
from .trainer.cuda import configure_cuda
from .trainer.data import sha256_file
from .trainer.model import FreshSampleCNN, model_definition_hash, model_hash


BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_utr50(sequence):
    if len(sequence) != 50 or any(base not in BASES for base in sequence):
        raise ValueError("production selection requires valid UTR50")
    values = np.zeros((1, 4, 50), dtype=np.float32)
    for index, base in enumerate(sequence):
        values[0, BASES[base], index] = 1.0
    return values


class FreshMemberScorer:
    def __init__(self, member, pool, config: EnsembleInferenceConfig):
        self.pool = pool
        self.seed = member.seed
        self.scaler = None
        self.device, self.dtype, self.gpu = configure_cuda(config.gpu_index, config.memory_fraction, config.dtype, member.seed)
        if sha256_file(member.checkpoint) != member.sha256:
            raise ValueError("selection checkpoint hash mismatch")
        checkpoint = load_checkpoint(member.checkpoint, self.device)
        self.scaler = checkpoint["scaler"]
        model = FreshSampleCNN()
        if checkpoint["model_definition_sha256"] != model_definition_hash(model):
            raise ValueError("selection model definition mismatch")
        model.load_state_dict(checkpoint["model_state"])
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        if model_hash(model) != member.model_hash:
            raise ValueError("selection model hash mismatch")
        self.model = model

    def score_index(self, index):
        values = encode_utr50(self.pool.utr50[index])
        normalized = predict_array(self.model, values, 1, self.device, self.dtype)
        return float(inverse_target_transform(normalized, self.scaler)[0])

    def __call__(self, candidate):
        return self.score_index(candidate.index)


class ProductionVScorer:
    def __init__(self, artifacts, memory_fraction=0.02):
        self.root = Path(artifacts)
        self.manifest = json.loads((self.root / "v_manifest.json").read_text())
        if self.manifest["outer_labels_read"] or self.manifest["outer_metrics_computed"] or self.manifest.get("calibrator_loss_type") != "log_loss":
            raise PermissionError("invalid or uncalibrated production V artifacts")
        for name, expected in self.manifest["files"].items():
            if sha256_file(self.root / name) != expected:
                raise ValueError("V artifact hash mismatch")
        folds = torch.load(self.root / "fold_models.pt", weights_only=False)
        platt = torch.load(self.root / "platt.pt", weights_only=False)
        self.device, self.dtype, self.gpu = configure_cuda(0, memory_fraction, "float32", 0)
        self.feature_names = tuple(self.manifest["selected"]["feature_names"])
        self.folds = []
        for artifact in folds:
            model = build_classifier(artifact["model_type"], len(self.feature_names))
            model.load_state_dict(artifact["model_state"])
            model.to(device=self.device, dtype=self.dtype)
            model.eval()
            self.folds.append((model, artifact["scaler"]))
        model = build_classifier("logistic", 1)
        model.load_state_dict(platt["model_state"])
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        self.platt = (model, platt["scaler"])
        self.last_item_forwards = len(self.folds) + 1

    def __call__(self, mean, std, value_range, support, index):
        available = {"proxy_mean": mean, "proxy_std": std, "proxy_range": value_range, "support": support}
        values = np.asarray([[available[name] for name in self.feature_names]], dtype=np.float64)
        raw = np.mean([predict_model(model, scaler, values, self.device, self.dtype)[0] for model, scaler in self.folds])
        calibrated = predict_model(self.platt[0], self.platt[1], np.asarray([[raw]]), self.device, self.dtype)[0]
        return float(1 - calibrated)


def run_production_grid(pool, inference_config, v_artifacts, output):
    if pool.synthetic or pool.manifest_path is None:
        raise PermissionError("production grid requires one non-synthetic manifest-backed pool")
    pool_manifest = json.loads(pool.manifest_path.read_text())
    prediction_manifest_path = Path(pool_manifest["prediction_manifest"]["path"])
    prediction_manifest = json.loads(prediction_manifest_path.read_text())
    if prediction_manifest["config_sha256"] != inference_config.sha256():
        raise ValueError("selection checkpoints differ from the pool inference lineage")
    members = [FreshMemberScorer(member, pool, inference_config) for member in inference_config.members]
    v_model = ProductionVScorer(v_artifacts)
    verifier_values = np.zeros(len(pool), dtype=np.float64)
    records = run_budget_grid(pool, members, verifier_values, v_model=v_model)
    metadata = {
        "pool_manifest_sha256": pool.manifest_sha256,
        "source_pools": [{"path": str(path), "sha256": sha256} for path, sha256 in pool.source_manifests],
        "inference_config_sha256": inference_config.sha256(),
        "v_manifest_sha256": sha256_file(Path(v_artifacts) / "v_manifest.json"),
        "actual_cuda_inference": True,
        "member_gpu": [member.gpu for member in members],
        "v_gpu": v_model.gpu,
        "outer_labels_read": False,
        "outer_metrics_computed": False,
        "E1_enabled": False,
    }
    return write_selection_records(records, output, metadata=metadata)
