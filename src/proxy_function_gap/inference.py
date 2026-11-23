import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .trainer.checkpoint import atomic_json_write, load_checkpoint, trainer_code_identity
from .trainer.config import TrainerConfig
from .trainer.cuda import configure_cuda
from .trainer.data import load_roles, preflight, sha256_file
from .trainer.engine import _code_commit
from .trainer.model import FreshSampleCNN, model_definition_hash, model_hash


@dataclass(frozen=True)
class EnsembleMember:
    seed: int
    checkpoint: Path
    sha256: str
    model_hash: str


@dataclass(frozen=True)
class EnsembleInferenceConfig:
    data_manifest: Path
    output_dir: Path
    members: tuple
    batch_size: int = 1024
    gpu_index: int = 0
    memory_fraction: float = 0.08
    dtype: str = "float32"

    def __post_init__(self):
        if len(self.members) != 3 or len({member.seed for member in self.members}) != 3:
            raise ValueError("exactly three unique fresh members are required")
        if self.batch_size <= 0 or not 0 < self.memory_fraction <= 1:
            raise ValueError("invalid inference batch or memory configuration")
        if any(len(member.sha256) != 64 or len(member.model_hash) != 64 for member in self.members):
            raise ValueError("member checkpoint and model SHA256 are required")

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text())
        allowed = {"data_manifest", "output_dir", "members", "batch_size", "gpu_index", "memory_fraction", "dtype"}
        if set(value) - allowed:
            raise ValueError(f"unknown inference config fields: {sorted(set(value) - allowed)}")
        members = tuple(
            EnsembleMember(int(item["seed"]), Path(item["checkpoint"]), item["sha256"], item["model_hash"])
            for item in value["members"]
        )
        return cls(
            data_manifest=Path(value["data_manifest"]),
            output_dir=Path(value["output_dir"]),
            members=members,
            batch_size=value.get("batch_size", 1024),
            gpu_index=value.get("gpu_index", 0),
            memory_fraction=value.get("memory_fraction", 0.08),
            dtype=value.get("dtype", "float32"),
        )

    def sha256(self):
        value = {
            "data_manifest": str(self.data_manifest),
            "output_dir": str(self.output_dir),
            "members": [
                {"seed": member.seed, "checkpoint": str(member.checkpoint), "sha256": member.sha256, "model_hash": member.model_hash}
                for member in self.members
            ],
            "batch_size": self.batch_size,
            "gpu_index": self.gpu_index,
            "memory_fraction": self.memory_fraction,
            "dtype": self.dtype,
        }
        import hashlib
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inverse_target_transform(values, scaler):
    if scaler.get("fit_role") != "train" or not np.isfinite(scaler["mean"]) or not np.isfinite(scaler["scale"]) or scaler["scale"] <= 0:
        raise ValueError("a finite train-only target scaler is required")
    return values * float(scaler["scale"]) + float(scaler["mean"])


def predict_array(model, values, batch_size, device, dtype):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            inputs = torch.from_numpy(values[start:start + batch_size]).to(device=device, dtype=dtype)
            if not inputs.is_cuda:
                raise RuntimeError("CPU inference fallback is forbidden")
            outputs.append(model(inputs).detach().cpu().numpy())
    torch.cuda.synchronize(device)
    return np.concatenate(outputs).astype(np.float64)


def run_ensemble_inference(config: EnsembleInferenceConfig):
    report = preflight(config.data_manifest)
    roles = load_roles(report)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    predictions = {role: [] for role in report.role_counts}
    member_reports = []
    for member in config.members:
        if sha256_file(member.checkpoint) != member.sha256:
            raise ValueError(f"checkpoint hash mismatch for seed {member.seed}")
        device, dtype, gpu = configure_cuda(config.gpu_index, config.memory_fraction, config.dtype, member.seed)
        checkpoint = load_checkpoint(member.checkpoint, device)
        for key, expected in (
            ("data_pack_sha256", report.data_pack_sha256),
            ("row_order_sha256", report.row_order_sha256),
            ("source_sha256", report.source_sha256),
            ("code_identity", trainer_code_identity()),
        ):
            if checkpoint[key] != expected:
                raise ValueError(f"inference identity mismatch for {key}")
        model = FreshSampleCNN()
        if checkpoint["model_definition_sha256"] != model_definition_hash(model):
            raise ValueError("inference model definition mismatch")
        model.load_state_dict(checkpoint["model_state"])
        model.to(device=device, dtype=dtype)
        model.eval()
        actual_model_hash = model_hash(model)
        if actual_model_hash != member.model_hash:
            raise ValueError(f"best model hash mismatch for seed {member.seed}")
        batch_count = 0
        for role, (x, _, _) in roles.items():
            values = inverse_target_transform(predict_array(model, x, config.batch_size, device, dtype), checkpoint["scaler"])
            predictions[role].append(values)
            batch_count += (len(x) + config.batch_size - 1) // config.batch_size
        member_reports.append({
            "seed": member.seed,
            "checkpoint_sha256": member.sha256,
            "initial_model_hash": checkpoint["initial_model_hash"],
            "model_hash": actual_model_hash,
            "model_state_sha256": checkpoint["model_state_sha256"],
            "target_scaler": dict(checkpoint["scaler"]),
            "gpu": gpu,
            "item_forward_by_role": {role: len(values) for role, values in ((r, x[0]) for r, x in roles.items())},
            "item_forward_total": sum(x[0].shape[0] for x in roles.values()),
            "batch_count": batch_count,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        })
        del model
        torch.cuda.empty_cache()

    files = {}
    for role, members in predictions.items():
        stacked = np.stack(members)
        values = {
            "row_id": roles[role][2],
            "ensemble_mean": stacked.mean(axis=0),
            "ensemble_std": stacked.std(axis=0, ddof=1),
            "ensemble_range": stacked.max(axis=0) - stacked.min(axis=0),
        }
        for index, member in enumerate(config.members):
            values[f"pred_seed_{member.seed}"] = stacked[index]
        path = config.output_dir / f"{role}_predictions.npz"
        temporary = config.output_dir / f"{role}_predictions.tmp.npz"
        np.savez(temporary, **values)
        temporary.replace(path)
        files[path.name] = {"sha256": sha256_file(path), "rows": len(stacked[0])}
    manifest = {
        "schema_version": "1.0",
        "config_sha256": config.sha256(),
        "code_commit": _code_commit(),
        "data_pack_sha256": report.data_pack_sha256,
        "row_order_sha256": report.row_order_sha256,
        "source_sha256": report.source_sha256,
        "synthetic": report.synthetic,
        "members": member_reports,
        "ensemble": {"mean": "member arithmetic mean", "std": "member sample std", "std_ddof": 1, "range": "member max-member min"},
        "files": files,
        "labels_in_output": False,
        "outer_labels_read": False,
        "outer_metrics_computed": False,
        "official_weights": False,
        "distillation": False,
        "evaluation_performed": False,
        "prediction_units": "measured_rl",
        "inverse_target_transform": True,
    }
    atomic_json_write(manifest, config.output_dir / "prediction_manifest.json")
    return manifest
