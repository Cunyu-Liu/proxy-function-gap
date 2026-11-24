import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .pool_manifest import load_pool_manifest
from .production_pool import SUPPORT_DEFINITION, train_support
from .production_selection import FreshMemberScorer, ProductionVScorer
from .q_head import pool_features, require_q_threshold_domain
from .selection_eval import SelectionRecord, run_budget_grid, write_selection_records
from .small_model import build_classifier, predict_model
from .trainer.cuda import configure_cuda
from .trainer.data import canonical_row_order_hash, sha256_file


@dataclass(frozen=True)
class OuterPool:
    pool_id: str
    pool_role: str
    synthetic: bool
    row_ids: tuple
    sequence_ids: tuple
    utr50: tuple
    group_ids: tuple
    support: np.ndarray
    proposal_probs: np.ndarray
    measured: None = None
    proxy_mean: np.ndarray = None
    proxy_std: np.ndarray = None
    proxy_range: np.ndarray = None

    def __len__(self):
        return len(self.row_ids)


def load_outer_pool(source_pool_manifest, pool_id):
    path = Path(source_pool_manifest).resolve()
    source = json.loads(path.read_text())
    role = "outer_designed" if pool_id.startswith("outer_designed_") else "outer_random" if pool_id.startswith("outer_random_") else None
    if role is None or pool_id not in source["pool_hashes"]:
        raise PermissionError("unknown outer fixed pool")
    validation_path = path.parent.parent / "final_validation.json"
    validation = json.loads(validation_path.read_text())
    if validation.get("pass") is not True:
        raise PermissionError("outer fixed-pool delivery is not verified")
    source_file = path.parent / source["files"][role]
    rows = []
    import gzip
    with gzip.open(source_file, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if row["pool_id"] == pool_id]
    if not rows or any(row["pool_hash"] != source["pool_hashes"][pool_id] or row["independent_evidence"] is not True for row in rows):
        raise ValueError("outer pool identity mismatch")
    forbidden = {"measured_rl", "measured_status", "predicted_mrl", "outer_regret", "outer_metric"}
    if any(forbidden & set(row) for row in rows):
        raise PermissionError("outer Learner view contains labels or metrics")
    production = json.loads(Path(source["source_production_manifest"]).read_text())
    train_path = Path(production["production_files"]["train"]["path"])
    with np.load(train_path, allow_pickle=False) as train:
        support = train_support([row["utr50"] for row in rows], train["utr50"].tolist())
    proposal = np.asarray([float(row["q_value"]) for row in rows])
    return OuterPool(
        pool_id=pool_id,
        pool_role=role,
        synthetic=False,
        row_ids=tuple(row["row_id"] for row in rows),
        sequence_ids=tuple(row["sequence_id"] for row in rows),
        utr50=tuple(row["utr50"] for row in rows),
        group_ids=tuple(row["cluster_id"] for row in rows),
        support=support,
        proposal_probs=proposal,
    )


class ProductionQScorer:
    def __init__(self, artifacts, memory_fraction=0.02):
        self.root = Path(artifacts)
        self.manifest = json.loads((self.root / "q_manifest.json").read_text())
        require_q_threshold_domain(self.manifest)
        if self.manifest.get("calibrator_loss_type") != "log_loss" or self.manifest["E1_enabled"]:
            raise PermissionError("Q artifact is not a finalized pre-E1 calibrator")
        for name, expected in self.manifest["files"].items():
            if sha256_file(self.root / name) != expected:
                raise ValueError("Q artifact hash mismatch")
        folds = torch.load(self.root / "q_fold_models.pt", weights_only=False)
        platt = torch.load(self.root / "q_platt.pt", weights_only=False)
        self.device, self.dtype, self.gpu = configure_cuda(0, memory_fraction, "float32", 0)
        self.folds = []
        for artifact in folds:
            model = build_classifier("logistic", 6)
            model.load_state_dict(artifact["model_state"])
            model.to(device=self.device, dtype=self.dtype)
            model.eval()
            self.folds.append((model, artifact["scaler"]))
        model = build_classifier("logistic", 1)
        model.load_state_dict(platt["model_state"])
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        self.platt = (model, platt["scaler"])
        self.threshold = self.manifest["selected_threshold"]

    def predict(self, pool, record):
        values = pool_features(pool, record).reshape(1, -1)
        raw = np.mean([predict_model(model, scaler, values, self.device, self.dtype)[0] for model, scaler in self.folds])
        probability = float(predict_model(self.platt[0], self.platt[1], np.asarray([[raw]]), self.device, self.dtype)[0])
        return {"pool_id": pool.pool_id, "seed": record.seed, "q_probability": probability, "threshold": self.threshold, "threshold_domain": self.manifest["threshold_domain"], "abstain": probability >= self.threshold, "q_model_item_forwards": len(self.folds) + 1}


def _require_verified(root):
    root = Path(root)
    for round_id in (1, 2, 3):
        candidates = [root / f"verification_round{round_id}.json", root.parent / f"verification_round{round_id}.json"]
        paths = [path for path in candidates if path.is_file()]
        if len(paths) != 1:
            raise PermissionError("pre-E1 verification path is missing or ambiguous")
        report = json.loads(paths[0].read_text())
        if report.get("pass") is not True or report.get("outer_labels_read") is not False:
            raise PermissionError("pre-E1 verification is incomplete")


def run_outer_grid(pool, inference_config, v_artifacts, q_artifacts, frozen_config, output):
    _require_verified(v_artifacts)
    _require_verified(q_artifacts)
    frozen = json.loads(Path(frozen_config).read_text())["selected"]
    if frozen["strategy"] != "fp-rej" or frozen["alpha"] != 20 or frozen["proposal_cap"] != 16:
        raise ValueError("outer selector differs from frozen dev configuration")
    members = [FreshMemberScorer(member, pool, inference_config) for member in inference_config.members]
    v_model = ProductionVScorer(v_artifacts)
    records = run_budget_grid(pool, members, np.zeros(len(pool)), v_model=v_model)
    for record in records:
        if record.regret is not None:
            raise PermissionError("outer Learner cannot compute regret")
    output = Path(output)
    metadata = {
        "pool_id": pool.pool_id,
        "pool_role": pool.pool_role,
        "pool_row_order_sha256": canonical_row_order_hash(pool.row_ids),
        "inference_config_sha256": inference_config.sha256(),
        "v_manifest_sha256": sha256_file(Path(v_artifacts) / "v_manifest.json"),
        "q_manifest_sha256": sha256_file(Path(q_artifacts) / "q_manifest.json"),
        "frozen_dev_config_sha256": sha256_file(frozen_config),
        "actual_cuda_inference": True,
        "member_gpu": [member.gpu for member in members],
        "v_gpu": v_model.gpu,
        "outer_labels_read": False,
        "outer_metrics_computed": False,
        "E1_enabled": False,
    }
    member_values = np.asarray([[member.score_index(index) for member in members] for index in range(len(pool))])
    pool = replace(
        pool,
        proxy_mean=member_values.mean(axis=1),
        proxy_std=member_values.std(axis=1, ddof=1),
        proxy_range=member_values.max(axis=1) - member_values.min(axis=1),
    )
    metadata["q_feature_member_item_forwards"] = int(member_values.size)
    config_hash = write_selection_records(records, output, metadata=metadata)
    q_model = ProductionQScorer(q_artifacts)
    decisions = []
    for record in records:
        if record.strategy == "fp-rej" and record.alpha == 20 and record.proposal_cap == 16:
            decisions.append(q_model.predict(pool, record))
    q_path = output.with_name(output.stem + "_q.json")
    q_path.write_text(json.dumps({"selection_config_sha256": config_hash, "decisions": decisions, "outer_labels_read": False}, indent=2) + "\n")
    return config_hash
