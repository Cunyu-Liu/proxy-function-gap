import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .selection_eval import SelectionRecord, selection_config_hash
from .trainer.checkpoint import atomic_json_write
from .trainer.data import sha256_file


def _labels(path):
    values = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not row["measured_rl"]:
                continue
            values.setdefault(row["sequence_id"], []).append(float(row["measured_rl"]))
    return {sequence_id: float(np.median(items)) for sequence_id, items in values.items()}


def evaluate_outer_pool(source_pool_manifest, pool_id, selection_path, q_path, output):
    source_path = Path(source_pool_manifest)
    source = json.loads(source_path.read_text())
    role = "outer_designed" if pool_id.startswith("outer_designed_") else "outer_random"
    vault_path = source_path.parent / "evaluator" / "vault_references.json"
    vault = json.loads(vault_path.read_text())[role]
    if vault["holder"] != "Evaluator" or sha256_file(vault["path"]) != vault["sha256"]:
        raise PermissionError("outer evaluator vault mismatch")
    labels = _labels(vault["path"])
    selection_path = Path(selection_path)
    payload = json.loads(selection_path.read_text())
    records = [SelectionRecord(**row) for row in payload["records"]]
    if selection_config_hash(records) != payload["selection_config_sha256"] or payload["execution_metadata"]["outer_labels_read"]:
        raise ValueError("outer selection hash or Learner boundary mismatch")
    q_payload = json.loads(Path(q_path).read_text())
    if q_payload["selection_config_sha256"] != payload["selection_config_sha256"]:
        raise ValueError("Q decisions do not match frozen outer selections")
    decisions = {(item["pool_id"], item["seed"]): item for item in q_payload["decisions"]}
    pool_labels = [labels[row["sequence_id"]] for row in _pool_rows(source_path, source, role, pool_id)]
    scored = []
    for record in records:
        abstain = record.abstain
        if record.strategy == "fp-rej" and record.alpha == 20 and record.proposal_cap == 16:
            decision = decisions.get((pool_id, record.seed))
            if decision is None:
                raise ValueError("missing frozen Q decision")
            abstain = record.abstain or decision["abstain"]
        if record.abstain:
            selector_regret = None
        else:
            if record.selected_id not in labels:
                raise ValueError("selected outer sequence is absent from Evaluator vault")
            selector_regret = float(max(pool_labels) - labels[record.selected_id])
        regret = None if abstain else selector_regret
        scored.append({**record.to_dict(), "selector_abstain": record.abstain, "selector_regret": selector_regret, "abstain": abstain, "regret": regret, "q_applied": record.strategy == "fp-rej" and record.alpha == 20 and record.proposal_cap == 16})
    result = {
        "schema_version": "1.0",
        "pool_id": pool_id,
        "pool_role": role,
        "selection_config_sha256": payload["selection_config_sha256"],
        "selection_sha256": sha256_file(selection_path),
        "q_sha256": sha256_file(q_path),
        "records": scored,
        "outer_metrics_computed": True,
        "labels_returned_to_learner": False,
        "E1_enabled": True,
    }
    atomic_json_write(result, output)
    return {"pool_id": pool_id, "records": len(scored), "labels_returned_to_learner": False}


def _pool_rows(source_path, source, role, pool_id):
    with gzip.open(source_path.parent / source["files"][role], "rt") as handle:
        return [json.loads(line) for line in handle if line.strip() and json.loads(line)["pool_id"] == pool_id]
