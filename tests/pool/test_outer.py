import csv
import gzip
import json

from proxy_function_gap.outer_evaluator import evaluate_outer_pool
from proxy_function_gap.selection_eval import SelectionRecord, selection_config_hash


def record(strategy, seed):
    return SelectionRecord(
        pool_id="outer_designed_000", strategy=strategy, alpha=20 if strategy == "fp-rej" else 0,
        proposal_cap=16, seed=seed, selected_index=0, selected_id="s0", regret=None,
        proxy_item_forwards=16 if strategy == "bon" else 0, ensemble_member_item_forwards=48,
        member_item_forwards={11: 16, 23: 16, 37: 16}, proposals=16, abstain=False,
        verifier_item_forwards=0 if strategy == "bon" else 16,
        v_model_item_forwards=0 if strategy == "bon" else 64,
    )


def test_outer_evaluator_separates_selector_and_q_regret(tmp_path):
    build = tmp_path / "build_a"
    learner = build / "learner"
    evaluator = build / "evaluator"
    learner.mkdir(parents=True)
    evaluator.mkdir()
    rows = [{"pool_id": "outer_designed_000", "sequence_id": "s0"}, {"pool_id": "outer_designed_000", "sequence_id": "s1"}]
    with gzip.open(learner / "outer.jsonl.gz", "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    labels = build / "labels.tsv.gz"
    with gzip.open(labels, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", "measured_rl"], delimiter="\t")
        writer.writeheader()
        writer.writerows([{"sequence_id": "s0", "measured_rl": 1}, {"sequence_id": "s1", "measured_rl": 3}, {"sequence_id": "outside_pool", "measured_rl": ""}])
    from proxy_function_gap.trainer.data import sha256_file
    (evaluator / "vault_references.json").write_text(json.dumps({"outer_designed": {"holder": "Evaluator", "path": str(labels), "sha256": sha256_file(labels)}}))
    source = build / "pool_manifest.json"
    source.write_text(json.dumps({"files": {"outer_designed": "learner/outer.jsonl.gz"}}))
    records = [record("fp-rej", 101), record("fp-rej", 102), record("bon", 101)]
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection_config_sha256": selection_config_hash(records), "execution_metadata": {"outer_labels_read": False}, "records": [row.to_dict() for row in records]}))
    decisions = tmp_path / "q.json"
    decisions.write_text(json.dumps({"selection_config_sha256": selection_config_hash(records), "decisions": [{"pool_id": "outer_designed_000", "seed": 101, "abstain": True}, {"pool_id": "outer_designed_000", "seed": 102, "abstain": False}]}))
    result = evaluate_outer_pool(source, "outer_designed_000", selection, decisions, tmp_path / "scored.json")
    assert result["labels_returned_to_learner"] is False
    scored = json.loads((tmp_path / "scored.json").read_text())["records"]
    assert scored[0]["regret"] is None and scored[0]["selector_regret"] == 2
    assert scored[1]["regret"] == 2 and scored[2]["regret"] == 2
