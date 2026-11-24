import dataclasses
import json

import numpy as np

from conftest import member_scorers, pool_factory
from proxy_function_gap.selection_eval import STRATEGIES, SelectionRecord, evaluate_strategy, run_budget_grid, selection_config_hash, write_selection_records


class FakeVModel:
    last_item_forwards = 4

    def __call__(self, mean, std, value_range, support, index):
        return 0.5


def make_record(pool, **overrides):
    values = {
        "pool_id": pool.pool_id,
        "strategy": "wsc",
        "alpha": 1,
        "proposal_cap": 2,
        "seed": 7,
        "selected_index": 0,
        "selected_id": pool.sequence_ids[0],
        "regret": None,
        "proxy_item_forwards": 0,
        "ensemble_member_item_forwards": 6,
        "member_item_forwards": {11: 2, 23: 2, 37: 2},
        "proposals": 2,
        "abstain": False,
        "verifier_item_forwards": 2,
        "verifier_member_item_forwards": 6,
        "v_model_item_forwards": 8,
    }
    values.update(overrides)
    return SelectionRecord(**values)


def test_verifier_member_and_v_forwards_are_counted_once(pool_factory, member_scorers):
    pool, _, _ = pool_factory("cost-verifier")
    record = evaluate_strategy(
        pool, member_scorers(pool), np.zeros(len(pool)), "wsc", 1, 2, 7, v_model=FakeVModel()
    )
    assert record.proxy_item_forwards == 0
    assert record.verifier_item_forwards == 2
    assert record.proxy_member_item_forwards == 0
    assert record.verifier_member_item_forwards == 6
    assert record.ensemble_member_item_forwards == 6
    assert record.v_model_item_forwards == 8
    assert record.q_model_item_forwards == 0
    assert record.total_model_item_forwards == 14
    assert record.total_model_item_forwards == 6 + 8
    assert record.to_dict()["total_model_item_forwards"] == 14


def test_proxy_member_forwards_use_the_same_total_ledger(pool_factory, member_scorers):
    pool, _, _ = pool_factory("cost-proxy")
    for strategy in ("bon", "lcb"):
        record = evaluate_strategy(pool, member_scorers(pool), np.zeros(len(pool)), strategy, 1, 2, 7)
        assert record.proxy_item_forwards == 2
        assert record.proxy_member_item_forwards == 6
        assert record.verifier_member_item_forwards == 0
        assert record.total_model_item_forwards == 6


def test_every_strategy_reconciles_member_attribution(pool_factory, member_scorers):
    pool, _, _ = pool_factory("cost-strategies")
    for strategy in STRATEGIES:
        record = evaluate_strategy(
            pool, member_scorers(pool), np.full(len(pool), 0.5), strategy, 1, 3, 7, v_model=FakeVModel()
        )
        assert sum(record.member_item_forwards.values()) == record.ensemble_member_item_forwards
        assert record.proxy_member_item_forwards + record.verifier_member_item_forwards == record.ensemble_member_item_forwards
        assert record.total_model_item_forwards == (
            record.ensemble_member_item_forwards + record.v_model_item_forwards + record.q_model_item_forwards
        )


def test_q_model_cost_is_included_without_counting_logical_calls(pool_factory):
    pool, _, _ = pool_factory("cost-q")
    record = make_record(pool, q_model_item_forwards=4)
    assert record.total_model_item_forwards == 18
    assert record.total_model_item_forwards != (
        record.proxy_item_forwards
        + record.verifier_item_forwards
        + record.ensemble_member_item_forwards
        + record.v_model_item_forwards
        + record.q_model_item_forwards
    )


def test_legacy_record_loader_preserves_cost_hash(pool_factory, member_scorers, tmp_path):
    from proxy_function_gap.pool_cli import _record

    pool, _, _ = pool_factory("cost-loader")
    record = evaluate_strategy(
        pool, member_scorers(pool), np.zeros(len(pool)), "wsc", 1, 2, 7, v_model=FakeVModel()
    )
    path = tmp_path / "selection.json"
    write_selection_records([record], path)
    value = json.loads(path.read_text())["records"][0]
    restored = _record(value)
    assert restored.proxy_member_item_forwards == record.proxy_member_item_forwards
    assert restored.verifier_member_item_forwards == record.verifier_member_item_forwards
    assert selection_config_hash([restored]) == selection_config_hash([record])


def test_grid_stops_on_actual_model_cost_not_nominal_cap(pool_factory, member_scorers):
    pool, _, _ = pool_factory("cost-grid")
    records = run_budget_grid(
        pool,
        member_scorers(pool),
        np.zeros(len(pool)),
        alphas=(1,),
        budgets=(1, 16),
        seeds=(7,),
        v_model=FakeVModel(),
        actual_cost_budget=13,
    )
    assert [record.strategy for record in records] == ["random", "bon", "lcb", "wsc"]
    assert [record.total_model_item_forwards for record in records] == [0, 3, 3, 7]
    assert sum(record.total_model_item_forwards for record in records) == 13


def test_unbounded_grid_remains_complete(pool_factory, member_scorers):
    pool, _, _ = pool_factory("cost-complete")
    records = run_budget_grid(pool, member_scorers(pool), np.full(len(pool), 0.5), v_model=FakeVModel())
    assert len(records) == 875


def test_production_grid_records_cost_budget_and_rerun_requirement(pool_factory, tmp_path, monkeypatch):
    import proxy_function_gap.production_selection as production

    pool, _, _ = pool_factory("cost-production")
    prediction_manifest_path = tmp_path / "predictions.json"
    prediction_manifest_path.write_text(json.dumps({"config_sha256": "lineage"}) + "\n")
    manifest_path = tmp_path / "pool.json"
    manifest_path.write_text(json.dumps({"prediction_manifest": {"path": str(prediction_manifest_path)}}) + "\n")
    pool = dataclasses.replace(pool, synthetic=False, manifest_path=manifest_path)
    v_artifacts = tmp_path / "v"
    v_artifacts.mkdir()
    (v_artifacts / "v_manifest.json").write_text("{}\n")

    class InferenceConfig:
        members = (object(), object(), object())

        def sha256(self):
            return "lineage"

    class FakeMember:
        gpu = "cpu-test"

        def __init__(self, member, pool, config):
            pass

    class FakeV:
        gpu = "cpu-test"

        def __init__(self, artifacts):
            pass

    captured = {}

    def fake_grid(pool, members, verifier_values, v_model=None, actual_cost_budget=None):
        captured["actual_cost_budget"] = actual_cost_budget
        return [make_record(pool)]

    monkeypatch.setattr(production, "FreshMemberScorer", FakeMember)
    monkeypatch.setattr(production, "ProductionVScorer", FakeV)
    monkeypatch.setattr(production, "run_budget_grid", fake_grid)
    output = tmp_path / "selection.json"
    production.run_production_grid(pool, InferenceConfig(), v_artifacts, output, actual_cost_budget=14)
    payload = json.loads(output.read_text())
    metadata = payload["execution_metadata"]
    assert captured["actual_cost_budget"] == 14
    assert metadata["actual_cost_budget"] == 14
    assert metadata["actual_cost_spent"] == 14
    assert metadata["grid_truncated_by_actual_cost"] is True
    assert metadata["cost_ledger_schema_version"] == "2.0"
    assert metadata["pre_cost_ledger_results_require_rerun"] is True
    assert payload["records"][0]["total_model_item_forwards"] == 14
