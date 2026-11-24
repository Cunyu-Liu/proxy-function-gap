import json
import sys

import numpy as np

from proxy_function_gap.selection_eval import SelectionRecord, selection_config_hash, write_selection_records


def _record(pool_id, strategy, alpha, seed, cnn, v, regret):
    return SelectionRecord(
        pool_id=pool_id, strategy=strategy, alpha=alpha, proposal_cap=16, seed=seed,
        selected_index=0, selected_id=f"{pool_id}-seq-000", regret=regret,
        proxy_item_forwards=16 if strategy in ("bon", "lcb") else 0,
        ensemble_member_item_forwards=cnn,
        member_item_forwards={11: cnn // 3, 23: cnn // 3, 37: cnn // 3},
        proposals=16, abstain=False, verifier_item_forwards=v // 4,
        v_model_item_forwards=v,
        proxy_member_item_forwards=48 if strategy in ("bon", "lcb") else 0,
        verifier_member_item_forwards=0 if strategy in ("random", "bon", "lcb") else cnn,
    )


def test_actual_cost_summary_keeps_full_support_tilt_over_budget(tmp_path, monkeypatch):
    import summarize_actual_cost_selection as summary

    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    q_root = tmp_path / "q"
    q_root.mkdir()
    alphas = summary.PRIMARY_ALPHA
    costs = {"random": (0, 0), "bon": (48, 0), "lcb": (48, 0), "wsc": (48, 64), "greedy": (48, 64), "fp-rej": (48, 64), "fp-tilt": (192, 256)}
    all_regrets = []
    for index in range(2):
        pool_id = f"dev_{index:03d}"
        records = [_record(pool_id, strategy, alpha, seed, *costs[strategy], regret=index + position / 10) for position, (strategy, alpha) in enumerate(alphas.items()) for seed in (101, 202, 303, 404, 505)]
        config_hash = write_selection_records(records, selection_dir / f"{pool_id}.json", metadata={"actual_cuda_inference": True, "outer_labels_read": False})
        decisions = [{"pool_id": pool_id, "seed": seed, "raw_q_probability": 0.4, "q_probability": 0.8 if seed == 101 else 0.2, "threshold": 0.75, "threshold_domain": "platt_calibrated_probability", "abstain": seed == 101, "q_model_item_forwards": 4} for seed in (101, 202, 303, 404, 505)]
        (selection_dir / f"{pool_id}_q.json").write_text(json.dumps({"selection_config_sha256": config_hash, "threshold_domain": "platt_calibrated_probability", "q_feature_member_item_forwards": 0, "decisions": decisions}) + "\n")
        all_regrets.extend(record.regret for record in records if record.strategy == "fp-rej")
    regrets = np.asarray(all_regrets)
    raw = np.full(len(regrets), 0.4)
    calibrated = np.tile([0.8, 0.2, 0.2, 0.2, 0.2], 2)
    np.savez(q_root / "q_oof_predictions.npz", raw_q=raw, calibrated_q=calibrated, regret=regrets, target=(regrets > 0.1).astype(float))
    manifest = {"threshold_domain": "platt_calibrated_probability", "selected_threshold": 0.75, "delta_regret": 0.1, "min_coverage": 0.5, "threshold_grid": [0.5, 0.75], "threshold_rows": [{"threshold": 0.5}, {"threshold": 0.75}]}
    (q_root / "q_manifest.json").write_text(json.dumps(manifest) + "\n")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"selected": {"strategy": "fp-rej", "alpha": 20, "proposal_cap": 16}}) + "\n")
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["summarize_actual_cost_selection.py", "--selection-dir", str(selection_dir), "--q-artifacts", str(q_root), "--frozen-config", str(frozen), "--role", "dev", "--count", "2", "--output", str(output)])
    summary.main()
    report = json.loads(output.read_text())
    rows = {row["strategy"]: row for row in report["configurations"]}
    assert len(rows) == 7
    assert rows["bon"]["mean_cnn_member_item_forwards"] == 48
    assert rows["fp-rej"]["mean_q_model_item_forwards"] == 4
    assert rows["fp-rej"]["v_plus_q"]["coverage"] == 0.8
    assert rows["fp-tilt"]["mean_cnn_member_item_forwards"] == 192
    assert rows["fp-tilt"]["selection_cnn_within_common_budget"] is False
    assert report["q_diagnostics"]["basis"] == "grouped_dev_oof"
    assert report["evidence_status"] == "development_only_not_scientific_result"
