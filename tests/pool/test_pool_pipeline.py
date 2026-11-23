import gzip
import json

import numpy as np
import pytest
import torch

from proxy_function_gap.inference import EnsembleInferenceConfig, inverse_target_transform, predict_array
from proxy_function_gap.pool_manifest import load_pool_manifest
from proxy_function_gap.production_pool import build_production_pool
from proxy_function_gap.q_head import QHeadConfig, fit_q_head, q_abstain, save_q_head_artifacts
from proxy_function_gap.selection_eval import SelectionRecord, run_budget_grid, write_selection_records
from proxy_function_gap.trainer.data import canonical_row_order_hash, sha256_file
from proxy_function_gap.trainer.synthetic import ZeroParameterModel
from proxy_function_gap.v_head import VHeadConfig, fit_v_head, save_v_head_artifacts


def test_pool_manifest_strict_schema(pool_factory):
    pool, _, _ = pool_factory("pool-a")
    assert len(pool) == 12
    assert pool.pool_role == "dev"
    assert pool.member_predictions.shape == (3, 12)
    assert np.isclose(pool.proxy_std[0], 0.1)


def test_pool_manifest_rejects_outer_role(pool_factory):
    _, manifest_path, manifest = pool_factory("pool-outer")
    manifest["pool_role"] = "outer"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PermissionError, match="outer"):
        from proxy_function_gap.pool_manifest import load_pool_manifest
        load_pool_manifest(manifest_path)


def test_budget_grid_counts_actual_three_member_forwards(pool_factory, member_scorers, tmp_path):
    pool, _, _ = pool_factory("pool-b")
    v_values = np.clip(1 - np.abs(pool.proxy_mean - pool.measured), 0, 1)
    records = run_budget_grid(
        pool,
        member_scorers(pool),
        v_values,
        alphas=(0, 1),
        budgets=(1, 3),
        seeds=(7, 11),
    )
    assert len(records) == 2 * 2 * 2 * 7
    by_strategy = {strategy: [item for item in records if item.strategy == strategy and item.proposal_cap == 3 and item.seed == 7 and item.alpha == 1] for strategy in {item.strategy for item in records}}
    assert all(item.ensemble_member_item_forwards == 0 for item in by_strategy["random"])
    for strategy in ("bon", "lcb", "wsc", "greedy"):
        assert all(item.ensemble_member_item_forwards == 9 for item in by_strategy[strategy])
    assert all(item.ensemble_member_item_forwards == 3 * len(pool) for item in by_strategy["fp-tilt"])
    assert all(item.ensemble_member_item_forwards <= 9 for item in by_strategy["fp-rej"])
    assert any(item.ensemble_member_item_forwards < 9 for item in by_strategy["fp-rej"])
    alpha_zero = [item for item in records if item.alpha == 0 and item.strategy in ("wsc", "fp-tilt", "fp-rej")]
    assert all(item.ensemble_member_item_forwards == 0 for item in alpha_zero)
    output = write_selection_records(records, tmp_path / "selection.json")
    manifest = json.loads((tmp_path / "selection.json").read_text())
    assert manifest["selection_config_sha256"] == output
    assert manifest["q_used_in_candidate_ranking"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA inference test")
def test_zero_model_cuda_inference_and_ensemble_statistics():
    model = ZeroParameterModel().cuda()
    values = np.zeros((4, 4, 50), dtype=np.float32)
    predictions = predict_array(model, values, 2, torch.device("cuda:0"), torch.float32)
    members = np.stack([predictions - 1, predictions + 1, predictions + 2])
    assert np.allclose(members.mean(axis=0), 2 / 3)
    assert np.allclose(members.std(axis=0, ddof=1), np.std(np.asarray([-1.0, 1.0, 2.0]), ddof=1))
    assert np.allclose(members.max(axis=0) - members.min(axis=0), 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA V/Q training tests")
def test_v_head_oof_selection_calibration_and_hashes(pool_factory, tmp_path):
    dev, _, _ = pool_factory("v-dev")
    calibration, _, _ = pool_factory("v-cal", role="calibration")
    config = VHeadConfig(deltas=(0.0, 0.25), model_types=("logistic", "mlp"), patience=1, min_delta=0.001)
    result = fit_v_head(dev, calibration, config)
    assert result["platt"]["loss_type"] == "log_loss"
    assert all(fold["loss_type"] == "squared_hinge" for fold in result["folds"])
    manifest = save_v_head_artifacts(result, tmp_path / "v")
    assert manifest["selected"]["delta"] in config.deltas
    assert manifest["selected"]["model_type"] in config.model_types
    assert len(manifest["candidate_scores"]) == 8
    assert manifest["dev_grouped_oof_only_for_selection"]
    assert manifest["calibration_only_for_final_probability_calibration"]
    assert len(manifest["fold_models"]) == 3
    assert all(item["best_epoch"] >= 1 for item in manifest["fold_models"])
    assert all(len(item["model_hash"]) == 64 for item in manifest["fold_models"])
    calibrated = np.load(tmp_path / "v" / "calibration_predictions.npz")
    assert np.all((calibrated["calibrated_v"] >= 0) & (calibrated["calibrated_v"] <= 1))
    assert len(calibrated["row_id"]) == len(calibration)
    assert set(manifest["files"]) == {"oof_predictions.npz", "calibration_predictions.npz", "fold_models.pt", "platt.pt"}


def q_examples(pool_factory, member_scorers, prefix, count, role):
    examples = []
    for index in range(count):
        pool, _, _ = pool_factory(f"{prefix}-{index}", role=role)
        selected = int(np.argmax(pool.proxy_mean))
        record = SelectionRecord(
            pool_id=pool.pool_id,
            strategy="fp-rej",
            alpha=5,
            proposal_cap=4,
            seed=101,
            selected_index=selected,
            selected_id=pool.sequence_ids[selected],
            regret=float(np.max(pool.measured) - pool.measured[selected]),
            proxy_item_forwards=0,
            ensemble_member_item_forwards=12,
            member_item_forwards={11: 4, 23: 4, 37: 4},
            proposals=4,
            abstain=False,
        )
        examples.append((pool, record))
    return examples


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Q training test")
def test_q_trained_after_frozen_selector_only_for_abstention(pool_factory, member_scorers, tmp_path):
    dev_examples = q_examples(pool_factory, member_scorers, "q-dev", 6, "dev")
    calibration_examples = q_examples(pool_factory, member_scorers, "q-cal", 3, "calibration")
    selector_hash = "f" * 64
    result = fit_q_head(dev_examples, calibration_examples, QHeadConfig(patience=1, min_delta=0.001), selector_hash)
    manifest = save_q_head_artifacts(result, tmp_path / "q")
    assert manifest["selection_config_sha256"] == selector_hash
    assert manifest["selector_frozen_before_q"]
    assert manifest["q_usage"].startswith("abstain/risk-coverage")
    assert 0 <= manifest["selected_threshold"] <= 1
    assert manifest["threshold_rows"]
    assert manifest["E1_enabled"] is False
    selected_before = dev_examples[0][1].selected_id
    selected_after = q_abstain(float(result["calibrated"][0]), manifest["selected_threshold"])
    assert isinstance(selected_after, bool)
    assert dev_examples[0][1].selected_id == selected_before
    assert set(manifest["files"]) == {"q_oof_predictions.npz", "q_calibration_predictions.npz", "q_fold_models.pt", "q_platt.pt"}


def production_fixture(pool_factory, tmp_path):
    pool, _, original = pool_factory("production", count=12)
    root = tmp_path / "production"
    prediction_path = root / "dev_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    for key in values:
        extra = np.asarray(["outside-pool"]) if key == "row_id" else np.asarray([0.0])
        values[key] = np.concatenate([values[key], extra])
    np.savez(prediction_path, **values)
    prediction_ids = values["row_id"].tolist()
    prediction_manifest_path = root / "prediction_manifest.json"
    prediction_manifest = json.loads(prediction_manifest_path.read_text())
    prediction_manifest["files"]["dev_predictions.npz"]["sha256"] = sha256_file(prediction_path)
    prediction_manifest["files"]["dev_predictions.npz"]["rows"] = 13
    prediction_manifest["synthetic"] = False
    prediction_manifest["prediction_units"] = "measured_rl"
    prediction_manifest["inverse_target_transform"] = True
    prediction_manifest["source_sha256"] = {"dev": original["label_source"]["sha256"]}
    prediction_manifest["row_order_sha256"] = {"dev": canonical_row_order_hash(prediction_ids)}
    prediction_manifest_path.write_text(json.dumps(prediction_manifest))

    train_path = root / "train.npz"
    np.savez(train_path, utr50=np.asarray(["A" * 50, "C" * 50]))
    production_manifest_path = root / "production_manifest.json"
    production_manifest = {
        "source_tsv": {"dev": original["label_source"]["path"]},
        "source_tsv_hashes": {"dev": original["label_source"]["sha256"]},
        "production_files": {"train": {"path": str(train_path), "sha256": sha256_file(train_path)}},
    }
    production_manifest_path.write_text(json.dumps(production_manifest))

    build = root / "source" / "build_a"
    learner = build / "learner"
    learner.mkdir(parents=True)
    source_pool_file = learner / "dev_pool.jsonl.gz"
    with gzip.open(source_pool_file, "wt") as handle:
        for index, row_id in enumerate(pool.row_ids):
            handle.write(json.dumps({
                "pool_id": "dev_000", "pool_role": "dev", "pool_hash": "p" * 64,
                "source_split_hash": original["split_hash"], "synthetic": False,
                "independent_evidence": True, "row_id": row_id,
                "sequence_id": pool.sequence_ids[index], "utr50": "A" * 50,
                "cluster_id": pool.group_ids[index], "q_value": 1 / len(pool),
            }) + "\n")
        handle.write(json.dumps({
            "pool_id": "dev_001", "pool_role": "dev", "pool_hash": "q" * 64,
            "source_split_hash": original["split_hash"], "synthetic": False,
            "independent_evidence": True, "row_id": "other-pool-row",
            "sequence_id": "other-sequence", "utr50": "C" * 50,
            "cluster_id": "other-cluster", "q_value": 1.0,
        }) + "\n")
    source_pool_manifest_path = build / "pool_manifest.json"
    source_pool_manifest_path.write_text(json.dumps({
        "synthetic": False, "E1_enabled": False, "source_split_hash": original["split_hash"],
        "pool_hashes": {"dev_000": "p" * 64}, "files": {"dev": "learner/dev_pool.jsonl.gz"},
        "source_production_manifest": str(production_manifest_path),
        "source_production_manifest_sha256": sha256_file(production_manifest_path),
    }))
    final_validation = build.parent / "final_validation.json"
    final_validation.write_text(json.dumps({"pass": True, "E1_enabled": False}))
    evaluator = build / "evaluator"
    evaluator.mkdir()
    (evaluator / "vault_references.json").write_text(json.dumps({
        "dev": {"holder": "Learner", "path": original["label_source"]["path"], "sha256": original["label_source"]["sha256"]}
    }))
    return source_pool_manifest_path, prediction_manifest_path, original


def test_production_pool_binds_frozen_roles_and_prediction_subsets(pool_factory, tmp_path):
    source, inference, _ = production_fixture(pool_factory, tmp_path)
    result = build_production_pool(source, "dev_000", inference, tmp_path / "built")
    pool = load_pool_manifest(result["manifest"])
    assert len(pool) == 12
    assert pool.synthetic is False
    assert np.all(pool.support == 1.0)
    assert pool.row_ids[0] == "production-row-000"
    assert result["outer_labels_read"] is False


def test_production_pool_rejects_label_and_candidate_tampering(pool_factory, tmp_path):
    source, inference, original = production_fixture(pool_factory, tmp_path)
    result = build_production_pool(source, "dev_000", inference, tmp_path / "built")
    manifest_path = tmp_path / "built" / "pool_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    other_labels = tmp_path / "other_labels.tsv.gz"
    other_labels.write_bytes((tmp_path / "production" / "labels.tsv.gz").read_bytes())
    manifest["label_source"] = {"path": str(other_labels), "sha256": sha256_file(other_labels)}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PermissionError, match="Learner"):
        load_pool_manifest(manifest_path)

    result = build_production_pool(source, "dev_000", inference, tmp_path / "built_second")
    second_path = tmp_path / "built_second" / "pool_manifest.json"
    manifest = json.loads(second_path.read_text())
    candidate_path = tmp_path / "built_second" / "candidates.tsv.gz"
    with gzip.open(candidate_path, "rt") as handle:
        lines = handle.readlines()
    lines[1] = lines[1].replace("production-row-000", "forged-row-000")
    with gzip.open(candidate_path, "wt") as handle:
        handle.writelines(lines)
    manifest["candidate_file"] = {"path": str(candidate_path), "sha256": sha256_file(candidate_path)}
    second_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity|coverage|row-order"):
        load_pool_manifest(second_path)


def test_combine_pools_preserves_disjoint_identities_and_provenance(pool_factory):
    from proxy_function_gap.pool_manifest import combine_pools
    first, _, _ = pool_factory("combine-a")
    second, _, _ = pool_factory("combine-b")
    combined = combine_pools([first, second], "combined")
    assert len(combined) == 24
    assert len(combined.source_manifests) == 2
    assert len(combined.manifest_sha256) == 64
    assert np.isclose(combined.proposal_probs.sum(), 1.0)
    with pytest.raises(ValueError, match="disjoint"):
        combine_pools([first, first], "duplicate")


def test_inverse_target_transform_restores_measured_units():
    values = np.asarray([-1.0, 0.0, 2.0])
    restored = inverse_target_transform(values, {"mean": 6.5, "scale": 2.0, "fit_role": "train"})
    assert np.allclose(restored, [4.5, 6.5, 10.5])
    with pytest.raises(ValueError, match="train-only"):
        inverse_target_transform(values, {"mean": 6.5, "scale": 2.0, "fit_role": "dev"})


def test_unlabeled_outer_selection_never_computes_regret(pool_factory, member_scorers):
    import dataclasses
    pool, _, _ = pool_factory("unlabeled")
    pool = dataclasses.replace(pool, measured=None)
    from proxy_function_gap.selection_eval import evaluate_strategy
    record = evaluate_strategy(pool, member_scorers(pool), np.zeros(len(pool)), "bon", 0, 2, 101)
    assert record.regret is None
    assert record.ensemble_member_item_forwards == 6


def test_production_candidates_use_utr50_and_length_mask(pool_factory, member_scorers):
    import dataclasses
    pool, _, _ = pool_factory("constraint")
    sequences = list(pool.utr50)
    sequences[0] = "A"
    members = pool.member_predictions.copy()
    members[:, 0] = 100.0
    pool = dataclasses.replace(pool, utr50=tuple(sequences), member_predictions=members)
    from proxy_function_gap.selection_eval import evaluate_strategy
    record = evaluate_strategy(pool, member_scorers(pool), np.zeros(len(pool)), "bon", 0, 16, 101)
    assert record.selected_index != 0
