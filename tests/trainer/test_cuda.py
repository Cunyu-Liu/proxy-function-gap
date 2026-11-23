import json

import pytest
import torch

from proxy_function_gap.trainer.checkpoint import atomic_torch_save, load_checkpoint
from proxy_function_gap.trainer.cli import main
from proxy_function_gap.trainer.config import TrainerConfig
from proxy_function_gap.trainer.cuda import configure_cuda, cuda_self_check
from proxy_function_gap.trainer.data import preflight, sha256_file
from proxy_function_gap.trainer.engine import (
    run_parallel_cuda_self_checks,
    run_parallel_training,
    run_synthetic_parallel_training,
    train_one,
)
from proxy_function_gap.trainer.model import FreshSampleCNN, model_hash
from proxy_function_gap.trainer.synthetic import ZeroParameterModel, make_synthetic_pack


def config_for(manifest, output, patience=1):
    return TrainerConfig.from_dict({
        "data_manifest": str(manifest),
        "output_root": str(output),
        "memory_fractions": [0.1, 0.1, 0.1],
        "batch_size": 4,
        "patience": patience,
        "min_delta": 0.0,
    })


def resume_references(run_dir, seeds=(11, 23, 37), name="best.pt"):
    return {
        str(seed): {
            "path": str(run_dir / f"seed_{seed}" / name),
            "sha256": sha256_file(run_dir / f"seed_{seed}" / name),
        }
        for seed in seeds
    }


def test_cuda_unavailable_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU training fallback is forbidden"):
        configure_cuda(0, 0.1, "float32", 0)


def test_train_cli_requires_explicit_execution_flag(tmp_path):
    manifest = make_synthetic_pack(tmp_path / "pack")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data_manifest": str(manifest), "output_root": str(tmp_path / "runs")}))
    with pytest.raises(PermissionError, match="execute-training"):
        main(["train", "--config", str(config)])
    with pytest.raises(PermissionError, match="formal training"):
        run_parallel_training(config_for(manifest, tmp_path / "other-runs"))
    with pytest.raises(PermissionError, match="formal training"):
        main(["train", "--config", str(config), "--execute-training"])


def test_preflight_cli_reports_not_ready_for_formal_training(tmp_path, capsys):
    manifest = make_synthetic_pack(tmp_path / "pack")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data_manifest": str(manifest), "output_root": str(tmp_path / "runs")}))
    assert main(["preflight", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["role_counts"]["train"] == 8
    assert len(report["row_order_sha256"]["train"]) == 64
    assert not report["ready_for_formal_training"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for trainer smoke")
def test_cuda_self_check_and_fresh_model_dtype():
    metadata = cuda_self_check(0, 0.1, "float32", 7)
    assert metadata["self_check"] == "PASS"
    assert metadata["actual_device"] == "cuda:0"
    assert metadata["result_dtype"] == "torch.float32"
    assert not metadata["scientific_result"]
    device = torch.device("cuda:0")
    model = FreshSampleCNN().to(device=device, dtype=torch.float32)
    inputs = torch.zeros((2, 4, 50), device=device, requires_grad=True)
    output = model(inputs)
    assert output.shape == (2,)
    assert output.is_cuda
    output.square().mean().backward()
    assert len(model_hash(model)) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for parallel self-check")
def test_three_process_single_gpu_self_checks(tmp_path):
    manifest = make_synthetic_pack(tmp_path / "pack")
    results = run_parallel_cuda_self_checks(config_for(manifest, tmp_path / "unused"))
    assert len(results) == 3
    assert {item["actual_device"] for item in results} == {"cuda:0"}
    assert {item["seed"] for item in results} == {11, 23, 37}
    assert {item["per_process_memory_fraction"] for item in results} == {0.1}
    assert all(item["self_check"] == "PASS" for item in results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for three-seed smoke")
def test_three_seed_synthetic_training_and_resume_full_pipeline(tmp_path):
    manifest = make_synthetic_pack(tmp_path / "pack")
    initial = run_synthetic_parallel_training(config_for(manifest, tmp_path / "run1"))
    assert {item["seed"] for item in initial} == {11, 23, 37}
    assert all(item["status"] == "completed" for item in initial)
    assert all(item["total_epochs"] == 2 for item in initial)
    assert all(item["synthetic_smoke"] and not item["scientific_result"] for item in initial)
    for summary in initial:
        run_dir = tmp_path / "run1" / f"seed_{summary['seed']}"
        history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
        assert len(history) == 2
        for record in history:
            assert {"train_online_loss", "train_eval_loss", "dev_eval_loss"} <= set(record)
            assert record["pack_sha256"] == summary["data_pack_sha256"]
            assert record["row_order_sha256"] == summary["row_order_sha256"]
            assert record["config_sha256"] == summary["effective_config_sha256"]
            assert record["shuffle_seed"] == summary["seed"] + record["epoch"]
            assert len(record["model_state_sha256"]) == 64
        latest = load_checkpoint(run_dir / "latest.pt", torch.device("cpu"))
        assert latest["status"] == "completed"
        assert latest["model_state_sha256"] == history[-1]["model_state_sha256"]

    resumed_config = TrainerConfig.from_dict({
        "data_manifest": str(manifest),
        "output_root": str(tmp_path / "run2"),
        "memory_fractions": [0.1, 0.1, 0.1],
        "batch_size": 4,
        "patience": 1,
        "min_delta": 0.0,
        "resume_from": resume_references(tmp_path / "run1"),
    })
    resumed = run_synthetic_parallel_training(resumed_config)
    assert len(resumed) == 3
    initial_by_seed = {item["seed"]: item for item in initial}
    for summary in resumed:
        assert summary["total_epochs"] == 2
        assert summary["history_records"] == 2
        assert summary["initial_model_hash"] == initial_by_seed[summary["seed"]]["initial_model_hash"]
        assert summary["resume_checkpoint_sha256"] == initial_by_seed[summary["seed"]]["best_checkpoint_sha256"]
        history = (tmp_path / "run2" / f"seed_{summary['seed']}" / "history.jsonl").read_text().splitlines()
        assert len(history) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for resume validation")
def test_completed_checkpoint_and_expected_hash_are_enforced(tmp_path):
    manifest = make_synthetic_pack(tmp_path / "pack")
    report = preflight(manifest)
    summary = train_one(config_for(manifest, tmp_path / "run1"), 11, 0.1, report, ZeroParameterModel)
    latest = tmp_path / "run1" / "seed_11" / "latest.pt"
    with pytest.raises(ValueError, match="completed checkpoint"):
        train_one(
            config_for(manifest, tmp_path / "run2"),
            11,
            0.1,
            report,
            ZeroParameterModel,
            resume_from=(latest, summary["latest_checkpoint_sha256"]),
        )
    best = tmp_path / "run1" / "seed_11" / "best.pt"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        train_one(
            config_for(manifest, tmp_path / "run3"),
            11,
            0.1,
            report,
            ZeroParameterModel,
            resume_from=(best, "0" * 64),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for resume identity checks")
@pytest.mark.parametrize(
    "field,value",
    [
        ("code_identity", {"sha256": "bad", "files": {}}),
        ("effective_config_sha256", "bad"),
        ("scaler", {"mean": 99}),
        ("data_pack_sha256", "bad"),
        ("row_order_sha256", {"train": "bad"}),
        ("source_sha256", {"train": "bad"}),
        ("split_manifest_sha256", "bad"),
        ("learner_allowlist_sha256", "bad"),
    ],
)
def test_resume_rejects_tampered_checkpoint_identities(tmp_path, field, value):
    manifest = make_synthetic_pack(tmp_path / "pack")
    report = preflight(manifest)
    train_one(config_for(manifest, tmp_path / "run1"), 11, 0.1, report, ZeroParameterModel)
    path = tmp_path / "run1" / "seed_11" / "best.pt"
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    checkpoint[field] = value
    atomic_torch_save(checkpoint, path)
    with pytest.raises(ValueError, match=f"resume identity mismatch: {field}"):
        train_one(
            config_for(manifest, tmp_path / "run2"),
            11,
            0.1,
            report,
            ZeroParameterModel,
            resume_from=(path, sha256_file(path)),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for model identity check")
def test_resume_rejects_different_model_definition(tmp_path):
    manifest = make_synthetic_pack(tmp_path / "pack")
    report = preflight(manifest)
    train_one(config_for(manifest, tmp_path / "run1"), 11, 0.1, report, ZeroParameterModel)
    path = tmp_path / "run1" / "seed_11" / "best.pt"
    with pytest.raises(ValueError, match="model_definition_sha256"):
        train_one(
            config_for(manifest, tmp_path / "run2"),
            11,
            0.1,
            report,
            FreshSampleCNN,
            resume_from=(path, sha256_file(path)),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for config identity check")
def test_resume_rejects_changed_effective_configuration(tmp_path):
    manifest = make_synthetic_pack(tmp_path / "pack")
    report = preflight(manifest)
    train_one(config_for(manifest, tmp_path / "run1"), 11, 0.1, report, ZeroParameterModel)
    path = tmp_path / "run1" / "seed_11" / "best.pt"
    with pytest.raises(ValueError, match="effective_config_sha256"):
        train_one(
            config_for(manifest, tmp_path / "run2", patience=2),
            11,
            0.1,
            report,
            ZeroParameterModel,
            resume_from=(path, sha256_file(path)),
        )
