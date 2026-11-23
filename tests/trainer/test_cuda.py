import json

import pytest
import torch

from proxy_function_gap.trainer.cli import main
from proxy_function_gap.trainer.config import TrainerConfig
from proxy_function_gap.trainer.cuda import configure_cuda, cuda_self_check
from proxy_function_gap.trainer.data import preflight
from proxy_function_gap.trainer.engine import run_parallel_cuda_self_checks, run_parallel_training, train_one
from proxy_function_gap.trainer.model import FreshSampleCNN, model_hash
from proxy_function_gap.trainer.synthetic import make_synthetic_pack


class ZeroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        return inputs[:, 0, 0] * 0 + self.weight


def config_for(manifest, output, patience=1):
    return TrainerConfig.from_dict({
        "data_manifest": str(manifest),
        "output_root": str(output),
        "memory_fractions": [0.1, 0.1, 0.1],
        "batch_size": 4,
        "patience": patience,
        "min_delta": 0.0,
    })


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
    output = model(torch.zeros((2, 4, 50), device=device))
    assert output.shape == (2,)
    assert output.is_cuda
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for trainer smoke")
def test_zero_loss_training_history_best_restore_and_resume(tmp_path):
    manifest = make_synthetic_pack(tmp_path / "pack")
    report = preflight(manifest)
    config = config_for(manifest, tmp_path / "run1")
    summary = train_one(config, 11, 0.1, report, model_factory=ZeroModel)
    assert summary["total_epochs"] == 2
    assert summary["best_epoch"] == 1
    assert summary["history_records"] == 2
    assert summary["stop_reason"] == "train_and_dev_plateau_min_delta_patience"
    assert summary["gpu"]["actual_device"] == "cuda:0"
    assert summary["gpu"]["dtype"] == "torch.float32"
    assert summary["synthetic_smoke"]
    assert not summary["scientific_result"]
    history = [json.loads(line) for line in (tmp_path / "run1" / "seed_11" / "history.jsonl").read_text().splitlines()]
    assert len(history) == 2
    assert all({"train_loss", "dev_loss", "epoch_seconds"} <= set(item) for item in history)

    resumed_config = TrainerConfig.from_dict({
        "data_manifest": str(manifest),
        "output_root": str(tmp_path / "run2"),
        "memory_fractions": [0.1, 0.1, 0.1],
        "batch_size": 4,
        "patience": 1,
        "min_delta": 0.0,
        "resume_from": {
            "11": str(tmp_path / "run1" / "seed_11" / "latest.pt"),
            "23": str(tmp_path / "run1" / "seed_11" / "latest.pt"),
            "37": str(tmp_path / "run1" / "seed_11" / "latest.pt"),
        },
    })
    resumed = train_one(resumed_config, 11, 0.1, report, model_factory=ZeroModel)
    assert resumed["total_epochs"] == 3
    assert resumed["history_records"] == 3
    assert resumed["resumed_from"].endswith("latest.pt")
    assert resumed["resume_checkpoint_sha256"] == summary["latest_checkpoint_sha256"]
    resumed_history = (tmp_path / "run2" / "seed_11" / "history.jsonl").read_text().splitlines()
    assert len(resumed_history) == 3
