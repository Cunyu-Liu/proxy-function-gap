import json
import multiprocessing as mp
import time

import pytest

from proxy_function_gap.trainer.config import TrainerConfig
from proxy_function_gap.trainer.engine import _parallel_results


def _review_worker(payload):
    if payload == 0:
        raise RuntimeError("injected worker failure")
    time.sleep(30)
    return payload


def test_default_parallel_config_has_three_seeds_and_fractions():
    config = TrainerConfig.from_dict({"data_manifest": "manifest.json", "output_root": "runs"})
    assert config.seeds == (11, 23, 37)
    assert config.memory_fractions == (0.2, 0.2, 0.2)
    assert config.gpu_index == 0


def test_four_process_config_is_explicitly_supported():
    config = TrainerConfig.from_dict({
        "data_manifest": "manifest.json",
        "output_root": "runs",
        "seeds": [1, 2, 3, 4],
        "memory_fractions": [0.1, 0.2, 0.3, 0.4],
    })
    assert len(config.seeds) == 4
    assert sum(config.memory_fractions) == 1.0


@pytest.mark.parametrize(
    "updates",
    [
        {"seeds": [1, 2]},
        {"seeds": [1, 1, 2]},
        {"memory_fractions": [0.5]},
        {"memory_fractions": [0.5, 0.5, 0.5]},
        {"memory_fractions": [0.0, 0.2, 0.2]},
    ],
)
def test_invalid_parallel_configs_fail_closed(updates):
    value = {"data_manifest": "manifest.json", "output_root": "runs"}
    value.update(updates)
    with pytest.raises(ValueError):
        TrainerConfig.from_dict(value)


@pytest.mark.parametrize("field", ["max_epochs", "max_steps", "fixed_steps"])
def test_epoch_or_step_caps_are_not_accepted(field):
    with pytest.raises(ValueError, match="unknown trainer config"):
        TrainerConfig.from_dict({"data_manifest": "m", "output_root": "r", field: 10})


def test_resume_requires_all_seed_checkpoints():
    with pytest.raises(ValueError, match="every seed"):
        TrainerConfig.from_dict({
            "data_manifest": "m",
            "output_root": "r",
            "resume_from": {"11": {"path": "seed11.pt", "sha256": "0" * 64}},
        })


def test_frozen_config_hash_covers_resource_and_optimization_fields():
    config = TrainerConfig.from_dict({"data_manifest": "m", "output_root": "r", "batch_size": 1024})
    frozen = config.frozen_config()
    assert frozen["seeds"] == [11, 23, 37]
    assert frozen["batch_size"] == 1024
    assert frozen["learning_rate"] == 0.001
    assert len(config.frozen_config_sha256()) == 64
    changed = TrainerConfig.from_dict({"data_manifest": "m", "output_root": "r", "batch_size": 512})
    assert changed.frozen_config_sha256() != config.frozen_config_sha256()


def test_worker_failure_terminates_other_processes_promptly():
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="injected worker failure"):
        _parallel_results(_review_worker, [0, 1, 2])
    assert time.monotonic() - started < 10
    assert not mp.active_children()
