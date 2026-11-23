import json
import random

import pytest

from proxy_function_gap.metrics import regret
from proxy_function_gap.run import execute, main
from proxy_function_gap.selection import Budget, select
from proxy_function_gap.synthetic import make_known_error_case


def test_known_proxy_error_structure_is_exact():
    case = make_known_error_case(123)
    for truth, bias, noise, proxy in zip(
        case.synthetic_truth, case.structured_bias, case.gaussian_noise, case.proxy_values
    ):
        assert proxy == pytest.approx(truth + bias + noise)
    assert case.structured_bias[0] > 0
    assert case.pool.feasible == (True, True, True, False)


def test_bon_finds_known_adversarial_decoy_and_incurs_synthetic_regret():
    case = make_known_error_case()
    assert case.proxy_values[3] > case.proxy_values[0] > case.proxy_values[1]
    result = select(case.pool, "bon", Budget(16), random.Random(1), proxy=case.proxy)
    assert result.selected_id == "decoy"
    assert regret(case.synthetic_truth, result.selected_index, case.pool.feasible) == pytest.approx(0.9)


def test_oracle_has_zero_regret_on_feasible_pool():
    case = make_known_error_case()
    best_index = case.pool.sequence_ids.index("best")
    assert regret(case.synthetic_truth, best_index, case.pool.feasible) == 0
    with pytest.raises(ValueError, match="must be feasible"):
        regret(case.synthetic_truth, 3, case.pool.feasible)


def test_runner_is_deterministic_and_config_hash_changes_with_config():
    config = {
        "synthetic_seed": 314159,
        "selection_seed": 271828,
        "budget": 8,
        "alpha": 5.0,
        "strategies": ["random", "bon", "wsc", "greedy", "v-only", "joint"],
    }
    first = execute(config)
    second = execute(config)
    assert first == second
    assert first["run_id"] == first["config_hash"][:16]
    changed = dict(config, budget=9)
    assert execute(changed)["config_hash"] != first["config_hash"]
    assert all(item["synthetic_regret"] >= 0 for item in first["selections"])


def test_cli_writes_the_same_json_it_prints(tmp_path, capsys):
    output = tmp_path / "e0.json"
    exit_code = main(["--budget", "3", "--output", str(output)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == captured.out
    report = json.loads(captured.out)
    assert report["config"]["budget"] == 3
    assert "not biological evidence" in report["interpretation"]
