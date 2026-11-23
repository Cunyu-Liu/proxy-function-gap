import json
import random

import pytest

from proxy_function_gap.metrics import regret
from proxy_function_gap.run import execute, main
from proxy_function_gap.selection import select
from proxy_function_gap.synthetic import SyntheticScenario, make_known_error_case


def base_config():
    return {
        "synthetic_seed": 314159,
        "selection_seed": 271828,
        "proposal_cap": 8,
        "trajectories": 3,
        "top_k": 2,
        "alpha": 5.0,
        "aggregation": "product",
        "zero_weight_policy": "abstain",
        "scenario": "adversarial",
        "abstain_below_q": None,
        "cache": False,
        "strategies": ["random", "bon", "wsc", "greedy", "fp-tilt", "fp-rej"],
    }


def test_adversarial_proxy_error_structure_is_exact():
    case = make_known_error_case(123, "adversarial")
    for signal, bias, noise, proxy in zip(
        case.proxy_signal, case.structured_bias, case.synthetic_proxy_noise, case.proxy_values
    ):
        assert proxy == pytest.approx(signal + bias + noise)
    assert case.structured_bias[0] > 0
    assert case.pool.feasible == (True, True, True, False)


def test_adversarial_bon_finds_decoy_with_known_regret():
    case = make_known_error_case()
    result = select(case.pool, "bon", 16, random.Random(1), proxy=case.proxy)
    assert result.selected_id == "decoy"
    assert regret(case.synthetic_truth, result.selected_index, case.pool.feasible) == pytest.approx(0.9)


def test_independent_truth_has_separate_proxy_signal():
    case = make_known_error_case(123, "independent")
    assert case.proxy_signal != case.synthetic_truth
    assert case.structured_bias == (0.0, 0.0, 0.0, 0.0)
    assert case.verifier_values == (0.80, 0.20, 0.70, 1.00)
    assert regret(case.synthetic_truth, 0, case.pool.feasible) == pytest.approx(0.9)


def test_oracle_proxy_has_zero_noise_and_zero_best_regret():
    case = make_known_error_case(123, "oracle")
    assert case.synthetic_proxy_noise == (0.0, 0.0, 0.0, 0.0)
    assert case.proxy_values == case.synthetic_truth
    result = select(case.pool, "bon", 16, random.Random(1), proxy=case.proxy)
    assert result.selected_id == "best"
    assert regret(case.synthetic_truth, result.selected_index, case.pool.feasible) == 0


def test_oracle_metric_and_infeasible_selection():
    case = make_known_error_case(scenario=SyntheticScenario.ORACLE)
    assert regret(case.synthetic_truth, 1, case.pool.feasible) == 0
    with pytest.raises(ValueError, match="must be feasible"):
        regret(case.synthetic_truth, 3, case.pool.feasible)


def test_runner_is_deterministic_and_hash_tracks_config():
    config = base_config()
    first = execute(config)
    second = execute(config)
    assert first == second
    assert first["method_family"] == "VGSC-inspired fixed-pool"
    assert first["run_id"] == first["config_hash"][:16]
    assert execute(dict(config, aggregation="min"))["config_hash"] != first["config_hash"]
    assert execute(dict(config, cache=True))["config_hash"] != first["config_hash"]


@pytest.mark.parametrize("scenario", ["adversarial", "independent", "oracle"])
def test_runner_supports_all_truth_scenarios(scenario):
    report = execute(dict(base_config(), scenario=scenario))
    assert report["synthetic_case"]["scenario"] == scenario
    for selection in report["selections"]:
        assert len(selection["synthetic_regret"]) == len(selection["selected_indices"])


def test_abstention_has_missing_regret_not_worst_value():
    report = execute(dict(base_config(), abstain_below_q=0.9, strategies=["fp-rej"]))
    selection = report["selections"][0]
    assert selection["abstain"]
    assert selection["coverage"] == 0.0
    assert selection["synthetic_regret"] == []


def test_cli_writes_the_same_json_it_prints(tmp_path, capsys):
    output = tmp_path / "e0.json"
    exit_code = main(["--proposal-cap", "3", "--output", str(output)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == captured.out
    report = json.loads(captured.out)
    assert report["config"]["proposal_cap"] == 3
    assert "not biological evidence" in report["interpretation"]
