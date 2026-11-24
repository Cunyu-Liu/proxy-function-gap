import numpy as np
import pytest
import torch

from proxy_function_gap.paper3_replay import (
    ReplayConfig,
    ReplayEvaluator,
    ReplayLearner,
    fit_ridge_cuda,
    select_posterior_mean_cuda,
)
from proxy_function_gap.bayesian_ridge import BayesianRidge
from proxy_function_gap.regret_acquisition import EvaluationVault, QueryLabelVault, UnlabeledPool


def replay_config(tmp_path, query_budget=2):
    return ReplayConfig(
        development_manifest=tmp_path / "development.json",
        query_manifest=tmp_path / "query.json",
        output_dir=tmp_path / "output",
        query_budget=query_budget,
        noise_variance=0.01,
        memory_fraction=0.05,
    )


def query_case(labels):
    pool = UnlabeledPool(("q0", "q1"), np.eye(2))
    return pool, QueryLabelVault(pool, labels), EvaluationVault(pool, labels)


def test_cuda_closed_form_fit_and_posterior_mean_match_numpy(tmp_path):
    config = replay_config(tmp_path)
    features = np.array([[1.0, 0.2], [0.1, 1.0], [0.8, -0.3]])
    labels = np.array([0.5, -0.2, 0.7])
    model, training = fit_ridge_cuda(features, labels, config)
    expected = BayesianRidge.fit(
        features,
        labels,
        prior_covariance=np.eye(2) * config.prior_variance,
        noise_variance=config.noise_variance,
    )
    assert model.mean == pytest.approx(expected.mean, abs=1e-10)
    assert model.covariance == pytest.approx(expected.covariance, abs=1e-10)
    assert training["training_device"] == "cuda:0"
    assert training["convergence"]["cpu_fallback"] is False
    selected, inference = select_posterior_mean_cuda(model, features, config)
    means = features @ model.mean
    assert selected == (int(np.argsort(-means, kind="stable")[0]),)
    assert inference["inference_device"] == "cuda:0"
    assert inference["posterior_mean_argmax"] is True


def test_learner_never_receives_evaluation_scores_and_full_q_opens_headroom(tmp_path):
    config = replay_config(tmp_path)
    pool, query_vault, evaluation_vault = query_case((0.0, 1.0))
    development_features = np.eye(2)
    development_labels = np.zeros(2)
    results = []
    for strategy in ("zero_label", "random_full_budget", "full_q"):
        learner_result = ReplayLearner(
            development_features, development_labels, pool, query_vault, config
        ).run(strategy)
        assert "regret" not in learner_result
        assert learner_result["evaluation_labels_read"] is False
        assert learner_result["per_round_evaluation_scores_read"] is False
        results.append((learner_result, ReplayEvaluator(evaluation_vault).evaluate(learner_result)))
    zero = results[0][1]["regret"]
    random_full = results[1][1]["regret"]
    full_q = results[2][1]["regret"]
    assert zero == pytest.approx(1.0)
    assert full_q == pytest.approx(0.0)
    assert random_full == full_q
    assert results[1][0]["unique_queries_used"] == results[2][0]["unique_queries_used"] == 2
    assert zero - full_q > config.min_repair_margin


def test_no_full_q_improvement_meets_stop_gate_without_relaxation(tmp_path):
    config = replay_config(tmp_path)
    pool, query_vault, evaluation_vault = query_case((1.0, 0.0))
    regrets = {}
    for strategy in ("zero_label", "full_q"):
        result = ReplayLearner(np.eye(2), np.zeros(2), pool, query_vault, config).run(strategy)
        regrets[strategy] = ReplayEvaluator(evaluation_vault).evaluate(result)["regret"]
    headroom = regrets["zero_label"] - regrets["full_q"]
    assert headroom == pytest.approx(0.0)
    assert not headroom > config.min_repair_margin


def test_repeat_query_is_free_and_ledger_is_per_run(tmp_path):
    config = replay_config(tmp_path)
    pool, vault, _ = query_case((0.0, 1.0))
    ledger_vault = QueryLabelVault(pool, (0.0, 1.0))
    from proxy_function_gap.regret_acquisition import UniqueQueryLedger
    ledger = UniqueQueryLedger(max_labels=2)
    assert ledger_vault.reveal(("q0", "q0"), ledger) == {"q0": 0.0}
    assert ledger_vault.reveal(("q0",), ledger) == {}
    assert ledger.used == 1
    learner = ReplayLearner(np.eye(2), np.zeros(2), pool, vault, config)
    assert learner.run("full_q")["unique_queries_used"] == 2
    assert learner.run("full_q")["unique_queries_used"] == 2


def test_preregistered_config_rejects_top_k_and_unknown_fields(tmp_path):
    with pytest.raises(ValueError, match="top_k=1"):
        ReplayConfig(tmp_path / "d", tmp_path / "q", tmp_path / "o", query_budget=2, top_k=2)
    path = tmp_path / "config.json"
    path.write_text('{"development_manifest":"d","query_manifest":"q","output_dir":"o","query_budget":2,"outer":true}')
    with pytest.raises(ValueError, match="unknown replay config fields"):
        ReplayConfig.from_json(path)
