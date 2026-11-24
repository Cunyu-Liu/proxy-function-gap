import numpy as np
import pytest

from proxy_function_gap.bayesian_ridge import BayesianRidge
from proxy_function_gap.regret_acquisition import (
    EvaluationVault,
    QueryLabelVault,
    UniqueQueryLedger,
    UnlabeledPool,
    acquisition_scores,
    expected_regret,
    greedy_query_batch,
    realized_regret,
    regret_acquisition_scores,
    select_posterior_mean,
)


def linear_gaussian_case(seed=17):
    rng = np.random.default_rng(seed)
    true_weights = np.array([0.8, -0.4, 0.6])
    train_features = rng.normal(size=(9, 3))
    train_labels = train_features @ true_weights + rng.normal(scale=0.2, size=9)
    model = BayesianRidge.fit(
        train_features,
        train_labels,
        prior_mean=np.array([0.1, -0.2, 0.3]),
        prior_covariance=np.diag([0.7, 1.1, 0.9]),
        noise_variance=0.04,
    )
    query_features = rng.normal(size=(5, 3))
    evaluation_features = rng.normal(size=(7, 3))
    return model, query_features, evaluation_features


def test_rank_one_update_matches_cholesky_brute_refit():
    model, query_features, _ = linear_gaussian_case()
    feature = query_features[0]
    label = 0.37
    updated = model.update(feature, label)
    brute = BayesianRidge.fit(
        np.vstack((model.train_features, feature)),
        np.append(model.train_labels, label),
        prior_mean=model.prior_mean,
        prior_covariance=model.prior_covariance,
        noise_variance=model.noise_variance,
    )
    assert updated.mean == pytest.approx(brute.mean, abs=1e-10)
    assert updated.covariance == pytest.approx(brute.covariance, abs=1e-10)
    assert updated.refit().mean == pytest.approx(brute.mean, abs=1e-10)
    assert updated.observation_count == model.observation_count + 1


def test_deployment_selector_is_posterior_mean_argmax_for_top_1_and_top_k():
    model, _, evaluation_features = linear_gaussian_case()
    means, _ = model.predict(evaluation_features)
    expected_order = np.argsort(-means, kind="stable")
    assert select_posterior_mean(model, evaluation_features) == (int(expected_order[0]),)
    assert select_posterior_mean(model, evaluation_features, top_k=3) == tuple(
        int(index) for index in expected_order[:3]
    )


def test_top_1_and_top_k_realized_regret_use_matching_oracle_budget():
    labels = np.array([3.0, 2.0, 1.0, 0.0])
    assert realized_regret(labels, (2,)) == pytest.approx(2.0)
    assert realized_regret(labels, (0,)) == pytest.approx(0.0)
    assert realized_regret(labels, (0, 3)) == pytest.approx(1.0)
    assert realized_regret(labels, (1, 2)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unique"):
        realized_regret(labels, (0, 0))


def test_expected_regret_supports_top_k_and_is_reproducible():
    model, _, evaluation_features = linear_gaussian_case()
    for top_k in (1, 2):
        first = expected_regret(
            model, evaluation_features, top_k, posterior_samples=64, rng=np.random.default_rng(9)
        )
        second = expected_regret(
            model, evaluation_features, top_k, posterior_samples=64, rng=np.random.default_rng(9)
        )
        assert first == pytest.approx(second)
        assert first >= 0.0


def test_regret_acquisition_is_deterministic_without_real_labels():
    model, query_features, evaluation_features = linear_gaussian_case()
    arguments = dict(
        top_k=2,
        posterior_samples=32,
        label_samples=8,
    )
    first = regret_acquisition_scores(
        model, query_features, evaluation_features, rng=np.random.default_rng(4), **arguments
    )
    second = regret_acquisition_scores(
        model, query_features, evaluation_features, rng=np.random.default_rng(4), **arguments
    )
    assert first == pytest.approx(second)
    assert np.isfinite(first).all()


def test_random_and_maximum_variance_baselines():
    model, query_features, evaluation_features = linear_gaussian_case()
    _, expected_variance = model.predict(query_features)
    variance = acquisition_scores(
        model, "max_variance", query_features, evaluation_features, rng=np.random.default_rng(3)
    )
    assert variance == pytest.approx(expected_variance)
    first = acquisition_scores(
        model, "random", query_features, evaluation_features, rng=np.random.default_rng(3)
    )
    second = acquisition_scores(
        model, "random", query_features, evaluation_features, rng=np.random.default_rng(3)
    )
    assert first == pytest.approx(second)
    assert np.all((0.0 <= first) & (first < 1.0))


def test_unique_query_billing_is_atomic_and_repeat_reveal_is_free():
    pool = UnlabeledPool(("q0", "q1", "q2"), np.eye(3))
    vault = QueryLabelVault(pool, (0.1, 0.2, 0.3))
    ledger = UniqueQueryLedger(max_labels=2)
    assert vault.reveal(("q0", "q0"), ledger) == {"q0": pytest.approx(0.1)}
    assert vault.reveal(("q0",), ledger) == {}
    assert ledger.used == 1
    assert vault.reveal(("q1",), ledger) == {"q1": pytest.approx(0.2)}
    with pytest.raises(ValueError, match="budget"):
        vault.reveal(("q2",), ledger)
    assert ledger.used == 2
    with pytest.raises(KeyError, match="unknown"):
        vault.reveal(("evaluation-only",), ledger)
    assert ledger.used == 2


def test_evaluation_features_are_unlabeled_and_vault_returns_only_regret():
    pool = UnlabeledPool(("e0", "e1", "e2"), np.eye(3))
    vault = EvaluationVault(pool, (1.0, 3.0, 2.0))
    assert not hasattr(pool, "labels")
    assert not hasattr(vault, "reveal")
    assert vault.regret(("e1",)) == pytest.approx(0.0)
    assert vault.regret(("e0",)) == pytest.approx(2.0)
    assert vault.regret(("e0", "e2")) == pytest.approx(1.0)


def test_batch_queries_are_unique_and_respect_near_duplicate_interface():
    model, query_features, evaluation_features = linear_gaussian_case()
    query_pool = UnlabeledPool(("q0", "q0-near", "q1", "q2", "q3"), query_features)
    evaluation_pool = UnlabeledPool(
        tuple(f"e{index}" for index in range(len(evaluation_features))), evaluation_features
    )

    def near_duplicate(left, right):
        return {left, right} == {"q0", "q0-near"}

    batch = greedy_query_batch(
        model,
        query_pool,
        evaluation_pool,
        "max_variance",
        batch_size=5,
        near_duplicate=near_duplicate,
    )
    assert len(batch.selected_ids) == len(set(batch.selected_ids)) == 4
    assert not {"q0", "q0-near"}.issubset(batch.selected_ids)


@pytest.mark.parametrize("top_k", [1, 2])
def test_posterior_and_label_sample_doubling_is_stable(top_k):
    model, query_features, evaluation_features = linear_gaussian_case(seed=29)
    coarse = regret_acquisition_scores(
        model,
        query_features,
        evaluation_features,
        top_k=top_k,
        posterior_samples=32,
        label_samples=8,
        rng=np.random.default_rng(123),
    )
    doubled = regret_acquisition_scores(
        model,
        query_features,
        evaluation_features,
        top_k=top_k,
        posterior_samples=64,
        label_samples=16,
        rng=np.random.default_rng(123),
    )
    assert doubled == pytest.approx(coarse, abs=0.08)
    assert np.argsort(-doubled, kind="stable")[0] == np.argsort(-coarse, kind="stable")[0]
