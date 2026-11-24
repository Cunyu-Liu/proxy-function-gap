from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .bayesian_ridge import BayesianRidge


class UnlabeledPool:
    def __init__(self, sequence_ids: Sequence[str], features):
        self.sequence_ids = tuple(sequence_ids)
        if not self.sequence_ids or any(not isinstance(item, str) or not item for item in self.sequence_ids):
            raise ValueError("sequence_ids must be non-empty strings")
        if len(set(self.sequence_ids)) != len(self.sequence_ids):
            raise ValueError("sequence_ids must be unique")
        features = np.array(features, dtype=np.float64, copy=True)
        if features.ndim != 2 or features.shape != (len(self.sequence_ids), features.shape[1]):
            raise ValueError("features must be a 2D array matching sequence_ids")
        if features.shape[1] == 0 or not np.isfinite(features).all():
            raise ValueError("features must be finite and have at least one column")
        features.setflags(write=False)
        self.features = features

    def __len__(self):
        return len(self.sequence_ids)

    def index(self, sequence_id):
        try:
            return self.sequence_ids.index(sequence_id)
        except ValueError as error:
            raise KeyError(sequence_id) from error


class UniqueQueryLedger:
    def __init__(self, max_labels=None):
        if max_labels is not None and (
            isinstance(max_labels, bool) or not isinstance(max_labels, int) or max_labels <= 0
        ):
            raise ValueError("max_labels must be a positive integer or None")
        self.max_labels = max_labels
        self._query_ids = set()

    @property
    def used(self):
        return len(self._query_ids)

    def __contains__(self, sequence_id):
        return sequence_id in self._query_ids

    def record(self, sequence_ids):
        new_ids = tuple(dict.fromkeys(sequence_ids))
        new_ids = tuple(sequence_id for sequence_id in new_ids if sequence_id not in self._query_ids)
        if self.max_labels is not None and self.used + len(new_ids) > self.max_labels:
            raise ValueError("unique query budget exceeded")
        self._query_ids.update(new_ids)
        return new_ids


class QueryLabelVault:
    def __init__(self, pool: UnlabeledPool, labels):
        labels = np.asarray(labels, dtype=np.float64)
        if labels.shape != (len(pool),) or not np.isfinite(labels).all():
            raise ValueError("query labels must be finite and match the pool")
        self._pool = pool
        self._labels = dict(zip(pool.sequence_ids, labels))

    def reveal(self, sequence_ids, ledger: UniqueQueryLedger):
        requested = tuple(sequence_ids)
        unknown = set(requested) - set(self._labels)
        if unknown:
            raise KeyError(f"unknown query ids: {sorted(unknown)}")
        new_ids = ledger.record(requested)
        return {sequence_id: float(self._labels[sequence_id]) for sequence_id in new_ids}


class EvaluationVault:
    def __init__(self, pool: UnlabeledPool, labels):
        labels = np.asarray(labels, dtype=np.float64)
        if labels.shape != (len(pool),) or not np.isfinite(labels).all():
            raise ValueError("evaluation labels must be finite and match the pool")
        self._pool = pool
        self._labels = np.array(labels, dtype=np.float64, copy=True)

    def regret(self, selected_ids):
        indices = tuple(self._pool.index(sequence_id) for sequence_id in selected_ids)
        return realized_regret(self._labels, indices)


@dataclass(frozen=True)
class AcquisitionBatch:
    selected_ids: tuple
    selected_indices: tuple
    scores: tuple


def select_posterior_mean(model: BayesianRidge, evaluation_features, top_k=1):
    features = np.asarray(evaluation_features, dtype=np.float64)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("evaluation_features must be a non-empty 2D array")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= len(features):
        raise ValueError("top_k must be between one and the evaluation pool size")
    means, _ = model.predict(features)
    order = np.argsort(-means, kind="stable")
    return tuple(int(index) for index in order[:top_k])


def realized_regret(labels, selected_indices):
    labels = np.asarray(labels, dtype=np.float64)
    selected = tuple(int(index) for index in selected_indices)
    if labels.ndim != 1 or len(labels) == 0 or not np.isfinite(labels).all():
        raise ValueError("labels must be a finite non-empty vector")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected_indices must be non-empty and unique")
    if any(index < 0 or index >= len(labels) for index in selected):
        raise ValueError("selected index is outside the evaluation pool")
    top_k = len(selected)
    best = np.argsort(-labels, kind="stable")[:top_k]
    return float(np.mean(labels[best]) - np.mean(labels[list(selected)]))


def expected_regret(model, evaluation_features, top_k=1, posterior_samples=64, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    selected = select_posterior_mean(model, evaluation_features, top_k)
    weights = model.sample_weights(posterior_samples, rng)
    outcomes = weights @ np.asarray(evaluation_features, dtype=np.float64).T
    oracle = np.argsort(-outcomes, axis=1, kind="stable")[:, :top_k]
    oracle_values = np.take_along_axis(outcomes, oracle, axis=1).mean(axis=1)
    regrets = oracle_values - outcomes[:, selected].mean(axis=1)
    return float(regrets.mean())


def regret_acquisition_scores(
    model,
    query_features,
    evaluation_features,
    top_k=1,
    posterior_samples=64,
    label_samples=16,
    rng=None,
):
    if isinstance(label_samples, bool) or not isinstance(label_samples, int) or label_samples <= 0:
        raise ValueError("label_samples must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    query_features = np.asarray(query_features, dtype=np.float64)
    if query_features.ndim != 2 or len(query_features) == 0:
        raise ValueError("query_features must be a non-empty 2D array")
    current_regret = expected_regret(model, evaluation_features, top_k, posterior_samples, rng)
    query_means, query_variances = model.predict(query_features)
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=len(query_features))
    scores = np.empty(len(query_features), dtype=np.float64)
    for query_index, feature in enumerate(query_features):
        local_rng = np.random.default_rng(int(seeds[query_index]))
        hypothetical_labels = query_means[query_index] + np.sqrt(query_variances[query_index]) * (
            local_rng.standard_normal(label_samples)
        )
        updated_regrets = []
        for label in hypothetical_labels:
            updated = model.update(feature, label)
            updated_regrets.append(
                expected_regret(updated, evaluation_features, top_k, posterior_samples, local_rng)
            )
        scores[query_index] = current_regret - np.mean(updated_regrets)
    return scores


def acquisition_scores(
    model,
    strategy,
    query_features,
    evaluation_features,
    top_k=1,
    posterior_samples=64,
    label_samples=16,
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng
    query_features = np.asarray(query_features, dtype=np.float64)
    if strategy == "random":
        if query_features.ndim != 2 or len(query_features) == 0:
            raise ValueError("query_features must be a non-empty 2D array")
        return rng.random(len(query_features))
    if strategy == "max_variance":
        _, variance = model.predict(query_features)
        return variance
    if strategy == "regret":
        return regret_acquisition_scores(
            model,
            query_features,
            evaluation_features,
            top_k,
            posterior_samples,
            label_samples,
            rng,
        )
    raise ValueError("strategy must be regret, random or max_variance")


def greedy_query_batch(
    model,
    query_pool: UnlabeledPool,
    evaluation_pool: UnlabeledPool,
    strategy,
    batch_size,
    seed=0,
    top_k=1,
    posterior_samples=64,
    label_samples=16,
    near_duplicate: Callable[[str, str], bool] = None,
    excluded_ids=(),
):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    excluded = set(excluded_ids)
    available = [index for index, item in enumerate(query_pool.sequence_ids) if item not in excluded]
    rng = np.random.default_rng(seed)
    selected_ids = []
    selected_indices = []
    selected_scores = []
    working_model = model
    while available and len(selected_ids) < batch_size:
        features = query_pool.features[available]
        scores = acquisition_scores(
            working_model,
            strategy,
            features,
            evaluation_pool.features,
            top_k,
            posterior_samples,
            label_samples,
            rng,
        )
        best_position = max(range(len(available)), key=lambda position: scores[position])
        best_index = available.pop(best_position)
        best_id = query_pool.sequence_ids[best_index]
        selected_ids.append(best_id)
        selected_indices.append(best_index)
        selected_scores.append(float(scores[best_position]))
        if near_duplicate is not None:
            available = [
                index
                for index in available
                if not near_duplicate(best_id, query_pool.sequence_ids[index])
                and not near_duplicate(query_pool.sequence_ids[index], best_id)
            ]
        if strategy == "regret" and available:
            fantasy_label = working_model.predict(query_pool.features[[best_index]])[0][0]
            working_model = working_model.update(query_pool.features[best_index], fantasy_label)
    return AcquisitionBatch(tuple(selected_ids), tuple(selected_indices), tuple(selected_scores))
