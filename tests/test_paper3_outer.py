import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from proxy_function_gap.paper3_headroom import kmer4
from proxy_function_gap.paper3_outer_formal import require_release
from proxy_function_gap.outer_learner import OuterPool
from proxy_function_gap.paper3_outer import (
    OuterProtocolConfig, cluster_bootstrap_mean, learner_curve,
    require_development_isolation, score_frozen_curve, split_group_indices, subset_pool,
)
from proxy_function_gap.regret_acquisition import QueryLabelVault, UnlabeledPool


def config(tmp_path):
    value = json.loads(Path("configs/paper3_outer_protocol_v1.json").read_text())
    value["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return OuterProtocolConfig.from_json(path)


@dataclass(frozen=True)
class MockPool:
    pool_id: str
    row_ids: tuple
    sequence_ids: tuple
    utr50: tuple
    group_ids: tuple
    measured: np.ndarray
    proxy_mean: np.ndarray


def mock_pool():
    sequences, groups, row_ids = [], [], []
    for group in range(16):
        bits = f"{group:04b}"
        sequence = "".join(("A" if bit == "1" else "C") * 10 for bit in bits) + "C" * 10
        for replicate in range(4):
            sequences.append(sequence)
            groups.append(f"group_{group}")
            row_ids.append(f"row_{group}_{replicate}")
    return MockPool(
        pool_id="calibration_test", row_ids=tuple(row_ids), sequence_ids=tuple(row_ids),
        utr50=tuple(sequences), group_ids=tuple(groups), measured=np.arange(64, dtype=float),
        proxy_mean=np.zeros(64),
    )


def test_config_rejects_outer_execution_and_retuning(tmp_path):
    protocol = config(tmp_path)
    value = protocol.to_dict()
    value["mode"] = "outer"
    path = tmp_path / "outer.json"
    path.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="only calibration"):
        OuterProtocolConfig.from_json(path)
    value = protocol.to_dict()
    value["noise_variance"] = 0.04
    path = tmp_path / "noise.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="hyperparameters"):
        OuterProtocolConfig.from_json(path)


def test_config_identity_ignores_output_path(tmp_path):
    protocol = config(tmp_path)
    value = protocol.to_dict()
    value["output_dir"] = str(tmp_path / "another_output")
    path = tmp_path / "another.json"
    path.write_text(json.dumps(value))
    assert OuterProtocolConfig.from_json(path).sha256() == protocol.sha256()


def test_unlabeled_group_split_is_balanced_isolated_and_label_invariant():
    pool = mock_pool()
    query, evaluation = split_group_indices(pool)
    assert len(query) == len(evaluation) == 32
    assert not {pool.group_ids[index] for index in query} & {pool.group_ids[index] for index in evaluation}
    assert split_group_indices(replace(pool, measured=pool.measured[::-1], proxy_mean=np.arange(64))) == (query, evaluation)


def test_qe_hamming_one_conflict_fails_closed():
    pool = mock_pool()
    query, evaluation = split_group_indices(pool)
    sequences = list(pool.utr50)
    changed = ("A" if sequences[evaluation[0]][0] == "C" else "C") + sequences[evaluation[0]][1:]
    sequences[query[0]] = changed
    contaminated = replace(pool, utr50=tuple(sequences))
    with pytest.raises(PermissionError, match="Hamming-1"):
        split_group_indices(contaminated)


def test_learner_uses_query_vault_and_evaluator_scores_only_frozen_ids(tmp_path):
    protocol = config(tmp_path)
    runtime = protocol.runtime(tmp_path)
    rng = np.random.default_rng(7)
    query_x = rng.normal(size=(32, 2))
    evaluation_x = rng.normal(size=(32, 2))
    query_ids = tuple(f"q_{index}" for index in range(32))
    evaluation_ids = tuple(f"e_{index}" for index in range(32))
    query = UnlabeledPool(query_ids, query_x)
    evaluation = UnlabeledPool(evaluation_ids, evaluation_x)
    labels = query_x[:, 0] - query_x[:, 1]
    vault = QueryLabelVault(query, labels)
    with pytest.raises(KeyError):
        vault.reveal(("e_0",), __import__("proxy_function_gap.regret_acquisition", fromlist=["UniqueQueryLedger"]).UniqueQueryLedger())
    sequences = tuple("".join(np.random.default_rng(1000 + index).choice(list("ACGT"), 50)) for index in range(32))
    curve = learner_curve(np.eye(2), np.zeros(2), query, vault, evaluation,
                          np.zeros(32), sequences, "random", protocol, runtime, 19)
    assert not curve["evaluation_labels_read"]
    assert len(set(curve["query_order"])) == 32
    assert [item["unique_queries_used"] for item in curve["records"]] == [0, 4, 8, 16, 32]
    evaluation_labels = evaluation_x[:, 0] + evaluation_x[:, 1]
    scored = score_frozen_curve(curve, evaluation_ids, evaluation_labels)
    expected = max(evaluation_labels) - evaluation_labels[0]
    assert scored["records"][0]["regret"] == pytest.approx(expected)
    assert scored["records"][-1]["training_device"] == "cuda:0"


def test_evaluator_rejects_query_id_in_evaluation_vault(tmp_path):
    curve = {"evaluation_labels_read": False, "evaluation_scores_returned_to_learner": False,
             "query_order": ["q0"], "records": []}
    with pytest.raises(PermissionError, match="query IDs"):
        score_frozen_curve(curve, ("q0",), np.asarray([1.0]))


def test_cluster_bootstrap_resamples_whole_clusters_not_rows():
    result = cluster_bootstrap_mean(np.asarray([-1.0, 1.0]))
    assert result["cluster_count"] == 2 and result["mean"] == 0
    assert result["ci_lower"] == -1 and result["ci_upper"] == 1
    assert not result["success_upper_below_zero"]
    assert result == cluster_bootstrap_mean(np.asarray([-1.0, 1.0]))
    negative = cluster_bootstrap_mean(np.asarray([-2.0]))
    assert negative["ci_upper"] == -2 and negative["success_upper_below_zero"]


def test_development_hamming_one_is_rejected():
    pool = mock_pool()
    query = replace(pool, group_ids=tuple(f"q_{item}" for item in pool.group_ids))
    development = replace(pool, utr50=(pool.utr50[0],), group_ids=("dev",))
    with pytest.raises(PermissionError):
        require_development_isolation(development, query, query)


def test_formal_outer_is_hard_locked_after_paper2_p0_failure(tmp_path):
    protocol = config(tmp_path)
    path = tmp_path / "release.json"
    path.write_text(json.dumps({"paper2_p0_pass": True, "paper3_outer_authorized": True,
                                "outer_labels_may_be_scored": True,
                                "protocol_config_sha256": protocol.sha256()}))
    with pytest.raises(PermissionError, match="hard locked"):
        require_release(path, protocol)


def test_formal_outer_release_hash_validation_is_reachable_only_after_unlock(tmp_path):
    protocol = config(tmp_path)
    import proxy_function_gap.paper3_outer_formal as formal
    formal.PAPER3_OUTER_HARD_LOCK = False
    try:
        path = tmp_path / "release.json"
        path.write_text(json.dumps({"paper2_p0_pass": True, "paper3_outer_authorized": True,
                                    "outer_labels_may_be_scored": True, "protocol_config_sha256": "wrong"}))
        with pytest.raises(PermissionError, match="exact Paper3 protocol"):
            require_release(path, protocol)
        path.write_text(json.dumps({"paper2_p0_pass": True, "paper3_outer_authorized": True,
                                    "outer_labels_may_be_scored": True,
                                    "protocol_config_sha256": protocol.sha256()}))
        assert require_release(path, protocol)["paper2_p0_pass"] is True
    finally:
        formal.PAPER3_OUTER_HARD_LOCK = True


def test_outer_unlabeled_subset_stays_label_free():
    base = mock_pool()
    pool = OuterPool(pool_id="outer_designed_test", pool_role="outer_designed", synthetic=False,
                     row_ids=base.row_ids, sequence_ids=base.sequence_ids, utr50=base.utr50,
                     group_ids=base.group_ids, support=np.zeros(64), proposal_probs=np.full(64, 1 / 64),
                     measured=None, proxy_mean=np.zeros(64), proxy_std=np.zeros(64), proxy_range=np.zeros(64))
    query_indices, evaluation_indices = split_group_indices(pool)
    query = subset_pool(pool, query_indices, "query")
    evaluation = subset_pool(pool, evaluation_indices, "evaluation")
    assert query.measured is None and evaluation.measured is None
    assert len(query) == len(evaluation) == 32
