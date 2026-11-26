import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from anchor import (
    HalfIndex, best_average_ranks, bootstrap_clusters, cap_regret,
    fusion_scores, holm_adjust, isolate_h1, make_proposals,
    oracle_decomposition, random_regret, spearman,
)
from evaluate import top_tail_metrics
from prepare import read_target_keys


def test_half_index_exact_h1_and_h2():
    index = HalfIndex()
    original = "A" * 50
    index.add(original)
    assert index.match(original) == (True, False)
    assert index.match("A" * 49 + "C") == (False, True)
    assert index.match("A" * 48 + "CC") == (False, False)


def test_h1_isolation_is_deterministic_and_fail_closed():
    values = ["A" * 50, "A" * 49 + "C", "C" * 50, "G" * 50]
    first = isolate_h1(values, 3)
    second = isolate_h1(list(reversed(values)), 3)
    assert first == second
    index = HalfIndex()
    for value in first:
        assert index.match(value) == (False, False)
        index.add(value)
    with pytest.raises(ValueError, match="insufficient"):
        isolate_h1(values, 4)


def test_average_ranks_and_fixed_fusion_handle_ties():
    ranks = best_average_ranks(np.asarray([3.0, 3.0, 1.0, 2.0]))
    assert np.allclose(ranks, [5.0 / 6.0, 5.0 / 6.0, 0.0, 1.0 / 3.0])
    cnn = np.asarray([3.0, 2.0, 1.0, 0.0])
    hepg2 = np.asarray([0.0, 1.0, 2.0, 3.0])
    pools = np.asarray([[0, 1, 2, 3]])
    fused, cnn_rank, hepg2_rank = fusion_scores(cnn, hepg2, pools)
    assert np.allclose(cnn_rank + hepg2_rank, 1.0)
    assert np.allclose(fused, 0.5)
    assert np.allclose(spearman(cnn, hepg2), -1.0)


def test_nested_proposals_and_repeat_linkage_shape():
    first = make_proposals(2, 64, [101, 202], 16)
    second = make_proposals(2, 64, [101, 202], 16)
    assert np.array_equal(first, second)
    assert first.shape == (2, 2, 16)
    for cap in (1, 2, 4, 8, 16):
        assert np.array_equal(first[:, :, :cap], second[:, :, :cap])


def test_selection_regret_tie_and_random_prefix():
    pools = np.asarray([[0, 1, 2, 3]])
    labels = np.asarray([1.0, 4.0, 3.0, 2.0])
    scores = np.asarray([5.0, 5.0, 0.0, 0.0])
    proposals = np.asarray([[[0, 1, 2, 3]]])
    regret, selected = cap_regret(scores, labels, pools, proposals, 4)
    assert selected.item() == 0
    assert regret.item() == 3.0
    random, random_selected = random_regret(labels, pools, proposals, 4)
    assert random_selected.item() == 0
    assert random.item() == 3.0


def test_bootstrap_and_holm():
    effect = bootstrap_clusters(np.asarray([-2.0, -1.0, -3.0, -0.5]), 17, 1000)
    assert effect["effect"] < 0 and effect["ci_high"] < 0
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
    assert np.all((adjusted >= 0) & (adjusted <= 1))


def test_oracle_opportunity_decomposition():
    pools = np.asarray([[0, 1, 2, 3]])
    labels = np.asarray([1.0, 4.0, 3.0, 2.0])
    scores = np.asarray([4.0, 0.0, 3.0, 2.0])
    proposals = np.asarray([[[0, 2, 0, 3]]])
    regret, selected = cap_regret(scores, labels, pools, proposals, 4)
    opportunity, headroom = oracle_decomposition(labels, pools, proposals, selected, 4)
    assert np.allclose(opportunity + headroom, regret)
    assert opportunity.item() == 2.0
    assert headroom.item() == 1.0


def test_preparation_target_reader_has_no_rl_access():
    tree = ast.parse(inspect.getsource(read_target_keys))
    subscripts = [node for node in ast.walk(tree) if isinstance(node, ast.Subscript)]
    assert not any(isinstance(node.value, ast.Name) and node.value.id == "row" and isinstance(node.slice, ast.Constant) and node.slice.value == "rl" for node in subscripts)


def test_top_tail_flip_direction():
    pools = np.asarray([[0, 1, 2, 3] + list(range(4, 64))])
    cnn = np.arange(64, dtype=float)[::-1]
    hepg2 = cnn.copy()
    hepg2[0], hepg2[1] = hepg2[1], hepg2[0]
    labels = cnn.copy()
    result = top_tail_metrics(cnn, hepg2, labels, pools)
    assert result["top1_flip_pools"] == 1
    assert result["cnn_top16_predictor_order_flipped_pairs"] == 1
    assert result["flipped_pairs_tcell_favor_cnn"] == 1
    assert result["flipped_pairs_tcell_favor_hepg2"] == 0


def test_frozen_config_contract():
    config = json.loads((Path(__file__).parents[1] / "configs" / "anchor_frozen.json").read_text())
    assert config["selection"]["caps"] == [1, 2, 4, 8, 16]
    assert config["pools"]["count"] == 16 and config["pools"]["size"] == 64
    assert config["methods"]["rank_fusion_50_50"].startswith("Secondary preregistered")
    assert config["primary_success"]["combined_repeat_linked_pool_bootstrap_replicates"] == 10000
    assert config["learner"]["target_tcell_labels_forbidden"] is True
    assert config["learner"]["residual_model"] is False
    assert "neither sequence-independent validation" in config["evidence_boundary"]
    assert config["no_push"] is True
