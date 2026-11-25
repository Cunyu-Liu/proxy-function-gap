import numpy as np

from persist_mrl import choose, conflict_types, kmer_features, metric_values, run_proposals


def test_variable_length_features_match_hand_example():
    assert np.isclose(kmer_features("AAAA")[:256].sum(), 1)
    assert kmer_features("AAAA")[0] == 1
    assert kmer_features("ATGA")[257] == 1
    assert kmer_features("TTTTT")[258] == 5 / 50
    assert kmer_features("ATGA")[256] == 1 / 4


def test_exact_and_hamming1_only_same_length():
    sample = {"train": {"AAAA"}, "dev": set(), "calibration": set()}
    assert conflict_types("AAAA", sample)[0]["type"] == "exact"
    assert conflict_types("AAAT", sample)[0]["type"] == "hamming1"
    assert conflict_types("AAA", sample) == []
    assert conflict_types("AATT", sample) == []


def test_nested_proposals_and_real_mrl_regret():
    labels = np.asarray([0., 1., 2.])
    scores = np.asarray([2., 1., 0.])
    ids = ["a", "b", "c"]
    values = run_proposals(labels, scores, ids, [7])
    order = np.random.default_rng(7).permutation(3)
    oracle = labels[order].max()
    assert values[0, 0] == oracle - labels[order[0]]
    best = sorted(order[:2], key=lambda i: (-scores[i], ids[i], int(i)))[0]
    assert values[0, 1] == oracle - labels[best]
    assert values[0, 2] == values[0, 1] - values[0, 0]


def test_lexicographic_tie_break():
    assert choose(np.asarray([1, 0]), np.asarray([1., 1.]), ["b", "a"]) == 1


def test_metrics_do_not_recalibrate():
    labels = np.asarray([1., 2., 4.])
    scores = labels + 2
    assert metric_values(labels, scores)[2] == 2
