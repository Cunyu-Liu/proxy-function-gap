import hashlib
import json

import numpy as np

from run_tie_audit_v1_1 import (
    AFFECTED_SEEDS,
    INPUT_HASHES,
    RESULTS,
    SEEDS,
    V1_ROOT,
    choose,
    load_frozen_inputs,
    order_for_seed,
    run_proposals,
    sha256_file,
    sha256_id,
)


def test_v1_is_lexicographic_not_stable_first():
    order = np.asarray([0, 1])
    scores = np.asarray([7.0, 7.0])
    ids = ["b", "a"]
    assert choose(order, scores, ids, rule="stable_first") == 0
    assert choose(order, scores, ids, rule="sequence_id") == 1


def test_sha256_rule_is_utf8_digest_ascending():
    assert sha256_id("120018B1") == hashlib.sha256(b"120018B1").hexdigest()
    order = np.asarray([1, 0])
    scores = np.asarray([7.0, 7.0])
    ids = ["b", "a"]
    expected = min(range(2), key=lambda i: (hashlib.sha256(ids[i].encode("utf-8")).hexdigest(), ids[i], i))
    assert choose(order, scores, ids, rule="sha256_sequence_id") == expected


def test_nine_exact_ties_and_five_stable_first_disagreements():
    ids, _, _, scores = load_frozen_inputs()
    tie_seeds = []
    affected = []
    for seed in SEEDS:
        order = order_for_seed(len(ids), seed)
        maximum = scores[order[:16]].max()
        if np.sum(scores[order[:16]] == maximum) < 2:
            continue
        tie_seeds.append(seed)
        if choose(order, scores, ids, 16, "stable_first") != choose(order, scores, ids, 16, "sequence_id"):
            affected.append(seed)
    assert len(tie_seeds) == 9
    assert affected == AFFECTED_SEEDS


def test_alternate_does_not_change_any_base_seed_effect():
    ids, _, labels, scores = load_frozen_inputs()
    v1 = run_proposals(labels, scores, ids, SEEDS)
    alternate = run_proposals(labels, scores, ids, SEEDS, rule="sha256_sequence_id")
    assert np.array_equal(v1, alternate)
    original = json.loads((V1_ROOT / "results" / "summary.json").read_text())
    assert v1[:, 2].mean() == original["primary"]["bon_minus_random_regret"]


def test_frozen_inputs_and_v1_1_delivery_are_consistent():
    for name, expected in INPUT_HASHES.items():
        assert sha256_file(V1_ROOT / name) == expected
    original = np.load(V1_ROOT / "results" / "bootstrap_replicates.npy")
    alternate = np.load(RESULTS / "bootstrap_v1_1.npy")
    summary = json.loads((RESULTS / "summary_v1_1.json").read_text())
    assert original.shape == alternate.shape == (10000, 8)
    assert np.array_equal(original[:, 1:5], alternate[:, 1:5])
    assert np.array_equal(original[:, 6], alternate[:, 6])
    assert summary["bootstrap_changed_effect_replicates"] == 78
    assert summary["bootstrap_seed_selection_changes"] == 79
    assert summary["version_selection_by_direction"] is False
    assert summary["statistical_compute"] == "CPU only; no model training or inference"
