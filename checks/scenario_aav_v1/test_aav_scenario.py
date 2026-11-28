import importlib.util
from pathlib import Path

import numpy as np
import torch

MODULE_PATH = Path(__file__).with_name("aav_scenario.py")
SPEC = importlib.util.spec_from_file_location("aav_scenario", MODULE_PATH)
aav = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aav)


def test_neighbor_keys_cover_exact_and_hamming_one():
    seq = "ACDEFGH"
    keys = aav.neighbor_keys(seq)
    assert keys[0] == seq
    assert len(keys) == 8
    assert aav.hamming_leq_one(seq, "ACDEFGA")
    assert not aav.hamming_leq_one(seq, "ACDEFAA")


def test_cross_isolation_rejects_exact_and_near():
    aav.assert_cross_isolation(["ACDEFGH"], ["IKLMNPQ"])
    try:
        aav.assert_cross_isolation(["ACDEFGH"], ["ACDEFGA"])
    except AssertionError:
        pass
    else:
        raise AssertionError("Hamming-one leakage was not rejected")


def test_components_keep_near_variants_together():
    sequences = ["ACDEFGH", "ACDEFGA", "ACDEFGT", "IKLMNPQ"]
    components = aav.hamming_components(sequences)
    assert sorted(len(component) for component in components) == [1, 3]


def test_equal_budget_selection_and_real_regret():
    y = np.array([0.0, 2.0, 1.0, 4.0, 3.0, 1.0, 0.0, 2.0])
    scores = np.array([0.0, 0.1, 0.9, 0.2, 0.8, 0.7, 0.1, 0.0])
    candidates = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    random_offsets = np.array([0, 3])
    result = aav.select_from_pools(y, scores, candidates, random_offsets)
    np.testing.assert_array_equal(result["bon_local"], [2, 0])
    np.testing.assert_array_equal(result["bon_y"], [1.0, 3.0])
    np.testing.assert_array_equal(result["random_y"], [0.0, 2.0])
    np.testing.assert_array_equal(result["oracle_y"], [4.0, 3.0])
    np.testing.assert_array_equal(result["bon_regret"], [3.0, 0.0])
    np.testing.assert_array_equal(result["random_regret"], [4.0, 1.0])


def test_cap_one_has_zero_paired_delta():
    y = np.array([1.0, 4.0, 2.0])
    scores = np.array([9.0, -1.0, 3.0])
    result = aav.select_from_pools(y, scores, np.arange(3).reshape(3, 1), np.zeros(3, dtype=int))
    np.testing.assert_array_equal(result["bon_regret"], result["random_regret"])


def test_encoding_and_cuda_or_cpu_forward_shape():
    tokens = aav.encode(["ACDEFGH", "IKLMNPQ"])
    assert tokens.shape == (2, 7)
    assert tokens.dtype == np.int64
    model = aav.SequenceCNN()
    output = model(torch.from_numpy(tokens))
    assert output.shape == (2,)
    assert aav.variant_id("ACDEFGH") != aav.variant_id("ACDEFGA")
