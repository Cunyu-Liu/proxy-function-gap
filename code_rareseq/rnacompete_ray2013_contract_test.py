#!/usr/bin/env python3
import gzip
import os
import tempfile

import numpy as np
import pandas as pd

from rnacompete_ray2013_freeze import (aggregate_sequence_labels, assign_balanced,
                                       frequency_tiers, load_experiment_labels,
                                       parse_clstr, split_roles)


def main():
    probes = pd.DataFrame({
        "sequence_id": ["a", "a", "a", "b", "c"],
        "probe_set": ["SetA", "SetA", "SetA", "SetB", "SetB"],
        "fold": [0, 0, 0, 1, 2],
    })
    aggregated = aggregate_sequence_labels(probes, [1.0, 3.0, np.nan, np.nan, 2.0]).set_index("sequence_id")
    assert aggregated.loc["a", "label_median"] == 2.0
    assert aggregated.loc["a", "n_finite"] == 2
    assert aggregated.loc["a", "n_nan"] == 1
    assert aggregated.loc["a", "n_probe_rows"] == 3
    assert aggregated.loc["a", "duplicate_label_range"] == 2.0
    assert np.isnan(aggregated.loc["b", "label_median"])
    assert aggregated.loc["b", "n_finite"] == 0
    assert aggregated.loc["b", "n_nan"] == 1
    assert aggregated.loc["c", "label_median"] == 2.0

    expected = {
        0: {"train": [2, 3, 4], "validation": [1], "test": [0]},
        4: {"train": [1, 2, 3], "validation": [0], "test": [4]},
    }
    for fold in range(5):
        roles = split_roles(fold)
        assert len(roles["train"]) == 3
        assert len(set(roles["train"] + roles["validation"] + roles["test"])) == 5
        if fold in expected:
            assert roles == expected[fold]

    weights = {"a": 8, "b": 7, "c": 6, "d": 5, "e": 4, "f": 3}
    first, loads = assign_balanced(weights)
    second, _ = assign_balanced(weights)
    assert first == second and sum(loads) == sum(weights.values())
    assert set(first.values()) == set(range(5))

    experiments = pd.DataFrame({
        "Motif_ID": ["e1", "e2", "e3", "e4", "e5"],
        "sequence_track_included": [True] * 5,
        "n_finite_unique": [500, 400, 400, 200, 100],
    })
    tiers = frequency_tiers(experiments)
    assert [tiers["e" + str(i)] for i in range(1, 6)] == ["head", "head", "mid", "mid", "tail"]

    with tempfile.TemporaryDirectory() as root:
        clstr = os.path.join(root, "clusters.clstr")
        with open(clstr, "w") as handle:
            handle.write(">Cluster 0\n0\t35nt, >aaa... *\n1\t35nt, >bbb... at 95%\n>Cluster 1\n0\t38nt, >ccc... *\n")
        assert parse_clstr(clstr) == {"aaa": 0, "bbb": 0, "ccc": 1}

        frozen = os.path.join(root, "frozen")
        source = os.path.join(root, "source")
        os.makedirs(frozen)
        os.makedirs(source)
        probes.to_csv(os.path.join(frozen, "probes.tsv.gz"), sep="\t", index=False)
        with gzip.open(os.path.join(source, "norm_data.txt.gz"), "wt") as handle:
            handle.write("Probe_ID\te1\np1\t1\np2\t3\np3\t NaN\np4\t NaN\np5\t2\n")
        probe_ids = pd.DataFrame({"probe_id": ["p1", "p2", "p3", "p4", "p5"]})
        merged = pd.concat([probes, probe_ids], axis=1)
        merged.to_csv(os.path.join(frozen, "probes.tsv.gz"), sep="\t", index=False)
        loaded = load_experiment_labels(source, frozen, "e1").set_index("sequence_id")
        assert loaded.loc["a", "label_median"] == 2.0
        assert np.isnan(loaded.loc["b", "label_median"])

    print("RNACOMPETE_CONTRACT_TEST_OK: NaN, median, strict3train, tiers, clusters, loader")


if __name__ == "__main__":
    main()
