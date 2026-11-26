#!/usr/bin/env python3
import json
import os
import tempfile

import numpy as np
import pandas as pd

from rnacompete_ray2013_baselines import (ARCHES, K_VALUES, BenchmarkData, aligned_prediction,
                                          audit_selection_boundaries, better_candidate,
                                          knn_predict, label_placeholders, macro_metrics, masked_mse, metrics_table,
                                          regression_metrics, replay_run, save_test_products, split_roles)
from rnacompete_ray2013_rnafm_embed import validate_cache


def synthetic_data(root):
    frozen = os.path.join(root, "frozen")
    embeddings = os.path.join(root, "embeddings")
    os.makedirs(frozen)
    os.makedirs(embeddings)
    n = 100
    tasks = [f"e{i:03d}" for i in range(244)]
    sequences = pd.DataFrame({"sequence_id": [f"s{i:03d}" for i in range(n)],
                              "rna_seq": ["ACGU"] * n,
                              "probe_set": np.where(np.arange(n) % 2, "SetA", "SetB"),
                              "fold": np.arange(n) % 5, "cluster_id": np.arange(n)})
    experiments = pd.DataFrame({"Motif_ID": tasks, "species_gene": [f"g{i}" for i in range(244)],
                                "rbp_fold": np.arange(244) % 5, "sequence_track_included": [True] * 244,
                                "frequency_tier": np.array(["head", "mid", "tail"])[np.arange(244) % 3],
                                "n_finite_unique": [90] * 244})
    sequences.to_csv(os.path.join(frozen, "sequence_folds.tsv.gz"), sep="\t", index=False)
    experiments.to_csv(os.path.join(frozen, "experiments.tsv"), sep="\t", index=False)
    with open(os.path.join(frozen, "split_manifest.json"), "w") as handle:
        json.dump({"version": "rnacompete-ray2013-v1.0-id90-strict3train", "status": "accepted"}, handle)
    rng = np.random.default_rng(4)
    emb = rng.normal(size=(n, 640)).astype(np.float16)
    labels = rng.normal(size=(n, 244)).astype(np.float32)
    labels[::17, 2] = np.nan
    np.save(os.path.join(embeddings, "rnafm.mean_masked.fp16.npy"), emb)
    with open(os.path.join(embeddings, "rnafm.ids"), "w") as handle:
        handle.write("\n".join(sequences.sequence_id) + "\n")
    cache = os.path.join(root, "labels.npz")
    np.savez(cache, labels=labels, sequence_ids=sequences.sequence_id.to_numpy(dtype=str),
             experiment_ids=experiments.Motif_ID.to_numpy(dtype=str))
    data = BenchmarkData(frozen, embeddings, cache)
    return data


def main():
    known = regression_metrics([1, 2, 3, np.nan], [1, 2, 3, 999])
    assert abs(known["spearman"] - 1) < 1e-12 and abs(known["pearson"] - 1) < 1e-12
    assert known["n"] == 3 and known["mse"] == 0
    constant = regression_metrics([1, 1, 1], [2, 3, 4])
    assert np.isnan(constant["spearman"]) and np.isnan(constant["pearson"])
    assert np.isfinite(constant["mse"])
    macro = macro_metrics(np.array([[1, 2], [2, 4], [3, 6]], dtype=float), np.array([1, 2, 3]))
    assert macro["n_valid_spearman"] == 2 and abs(macro["macro_spearman"] - 1) < 1e-12
    aligned = aligned_prediction(np.array([[1], [2]]), 3)
    assert aligned.shape == (2, 3) and np.array_equal(aligned, [[1, 1, 1], [2, 2, 2]])

    observed, placeholders = label_placeholders(np.array([[1.0, np.nan, -2.0]]))
    assert observed.tolist() == [[True, False, True]]
    assert np.isfinite(placeholders).all() and placeholders.tolist() == [[1.0, 0.0, -2.0]]

    import torch
    loss = masked_mse(torch.tensor([[1.0, 4.0]]), torch.tensor([[0.0, 0.0]]),
                      torch.tensor([[1.0, 0.0]]))
    assert float(loss) == 1.0
    try:
        masked_mse(torch.tensor([[1.0]]), torch.tensor([[0.0]]), torch.tensor([[0.0]]))
        raise AssertionError("all-NaN mask was not rejected")
    except ValueError:
        pass

    assert split_roles(0) == {"train": [2, 3, 4], "validation": [1], "test": [0]}
    candidate = {"macro_spearman": 0.5, "macro_mse": 1.0, "regularization": 0.01}
    same_lower_reg = {"macro_spearman": 0.5, "macro_mse": 1.0, "regularization": 0.001}
    assert better_candidate(same_lower_reg, candidate)
    assert not better_candidate(candidate, same_lower_reg)

    with tempfile.TemporaryDirectory() as root:
        data = synthetic_data(root)
        for track in ("sequence", "rbp_ood"):
            context = data.context(track, 0, include_test=False)
            assert set(context["rows"]) == {"train", "validation"}
            assert set(context["tasks"]) == {"train", "validation"}
            assert len(context["roles"]["train"]) == 3
            full = data.context(track, 0, include_test=True)
            assert set(full["rows"]) == {"train", "validation", "test"}
            if track == "sequence":
                assert set(full["rows"]["train"]) & set(full["rows"]["test"]) == set()
                assert len(full["tasks"]["train"]) == 244
            else:
                assert len(full["rows"]["train"]) == 100
                assert len(full["tasks"]["train"]) == 146
                assert len(full["tasks"]["validation"]) == 49
                assert len(full["tasks"]["test"]) == 49
            knn = knn_predict(data, context, "validation", K_VALUES, torch.device("cpu"))
            assert set(knn) == set(K_VALUES)
            expected_width = 1 if track == "rbp_ood" else len(context["tasks"]["validation"])
            expected_rows = len(context["rows"]["validation"])
            for prediction in knn.values():
                assert prediction.shape == (expected_rows, expected_width)
                assert aligned_prediction(prediction, len(context["tasks"]["validation"])).shape == (
                    expected_rows, len(context["tasks"]["validation"]))
            assert all(np.isfinite(prediction).all() for prediction in knn.values())

        context = data.context("sequence", 0, include_test=True)
        prediction = np.asarray(data.embeddings[context["rows"]["test"], :244], dtype=np.float32)
        run = os.path.join(root, "out", "knn", "sequence_f0")
        os.makedirs(run)
        table = save_test_products(data, context, prediction, run, {"version": "test"})
        assert len(table) == 244 * 3
        with open(os.path.join(run, ".eval_complete"), "w") as handle:
            handle.write("done\n")
        replay = replay_run(data, "sequence", "knn", 0, None, os.path.join(root, "out"))
        assert replay["replay_passed"] and replay["n_metric_rows"] == 244 * 3

        emb_path = os.path.join(root, "embeddings", "rnafm.mean_masked.fp16.npy")
        ids_path = os.path.join(root, "embeddings", "rnafm.ids")
        checked = validate_cache(emb_path, ids_path, data.sequences.sequence_id.tolist())
        assert checked["rows"] == 100 and checked["dim"] == 640
        bad = np.asarray(np.load(emb_path), copy=True)
        bad[0, 0] = np.nan
        np.save(emb_path, bad)
        try:
            validate_cache(emb_path, ids_path, data.sequences.sequence_id.tolist())
            raise AssertionError("non-finite embedding was not rejected")
        except ValueError:
            pass

        gate_root = os.path.join(root, "selection_audit")
        os.makedirs(os.path.join(gate_root, "selections"))
        for track in ("sequence", "rbp_ood"):
            for arch in ("linear", "mlp", "knn"):
                with open(os.path.join(gate_root, "selections", f"{track}_{arch}.json"), "w") as handle:
                    json.dump({"status": "frozen", "test_used": False, "selection_fold": 0}, handle)
        candidate = os.path.join(gate_root, "selection_runs", "candidate")
        os.makedirs(candidate)
        with open(os.path.join(candidate, "config.json"), "w") as handle:
            json.dump({"test_read_during_training": False}, handle)
        audited = audit_selection_boundaries(gate_root)
        assert audited["n_selections"] == 6 and audited["selection_tree_test_artifacts"] == 0
        with open(os.path.join(candidate, "test_predictions.npz"), "wb") as handle:
            handle.write(b"forbidden")
        try:
            audit_selection_boundaries(gate_root)
            raise AssertionError("test artifact in selection tree was not rejected")
        except ValueError:
            pass

    print("RNACOMPETE_MODEL_CONTRACT_TEST_OK: NaN mask, metrics, split boundaries, IDs, replay, ties")


if __name__ == "__main__":
    main()
