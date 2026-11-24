import csv
import gzip
import json

import numpy as np
import pytest

from proxy_function_gap.schlusser_te import (
    COMPARISONS,
    METHODS,
    CandidateTable,
    ModelScores,
    build_score_match_maps,
    cluster_bootstrap,
    evaluate_round,
    load_config,
    make_selections,
    nearest_score_match,
    paired_seeds,
    render_result,
)
from proxy_function_gap.trainer.data import sha256_file


def config():
    return {
        "schema_version": "1.0",
        "dataset": "synthetic",
        "protocol_document": "protocol",
        "output_root": "output",
        "gate0_root": "gate0",
        "learner": {},
        "evaluator_vault": {},
        "authorized_model_execution": True,
        "proxy_endpoint": "MRL",
        "deployment_endpoint": "TE",
        "models": {},
        "caps": [1, 2, 4, 8, 16],
        "paired_seeds": {"start": 101, "step": 101, "count": 100},
        "methods": list(METHODS),
        "primary": {"method": "bon_cnn", "reference": "random", "proposal_cap": 16},
        "score_matching": {},
        "bootstrap": {"seed": 20261124, "replicates": 20, "percentile_interval": [2.5, 97.5]},
        "execution": {"rounds": 3, "single_unblinding_without_result_driven_revision": True},
        "gpu": {},
    }


def table_and_scores():
    table = CandidateTable(
        tuple(f"row_{index}" for index in range(20)),
        tuple(f"gene_{index // 2}" for index in range(20)),
        tuple(f"transcript_{index}" for index in range(20)),
        tuple(("A" if index % 2 else "C") * 50 for index in range(20)),
    )
    cnn = np.asarray([np.linspace(0, 1, 20), np.linspace(0.1, 1.1, 20), np.linspace(0.2, 1.2, 20)])
    kmer = np.asarray([np.linspace(1, 0, 20), np.linspace(1.1, 0.1, 20), np.linspace(1.2, 0.2, 20)])
    return table, ModelScores(cnn, kmer, {})


def test_frozen_config_and_seed_formula(tmp_path):
    value = config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    assert load_config(path) == value
    assert paired_seeds(value)[:5] == [101, 202, 303, 404, 505]
    assert paired_seeds(value)[-1] == 10100


def test_paired_selection_cap_one_and_full_grid():
    table, scores = table_and_scores()
    value = config()
    records = make_selections(table, scores, value)
    assert len(records) == 100 * 5 * 4
    by_key = {(row["seed"], row["proposal_cap"], row["method"]): row for row in records}
    for seed in paired_seeds(value):
        reference = by_key[(seed, 1, "random")]
        for method in METHODS:
            assert by_key[(seed, 1, method)]["selected_index"] == reference["selected_index"]
        for cap in value["caps"]:
            proposals = by_key[(seed, cap, "random")]["proposal_indices"]
            for method in METHODS:
                assert by_key[(seed, cap, method)]["proposal_indices"] == proposals
                assert by_key[(seed, cap, method)]["regret"] is None
        for cap in (2, 4, 8, 16):
            assert by_key[(seed, cap, "random")]["selected_index"] == reference["selected_index"]


def test_score_match_excludes_same_sequence_and_is_label_free():
    sequences = ("A" * 50, "A" * 50, "C" * 50, "G" * 50)
    mean = np.asarray([1.0, 1.0, 1.1, 3.0])
    assert nearest_score_match(mean, sequences, 0) == 2
    table, scores = table_and_scores()
    records = make_selections(table, scores, config())
    for row in records:
        if "score_match" in row:
            selected = row["selected_index"]
            matched = row["score_match"]["matched_index"]
            assert table.utr50[selected] != table.utr50[matched]
            assert "TE" not in json.dumps(row)
            assert row["regret"] is None


def test_cluster_bootstrap_recomputes_all_paired_effects():
    table, scores = table_and_scores()
    value = config()
    records = make_selections(table, scores, value)
    labels = np.linspace(0.5, 2.5, len(table))
    for unit in ("gene", "transcript"):
        result = cluster_bootstrap(table, scores, labels, records, value, unit)
        assert result["replicates"] == 20
        assert len(result["mean_regret"]) == len(METHODS) * 5
        assert len(result["paired_regret_difference"]) == len(COMPARISONS) * 5
        for comparison in COMPARISONS:
            low, high = result["paired_regret_difference"][f"{comparison}|1"]
            assert low == pytest.approx(0)
            assert high == pytest.approx(0)
        for method in ("bon_cnn", "bon_kmer"):
            interval = result["score_matched_regret_difference_conditional_on_frozen_match_map"][method]
            assert len(interval) == 2 and all(np.isfinite(interval))


def test_evaluator_summary_and_report_keep_cross_endpoint_boundary(tmp_path):
    table, scores = table_and_scores()
    value = config()
    value["output_root"] = str(tmp_path)
    value["bootstrap"]["replicates"] = 5
    learner = tmp_path / "learner_round_1"
    learner.mkdir()
    maps = build_score_match_maps(table, scores)
    records = make_selections(table, scores, value, maps)
    (learner / "selection.json").write_text(json.dumps({"records": records}))
    (learner / "score_match_maps.json").write_text(json.dumps({"maps": maps}))
    (learner / "learner_manifest.json").write_text(json.dumps({"outer_labels_read": False, "outer_metrics_computed": False, "actual_cuda_inference": True, "prediction_array_sha256": __import__("proxy_function_gap.schlusser_te", fromlist=["_array_hash"])._array_hash([scores.cnn_members, scores.kmer_members])}))
    np.savez(learner / "proxy_scores.npz", row_id=np.asarray(table.row_ids), cnn_members=scores.cnn_members, kmer_members=scores.kmer_members)
    vault = tmp_path / "vault.tsv.gz"
    with gzip.open(vault, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "TE"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, row_id in enumerate(table.row_ids):
            writer.writerow({"row_id": row_id, "TE": index / 10})
    value["evaluator_vault"] = {"path": str(vault), "sha256": sha256_file(vault)}
    summary = evaluate_round(value, table, 1)
    assert summary["outer_metrics_computed"] is True
    assert summary["labels_returned_to_learner"] is False
    assert summary["endpoint_alignment_descriptive"]["cross_endpoint_absolute_error_computed"] is False
    cap_one = [row for row in summary["paired_comparisons"] if row["proposal_cap"] == 1]
    assert all(row["mean_regret_difference_left_minus_right"] == 0 for row in cap_one)
    report = render_result(value, {"pass": True})
    text = report.read_text()
    assert "跨终点部署失配" in text
    assert "不可识别边界" in text
    assert "BoN-CNN−Random" in text
