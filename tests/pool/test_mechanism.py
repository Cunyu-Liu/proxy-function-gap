import csv
import gzip
import json

from proxy_function_gap.mechanism import evaluate_mechanism, sequence_features


def test_mechanism_score_matching_and_features(tmp_path):
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    with gzip.open(evaluator / "labels.tsv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", "measured_rl"], delimiter="\t")
        writer.writeheader(); writer.writerows([{"sequence_id": "s0", "measured_rl": 1}, {"sequence_id": "s1", "measured_rl": 3}])
    scores = tmp_path / "scores.json"
    records = [
        {"sequence_id": "s0", "proxy_mean": 2.0, "proxy_std": 0.1, "support": 0.5, "gc": 0.5, "uAUG": False, "longest_poly_u": 1},
        {"sequence_id": "s1", "proxy_mean": 2.1, "proxy_std": 0.2, "support": 0.6, "gc": 0.6, "uAUG": True, "longest_poly_u": 2},
    ]
    scores.write_text(json.dumps({"metadata": {"outer_labels_read": False, "actual_cuda_inference": True}, "records": records}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"records": [{"strategy": "bon", "proposal_cap": 16, "selected_id": "s0"}]}))
    report = evaluate_mechanism(tmp_path, "pool", scores, selection, tmp_path / "report.json")
    assert report["score_matched"]["optimized_mean_abs_error"] == 1
    assert report["score_matched"]["matched_mean_abs_error"] == 0.8999999999999999
    assert report["labels_returned_to_learner"] is False
    features = sequence_features("ATGTTTT")
    assert features["uAUG"] and features["longest_poly_u"] == 4
