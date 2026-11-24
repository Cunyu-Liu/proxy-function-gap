import json

from summarize_strict_h1_paper1 import cluster_bootstrap, summarize


def test_strict_h1_paired_effects_and_cluster_interval(tmp_path):
    context = "n50_hek293t_r1"
    root = tmp_path / context
    for name in ("evaluation", "mechanism", "cross_architecture"):
        (root / name).mkdir(parents=True)
    pool_id = f"castillo_{context}_000"
    records = []
    for strategy, regret in (("random", 2.0), ("bon", 1.5)):
        records.append({"pool_id": pool_id, "strategy": strategy, "alpha": 0, "proposal_cap": 1, "seed": 101, "regret": regret})
    (root / "evaluation" / f"{pool_id}.json").write_text(json.dumps({"records": records}))
    score = {"optimized_mean_abs_error": 3.0, "matched_mean_abs_error": 2.0, "abs_error_difference": 1.0, "mean_proxy_gap": 0.01}
    (root / "mechanism" / f"{pool_id}.json").write_text(json.dumps({"score_matched": score}))
    common = {"fresh_mean_abs_error": 2.5, "kmer_mean_abs_error": 3.0, "architecture_error_correlation": 0.9, "mean_architecture_gap": 0.2, "low_gap_high_error_count": 1}
    (root / "cross_architecture" / f"{pool_id}.json").write_text(json.dumps(common))
    (root / "context_manifest.json").write_text("{}")
    (root / "evaluation" / "evaluation_manifest.json").write_text("{}")
    (root / "cross_architecture" / "summary.json").write_text("{}")
    config = {
        "analysis_root": str(tmp_path),
        "protocol": "strict_h1_retrospective_protocol_sensitivity",
        "evidence_boundary": "retrospective",
        "pool_count": 1,
        "contexts": {context: "unused"},
        "selection": {"proposal_caps": [1], "seeds": [101]},
        "bootstrap": {"seed": 20261124, "replicates": 100},
    }
    report = summarize(config)["contexts"][context]
    assert report["pressure"]["rows"][0]["paired_bon_minus_random"] == -0.5
    assert report["pressure"]["rows"][0]["paired_cluster_bootstrap_95_percent_interval"] == [-0.5, -0.5]
    assert report["score_matching"]["paired_optimized_minus_matched"] == 1.0
    assert report["common_error"]["paired_cnn_minus_kmer"] == -0.5
    assert report["common_error"]["per_pool"][0]["low_gap_high_error_count"] == 1


def test_cluster_bootstrap_is_deterministic():
    assert cluster_bootstrap([1, 2, 4], 7, 100) == cluster_bootstrap([1, 2, 4], 7, 100)
