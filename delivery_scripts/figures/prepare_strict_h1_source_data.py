#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STRICT = ROOT / "reports" / "source_data" / "paper1_strict_h1_v3"
SUMMARY = STRICT / "analysis" / "strict_h1_paired_summary.json"
MULTIPLICITY = STRICT / "analysis" / "strict_h1_multiplicity_sensitivity.json"
REPORT = ROOT / "reports" / "docs" / "PAPER1_STRICT_H1_RESULT.md"
OUT = HERE / "source_data"

CONTEXTS = {
    "n50_hek293t_r1": "N50 HEK293T",
    "defined_hepg2_r1": "Defined HepG2 r1",
    "defined_tcell_r1": "Defined T-cell r1",
    "defined_tcell_r2": "Defined T-cell r2",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pool_index(pool_id: str) -> int:
    return int(pool_id.rsplit("_", 1)[1])


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {name}")
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    multiplicity = json.loads(MULTIPLICITY.read_text())
    audit = json.loads((STRICT / "analysis" / "final_audit.json").read_text())
    assert audit["pass"] and not audit["new_blind_test"]
    assert payload["new_blind_test"] is False
    assert multiplicity["status"] == "PASS" and multiplicity["source"]["sha256"] == digest(SUMMARY)
    assert multiplicity["conclusions"]["tcell_r1_r2_caps_2_to_16_all_20_bonferroni_intervals_exclude_zero"]
    assert multiplicity["conclusions"]["hepg2_cap8_all_20_bonferroni_interval_includes_zero"]
    assert multiplicity["conclusions"]["hepg2_cap8_within_context_5_bonferroni_interval_includes_zero"]
    outputs = {
        "strict_h1_figure2_paired_pressure.csv": [],
        "strict_h1_figure2_pool_effects.csv": [],
        "strict_h1_figure3_score_matching.csv": [],
        "strict_h1_figure3_pool_effects.csv": [],
        "strict_h1_figure4_cross_architecture_summary.csv": [],
        "strict_h1_figure4_pool_effects.csv": [],
    }
    for context_id, context in CONTEXTS.items():
        result = payload["contexts"][context_id]
        pressure = result["pressure"]
        for row in pressure["rows"]:
            low, high = row["paired_cluster_bootstrap_95_percent_interval"]
            outputs["strict_h1_figure2_paired_pressure.csv"].append({
                "context": context,
                "proposal_cap": row["proposal_cap"],
                "random_mean_regret": row["random_mean_regret"],
                "bon_mean_regret": row["bon_mean_regret"],
                "paired_bon_minus_random": row["paired_bon_minus_random"],
                "ci_low": low,
                "ci_high": high,
                "outer_pools": row["outer_pools"],
            })
        for row in pressure["per_pool"]:
            outputs["strict_h1_figure2_pool_effects.csv"].append({
                "context": context,
                "pool_index": pool_index(row["pool_id"]),
                "proposal_cap": row["proposal_cap"],
                "random_mean_regret": row["random_mean_regret"],
                "bon_mean_regret": row["bon_mean_regret"],
                "paired_bon_minus_random": row["paired_bon_minus_random"],
            })
        score = result["score_matching"]
        low, high = score["paired_cluster_bootstrap_95_percent_interval"]
        outputs["strict_h1_figure3_score_matching.csv"].append({
            "context": context,
            "optimized_mean_abs_error": score["optimized_mean_abs_error"],
            "matched_mean_abs_error": score["matched_mean_abs_error"],
            "difference": score["paired_optimized_minus_matched"],
            "ci_low": low,
            "ci_high": high,
            "outer_pools": 16,
        })
        for row in score["per_pool"]:
            outputs["strict_h1_figure3_pool_effects.csv"].append({
                "context": context,
                "pool_index": pool_index(row["pool_id"]),
                "optimized_mean_abs_error": row["optimized_mean_abs_error"],
                "matched_mean_abs_error": row["matched_mean_abs_error"],
                "paired_optimized_minus_matched": row["paired_optimized_minus_matched"],
                "mean_proxy_gap": row["mean_proxy_gap"],
            })
        common = result["common_error"]
        gap_low, gap_high = common["paired_cluster_bootstrap_95_percent_interval"]
        cor_low, cor_high = common["correlation_cluster_bootstrap_95_percent_interval"]
        outputs["strict_h1_figure4_cross_architecture_summary.csv"].append({
            "context": context,
            "cnn_mean_abs_error": common["cnn_mean_abs_error"],
            "kmer_mean_abs_error": common["kmer_mean_abs_error"],
            "paired_cnn_minus_kmer": common["paired_cnn_minus_kmer"],
            "difference_ci_low": gap_low,
            "difference_ci_high": gap_high,
            "architecture_error_correlation": common["mean_architecture_error_correlation"],
            "correlation_ci_low": cor_low,
            "correlation_ci_high": cor_high,
            "outer_pools": 16,
            "candidate_count": 1024,
        })
        for row in common["per_pool"]:
            outputs["strict_h1_figure4_pool_effects.csv"].append({
                "context": context,
                "pool_index": pool_index(row["pool_id"]),
                "cnn_mean_abs_error": row["cnn_mean_abs_error"],
                "kmer_mean_abs_error": row["kmer_mean_abs_error"],
                "paired_cnn_minus_kmer": row["paired_cnn_minus_kmer"],
                "architecture_error_correlation": row["architecture_error_correlation"],
                "mean_architecture_gap": row["mean_architecture_gap"],
            })
    for name, rows in outputs.items():
        write_csv(name, rows)
    forbidden = {"sequence", "sequence_id", "utr50", "measured_rl", "selected_id", "outer_label"}
    for name, rows in outputs.items():
        assert not forbidden.intersection(rows[0]), (name, forbidden.intersection(rows[0]))
    manifest = {
        "protocol": payload["protocol"],
        "evidence_boundary": payload["evidence_boundary"],
        "new_blind_test": False,
        "source_commit": audit["source_commit"],
        "statistical_method": "Within-pool paired effects; pool-cluster percentile bootstrap, 10,000 draws, base seed 20261124 with frozen context/cap/metric offsets from summarize_strict_h1_paper1.py.",
        "candidate_privacy": "Only pool-level aggregates and anonymous pool indexes are exported; no sequences, sequence IDs, measured-MRL labels, or candidate-level errors.",
        "source_files": {
            str(SUMMARY.relative_to(ROOT)): digest(SUMMARY),
            str(MULTIPLICITY.relative_to(ROOT)): digest(MULTIPLICITY),
            str(REPORT.relative_to(ROOT)): digest(REPORT),
            str((STRICT / "analysis" / "final_audit.json").relative_to(ROOT)): digest(STRICT / "analysis" / "final_audit.json"),
        },
        "checks": {
            "pressure_rows": len(outputs["strict_h1_figure2_paired_pressure.csv"]),
            "multiplicity_rows": len(multiplicity["rows"]),
            "pressure_pool_rows": len(outputs["strict_h1_figure2_pool_effects.csv"]),
            "score_pool_rows": len(outputs["strict_h1_figure3_pool_effects.csv"]),
            "common_error_pool_rows": len(outputs["strict_h1_figure4_pool_effects.csv"]),
            "source_validation_rounds": audit["summary_validation_rounds"],
            "labels_or_sequences_exported": False,
        },
        "files": {
            name: {"rows": len(rows), "sha256": digest(OUT / name)}
            for name, rows in outputs.items()
        },
    }
    (OUT / "strict_h1_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["checks"], indent=2))


if __name__ == "__main__":
    main()
