#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "strict_h1_paired_summary.json"
OUTPUT = HERE / "strict_h1_multiplicity_sensitivity.json"
CONTEXT_OFFSET = {
    "n50_hek293t_r1": 0,
    "defined_hepg2_r1": 1,
    "defined_tcell_r1": 2,
    "defined_tcell_r2": 3,
}
CAP_INDEX = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
BASE_SEED = 20261124
REPLICATES = 10000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: np.ndarray, seed: int, comparisons: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(REPLICATES, len(values)))
    samples = values[indices].mean(axis=1)
    tail_probability = 0.05 / (2 * comparisons)
    return np.percentile(
        samples,
        [100 * tail_probability, 100 * (1 - tail_probability)],
    ).tolist()


def excludes_zero(bounds: list[float]) -> bool:
    return bounds[0] > 0 or bounds[1] < 0


def build(summary_path: Path = SUMMARY) -> dict:
    source = json.loads(summary_path.read_text())
    assert source["new_blind_test"] is False
    assert source["bootstrap"]["replicates"] == REPLICATES
    assert source["bootstrap"]["seed"] == BASE_SEED
    rows = []
    for context_id, offset in CONTEXT_OFFSET.items():
        pressure = source["contexts"][context_id]["pressure"]
        for summary_row in pressure["rows"]:
            cap = summary_row["proposal_cap"]
            pool_rows = [row for row in pressure["per_pool"] if row["proposal_cap"] == cap]
            assert len(pool_rows) == summary_row["outer_pools"] == 16
            values = np.asarray([row["paired_bon_minus_random"] for row in pool_rows], dtype=np.float64)
            seed = BASE_SEED + offset * 100 + CAP_INDEX[cap]
            nominal = interval(values, seed, 1)
            np.testing.assert_allclose(
                nominal,
                summary_row["paired_cluster_bootstrap_95_percent_interval"],
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                values.mean(), summary_row["paired_bon_minus_random"], rtol=0, atol=1e-12
            )
            familywise = interval(values, seed, 20)
            within_context = interval(values, seed, 5)
            rows.append({
                "context": context_id,
                "proposal_cap": cap,
                "outer_pools": len(values),
                "paired_bon_minus_random": float(values.mean()),
                "nominal_95_percent_interval": nominal,
                "nominal_95_percent_excludes_zero": excludes_zero(nominal),
                "bonferroni_20_comparisons_99_75_percent_interval": familywise,
                "bonferroni_20_comparisons_excludes_zero": excludes_zero(familywise),
                "bonferroni_within_context_5_comparisons_99_percent_interval": within_context,
                "bonferroni_within_context_5_comparisons_excludes_zero": excludes_zero(within_context),
            })
    assert len(rows) == 20
    tcell_rows = [
        row for row in rows
        if row["context"] in {"defined_tcell_r1", "defined_tcell_r2"} and row["proposal_cap"] >= 2
    ]
    hepg2_cap8 = next(
        row for row in rows if row["context"] == "defined_hepg2_r1" and row["proposal_cap"] == 8
    )
    assert len(tcell_rows) == 8
    assert all(row["bonferroni_20_comparisons_excludes_zero"] for row in tcell_rows)
    assert all(row["paired_bon_minus_random"] < 0 for row in tcell_rows)
    assert not hepg2_cap8["bonferroni_20_comparisons_excludes_zero"]
    assert not hepg2_cap8["bonferroni_within_context_5_comparisons_excludes_zero"]
    assert hepg2_cap8["nominal_95_percent_excludes_zero"]
    return {
        "schema_version": "1.0",
        "analysis": "strict_h1_multiplicity_sensitivity",
        "status": "PASS",
        "source": {
            "path": summary_path.name,
            "sha256": digest(summary_path),
            "protocol": source["protocol"],
            "new_blind_test": False,
        },
        "method": {
            "effect": source["effect_signs"]["pressure"],
            "cluster": "pool",
            "bootstrap": "percentile bootstrap of complete-pool paired effects",
            "replicates": REPLICATES,
            "base_seed": BASE_SEED,
            "seed_rule": "base seed + 100 * context offset + cap index; frozen offsets from the aggregate summary",
            "nominal_confidence_percent": 95,
            "bonferroni_all_contexts": {
                "comparisons": 20,
                "confidence_percent": 99.75,
                "percentiles": [0.125, 99.875],
            },
            "bonferroni_within_context": {
                "comparisons": 5,
                "confidence_percent": 99,
                "percentiles": [0.5, 99.5],
            },
            "interpretation": "Nominal 95% CIs are retained as the primary descriptive intervals. Bonferroni intervals are conservative sensitivity analyses, not a new confirmatory analysis.",
        },
        "comparison_scope": {
            "random_baseline": "Random is always a single-candidate baseline at every row, including cap 1.",
            "bon_budget": "BoN uses the stated proposal cap of 1, 2, 4, 8, or 16 candidates.",
            "pressure_curve": "The pressure curve varies the BoN candidate budget against the same single-candidate Random baseline; it is not an equal-candidate-budget comparison.",
        },
        "rows": rows,
        "conclusions": {
            "tcell_r1_r2_caps_2_to_16_all_20_bonferroni_intervals_exclude_zero": True,
            "tcell_r1_r2_caps_2_to_16_count": 8,
            "hepg2_cap8_nominal_interval_excludes_zero": True,
            "hepg2_cap8_all_20_bonferroni_interval_includes_zero": True,
            "hepg2_cap8_within_context_5_bonferroni_interval_includes_zero": True,
            "hepg2_cap8_interpretation": "Exploratory description only; not multiplicity-robust evidence.",
            "evidence_boundary": "Retrospective protocol sensitivity; conservative multiplicity sensitivity does not create prospective or confirmatory evidence.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build(args.summary)
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.check:
        assert args.output.read_text() == encoded
        print(f"PASS: {args.output} is reproducible")
    else:
        args.output.write_text(encoded)
        print(json.dumps(payload["conclusions"], indent=2))


if __name__ == "__main__":
    main()
