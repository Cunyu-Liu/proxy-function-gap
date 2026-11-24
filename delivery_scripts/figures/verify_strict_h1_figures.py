#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data"
STRICT = HERE.parent / "reports" / "source_data" / "paper1_strict_h1_v3" / "analysis"
MULTIPLICITY = STRICT / "strict_h1_multiplicity_sensitivity.json"
MULTIPLICITY_SCRIPT = STRICT / "compute_strict_h1_multiplicity_sensitivity.py"
BASENAMES = [
    "Figure2_strict_h1_selection_pressure",
    "Figure3_strict_h1_score_matching",
    "Figure4_strict_h1_common_proxy_error",
]
SCRIPTS = ["plot_strict_h1_figure2.py", "plot_strict_h1_figure3.py", "plot_strict_h1_figure4.py"]
LEGACY = [
    "Figure2_selection_pressure", "Figure3_score_matching", "Figure4_common_proxy_error",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, tolerance=1e-10):
    assert np.allclose(actual, expected, rtol=0, atol=tolerance), (actual, expected)


CONTEXT_OFFSET = {
    "N50 HEK293T": 0,
    "Defined HepG2 r1": 1,
    "Defined T-cell r1": 2,
    "Defined T-cell r2": 3,
}
CAP_INDEX = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}


def bootstrap(values, seed, comparisons=1):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    samples = values[indices].mean(axis=1)
    tail = 100 * 0.05 / (2 * comparisons)
    return np.percentile(samples, [tail, 100 - tail])


def round_one() -> dict:
    pressure = pd.read_csv(DATA / "strict_h1_figure2_paired_pressure.csv")
    pressure_pools = pd.read_csv(DATA / "strict_h1_figure2_pool_effects.csv")
    multiplicity_payload = json.loads(MULTIPLICITY.read_text())
    assert multiplicity_payload["status"] == "PASS" and len(multiplicity_payload["rows"]) == 20
    assert multiplicity_payload["source"]["sha256"] == digest(STRICT / "strict_h1_paired_summary.json")
    context_ids = {
        "N50 HEK293T": "n50_hek293t_r1",
        "Defined HepG2 r1": "defined_hepg2_r1",
        "Defined T-cell r1": "defined_tcell_r1",
        "Defined T-cell r2": "defined_tcell_r2",
    }
    multiplicity_rows = {(row["context"], row["proposal_cap"]): row for row in multiplicity_payload["rows"]}
    assert len(pressure) == 20 and len(pressure_pools) == 320
    for (context, cap), rows in pressure_pools.groupby(["context", "proposal_cap"]):
        current = pressure[(pressure.context == context) & (pressure.proposal_cap == cap)].iloc[0]
        adjusted = multiplicity_rows[(context_ids[context], int(cap))]
        assert len(rows) == 16
        close(rows.random_mean_regret.mean(), current.random_mean_regret)
        close(rows.bon_mean_regret.mean(), current.bon_mean_regret)
        close(rows.bon_mean_regret - rows.random_mean_regret, rows.paired_bon_minus_random)
        close(rows.paired_bon_minus_random.mean(), current.paired_bon_minus_random)
        seed = 20261124 + CONTEXT_OFFSET[context] * 100 + CAP_INDEX[cap]
        close(bootstrap(rows.paired_bon_minus_random, seed), [current.ci_low, current.ci_high])
        close(bootstrap(rows.paired_bon_minus_random, seed, 20),
              adjusted["bonferroni_20_comparisons_99_75_percent_interval"])
        close(bootstrap(rows.paired_bon_minus_random, seed, 5),
              adjusted["bonferroni_within_context_5_comparisons_99_percent_interval"])
    for context in ("Defined T-cell r1", "Defined T-cell r2"):
        rows = pressure[(pressure.context == context) & (pressure.proposal_cap >= 2)]
        assert (rows.ci_high < 0).all()
        for cap in (2, 4, 8, 16):
            adjusted = multiplicity_rows[(context_ids[context], cap)]
            assert adjusted["bonferroni_20_comparisons_99_75_percent_interval"][1] < 0
            assert adjusted["bonferroni_within_context_5_comparisons_99_percent_interval"][1] < 0
    hepg2 = pressure[(pressure.context == "Defined HepG2 r1") & (pressure.proposal_cap >= 2)]
    assert (hepg2.ci_high < 0).sum() == 1
    assert hepg2[hepg2.proposal_cap == 8].ci_high.iloc[0] < 0
    assert hepg2[hepg2.proposal_cap == 16].ci_high.iloc[0] > 0
    hepg2_cap8 = multiplicity_rows[("defined_hepg2_r1", 8)]
    assert hepg2_cap8["bonferroni_20_comparisons_99_75_percent_interval"][0] < 0 < hepg2_cap8["bonferroni_20_comparisons_99_75_percent_interval"][1]
    assert hepg2_cap8["bonferroni_within_context_5_comparisons_99_percent_interval"][0] < 0 < hepg2_cap8["bonferroni_within_context_5_comparisons_99_percent_interval"][1]
    n50 = pressure[pressure.context == "N50 HEK293T"]
    assert ((n50.ci_low <= 0) & (n50.ci_high >= 0)).all()

    score = pd.read_csv(DATA / "strict_h1_figure3_score_matching.csv")
    score_pools = pd.read_csv(DATA / "strict_h1_figure3_pool_effects.csv")
    assert len(score) == len(score_pools) // 16 == 4
    for context, rows in score_pools.groupby("context"):
        current = score[score.context == context].iloc[0]
        close(rows.optimized_mean_abs_error.mean(), current.optimized_mean_abs_error)
        close(rows.matched_mean_abs_error.mean(), current.matched_mean_abs_error)
        close(rows.paired_optimized_minus_matched.mean(), current.difference)
        close(bootstrap(rows.paired_optimized_minus_matched, 20261124 + 1000 + CONTEXT_OFFSET[context]),
              [current.ci_low, current.ci_high])
    assert ((score.ci_low < 0) & (score.ci_high > 0)).all()

    common = pd.read_csv(DATA / "strict_h1_figure4_cross_architecture_summary.csv")
    common_pools = pd.read_csv(DATA / "strict_h1_figure4_pool_effects.csv")
    assert len(common) == len(common_pools) // 16 == 4
    for context, rows in common_pools.groupby("context"):
        current = common[common.context == context].iloc[0]
        close(rows.cnn_mean_abs_error.mean(), current.cnn_mean_abs_error)
        close(rows.kmer_mean_abs_error.mean(), current.kmer_mean_abs_error)
        close(rows.paired_cnn_minus_kmer.mean(), current.paired_cnn_minus_kmer)
        close(rows.architecture_error_correlation.mean(), current.architecture_error_correlation)
        close(bootstrap(rows.paired_cnn_minus_kmer, 20261124 + 2000 + CONTEXT_OFFSET[context]),
              [current.difference_ci_low, current.difference_ci_high])
        close(bootstrap(rows.architecture_error_correlation, 20261124 + 3000 + CONTEXT_OFFSET[context]),
              [current.correlation_ci_low, current.correlation_ci_high])
    assert (common.difference_ci_high < 0).all()
    close(common.cnn_mean_abs_error - common.kmer_mean_abs_error, common.paired_cnn_minus_kmer)
    return {
        "status": "PASS",
        "pressure_effects_and_intervals_recomputed": 20,
        "bonferroni_intervals_recomputed": 40,
        "tcell_all_20_bonferroni_negative_intervals": 8,
        "hepg2_cap8_multiplicity_adjusted_intervals_include_zero": True,
        "score_matching_intervals_recomputed": 4,
        "common_error_and_correlation_intervals_recomputed": 8,
        "tcell_negative_interval_caps_2_to_16": 8,
        "hepg2_only_cap8_interval_excludes_zero": True,
        "n50_all_intervals_include_zero": True,
        "score_matching_all_intervals_include_zero": True,
        "cnn_lower_mae_intervals_exclude_zero": 4,
    }


def round_two() -> dict:
    forbidden = {"sequence", "sequence_id", "utr50", "measured_rl", "selected_id", "outer_label"}
    csv_files = sorted(DATA.glob("strict_h1_*.csv"))
    for path in csv_files:
        columns = set(pd.read_csv(path, nrows=1).columns)
        assert not forbidden.intersection(columns), (path, forbidden.intersection(columns))
    expected_svg_text = {
        BASENAMES[0]: ["Strict-H1 pressure", "-1.243", "+0.097", "-0.670", "Nominal 95%", "all-20 Bonferroni", "exploratory", "not a new blind test"],
        BASENAMES[1]: ["Strict-H1 score matching", "+0.174", "+0.326", "-0.303", "+0.148", "-0.495", "nominal", "intervals include zero"],
        BASENAMES[2]: ["Strict-H1 common proxy errors", "0.878", "0.933", "-0.254", "-0.152", "nominal", "exclude zero"],
    }
    dimensions, layout = {}, {}
    for base in BASENAMES:
        svg, pdf, png = [HERE / f"{base}.{suffix}" for suffix in ("svg", "pdf", "png")]
        for path in (svg, pdf, png):
            assert path.is_file() and path.stat().st_size > 1000
        ET.parse(svg)
        svg_text = svg.read_text()
        for value in expected_svg_text[base]:
            assert value in svg_text, (base, value)
        document = fitz.open(pdf)
        assert len(document) == 1
        page = document[0]
        words = page.get_text("words")
        assert words
        for word in words:
            x0, y0, x1, y1 = word[:4]
            assert x0 >= -0.5 and y0 >= -0.5 and x1 <= page.rect.width + 0.5 and y1 <= page.rect.height + 0.5, (base, word)
        layout[base] = {"pdf_words": len(words), "all_pdf_text_within_page": True}
        document.close()
        with Image.open(png) as image:
            image.verify()
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            assert all(abs(value - 300) < 0.01 for value in dpi), (base, dpi)
            dimensions[base] = {"pixels": list(image.size), "dpi": list(dpi)}
    for base in LEGACY:
        for suffix in ("svg", "pdf", "png"):
            assert (HERE / f"{base}.{suffix}").is_file()
    return {
        "status": "PASS",
        "strict_csv_files_without_forbidden_columns": len(csv_files),
        "artifacts_per_figure": 3,
        "legacy_figures_preserved": LEGACY,
        "dimensions": dimensions,
        "layout": layout,
    }


def round_three() -> dict:
    targets = [HERE / f"{base}.{suffix}" for base in BASENAMES for suffix in ("png", "svg")]
    csv_files = sorted(DATA.glob("strict_h1_*.csv"))
    tracked = targets + csv_files + [MULTIPLICITY]
    before = {str(path): digest(path) for path in tracked}
    subprocess.run([sys.executable, str(MULTIPLICITY_SCRIPT)], check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(HERE / "prepare_strict_h1_source_data.py")], check=True, capture_output=True, text=True)
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(HERE / script)], check=True, capture_output=True, text=True)
    after = {str(path): digest(path) for path in tracked}
    assert before == after, [name for name in before if before[name] != after[name]]
    return {
        "status": "PASS",
        "rerun_scripts": [MULTIPLICITY_SCRIPT.name, "prepare_strict_h1_source_data.py"] + SCRIPTS,
        "byte_identical_png_svg_csv_json_files": len(tracked),
        "pdf_checked_in_round": 2,
    }


def main() -> None:
    report = {"round_1_numeric_recalculation": round_one()}
    print("Round 1: strict-H1 numeric recalculation PASS", flush=True)
    report["round_2_artifact_and_layout"] = round_two()
    print("Round 2: strict-H1 artifact and layout PASS", flush=True)
    report["round_3_clean_rerun"] = round_three()
    print("Round 3: strict-H1 deterministic rerun PASS", flush=True)
    report["round_2_artifact_and_layout"] = round_two()
    report["evidence_boundary"] = "Retrospective strict-H1 protocol sensitivity; not a new blind test."
    report["status"] = "PASS"
    (HERE / "strict_h1_validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(HERE / "strict_h1_validation_report.json")


if __name__ == "__main__":
    main()
