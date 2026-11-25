#!/usr/bin/env python3
"""Independent, standard-library CPU verification for the 2026-11-26 V5 preprint set."""

import argparse
import array
import ast
import csv
import hashlib
import json
import math
import platform
import re
import statistics
import struct
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from delivery_scripts.publication_paths import EVIDENCE_ROOT, OUTPUT_ROOT

ROOT = EVIDENCE_ROOT
DOCS = ROOT / "deliverables/reports/docs"
SOURCES = ROOT / "deliverables/reports/source_data"
FIGURES = ROOT / "deliverables/figures/source_data"
PAPER_DIR = ROOT / "deliverables/paper"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
OUT = args.output.resolve() if args.output else OUTPUT_ROOT / "docs" / f"UNIFIED_PREPRINT_SOURCE_VERIFICATION_V5_20261126_ROUND{args.round}.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
TOL = 1e-9

checks = []
inputs = {}


def add(name, ok, detail):
    checks.append({"name": name, "pass": bool(ok), "detail": detail})


def close(a, b, tol=TOL):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def mean(values):
    return statistics.fmean(float(v) for v in values)


def load_json(path):
    rel = path.relative_to(ROOT).as_posix()
    inputs[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def load_table(path, delimiter=","):
    rel = path.relative_to(ROOT).as_posix()
    inputs[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def load_text(path):
    rel = path.relative_to(ROOT).as_posix()
    inputs[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path.read_text(encoding="utf-8")


def load_npy_float64(path):
    rel = path.relative_to(ROOT).as_posix()
    inputs[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError(f"Not an NPY file: {path}")
        major, minor = handle.read(2)
        header_size = struct.unpack("<H" if major == 1 else "<I", handle.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(handle.read(header_size).decode("latin1").strip())
        if header["descr"] not in {"<f8", "=f8"} or header["fortran_order"]:
            raise ValueError(f"Expected native/little-endian C-order float64: {path}")
        shape = header["shape"]
        values = array.array("d")
        values.frombytes(handle.read())
        if sys.byteorder != "little":
            values.byteswap()
    if len(values) != math.prod(shape):
        raise ValueError(f"NPY payload size mismatch: {path}")
    return shape, values


def percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    numerator = sum((float(x) - mx) * (float(y) - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((float(x) - mx) ** 2 for x in xs)
        * sum((float(y) - my) ** 2 for y in ys)
    )
    return numerator / denominator


def ranks(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            result[ordered[k][0]] = rank
        i = j
    return result


def grouped(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def includes_zero(interval):
    return float(interval[0]) <= 0.0 <= float(interval[1])


def holm_adjust(rows, p_field="p"):
    ordered = sorted(rows, key=lambda row: (float(row[p_field]), row["context"], int(row["cap"])))
    running = 0.0
    adjusted = {}
    m = len(ordered)
    for index, row in enumerate(ordered):
        running = max(running, min(1.0, (m - index) * float(row[p_field])))
        adjusted[(row["context"], int(row["cap"]))] = running
    return adjusted


# Paper 1: recompute all strict-H1 pressure means from the 320 pool rows.
pressure_pool = load_table(FIGURES / "strict_h1_figure2_pool_effects.csv")
pressure_summary = load_table(FIGURES / "strict_h1_figure2_paired_pressure.csv")
raw_strict = load_json(SOURCES / "paper1_strict_h1_v3/analysis/strict_h1_paired_summary.json")
key_to_context = {
    "n50_hek293t_r1": "N50 HEK293T",
    "defined_hepg2_r1": "Defined HepG2 r1",
    "defined_tcell_r1": "Defined T-cell r1",
    "defined_tcell_r2": "Defined T-cell r2",
}
lookup = {(r["context"], int(r["proposal_cap"])): r for r in pressure_summary}
raw_ok = True
figure_ok = True
for key, context in key_to_context.items():
    raw_rows = raw_strict["contexts"][key]["pressure"]["rows"]
    raw_pools = raw_strict["contexts"][key]["pressure"]["per_pool"]
    for cap in (1, 2, 4, 8, 16):
        selected = [r for r in raw_pools if int(r["proposal_cap"]) == cap]
        assert len(selected) == 16
        recomputed = {
            "random_mean_regret": mean(r["random_mean_regret"] for r in selected),
            "bon_mean_regret": mean(r["bon_mean_regret"] for r in selected),
            "paired_bon_minus_random": mean(r["paired_bon_minus_random"] for r in selected),
        }
        raw = next(r for r in raw_rows if int(r["proposal_cap"]) == cap)
        raw_ok &= all(close(recomputed[field], raw[field]) for field in recomputed)
        figure = lookup[(context, cap)]
        figure_ok &= all(close(recomputed[field], figure[field]) for field in recomputed)
        csv_selected = [
            r for r in pressure_pool
            if r["context"] == context and int(r["proposal_cap"]) == cap
        ]
        figure_ok &= len(csv_selected) == 16
        figure_ok &= close(mean(r["paired_bon_minus_random"] for r in csv_selected), recomputed["paired_bon_minus_random"])
add("strict_h1_raw_per_pool_to_aggregate", raw_ok, "80/80 context-cap means recomputed from raw aggregate per-pool records")
add("strict_h1_figure_to_source", figure_ok, "20 summary rows and 320 pool rows agree with the aggregate JSON")

multiplicity = load_json(SOURCES / "paper1_strict_h1_v3/analysis/strict_h1_multiplicity_sensitivity.json")
tcell_adjusted = [
    r for r in multiplicity["rows"]
    if r["context"] in {"defined_tcell_r1", "defined_tcell_r2"}
    and int(r["proposal_cap"]) in {2, 4, 8, 16}
]
add(
    "tcell_bonferroni_boundary",
    len(tcell_adjusted) == 8
    and all(r["bonferroni_20_comparisons_99_75_percent_interval"][1] < 0 for r in tcell_adjusted),
    "All eight T-cell cap 2–16 all-20 Bonferroni intervals are negative",
)
hepg2_cap8 = next(
    r for r in multiplicity["rows"]
    if r["context"] == "defined_hepg2_r1" and int(r["proposal_cap"]) == 8
)
add(
    "hepg2_cap8_exploratory",
    includes_zero(hepg2_cap8["bonferroni_20_comparisons_99_75_percent_interval"])
    and includes_zero(hepg2_cap8["bonferroni_within_context_5_comparisons_99_percent_interval"]),
    "Both adjusted intervals include zero; no confirmatory claim is licensed",
)
n50_rows = [r for r in pressure_summary if r["context"] == "N50 HEK293T"]
add("n50_no_sustained_gain", all(includes_zero((r["ci_low"], r["ci_high"])) for r in n50_rows), "All five nominal intervals include zero")

# V5 equal-candidate-budget sensitivity: independently aggregate all three all-20 families.
budget_root = SOURCES / "paper1_budget_matched_v1"
budget = load_json(budget_root / "results.json")
budget_config = load_json(budget_root / "frozen_config.json")
budget_pools = load_table(budget_root / "pool_effects.tsv", "\t")
budget_summary = {(r["context"], int(r["cap"])): r for r in budget["context_summaries"]}
budget_pool_ok = len(budget_pools) == 320
for row in budget_summary.values():
    selected = [r for r in budget_pools if r["context"] == row["context"] and int(r["cap"]) == row["cap"]]
    budget_pool_ok &= len(selected) == 16
    for pool_field, summary_field in (
        ("random1_regret", "random1_regret"),
        ("bon_regret", "bon_regret"),
        ("random_n_regret", "random_n_regret"),
        ("bon_minus_random1", "bon_minus_random1"),
        ("bon_minus_random_n", "bon_minus_random_n"),
        ("random_n_minus_random1", "random_n_minus_random1"),
    ):
        budget_pool_ok &= close(mean(r[pool_field] for r in selected), row[summary_field])
add("v5_budget_320_pool_rows_recompute", budget_pool_ok, "All 20 summaries and six means/effects per context-cap recomputed from 320 de-identified pool rows")

budget_family_ok = True
budget_holm_ok = True
for family, rows in budget["families"].items():
    budget_family_ok &= len(rows) == 20
    recomputed_holm = holm_adjust(rows)
    for row in rows:
        summary = budget_summary[(row["context"], int(row["cap"]))]
        budget_family_ok &= close(row["effect"], summary[family])
        budget_family_ok &= close(row["effect"], mean(
            r[family] for r in budget_pools
            if r["context"] == row["context"] and int(r["cap"]) == row["cap"]
        ))
        budget_holm_ok &= close(recomputed_holm[(row["context"], int(row["cap"]))], row["holm_p"])
        if int(row["cap"]) == 1:
            budget_family_ok &= close(row["effect"], 0.0) and close(row["p"], 1.0) and close(row["holm_p"], 1.0)
add("v5_budget_all60_effects_and_holm", budget_family_ok and budget_holm_ok, "60/60 effects independently aggregated and all three complete all-20 Holm families recomputed, including cap1 p=1")

equal_rows = budget["families"]["bon_minus_random_n"]
equal_positive = {(r["context"], int(r["cap"])) for r in equal_rows if float(r["holm_p"]) < 0.05}
expected_equal_positive = {("defined_tcell_r1", cap) for cap in (2, 4, 8, 16)} | {("defined_tcell_r2", 4), ("defined_hepg2_r1", 2)}
add(
    "v5_equal_budget_primary_claim",
    equal_positive == expected_equal_positive
    and all(next(r for r in equal_rows if r["context"] == "defined_tcell_r1" and int(r["cap"]) == cap)["nominal_95_ci"][1] < 0 for cap in (2, 4, 8, 16))
    and all(float(r["holm_p"]) >= 0.05 for r in budget["families"]["random_n_minus_random1"]),
    "Holm-positive set is exactly T-cell r1 cap2/4/8/16, T-cell r2 cap4, and HepG2 cap2; no Random(N)-Random(1) row survives",
)
add(
    "v5_budget_protocol_and_existing_rounds",
    budget["new_blind_test"] is False
    and budget_config["random_n"]["bit_generator"] == "PCG64"
    and budget_config["random_n"]["spawn_key"] == ["context_index", "pool_index", "seed_index", "cap_index"]
    and budget_config["bootstrap"]["replicates"] == 10000
    and budget_config["include_cap1_structural_zero"] is True
    and load_json(budget_root / "validation/independent_round3.json")["status"] == "PASS",
    "Record-specific PCG64 child streams, 10,000 complete-pool bootstraps, all-20 cap1 inclusion, and existing independent raw checks confirmed",
)

# V5 sequence fixed-effect and family sensitivities; no causal interpretation.
sequence_root = SOURCES / "paper1_sequence_ablation_v1"
sequence = load_json(sequence_root / "analysis/results/summary.json")
sequence_config = load_json(sequence_root / "frozen/frozen_analysis.json")
sequence_primary_tsv = load_table(sequence_root / "analysis/results/primary_paired.tsv", "\t")
sequence_family_tsv = load_table(sequence_root / "analysis/results/sequence_family_sensitivity.tsv", "\t")
sequence_pools = load_table(sequence_root / "analysis/results/pool_effects.tsv", "\t")
sequence_primary_ok = len(sequence["primary"]) == len(sequence_primary_tsv) == 20 and len(sequence_pools) == 320
for row, tsv in zip(sequence["primary"], sequence_primary_tsv):
    sequence_primary_ok &= row["context"] == tsv["context"] and int(row["cap"]) == int(tsv["cap"])
    for field in ("unadjusted_effect", "unadjusted_ci_low", "unadjusted_ci_high", "adjusted_effect", "adjusted_ci_low", "adjusted_ci_high", "adjusted_holm_p"):
        sequence_primary_ok &= close(row[field], tsv[field])
    selected = [r for r in sequence_pools if r["context"] == row["context"] and int(r["cap"]) == row["cap"]]
    sequence_primary_ok &= len(selected) == 16 and close(mean(r["mean_delta_regret"] for r in selected), row["unadjusted_effect"])
add("v5_sequence_primary_tsv_json_pool_recompute", sequence_primary_ok, "20 primary TSV/JSON rows match; all 20 unadjusted effects independently recomputed from 320 pool rows")
sequence_holm_ok = all(
    close(holm_adjust(sequence["primary"], "adjusted_p")[(r["context"], int(r["cap"]))], r["adjusted_holm_p"])
    for r in sequence["primary"]
)
sequence_positive = {(r["context"], int(r["cap"])) for r in sequence["primary"] if float(r["adjusted_holm_p"]) < 0.05}
sequence_nominal = {(r["context"], int(r["cap"])) for r in sequence["primary"] if float(r["adjusted_ci_high"]) < 0}
add(
    "v5_sequence_adjusted_attenuation",
    sequence_holm_ok
    and not sequence_positive
    and sequence_nominal == {("defined_tcell_r1", 4), ("defined_tcell_r1", 8)}
    and next(r for r in sequence["primary"] if r["context"] == "defined_tcell_r2" and int(r["cap"]) == 4)["adjusted_effect"] > 0,
    "Adjusted all-20 Holm independently recomputed; no survivor, only T-cell r1 cap4/8 nominal negative intervals, r2 cap4 sign reversal retained",
)
sequence_family_ok = len(sequence["sequence_family_sensitivity"]) == len(sequence_family_tsv) == 40
for row, tsv in zip(sequence["sequence_family_sensitivity"], sequence_family_tsv):
    sequence_family_ok &= row["family"] == tsv["family"] and row["context"] == tsv["context"] and int(row["cap"]) == int(tsv["cap"])
    for field in ("effect", "ci_low", "ci_high", "p", "holm_p"):
        sequence_family_ok &= close(row[field], tsv[field])
family_holm_ok = True
for family in ("no_uAUG", "no_polyU_upper_quartile"):
    rows = [r for r in sequence["sequence_family_sensitivity"] if r["family"] == family]
    recomputed = holm_adjust(rows)
    family_holm_ok &= len(rows) == 20 and all(close(recomputed[(r["context"], int(r["cap"]))], r["holm_p"]) for r in rows)
family_positive = {(r["family"], r["context"], int(r["cap"])) for r in sequence["sequence_family_sensitivity"] if float(r["holm_p"]) < 0.05}
expected_family_positive = {("no_polyU_upper_quartile", "defined_tcell_r1", 4), ("no_polyU_upper_quartile", "defined_tcell_r2", 8)}
add(
    "v5_sequence_families_and_noncausal_boundary",
    sequence_family_ok and family_holm_ok and family_positive == expected_family_positive
    and sequence["new_blind_test"] is False and sequence["causal_claim"] is False
    and sequence_config["poly_u_q75_nearest_rank"] == 3
    and sequence_config["poly_u_upper_quartile_candidate_count"] == 1591
    and sequence_config["features"] == ["gc", "uAUG", "longest_poly_u", "support"]
    and sequence_config["bootstrap"]["replicates"] == 10000
    and load_json(sequence_root / "verification/round1_raw_fields.json")["pass"] is True
    and load_json(sequence_root / "verification/round2_independent.json")["pass"] is True
    and load_json(sequence_root / "verification/round3_local_delivery.json")["pass"] is True,
    "40 family rows and both all-20 Holm families match; exactly two polyU-excluded survivors and no uAUG survivor; existing raw/independent/delivery checks PASS; causal_claim=false",
)

score_pool = load_table(FIGURES / "strict_h1_figure3_pool_effects.csv")
score_summary = load_table(FIGURES / "strict_h1_figure3_score_matching.csv")
score_ok = True
for row in score_summary:
    pool_rows = [r for r in score_pool if r["context"] == row["context"]]
    score_ok &= len(pool_rows) == 16
    for aggregate_field, pool_field in (
        ("optimized_mean_abs_error", "optimized_mean_abs_error"),
        ("matched_mean_abs_error", "matched_mean_abs_error"),
        ("difference", "paired_optimized_minus_matched"),
    ):
        score_ok &= close(mean(r[pool_field] for r in pool_rows), row[aggregate_field])
add("score_matching_recompute", score_ok, "12 means recomputed from 64 pool rows")
add("score_matching_null_boundary", all(includes_zero((r["ci_low"], r["ci_high"])) for r in score_summary), "All four intervals include zero; directions are mixed")

arch_pool = load_table(FIGURES / "strict_h1_figure4_pool_effects.csv")
arch_summary = load_table(FIGURES / "strict_h1_figure4_cross_architecture_summary.csv")
arch_ok = True
for row in arch_summary:
    pool_rows = [r for r in arch_pool if r["context"] == row["context"]]
    arch_ok &= len(pool_rows) == 16
    for field in (
        "cnn_mean_abs_error", "kmer_mean_abs_error", "paired_cnn_minus_kmer",
        "architecture_error_correlation",
    ):
        arch_ok &= close(mean(r[field] for r in pool_rows), row[field])
add("cross_architecture_recompute", arch_ok, "16 means recomputed from 64 pool rows")
add(
    "cross_architecture_claim_boundary",
    all(float(r["architecture_error_correlation"]) > 0.87 for r in arch_summary)
    and all(not includes_zero((r["difference_ci_low"], r["difference_ci_high"])) for r in arch_summary),
    "Error correlations 0.878–0.962 and all paired MAE intervals exclude zero; no causal apportionment is computed",
)

historical = load_table(FIGURES / "figure2_pressure.csv")
designed = sorted(
    [r for r in historical if r["context"] == "Sample designed"],
    key=lambda r: int(r["proposal_cap"]),
)
random_sample = [r for r in historical if r["context"] == "Sample random"]
add(
    "sample_historical_context",
    [int(r["proposal_cap"]) for r in designed] == [1, 2, 4, 8, 16]
    and all(float(a["mean_regret"]) > float(b["mean_regret"]) for a, b in zip(designed, designed[1:]))
    and len(random_sample) == 5
    and all(int(r["outer_pools"]) == 3 for r in random_sample),
    "Designed regret decreases 1.905→0.681; random has only three eligible pools and is non-monotone",
)

vgsc = load_table(FIGURES / "extended_data_vgsc.csv")
fp = next(r for r in vgsc if r["comparison"] == "FP-REJ without Q minus BoN")
add("vgsc_falsification", close(fp["difference"], 0.34315679237874996) and float(fp["ci_low"]) > 0, "Designed FP-REJ is worse than BoN: +0.343 [0.225, 0.467]")
noise = load_table(FIGURES / "extended_data_noise.csv")
noise16 = sorted(
    [r for r in noise if int(r["proposal_cap"]) == 16],
    key=lambda r: float(r["noise_sigma"]),
)
add("noise_nonmonotone", [round(float(r["mean_regret"]), 3) for r in noise16] == [6.519, 6.018, 6.143, 6.205, 6.526], "No monotone degradation; sigma-zero rows retained")

# Paper 2: configuration, paired effects, Q discrimination, and exploratory coverage.
baseline_designed = load_json(SOURCES / "paper2_baselines_v1/outer_designed_summary.json")
baseline_random = load_json(SOURCES / "paper2_baselines_v1/outer_random_summary.json")
methods = {"bon", "lcb", "distance", "com", "diversity_lcb"}
configs16 = {
    r["method"]: r for r in baseline_designed["configurations"]
    if int(r["proposal_cap"]) == 16
}
add(
    "paper2_equal_cnn_budget",
    set(configs16) == methods
    and all(int(r["mean_cnn_member_item_forwards"]) == 48 for r in configs16.values()),
    "All five designed cap-16 selectors use 48 CNN member item-forwards",
)
designed_top1 = [
    r for r in baseline_designed["paired_comparisons"]
    if r["metric"] == "regret" and r["method"] in {"com", "diversity_lcb"}
]
add("paper2_no_designed_advantage", all(includes_zero(r["pool_cluster_bootstrap_95_percent_interval"]) for r in designed_top1), "COM-style and diversity-LCB top-1 intervals include zero")
random_com = next(
    r for r in baseline_random["paired_comparisons"]
    if r["metric"] == "regret" and r["method"] == "com" and r["reference"] == "bon"
)
add("paper2_random_com_worse", random_com["mean_difference"] > 0 and random_com["pool_cluster_bootstrap_95_percent_interval"][0] > 0, "+1.811 [0.123, 2.660]; three-pool secondary result")
add(
    "paper2_retrospective_status",
    "retrospective" in baseline_designed["evidence_status"]
    and baseline_designed["labels_returned_to_learner"] is False,
    "Previously viewed outer sensitivity evidence, not a new blind test",
)
coverage = {float(r["coverage"]): r for r in baseline_designed["risk_coverage"] if r["method"] == "bon"}
add(
    "uncertainty_old_outer_retrospective_only",
    close(coverage[0.1]["mean_regret"], 0.4589022477825003)
    and close(coverage[1.0]["mean_regret"], 0.6723771959983751)
    and all("not a deployment threshold" in r["threshold_source"] for r in coverage.values()),
    "Old outer empirical quantiles remain retrospective sensitivity results, not blind or deployment-threshold evidence",
)
frozen_risk = load_json(SOURCES / "paper2_risk_frozen_v1/frozen_protocol.json")
selected_dev = next(
    r for r in frozen_risk["development_grid"]
    if close(r["quantile"], frozen_risk["selected_quantile"])
)
add(
    "paper2_frozen_dev_lock",
    frozen_risk["threshold_locked"] is True
    and close(frozen_risk["selected_quantile"], 0.50)
    and close(frozen_risk["selected_threshold"], 0.10402689717871444)
    and close(selected_dev["threshold"], frozen_risk["selected_threshold"])
    and close(selected_dev["dev"]["coverage"], 0.500),
    "Development selected q=0.50 and froze threshold 0.10402689717871444",
)
cal = frozen_risk["calibration"]
gate = frozen_risk["calibration_gate"]
add(
    "paper2_frozen_calibration_failure",
    cal["decisions"] == 80
    and cal["accepted_decisions"] == 34
    and cal["accepted_pools"] == 15
    and close(cal["coverage"], 0.425)
    and close(cal["accepted_mean_regret"], 2.8668831721861756)
    and close(cal["full_bon_mean_regret"], 2.8614633527372497)
    and close(cal["regret_reduction"], -0.005419819448925889)
    and close(cal["regret_reduction"], cal["full_bon_mean_regret"] - cal["accepted_mean_regret"])
    and close(cal["regret_reduction_95ci"][0], -0.5092712668595158)
    and close(cal["regret_reduction_95ci"][1], 0.6286230065478566)
    and includes_zero(cal["regret_reduction_95ci"])
    and gate["pass"] is False
    and gate["checks"]["positive_point_reduction"] is False
    and gate["checks"]["reduction_ci_excludes_zero"] is False,
    "Calibration coverage=0.425; accepted=2.866883 versus full-BoN=2.861463; reduction=-0.005420 [-0.509271,0.628623]; both effect gates fail",
)
risk_rounds = [
    load_json(SOURCES / f"paper2_risk_frozen_v1/verification_round{i}_{suffix}.json")
    for i, suffix in ((1, "raw"), (2, "independent"), (3, "delivery"))
]
outer_refusal = load_text(SOURCES / "paper2_risk_frozen_v1/outer_refused.log")
add(
    "paper2_outer_hard_lock_not_scored",
    frozen_risk["outer_scoring_authorized"] is False
    and frozen_risk["outer_scored"] is False
    and all(r["outer_scored"] is False for r in risk_rounds)
    and all(r["pass"] is True for r in risk_rounds)
    and "PermissionError" in outer_refusal
    and "calibration gate has not authorized outer scoring" in outer_refusal,
    "All frozen verification rounds confirm outer_scored=false; score-outer was refused by PermissionError",
)
add(
    "paper2_frozen_evidence_boundary",
    "previously viewed" in frozen_risk["evidence_boundary"]
    and "not an untouched blind test" in frozen_risk["evidence_boundary"]
    and frozen_risk["primary_endpoint"] == "full-BoN mean regret minus accepted mean regret; positive favors ensemble-std abstention",
    "Historical outer curves were previously viewed and cannot be promoted to untouched blind evidence",
)
risk_pairs = load_table(SOURCES / "paper2_risk_prediction/pool_risk_regret_pairs.csv")
risk_analysis = load_json(SOURCES / "paper2_risk_prediction/risk_analysis_v2.json")
for role in ("dev", "calibration"):
    rows = [r for r in risk_pairs if r["role"] == role]
    q = [float(r["q_probability"]) for r in rows]
    regret = [float(r["bon_cap16_regret"]) for r in rows]
    rho = pearson(ranks(q), ranks(regret))
    expected = risk_analysis["roles"][role]["tests"]["q_vs_bon"]["spearman"]
    add(f"q_{role}_spearman_recompute", close(rho, expected), f"n={len(rows)}, rho={rho:.12f}")
calibration_coverage = risk_analysis["roles"]["calibration"]["q_vs_bon_risk_coverage"]
add(
    "q_abstention_negative",
    all(float(r["reduction_vs_random"]) < 0 for r in calibration_coverage if float(r["abstain_fraction"]) > 0),
    "Calibration Q-based abstention does not reduce accepted regret",
)

# Schlusser: frozen primary and three-round identity.
schlusser_rounds = [
    load_json(SOURCES / f"schlusser2024_hepg2_te_confirmatory_v1/evaluator_round_{i}/summary.json")
    for i in (1, 2, 3)
]
primary = schlusser_rounds[0]["primary"]
paired = primary["paired_comparison"]
add(
    "schlusser_primary_null",
    close(primary["configuration"]["mean_regret"], 4864.008158531935)
    and close(paired["mean_regret_difference_left_minus_right"], 0.6266078320630186)
    and includes_zero(paired["gene_cluster_bootstrap_95_interval"])
    and includes_zero(paired["transcript_cluster_bootstrap_95_interval"]),
    "No evidence of BoN-CNN TE improvement; neither a positive BoN effect nor a precise Random advantage is claimed",
)
add(
    "schlusser_endpoint_alignment",
    close(schlusser_rounds[0]["endpoint_alignment_descriptive"]["spearman_cnn_MRL_vs_TE"], 0.0720729, 1e-7)
    and schlusser_rounds[0]["deployment_mismatch"] == "cross_endpoint_not_same_endpoint_MRL_replication",
    "MRL–TE Spearman 0.072; not same-endpoint replication or protein-output validation",
)
add(
    "schlusser_three_rounds",
    all(json.dumps(r, sort_keys=True) == json.dumps(schlusser_rounds[0], sort_keys=True) for r in schlusser_rounds[1:]),
    "All three evaluator summaries are semantically identical",
)

# Leppek: recompute from construct-level measured fields, including ranks and top-k.
leppek_rows = load_table(SOURCES / "leppek2022_cds_mechanism_audit_v5/cds_construct_endpoints.tsv", "\t")
leppek_summary = load_json(SOURCES / "leppek2022_cds_mechanism_audit_v5/analysis_summary.json")
half_life = [float(r["cds_in_solution_half_life"]) for r in leppek_rows]
expression6 = [float(r["nluc_fluc_6h"]) for r in leppek_rows]
expected_corr = next(
    r for r in leppek_summary["correlations"]
    if r["left_endpoint"] == "cds_in_solution_half_life" and r["right_endpoint"] == "nluc_fluc_6h"
)
add(
    "leppek_correlation_from_fields",
    len(leppek_rows) == 24
    and close(pearson(half_life, expression6), expected_corr["pearson_r"])
    and close(pearson(ranks(half_life), ranks(expression6)), expected_corr["spearman_rho"]),
    "Pearson −0.192 and Spearman −0.307 recomputed from 24 construct rows",
)
reversal_count = sum((a - b) * (c - d) < 0 for a, b, c, d in (
    (half_life[i], half_life[j], expression6[i], expression6[j])
    for i, j in combinations(range(24), 2)
))
rank_half = sorted(range(24), key=lambda i: half_life[i], reverse=True)
rank_expr = sorted(range(24), key=lambda i: expression6[i], reverse=True)
overlap = len(set(rank_half[:6]) & set(rank_expr[:6]))
expected_rank = next(
    r for r in leppek_summary["rank_summaries"]
    if r["left_endpoint"] == "cds_in_solution_half_life" and r["right_endpoint"] == "nluc_fluc_6h"
)
add(
    "leppek_reversals_from_fields",
    reversal_count == 171 and overlap == 1
    and reversal_count == expected_rank["discordant_pairs"]
    and overlap == expected_rank["top_k_overlap_n"],
    "171/276 discordant pairs and top-6 overlap 1/6 recomputed independently",
)
add(
    "leppek_descriptive_only",
    leppek_summary["outer_results_accessed"] is False
    and leppek_summary["model_training_or_inference"] is False
    and leppek_summary["predicted_expression_used_as_measured"] is False,
    "Exploratory descriptive audit; predicted expression never treated as measured",
)

# PERSIST: independent-study same-endpoint retrospective external evaluation.
persist_root = SOURCES / "persist_mrl_external_v1"
persist_summary = load_json(persist_root / "results/summary.json")
persist_inclusion = load_json(persist_root / "protocol/inclusion_audit.json")
persist_predictions = load_table(persist_root / "predictions/predictions.tsv", "\t")
persist_labels = load_table(persist_root / "protocol/evaluator_labels.tsv", "\t")
persist_eligible = load_table(persist_root / "protocol/eligible_sequences.tsv", "\t")
persist_excluded = load_table(persist_root / "protocol/excluded_constructs.tsv", "\t")
persist_per_seed = load_table(persist_root / "results/per_seed.tsv", "\t")
persist_cuda = load_json(persist_root / "predictions/cuda_inference.json")
persist_report = load_text(DOCS / "PERSIST_MRL_EXTERNAL_V1.md")
persist_ids = {row["sequence_id"] for row in persist_eligible}
persist_by_id = {row["sequence_id"]: row for row in persist_predictions}
labels_by_id = {row["sequence_id"]: row for row in persist_labels}
add(
    "persist_frozen_cohort_and_isolation",
    len(persist_eligible) == len(persist_predictions) == len(persist_labels) == 59
    and len(persist_ids) == 59
    and len({row["utr"] for row in persist_eligible}) == 49
    and persist_ids == set(persist_by_id) == set(labels_by_id)
    and all(persist_by_id[key]["utr"] == labels_by_id[key]["utr"] for key in persist_ids)
    and not persist_excluded
    and persist_inclusion["frozen_group_rows"] == persist_inclusion["remaining_constructs"] == 59
    and persist_inclusion["frozen_group_unique_utrs"] == persist_inclusion["remaining_utr_clusters"] == 49
    and persist_inclusion["exact_50nt_constructs"] == persist_inclusion["excluded_constructs"] == persist_inclusion["excluded_utr_clusters"] == 0
    and persist_inclusion["conflicts"] == {},
    "59 constructs/49 unique UTRs; exact and equal-length Hamming-1 conflicts are zero; no exclusions",
)
prediction_values = [float(row["prediction"]) for row in persist_predictions]
mrl_values = [float(labels_by_id[row["sequence_id"]]["mrl"]) for row in persist_predictions]
members_ok = all(
    math.isclose(mean(float(row[f"member_{seed}"]) for seed in (101, 202, 303)), float(row["prediction"]), rel_tol=1e-6, abs_tol=1e-6)
    for row in persist_predictions
)
persist_secondary = persist_summary["secondary"]
add(
    "persist_prediction_and_correlation_recompute",
    members_ok
    and close(pearson(ranks(prediction_values), ranks(mrl_values)), persist_secondary["spearman"]["value"])
    and close(pearson(prediction_values, mrl_values), persist_secondary["pearson"]["value"])
    and close(mean(abs(x - y) for x, y in zip(prediction_values, mrl_values)), persist_secondary["mae"]["value"]),
    f"Three-member means, Spearman={persist_secondary['spearman']['value']:.12f}, Pearson and MAE recomputed from 59 constructs",
)
seeds = [int(row["algorithm_seed"]) for row in persist_per_seed]
delta_values = [float(row["delta"]) for row in persist_per_seed]
random_values = [float(row["random_regret"]) for row in persist_per_seed]
bon_values = [float(row["bon_regret"]) for row in persist_per_seed]
persist_primary = persist_summary["primary"]
add(
    "persist_primary_from_200_paired_seeds",
    seeds == list(range(20261125, 20261325))
    and all(close(float(row["bon_regret"]) - float(row["random_regret"]), row["delta"]) for row in persist_per_seed)
    and close(mean(delta_values), persist_primary["bon_minus_random_regret"])
    and close(mean(random_values), persist_secondary["random_top1_regret"]["value"])
    and close(mean(bon_values), persist_secondary["bon_top1_regret"]["value"])
    and close(mean(float(row["random_oracle_coverage"]) for row in persist_per_seed), persist_secondary["random_oracle_coverage"]["value"])
    and close(mean(float(row["bon_oracle_coverage"]) for row in persist_per_seed), persist_secondary["bon_oracle_coverage"]["value"])
    and persist_primary["bon_minus_random_regret"] > 0
    and persist_primary["significant_benefit"] is False,
    "200 paired seeds reproduce +0.10472255085 BoN-minus-Random regret and both regret/coverage means; no BoN benefit",
)
bootstrap_shape, bootstrap_values = load_npy_float64(persist_root / "results/bootstrap_replicates.npy")
bootstrap_columns = [bootstrap_values[index::8] for index in range(8)]
bootstrap_targets = [
    persist_primary,
    persist_secondary["spearman"],
    persist_secondary["pearson"],
    persist_secondary["mae"],
    persist_secondary["random_top1_regret"],
    persist_secondary["bon_top1_regret"],
    persist_secondary["random_oracle_coverage"],
    persist_secondary["bon_oracle_coverage"],
]
bootstrap_ok = bootstrap_shape == (10000, 8)
for column, target in zip(bootstrap_columns, bootstrap_targets):
    expected = target["ci95"]
    bootstrap_ok &= close(percentile(column, 0.025), expected[0])
    bootstrap_ok &= close(percentile(column, 0.975), expected[1])
sign_p = min(1.0, 2 * (sum(value <= 0 for value in bootstrap_columns[0]) + 1) / (len(bootstrap_columns[0]) + 1))
add(
    "persist_bootstrap_intervals_and_sign_p_recompute",
    bootstrap_ok
    and close(sign_p, persist_primary["p_two_sided_sign"])
    and includes_zero(persist_primary["ci95"])
    and persist_primary["p_two_sided_sign"] >= 0.05,
    "All eight 10,000-replicate UTR-cluster percentile intervals and two-sided sign p=0.245175 recomputed from NPY",
)
persist_rounds = [
    load_json(persist_root / "verification/round1_execution.json"),
    load_json(persist_root / "verification/round2_independent.json"),
    load_json(persist_root / "verification/round3_delivery.json"),
    load_json(persist_root / "delivery_verification.json"),
]
add(
    "persist_cuda_and_existing_three_round_verification",
    persist_cuda["actual_device"] == "cuda:0"
    and persist_cuda["cpu_fallback"] is False
    and persist_cuda["labels_read"] is False
    and persist_cuda["deterministic_algorithms"] is True
    and persist_cuda["tf32"] is False
    and len(persist_cuda["members"]) == 3
    and all(member["item_forwards"] == 59 for member in persist_cuda["members"])
    and all(record["status"] == "PASS" for record in persist_rounds)
    and close(persist_rounds[1]["primary_recomputed"], persist_primary["bon_minus_random_regret"])
    and persist_rounds[1]["bootstrap_replicates_recomputed"] == 10000,
    "Frozen three-member variable-length MLP inference used deterministic CUDA without fallback; original raw/independent/delivery checks PASS",
)
add(
    "persist_evidence_boundary_and_negative_result",
    persist_summary["constructs"] == 59
    and persist_summary["utr_clusters"] == 49
    and persist_summary["algorithm_seeds"] == 200
    and persist_summary["negative_results_retained"] is True
    and persist_summary["outer_tuning"] is False
    and "not a blind test" in persist_summary["evidence_boundary"]
    and "不是盲测或前瞻性验证" in persist_report
    and "+0.104723" in persist_report
    and "−0.100964 至 0.212706" in persist_report
    and "p=0.245175" in persist_report
    and "−0.008447" in persist_report,
    "Retrospective independent-study same-endpoint evaluation, not blind/prospective; negative uncertain result retained",
)

# V5 PERSIST source-confirmed tie rule and SHA256 sensitivity.
tie_root = ROOT / "persist_mrl_external_v1/tie_audit_v1_1"
tie_summary = load_json(tie_root / "results/summary_v1_1.json")
tie_comparison = load_json(tie_root / "results/comparison_v1_vs_v1_1.json")
tie_delivery = load_json(tie_root / "verification/round3_delivery.json")
tie_hashes = load_json(tie_root / "frozen_code_hashes.json")
tie_source = load_text(tie_root / "source_snapshot/persist_mrl.py")
tie_affected = load_table(tie_root / "results/affected_seeds.tsv", "\t")
tie_report = load_text(ROOT / "PERSIST_TIE_AUDIT_V1_1.md")
tie_primary_same = all(
    close(tie_comparison["v1"][field], tie_comparison["v1_1"][field])
    for field in ("bon_minus_random_regret", "p_two_sided_sign")
) and all(close(a, b) for a, b in zip(tie_comparison["v1"]["ci95"], tie_comparison["v1_1"]["ci95"]))
add(
    "v5_persist_tie_source_and_primary_invariance",
    "sorted(candidates, key=lambda i: (-scores[i], ids[i], int(i)))[0]" in tie_source
    and tie_hashes["actual_v1_tie_rule"]["preregistered"] is True
    and tie_hashes["actual_v1_tie_rule"]["documentation_gap"] is False
    and tie_hashes["files"]["persist_mrl.py"]["sha256"] == "94f667185a48abd035fc757c93d0fef6e455e4e1bc4d4bd981d0b56f999334da"
    and tie_summary["exact_tie_seed_count"] == 9
    and tie_summary["order_sensitive_seed_count"] == 5
    and len(tie_affected) == 5
    and all(row["selection_changed"] == "false" and close(row["v1_delta"], row["v1_1_delta"]) for row in tie_affected)
    and tie_primary_same
    and close(tie_comparison["v1_1_minus_v1_point"], 0.0)
    and close(tie_summary["primary"]["bon_minus_random_regret"], persist_primary["bon_minus_random_regret"])
    and tie_summary["version_selection_by_direction"] is False,
    "Source confirms prediction descending/Sequence ID/row order; five order-sensitive original selections and effect/CI/p are unchanged under SHA256 alternative",
)
add(
    "v5_persist_tie_adverse_details_and_history_retained",
    tie_summary["bootstrap_seed_selection_changes"] == 79
    and tie_summary["bootstrap_changed_effect_replicates"] == 78
    and tie_comparison["bootstrap_max_absolute_change"] > 0
    and tie_delivery["status"] == "PASS"
    and tie_delivery["original_v1_data_or_results_changed"] is False
    and "delivery_verification.json" in tie_delivery["original_v1_preexisting_manifest_mismatches"]
    and "初次检查" in tie_report
    and "79" in tie_report and "78" in tie_report,
    "Bootstrap changes and the pre-existing delivery_verification.json manifest mismatch remain disclosed; original scientific data/results unchanged",
)

# Paper 3 V4: all 48 gates, curve means, numerical convergence, and outer isolation.
v4 = load_json(SOURCES / "paper3_headroom_v4/headroom_v4_validation/aggregate.json")
case_ok = len(v4["cases"]) == 48
for case in v4["cases"]:
    case_ok &= close(
        case["repair_headroom"],
        float(case["zero_label_regret"]) - float(case["full_q_regret"]),
    )
add("paper3_all_gate_arithmetic", case_ok, "48/48 zero/full-Q headroom values recomputed")
representation_ok = True
recomputed_counts = {}
for representation in ("summary4", "kmer4", "cnn64"):
    cases = [c for c in v4["cases"] if c["representation"] == representation]
    recomputed_counts[representation] = sum(c["gate_pass"] for c in cases)
    summary = next(r for r in v4["representation_summary"] if r["representation"] == representation)
    representation_ok &= len(cases) == 16
    representation_ok &= recomputed_counts[representation] == summary["headroom_present_count"]
    representation_ok &= close(mean(c["repair_headroom"] for c in cases), summary["headroom_mean"])
    representation_ok &= close(max(c["repair_headroom"] for c in cases), summary["headroom_max"])
add("paper3_representation_summary", representation_ok and recomputed_counts == {"summary4": 0, "kmer4": 5, "cnn64": 1}, f"Positive gates: {recomputed_counts}; kmer4 maximum 3.686 and cnn64 maximum 5.935")

curve_records = v4["curve_records"]
curve_ok = True
curve_lookup = {}
for summary in v4["curve_summary"]:
    rep, strategy, budget = summary["representation"], summary["strategy"], int(summary["budget"])
    curve_lookup[(rep, strategy, budget)] = float(summary["pool_mean_regret_mean"])
    selected = [r for r in curve_records if r["representation"] == rep and r["strategy"] == strategy and int(r["budget"]) == budget]
    pool_values = defaultdict(list)
    for record in selected:
        pool_values[record["query_pool_id"]].append(float(record["regret"]))
    recomputed = mean(mean(values) for values in pool_values.values())
    curve_ok &= close(recomputed, summary["pool_mean_regret_mean"])
add("paper3_curve_recompute", curve_ok, "72 pooled representation/strategy/budget cells recomputed from 288 curve records and pool means")
expected_curves = {
    ("kmer4", "random", 16): 3.497016,
    ("kmer4", "max_variance", 16): 2.644595,
    ("kmer4", "expected_improvement", 16): 1.220084,
    ("kmer4", "regret_analytic", 16): 2.131689,
    ("kmer4", "max_variance", 32): 2.068558,
    ("kmer4", "regret_analytic", 32): 2.131689,
}
add(
    "paper3_simple_ei_and_no_dominance",
    all(close(curve_lookup[key], value, 1e-6) for key, value in expected_curves.items())
    and v4["acquisition_superiority_claimed"] is False,
    "EI@16=1.220 < random 3.497 and regret-analytic 2.132; max-variance is best at 32 labels",
)
convergence = v4["regret_convergence_summary"]
add(
    "paper3_v4_numerical_not_biological",
    convergence["acquisition_step_count"] == 384
    and convergence["converged_count"] == 384
    and v4["posterior_dependence"] == "diagonal_independent_fantasy_posterior_approximation",
    "384/384 numerical convergence concerns the diagonal-independent approximation, not acquisition efficacy",
)
add(
    "paper3_outer_untouched",
    v4["outer_labels_read"] is False
    and v4["outer_metrics_computed"] is False
    and "not a new blind test" in v4["scope"],
    "Development/calibration diagnostics only; no permanent outer confirmation",
)
v4_rounds = load_json(SOURCES / "paper3_headroom_v4/headroom_v4_validation/three_round_validation.json")
add(
    "paper3_three_rounds",
    v4_rounds["scientific_results_identical"] is True
    and len(set(v4_rounds["scientific_sha256_each"])) == 1
    and v4_rounds["case_count_each"] == [48, 48, 48],
    "Three scientific hashes are identical across all 48 cases",
)

# Delivery text gates run only after the V5 manuscripts and narrative documents exist.
paper_paths = [
    PAPER_DIR / "UNIFIED_PREPRINT_MAIN_V5_20261126.md",
    PAPER_DIR / "UNIFIED_PREPRINT_SUPPLEMENT_S2_V5_20261126.md",
    PAPER_DIR / "UNIFIED_PREPRINT_SUPPLEMENT_S3_V5_20261126.md",
]
story_path = DOCS / "SCIENTIFIC_STORY_V5_20261126.md"
ledger_path = DOCS / "UNIFIED_PREPRINT_CLAIM_EVIDENCE_LEDGER_V5_20261126.md"
if all(path.exists() for path in paper_paths + [story_path, ledger_path]):
    texts = {path.name: load_text(path) for path in paper_paths + [story_path, ledger_path]}
    combined = "\n".join(texts.values())
    normalized_combined = re.sub(r"\s+", "", combined).lower()
    main = texts[paper_paths[0].name]
    s2 = texts[paper_paths[1].name]
    s3 = texts[paper_paths[2].name]
    story = texts[story_path.name]
    ledger = texts[ledger_path.name]
    required_main = [
        "## Abstract", "## 1. Introduction", "## 3. Results", "## 4. Discussion",
        "## 5. Limitations", "## 6. Methods", "## Figure legends", "## References",
        "retrospective", "not a new blind test", "no new wet-lab",
    ]
    add("delivery_main_structure", all(token.lower() in main.lower() for token in required_main), "Required preprint sections and evidence boundaries are present")
    add(
        "delivery_frozen_calibration_values",
        all(token in main and token in s2 for token in (
            "q = 0.50", "0.104026897", "0.425", "2.866883", "2.861463",
            "−0.005420", "[−0.509271, 0.628623]",
        )),
        "Main and S2 contain the selected development lock and every required calibration value",
    )
    add(
        "delivery_frozen_gate_and_outer_lock",
        "retrospective signal did not pass the frozen calibration gate" in s2
        and "outer_scoring_authorized=false" in s2
        and "outer_scored=false" in s2
        and "PermissionError" in s2
        and "Outer was not scored" in main
        and "hard lock" in main,
        "V5 states calibration failure, hard-lock refusal, and absence of outer scoring",
    )
    add(
        "delivery_old_outer_not_blind",
        "do not constitute blind evidence" in main
        and "not blind evidence" in s2
        and "旧 outer 经验分位点不构成盲态证据" in story,
        "Historical outer empirical quantiles are explicitly excluded as blind evidence",
    )
    add(
        "delivery_narrative_roles",
        "strongest current article is the paper 1 phenomenon/mechanism study" in main.lower()
        and "study of limitations and falsification" in s2
        and "standalone diagnostic supplement" in s3
        and "主文：Paper 1 现象/机制" in story
        and "S2：限制与证伪" in story
        and "S3：诊断" in story,
        "Paper 1 phenomenon/mechanism is the main article; S2 is limitations/falsification and S3 is diagnostic",
    )
    add(
        "delivery_chinese_narrative_and_ledger",
        all("回顾性信号未通过冻结calibration门禁" in doc for doc in (story, ledger))
        and all("outer 硬锁未评分" in doc for doc in (story, ledger))
        and all("PAPER2_RISK_FROZEN_V1.md" in doc for doc in (story, ledger))
        and all(token in story and token in ledger for token in (
            "0.104026897", "0.425", "2.866883", "2.861463", "−0.005420", "−0.509271,0.628623",
        )),
        "Chinese story and ledger carry the frozen negative conclusion and numerical anchors",
    )
    add(
        "delivery_persist_values_and_contract_boundary",
        all(token in main for token in (
            "59 eligible constructs spanning 49 unique 5′ UTRs", "+0.104723",
            "[−0.100964, 0.212706]", "p=0.245175", "−0.008447",
            "frozen variable-length 4-mer MLP", "not a blind test",
            "does not directly contradict the T-cell", "not proof of universal proxy failure",
            "training lineage, architecture, and construct", "Fig. 13",
        ))
        and all(token in story and token in ledger for token in (
            "59", "49", "exact", "Hamming-1", "+0.104723", "0.245175", "−0.008447",
            "变长", "MLP", "CUDA", "不是盲测", "直接反驳", "普遍失效",
            "训练谱系", "架构", "构建体", "PERSIST_MRL_EXTERNAL_V1.md",
        )),
        "Main, Chinese story, and C13/C14 ledger retain all PERSIST values and training-lineage/architecture/construct boundaries",
    )
    add(
        "delivery_persist_completed_not_missing",
        "completed small retrospective replication" in main
        and "PERSIST 完整 MLP 契约无可检测迁移" in story
        and "PERSIST 完整变长 4-mer MLP 契约无可检测迁移" in ledger
        and "There is no independent-study, same-endpoint MRL replication" not in combined
        and "独立同终点复验仍缺失" not in combined
        and "独立同终点复验尚缺失" not in combined,
        "The completed retrospective replication is described as showing no transferable ranking signal, not as missing",
    )
    forbidden = [
        "剩余阳性风险信号", "待outer确认", "remaining positive risk-control signal",
        "positive risk-control signal", "risk–coverage signal", "risk-coverage signal",
        "signal awaiting confirmation", "awaits a frozen-threshold test",
        "pending uncertainty abstention", "pending risk–coverage", "pending risk-coverage",
        "pending independent risk-control evaluation", "pending external/frozen-threshold confirmation",
        "requires frozen-threshold confirmation", "positive exit", "正结果出口", "阳性出口",
        "There is no independent-study, same-endpoint MRL replication",
        "独立同终点复验仍缺失", "独立同终点复验尚缺失",
        "PERSIST supports BoN benefit", "PERSIST proves universal proxy failure",
    ]
    present_forbidden = [phrase for phrase in forbidden if re.sub(r"\s+", "", phrase).lower() in normalized_combined]
    add("delivery_forbidden_wording_absent", not present_forbidden, f"Forbidden stale wording found: {present_forbidden}" if present_forbidden else "No stale positive-exit or awaiting-outer wording is present")
    add(
        "delivery_key_values",
        all(token in combined for token in ("0.878–0.962", "171/276", "0.626608", "0.458902", "1.220084", "384/384", "+0.104723", "0.245175", "−0.008447")),
        "All prior anchors plus PERSIST effect, sign p value, and Spearman appear in the delivered manuscript set",
    )
    add("delivery_references", main.count("\n[") >= 15, "At least 15 numbered references are present")
    figure_links = []
    for path in paper_paths:
        for link in re.findall(r"`([^`]*figures/[^`]+\.pdf)`", texts[path.name]):
            figure_links.append((path, link, (path.parent / link).resolve().exists()))
    add(
        "delivery_figure_links",
        len(figure_links) == 17 and all(item[2] for item in figure_links),
        f"{sum(item[2] for item in figure_links)}/{len(figure_links)} referenced PDF figures resolve locally",
    )
    add(
        "delivery_negative_results_retained",
        all(token in main for token in ("N50", "HepG2", "Q therefore did not discriminate", "no complex acquisition strategy dominated", "+0.104723", "[−0.100964, 0.212706]", "−0.008447"))
        and "Every strategy and every budget had regret 3.686476" in s3
        and "both effect gates failed" in s2.lower(),
        "N50/HepG2 and PERSIST boundaries, Q failure, frozen calibration failure, acquisition non-dominance, and summary4 stagnation are retained",
    )
    add(
        "delivery_v5_equal_budget_and_composition_claims",
        all(token in main for token in (
            "BoN(N) regret minus same-prefix Random(N) regret", "−0.799949", "−1.448756",
            "−1.579340", "−1.705760", "only cap 4 survived Holm", "0.006399",
            "Holm p=0.055994", "No Random(N)-minus-Random(1) contrast survived",
            "PCG64", "SeedSequence(20261125)", "10,000 complete-pool", "cap-1 structural zeros",
            "−0.215", "−0.486", "−0.484", "−0.569", "+0.043", "no adjusted effect surviving Holm",
            "pool indicators", "longest_poly_u >= 3", "not evidence that uAUG or poly-U causally mediates",
            "Fig. 14", "Fig. 15", "Figure14_v3_equal_budget_value.pdf", "Figure15_v3_sequence_composition_sensitivity.pdf",
        ))
        and all(token in story and token in ledger for token in (
            "PAPER1_BUDGET_MATCHED_V1.md", "PAPER1_SEQUENCE_ABLATION_V1.md", "0.004000",
            "0.006399", "0.055994", "回顾性敏感性", "因果", "−0.005420",
        )),
        "V5 main/story/ledger prioritize equal-budget positive results and retain complete attenuation, family, N50/HepG2, and noncausal boundaries",
    )
    add(
        "delivery_v5_tie_and_independent_gates",
        all(token in main for token in (
            "prediction descending, then ascending `Sequence ID`, then row position",
            "SHA256 alternative", "primary effect, 95% interval, or sign p value",
            "requires its own gate", "PERSIST_TIE_AUDIT_V1_1.md",
        ))
        and all(token in story and token in ledger for token in (
            "PERSIST_TIE_AUDIT_V1_1.md", "prediction descending", "Sequence ID", "row order",
            "SHA256", "主效应", "独立门禁",
        )),
        "Source-confirmed tie rule, SHA256 primary invariance, and independent risk/repair gates appear throughout V5",
    )
    v2_hashes = {
        "UNIFIED_PREPRINT_MAIN_V2_20261125.md": "ea3bc54191941cc42b22adcd352816f64f46b75fa636e5866b1b97b398fb0808",
        "UNIFIED_PREPRINT_SUPPLEMENT_S2_V2_20261125.md": "ed41c49328712ca8b1b8cef5f39ddfdbbd2776bafe9076da8e5eef7499359659",
        "UNIFIED_PREPRINT_SUPPLEMENT_S3_V2_20261125.md": "cdd28e9005aeafe7827249ee440a37bdccb6d111762ac99205c97035007458ef",
    }
    v2_ok = True
    for name, expected_hash in v2_hashes.items():
        path = PAPER_DIR / name
        v2_ok &= path.exists() and hashlib.sha256(load_text(path).encode("utf-8")).hexdigest() == expected_hash
    historical_paths = [
        DOCS / "SCIENTIFIC_STORY_20261124.md",
        DOCS / "UNIFIED_PREPRINT_CLAIM_EVIDENCE_LEDGER_20261125.md",
        DOCS / "UNIFIED_PREPRINT_THREE_ROUND_AUDIT_20261125.md",
        DOCS / "UNIFIED_PREPRINT_SOURCE_VERIFICATION_20261125.json",
    ]
    v2_ok &= all(path.exists() for path in historical_paths)
    add("delivery_v2_history_preserved", v2_ok, "V2 manuscript hashes match the prior audit and historical story/ledger/audit outputs remain present")
    v31_hashes = {
        "UNIFIED_PREPRINT_MAIN_V3.1_20261125.md": "c09d601f5f81bc950f1988c516f89e14161d0bb6b600fe0fb278852f03fd6e4e",
        "UNIFIED_PREPRINT_SUPPLEMENT_S2_V3.1_20261125.md": "a6d4443b3b60ee814b1552e27f21bf81fea5a7a72d838a0ca4bcc6d9841e4700",
        "UNIFIED_PREPRINT_SUPPLEMENT_S3_V3.1_20261125.md": "17c63ec4a33c5e562ee38ed611272007dfc99ee6965ac73dfc7017e6abbe957d",
    }
    v31_ok = True
    for name, expected_hash in v31_hashes.items():
        for path in (PAPER_DIR / name, PAPER_DIR / "history/V3.1_20261125" / name):
            v31_ok &= path.exists() and hashlib.sha256(load_text(path).encode("utf-8")).hexdigest() == expected_hash
    v31_ok &= all((PAPER_DIR / "history/V3.1_20261125" / name).exists() for name in (
        "UNIFIED_PREPRINT_MAIN_V3.1_20261125.pdf",
        "UNIFIED_PREPRINT_SUPPLEMENT_S2_V3.1_20261125.pdf",
        "UNIFIED_PREPRINT_SUPPLEMENT_S3_V3.1_20261125.pdf",
        "UNIFIED_PREPRINT_V3.1_RELEASE_MANIFEST.json",
        "build_unified_preprint_v3_1.py",
    ))
    add("delivery_v31_history_preserved", v31_ok, "Original and archived V3.1 Markdown hashes match; V3.1 PDFs, manifest, and build script remain archived")
    v4_hashes = {
        PAPER_DIR / "UNIFIED_PREPRINT_MAIN_V4_20261125.md": "2181b45a8db8ed6ef8c57130da0c98ff8833205220001eb3c9dcf2d26fa5942b",
        PAPER_DIR / "UNIFIED_PREPRINT_SUPPLEMENT_S2_V4_20261125.md": "99d69f3a71fa27e026ce8bacd91d555bf457467ac3f88566e438c9859e701ac5",
        PAPER_DIR / "UNIFIED_PREPRINT_SUPPLEMENT_S3_V4_20261125.md": "d025cb5adf08facc1f0d0a913e08b2318755119f3feb968e0446940d01bcfdda",
        PAPER_DIR / "build_unified_preprint_v4.py": "f9585743e180e8edd9f935ed3ec0ceeb6ed9ed69b558eedffd8d752847d0b7b4",
        DOCS / "SCIENTIFIC_STORY_V4_20261125.md": "2de290f2685f8199cb452cbac58d6a631981db634a3775f9241bdb36f9b1dd83",
        DOCS / "UNIFIED_PREPRINT_CLAIM_EVIDENCE_LEDGER_V4_20261125.md": "ccc88e1f9f3560f02231e4fdae6dd089a72908f995dbc21e1e10e06ee2f2e304",
        DOCS / "verify_unified_preprint_v4_20261125.py": "d8458056050669305fe5a399709e4513c63364263c57a118cce6293af31e1abc",
    }
    v4_ok = all(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in v4_hashes.items())
    v4_ok &= all(path.exists() for path in (
        PAPER_DIR / "UNIFIED_PREPRINT_MAIN_V4_20261125.pdf",
        PAPER_DIR / "UNIFIED_PREPRINT_SUPPLEMENT_S2_V4_20261125.pdf",
        PAPER_DIR / "UNIFIED_PREPRINT_SUPPLEMENT_S3_V4_20261125.pdf",
        PAPER_DIR / "UNIFIED_PREPRINT_V4_RELEASE_MANIFEST.json",
        DOCS / "UNIFIED_PREPRINT_SOURCE_VERIFICATION_V4_20261125_ROUND1.json",
        DOCS / "UNIFIED_PREPRINT_SOURCE_VERIFICATION_V4_20261125_ROUND2.json",
        DOCS / "UNIFIED_PREPRINT_SOURCE_VERIFICATION_V4_20261125_ROUND3.json",
    ))
    add("delivery_v4_history_preserved", v4_ok, "All V4 Markdown, builder, story, ledger and verifier hashes match; V4 PDFs, manifest and three-round outputs remain present")
else:
    add("delivery_files_present", False, "One or more V5 manuscript or narrative files are missing")

result = {
    "schema_version": "unified-preprint-source-verification-v5",
    "verification_round": args.round,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_configuration": {
        "mode": "read-only independent statistics on local frozen artifacts",
        "statistics_device": "CPU",
        "gpu_required": False,
        "new_training_or_inference": False,
        "remote_access_or_unblinding": False,
        "python": sys.version,
        "platform": platform.platform(),
        "tolerance": TOL,
    },
    "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
    "check_count": len(checks),
    "passed_count": sum(c["pass"] for c in checks),
    "checks": checks,
    "input_sha256": dict(sorted(inputs.items())),
}
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"status={result['status']} checks={result['passed_count']}/{result['check_count']}")
for check in checks:
    print(f"{'PASS' if check['pass'] else 'FAIL'} {check['name']}: {check['detail']}")
print(f"output={OUT}")
sys.exit(0 if result["status"] == "PASS" else 1)
