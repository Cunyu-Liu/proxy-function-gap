#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def number(value):
    return f"{value:.6f}"


def interval(row, key="nominal_95_ci"):
    return f"[{number(row[key][0])}, {number(row[key][1])}]"


def pvalue(value):
    return f"{value:.6f}"


def render(results, audit):
    lines = [
        "# Paper 1 strict-H1 equal-budget Random(N) sensitivity",
        "",
        "## Evidence status",
        "",
        "This is a retrospective sensitivity analysis of the four previously inspected strict-H1 Castillo outer contexts, not a new blind test or prospective confirmation. No model was trained, calibrated, tuned, or run. All new computation was CPU-only reconstruction and statistics using frozen proposals, CNN predictions, selections, and measured labels. Existing strict-H1 results were not overwritten.",
        "",
        "The analysis included all 16 pools per context, original seeds 101/202/303/404/505, and caps 1/2/4/8/16. Random(N) uniformly selected one position from the exact same nested proposal prefix used by BoN(N), using an independent record-specific PCG64 child stream. Negative regret differences favor the first-named strategy. Full selected-ID/regret audits remain remote; local source data are de-identified aggregates.",
        "",
        "## Raw and independent verification",
        "",
        f"All {audit['original_grid_records']} original Random/BoN selection records ({audit['original_paired_rows']} pairs) matched reconstructed selected IDs, indices, and regrets. All {audit['new_random_n_records']} Random(N) selections were independently checked. All {audit['cap1_random_n_id_regret_matches']} cap-1 rows exactly reproduced Random(1), and all 1,600 seed-level decomposition identities passed. Three complete all-20 families retained cap-1 structural zeros with p=1.",
        "",
        "Three real validation rounds were executed: raw reconstruction/tests, a separately implemented independent recomputation, and final delivery/hash verification. Failed environment, test, and numerical-precision attempts are retained under `logs/`; no failure was hidden or converted into a protocol change.",
        "",
        "## Main finding",
        "",
        "At equal candidate budget, proxy ranking shows context-dependent value rather than a universal benefit or harm. All four T-cell r1 cap 2–16 effects are negative and survive that contrast's all-20 Holm correction. T-cell r2 is negative at every cap 2–16, but only cap 4 survives Holm; caps 2, 8, and 16 remain nominal negative results without multiplicity-robust support. HepG2 cap 2 survives Holm, cap 4 is nominally negative but Holm p=0.055994, cap 8 includes zero, and cap 16 is nominally negative but not corrected-significant. N50 has no nominally supported equal-budget effect and is directionally unfavorable at caps 8 and 16.",
        "",
        "No Random(N)−Random(1) contrast survives its all-20 Holm correction. HepG2 cap 2 is nominally positive (higher regret for the Random(2) realization), but this is not corrected-significant. Because every prefix position has the same uniform proposal distribution, increasing N does not change the expectation of this independently randomized contrast; observed nonzero estimates are paired randomization realizations and Monte Carlo variability, not evidence of a systematic candidate-budget effect.",
        "",
        "## Mean measured-MRL regret",
        "",
        "| Context | Cap | BoN(N) | Random(N) | Random(1) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["context_summaries"]:
        lines.append(f"| {row['context']} | {row['cap']} | {number(row['bon_regret'])} | {number(row['random_n_regret'])} | {number(row['random1_regret'])} |")
    for family, title, description in [
        ("bon_minus_random_n", "Equal-budget proxy-ranking value: BoN(N)−Random(N)", "Negative values favor proxy ranking at the same N."),
        ("random_n_minus_random1", "Candidate-budget realization: Random(N)−Random(1)", "Negative values favor the Random(N) realization; all cap-1 rows are structural zeros."),
        ("bon_minus_random1", "Original pressure contrast retained: BoN(N)−Random(1)", "These estimates and nominal intervals reproduce the frozen strict-H1 summary; the old baseline remains one candidate."),
    ]:
        lines.extend([
            "",
            f"## {title}",
            "",
            description,
            "",
            "Intervals are nominal 95% complete-pool cluster percentile bootstrap intervals with 10,000 replicates. Holm and Bonferroni p-values use the separate complete all-20 family, including cap 1. `results.json` additionally retains the all-20 Bonferroni 99.75% intervals from the same draws.",
            "",
            "| Context | Cap | Effect | Nominal 95% CI | Holm p | Bonferroni p |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in results["families"][family]:
            lines.append(f"| {row['context']} | {row['cap']} | {number(row['effect'])} | {interval(row)} | {pvalue(row['holm_p'])} | {pvalue(row['bonferroni_p'])} |")
    lines.extend([
        "",
        "## Interpretation boundaries",
        "",
        "The original BoN−Random(1) conclusion is unchanged: both T-cell replicates improve at caps 2–16, N50 has no sustained benefit, and HepG2 is non-monotone. The equal-budget supplement shows that the old contrast is not merely a change in candidate count, but it does not make every T-cell r2 cap multiplicity-robust. The absence of corrected Random(N)−Random(1) effects is reported in full, as are positive or null equal-budget estimates in N50 and HepG2.",
        "",
        "These are retrospective sensitivity results from already inspected outer contexts. They are not independent biological replication, new blind validation, or proof of a universal proxy-ranking advantage. Bootstrap p-values are descriptive sensitivity quantities and do not convert the analysis into a prospective confirmatory design.",
        "",
        "## Files and reproducibility",
        "",
        "Remote full artifacts: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_budget_matched_v1`. New workspace: `/home/proxy-function-gap-paper1-budget`. Local aggregates, code, protocol, validation logs, and hashes are under `deliverables/reports/source_data/paper1_budget_matched_v1/`. `analysis/paired_records.tsv` remains remote because it contains selected sequence IDs and sequence-level regrets; it is not part of the local export.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    results = json.loads((args.root / "analysis/results.json").read_text())
    audit = json.loads((args.root / "analysis/selection_audit.json").read_text())
    encoded = render(results, audit)
    if args.check:
        if args.output.read_text() != encoded:
            raise ValueError("report is not reproducible")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(encoded)
    print(json.dumps({"status": "PASS", "report": str(args.output), "check": args.check}))


if __name__ == "__main__":
    main()
