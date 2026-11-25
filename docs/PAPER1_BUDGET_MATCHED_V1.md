# Paper 1 strict-H1 equal-budget Random(N) sensitivity

## Evidence status

This is a retrospective sensitivity analysis of the four previously inspected strict-H1 Castillo outer contexts, not a new blind test or prospective confirmation. No model was trained, calibrated, tuned, or run. All new computation was CPU-only reconstruction and statistics using frozen proposals, CNN predictions, selections, and measured labels. Existing strict-H1 results were not overwritten.

The analysis included all 16 pools per context, original seeds 101/202/303/404/505, and caps 1/2/4/8/16. Random(N) uniformly selected one position from the exact same nested proposal prefix used by BoN(N), using an independent record-specific PCG64 child stream. Negative regret differences favor the first-named strategy. Full selected-ID/regret audits remain remote; local source data are de-identified aggregates.

## Raw and independent verification

All 3200 original Random/BoN selection records (1600 pairs) matched reconstructed selected IDs, indices, and regrets. All 1600 Random(N) selections were independently checked. All 320 cap-1 rows exactly reproduced Random(1), and all 1,600 seed-level decomposition identities passed. Three complete all-20 families retained cap-1 structural zeros with p=1.

Three real validation rounds were executed: raw reconstruction/tests, a separately implemented independent recomputation, and final delivery/hash verification. Failed environment, test, and numerical-precision attempts are retained under `logs/`; no failure was hidden or converted into a protocol change.

## Main finding

At equal candidate budget, proxy ranking shows context-dependent value rather than a universal benefit or harm. All four T-cell r1 cap 2–16 effects are negative and survive that contrast's all-20 Holm correction. T-cell r2 is negative at every cap 2–16, but only cap 4 survives Holm; caps 2, 8, and 16 remain nominal negative results without multiplicity-robust support. HepG2 cap 2 survives Holm, cap 4 is nominally negative but Holm p=0.055994, cap 8 includes zero, and cap 16 is nominally negative but not corrected-significant. N50 has no nominally supported equal-budget effect and is directionally unfavorable at caps 8 and 16.

No Random(N)−Random(1) contrast survives its all-20 Holm correction. HepG2 cap 2 is nominally positive (higher regret for the Random(2) realization), but this is not corrected-significant. Because every prefix position has the same uniform proposal distribution, increasing N does not change the expectation of this independently randomized contrast; observed nonzero estimates are paired randomization realizations and Monte Carlo variability, not evidence of a systematic candidate-budget effect.

## Mean measured-MRL regret

| Context | Cap | BoN(N) | Random(N) | Random(1) |
|---|---:|---:|---:|---:|
| n50_hek293t_r1 | 1 | 6.422002 | 6.422002 | 6.422002 |
| n50_hek293t_r1 | 2 | 6.340792 | 6.667535 | 6.422002 |
| n50_hek293t_r1 | 4 | 6.616574 | 7.035745 | 6.422002 |
| n50_hek293t_r1 | 8 | 6.705102 | 6.519020 | 6.422002 |
| n50_hek293t_r1 | 16 | 6.518881 | 6.382942 | 6.422002 |
| defined_hepg2_r1 | 1 | 4.463451 | 4.463451 | 4.463451 |
| defined_hepg2_r1 | 2 | 4.331587 | 5.047212 | 4.463451 |
| defined_hepg2_r1 | 4 | 4.041576 | 5.000674 | 4.463451 |
| defined_hepg2_r1 | 8 | 3.559375 | 4.203125 | 4.463451 |
| defined_hepg2_r1 | 16 | 3.793750 | 4.880636 | 4.463451 |
| defined_tcell_r1 | 1 | 4.798553 | 4.798553 | 4.798553 |
| defined_tcell_r1 | 2 | 4.121807 | 4.921756 | 4.798553 |
| defined_tcell_r1 | 4 | 3.642443 | 5.091199 | 4.798553 |
| defined_tcell_r1 | 8 | 3.599316 | 5.178656 | 4.798553 |
| defined_tcell_r1 | 16 | 3.555181 | 5.260940 | 4.798553 |
| defined_tcell_r2 | 1 | 6.398339 | 6.398339 | 6.398339 |
| defined_tcell_r2 | 2 | 5.421317 | 6.102141 | 6.398339 |
| defined_tcell_r2 | 4 | 5.254170 | 6.269046 | 6.398339 |
| defined_tcell_r2 | 8 | 4.888535 | 5.829409 | 6.398339 |
| defined_tcell_r2 | 16 | 5.043377 | 6.130317 | 6.398339 |

## Equal-budget proxy-ranking value: BoN(N)−Random(N)

Negative values favor proxy ranking at the same N.

Intervals are nominal 95% complete-pool cluster percentile bootstrap intervals with 10,000 replicates. Holm and Bonferroni p-values use the separate complete all-20 family, including cap 1. `results.json` additionally retains the all-20 Bonferroni 99.75% intervals from the same draws.

| Context | Cap | Effect | Nominal 95% CI | Holm p | Bonferroni p |
|---|---:|---:|---:|---:|---:|
| n50_hek293t_r1 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 2 | -0.326743 | [-1.157441, 0.449406] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 4 | -0.419171 | [-1.060154, 0.192942] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 8 | 0.186082 | [-1.059415, 1.397352] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 16 | 0.135939 | [-0.989306, 1.135124] | 1.000000 | 1.000000 |
| defined_hepg2_r1 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| defined_hepg2_r1 | 2 | -0.715625 | [-1.143750, -0.293750] | 0.006399 | 0.007999 |
| defined_hepg2_r1 | 4 | -0.959098 | [-1.578833, -0.298963] | 0.055994 | 0.079992 |
| defined_hepg2_r1 | 8 | -0.643750 | [-1.334375, 0.043750] | 0.588541 | 1.000000 |
| defined_hepg2_r1 | 16 | -1.086886 | [-2.084375, -0.118105] | 0.303970 | 0.607939 |
| defined_tcell_r1 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| defined_tcell_r1 | 2 | -0.799949 | [-1.201008, -0.449730] | 0.004000 | 0.004000 |
| defined_tcell_r1 | 4 | -1.448756 | [-1.809149, -1.094514] | 0.004000 | 0.004000 |
| defined_tcell_r1 | 8 | -1.579340 | [-1.975375, -1.183081] | 0.004000 | 0.004000 |
| defined_tcell_r1 | 16 | -1.705760 | [-2.163445, -1.211914] | 0.004000 | 0.004000 |
| defined_tcell_r2 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| defined_tcell_r2 | 2 | -0.680824 | [-1.248022, -0.135771] | 0.177582 | 0.295970 |
| defined_tcell_r2 | 4 | -1.014876 | [-1.544490, -0.465832] | 0.006399 | 0.007999 |
| defined_tcell_r2 | 8 | -0.940873 | [-1.738304, -0.151800] | 0.222178 | 0.403960 |
| defined_tcell_r2 | 16 | -1.086941 | [-1.833685, -0.318589] | 0.064994 | 0.099990 |

## Candidate-budget realization: Random(N)−Random(1)

Negative values favor the Random(N) realization; all cap-1 rows are structural zeros.

Intervals are nominal 95% complete-pool cluster percentile bootstrap intervals with 10,000 replicates. Holm and Bonferroni p-values use the separate complete all-20 family, including cap 1. `results.json` additionally retains the all-20 Bonferroni 99.75% intervals from the same draws.

| Context | Cap | Effect | Nominal 95% CI | Holm p | Bonferroni p |
|---|---:|---:|---:|---:|---:|
| n50_hek293t_r1 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 2 | 0.245533 | [-0.476752, 0.992099] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 4 | 0.613742 | [-0.269123, 1.485210] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 8 | 0.097018 | [-1.040120, 1.309749] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 16 | -0.039060 | [-1.033282, 0.960963] | 1.000000 | 1.000000 |
| defined_hepg2_r1 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| defined_hepg2_r1 | 2 | 0.583761 | [0.090011, 1.065625] | 0.423958 | 0.423958 |
| defined_hepg2_r1 | 4 | 0.537223 | [-0.038321, 1.155649] | 1.000000 | 1.000000 |
| defined_hepg2_r1 | 8 | -0.260326 | [-0.998030, 0.399266] | 1.000000 | 1.000000 |
| defined_hepg2_r1 | 16 | 0.417184 | [-0.181873, 1.030654] | 1.000000 | 1.000000 |
| defined_tcell_r1 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| defined_tcell_r1 | 2 | 0.123204 | [-0.275322, 0.530150] | 1.000000 | 1.000000 |
| defined_tcell_r1 | 4 | 0.292646 | [-0.146546, 0.737130] | 1.000000 | 1.000000 |
| defined_tcell_r1 | 8 | 0.380103 | [-0.201619, 0.948354] | 1.000000 | 1.000000 |
| defined_tcell_r1 | 16 | 0.462388 | [-0.092958, 0.968736] | 1.000000 | 1.000000 |
| defined_tcell_r2 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| defined_tcell_r2 | 2 | -0.296198 | [-1.032263, 0.353068] | 1.000000 | 1.000000 |
| defined_tcell_r2 | 4 | -0.129292 | [-0.901305, 0.641162] | 1.000000 | 1.000000 |
| defined_tcell_r2 | 8 | -0.568930 | [-1.300954, 0.146161] | 1.000000 | 1.000000 |
| defined_tcell_r2 | 16 | -0.268021 | [-0.941300, 0.443352] | 1.000000 | 1.000000 |

## Original pressure contrast retained: BoN(N)−Random(1)

These estimates and nominal intervals reproduce the frozen strict-H1 summary; the old baseline remains one candidate.

Intervals are nominal 95% complete-pool cluster percentile bootstrap intervals with 10,000 replicates. Holm and Bonferroni p-values use the separate complete all-20 family, including cap 1. `results.json` additionally retains the all-20 Bonferroni 99.75% intervals from the same draws.

| Context | Cap | Effect | Nominal 95% CI | Holm p | Bonferroni p |
|---|---:|---:|---:|---:|---:|
| n50_hek293t_r1 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 2 | -0.081210 | [-0.731437, 0.535447] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 4 | 0.194572 | [-0.508385, 0.861242] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 8 | 0.283100 | [-0.627535, 1.254418] | 1.000000 | 1.000000 |
| n50_hek293t_r1 | 16 | 0.096879 | [-1.038838, 1.243472] | 1.000000 | 1.000000 |
| defined_hepg2_r1 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| defined_hepg2_r1 | 2 | -0.131864 | [-0.728140, 0.482548] | 1.000000 | 1.000000 |
| defined_hepg2_r1 | 4 | -0.421875 | [-1.018828, 0.156250] | 1.000000 | 1.000000 |
| defined_hepg2_r1 | 8 | -0.904076 | [-1.707755, -0.081250] | 0.374363 | 0.623938 |
| defined_hepg2_r1 | 16 | -0.669701 | [-1.627686, 0.306250] | 1.000000 | 1.000000 |
| defined_tcell_r1 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| defined_tcell_r1 | 2 | -0.676745 | [-0.944416, -0.388053] | 0.004000 | 0.004000 |
| defined_tcell_r1 | 4 | -1.156110 | [-1.629469, -0.708590] | 0.004000 | 0.004000 |
| defined_tcell_r1 | 8 | -1.199237 | [-1.643140, -0.738305] | 0.004000 | 0.004000 |
| defined_tcell_r1 | 16 | -1.243372 | [-1.709525, -0.768757] | 0.004000 | 0.004000 |
| defined_tcell_r2 | 1 | 0.000000 | [0.000000, 0.000000] | 1.000000 | 1.000000 |
| defined_tcell_r2 | 2 | -0.977022 | [-1.612004, -0.334966] | 0.036396 | 0.051995 |
| defined_tcell_r2 | 4 | -1.144168 | [-1.897785, -0.384847] | 0.036396 | 0.051995 |
| defined_tcell_r2 | 8 | -1.509803 | [-2.131752, -0.868396] | 0.004000 | 0.004000 |
| defined_tcell_r2 | 16 | -1.354962 | [-2.082810, -0.653026] | 0.004000 | 0.004000 |

## Interpretation boundaries

The original BoN−Random(1) conclusion is unchanged: both T-cell replicates improve at caps 2–16, N50 has no sustained benefit, and HepG2 is non-monotone. The equal-budget supplement shows that the old contrast is not merely a change in candidate count, but it does not make every T-cell r2 cap multiplicity-robust. The absence of corrected Random(N)−Random(1) effects is reported in full, as are positive or null equal-budget estimates in N50 and HepG2.

These are retrospective sensitivity results from already inspected outer contexts. They are not independent biological replication, new blind validation, or proof of a universal proxy-ranking advantage. Bootstrap p-values are descriptive sensitivity quantities and do not convert the analysis into a prospective confirmatory design.

## Files and reproducibility

Remote full artifacts: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_budget_matched_v1`. New workspace: `/home/proxy-function-gap-paper1-budget`. Local aggregates, code, protocol, validation logs, and hashes are under `deliverables/reports/source_data/paper1_budget_matched_v1/`. `analysis/paired_records.tsv` remains remote because it contains selected sequence IDs and sequence-level regrets; it is not part of the local export.
