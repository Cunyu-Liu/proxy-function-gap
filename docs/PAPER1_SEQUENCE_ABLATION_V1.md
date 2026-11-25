# Paper 1 sequence mechanism ablation v1

## Status and evidence boundary

This is a **retrospective/exploratory mechanism sensitivity analysis** of the four previously inspected strict-H1 Castillo outer contexts. It is not a new blind test, prospective confirmation, or causal analysis. No model was retrained, recalibrated, tuned, or rerun; all computation was CPU-only. The analysis does not change the historical outer-viewing boundary.

The analysis used the original 4,096 candidates, 1,600 BoN/Random selections, and sequence-level measured MRL on px2. Local source data deliberately omit sequence strings and sequence-level measured labels; raw inputs, the remote-only paired table, frozen configuration, results, code, logs, and hashes remain in the remote artifact.

## Frozen design

The four fixed covariates were continuous GC fraction, the existing uAUG indicator, longest contiguous poly-U length (T=U), and the existing train-support value (`support`, used unchanged). For each context and cap, the within-pool/seed outcome was BoN regret minus Random regret; negative values favor BoN.

For caps 2–16, ordinary least squares included all 16 pool indicators and the four BoN-minus-Random feature contrasts. The adjusted effect was the mean pool intercept, representing the estimated paired effect at zero feature contrast. Pool-cluster bootstrap used 10,000 replicates and seed 20261125. All 20 context–cap comparisons, including deterministic cap 1, were retained for Holm and Bonferroni corrections.

The longest-poly-U upper-quartile cutoff was frozen from all candidate sequence features without using MRL, selection, regret, or mechanism results. The nearest-rank 75th percentile was **3**, and the upper-quartile family was `longest_poly_u >= 3`; 1,591/4,096 candidates belonged to it because of ties. Family sensitivities removed a complete BoN/Random pair when either selected sequence was uAUG-positive or in that poly-U family. Candidates were not reranked, replaced, or re-optimized.

## Primary paired results

Effects are BoN−Random regret. Both unadjusted and adjusted effects use pool-cluster 95% percentile intervals. `Holm-p` below applies to the adjusted effect over all 20 comparisons; the full table also retains unadjusted and Bonferroni p-values in `analysis/results/primary_paired.tsv`.

| Context | Cap | Unadjusted effect [95% CI] | Adjusted effect [95% CI] | Adjusted Holm-p |
|---|---:|---:|---:|---:|
| N50 HEK293T | 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 |
| N50 HEK293T | 2 | −0.081 [−0.738, +0.550] | −0.108 [−0.593, +0.405] | 1.000 |
| N50 HEK293T | 4 | +0.195 [−0.504, +0.882] | +0.202 [−0.574, +0.977] | 1.000 |
| N50 HEK293T | 8 | +0.283 [−0.624, +1.223] | +0.001 [−0.767, +1.060] | 1.000 |
| N50 HEK293T | 16 | +0.097 [−1.014, +1.250] | +0.408 [−0.551, +1.453] | 1.000 |
| Defined HepG2 r1 | 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 |
| Defined HepG2 r1 | 2 | −0.132 [−0.747, +0.456] | +0.465 [−0.206, +1.230] | 1.000 |
| Defined HepG2 r1 | 4 | −0.422 [−1.019, +0.153] | +0.879 [+0.103, +1.612] | 0.524 |
| Defined HepG2 r1 | 8 | −0.904 [−1.686, −0.069] | −0.192 [−1.036, +0.914] | 1.000 |
| Defined HepG2 r1 | 16 | −0.670 [−1.635, +0.344] | +0.233 [−0.764, +1.480] | 1.000 |
| Defined T-cell r1 | 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 |
| Defined T-cell r1 | 2 | −0.677 [−0.944, −0.390] | −0.215 [−0.481, +0.094] | 1.000 |
| Defined T-cell r1 | 4 | −1.156 [−1.634, −0.702] | −0.486 [−0.941, −0.028] | 0.662 |
| Defined T-cell r1 | 8 | −1.199 [−1.653, −0.733] | −0.484 [−0.855, −0.041] | 0.642 |
| Defined T-cell r1 | 16 | −1.243 [−1.697, −0.773] | −0.569 [−1.296, +0.098] | 1.000 |
| Defined T-cell r2 | 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 |
| Defined T-cell r2 | 2 | −0.977 [−1.614, −0.360] | −0.135 [−0.593, +0.460] | 1.000 |
| Defined T-cell r2 | 4 | −1.144 [−1.892, −0.383] | +0.043 [−0.850, +0.962] | 1.000 |
| Defined T-cell r2 | 8 | −1.510 [−2.134, −0.870] | −0.576 [−1.787, +0.696] | 1.000 |
| Defined T-cell r2 | 16 | −1.355 [−2.085, −0.657] | −0.329 [−1.582, +0.863] | 1.000 |

The original unadjusted T-cell improvements were therefore **not preserved in full after feature adjustment**. Effects attenuated in both replicates. Only T-cell r1 caps 4 and 8 retained nominal negative adjusted intervals; neither survived multiplicity correction. T-cell r2 had no adjusted interval below zero, and its cap-4 point estimate reversed sign. HepG2 cap 8, the strongest unadjusted HepG2 result, also attenuated toward zero. N50 remained without a sustained benefit. HepG2 cap 4 and several other sign reversals are retained as unfavorable sensitivity findings, not interpreted as new benefits or harms.

## Sequence-family sensitivities

Each row reports retained pairs/pools and the unadjusted BoN−Random regret effect. Corrections are over all 20 comparisons within each family.

### Excluding pairs containing uAUG

| Context | Cap | Pairs/pools | Effect [95% CI] | Holm-p |
|---|---:|---:|---:|---:|
| N50 | 1 | 38/16 | 0.000 [0.000, 0.000] | 1.000 |
| N50 | 2 | 37/16 | +0.132 [−0.266, +0.544] | 1.000 |
| N50 | 4 | 37/16 | +1.093 [+0.021, +2.459] | 0.864 |
| N50 | 8 | 36/16 | +0.828 [−0.184, +1.781] | 1.000 |
| N50 | 16 | 36/15 | +1.029 [−0.241, +2.569] | 1.000 |
| HepG2 | 1 | 41/15 | 0.000 [0.000, 0.000] | 1.000 |
| HepG2 | 2 | 40/15 | +0.417 [−0.114, +1.122] | 1.000 |
| HepG2 | 4 | 41/15 | +0.557 [+0.021, +1.161] | 0.864 |
| HepG2 | 8 | 40/15 | +0.155 [−0.737, +1.028] | 1.000 |
| HepG2 | 16 | 37/15 | +0.281 [−1.073, +1.533] | 1.000 |
| T-cell r1 | 1 | 35/16 | 0.000 [0.000, 0.000] | 1.000 |
| T-cell r1 | 2 | 35/16 | −0.179 [−0.380, −0.006] | 0.864 |
| T-cell r1 | 4 | 35/16 | −0.221 [−0.563, +0.219] | 1.000 |
| T-cell r1 | 8 | 33/16 | −0.055 [−0.512, +0.491] | 1.000 |
| T-cell r1 | 16 | 32/16 | −0.233 [−0.760, +0.404] | 1.000 |
| T-cell r2 | 1 | 32/15 | 0.000 [0.000, 0.000] | 1.000 |
| T-cell r2 | 2 | 32/15 | −0.059 [−0.507, +0.321] | 1.000 |
| T-cell r2 | 4 | 32/15 | −0.419 [−1.222, +0.432] | 1.000 |
| T-cell r2 | 8 | 31/15 | −0.524 [−1.452, +0.415] | 1.000 |
| T-cell r2 | 16 | 30/15 | −0.548 [−1.623, +0.552] | 1.000 |

No uAUG-excluded comparison survived Holm correction. In particular, the T-cell r1/r2 effects were substantially smaller and all intervals at caps 4–16 included zero.

### Excluding the blind upper-quartile poly-U family

| Context | Cap | Pairs/pools | Effect [95% CI] | Holm-p |
|---|---:|---:|---:|---:|
| N50 | 1 | 26/15 | 0.000 [0.000, 0.000] | 1.000 |
| N50 | 2 | 19/10 | −0.179 [−1.654, +1.102] | 1.000 |
| N50 | 4 | 15/9 | −0.714 [−2.648, +0.935] | 1.000 |
| N50 | 8 | 11/6 | −1.136 [−3.748, +0.849] | 1.000 |
| N50 | 16 | 13/8 | −0.706 [−3.237, +1.667] | 1.000 |
| HepG2 | 1 | 56/16 | 0.000 [0.000, 0.000] | 1.000 |
| HepG2 | 2 | 42/14 | −0.192 [−0.625, +0.307] | 1.000 |
| HepG2 | 4 | 39/14 | −0.056 [−0.652, +0.704] | 1.000 |
| HepG2 | 8 | 38/14 | −1.433 [−2.894, −0.062] | 0.557 |
| HepG2 | 16 | 35/15 | −1.008 [−2.142, +0.085] | 0.946 |
| T-cell r1 | 1 | 59/16 | 0.000 [0.000, 0.000] | 1.000 |
| T-cell r1 | 2 | 51/16 | −0.278 [−0.582, +0.034] | 0.946 |
| T-cell r1 | 4 | 44/16 | −0.950 [−1.577, −0.363] | 0.016 |
| T-cell r1 | 8 | 42/16 | −0.920 [−1.521, −0.307] | 0.061 |
| T-cell r1 | 16 | 42/14 | −1.190 [−1.885, −0.388] | 0.102 |
| T-cell r2 | 1 | 59/15 | 0.000 [0.000, 0.000] | 1.000 |
| T-cell r2 | 2 | 49/15 | −0.909 [−1.655, −0.157] | 0.246 |
| T-cell r2 | 4 | 44/15 | −0.989 [−2.040, +0.078] | 0.946 |
| T-cell r2 | 8 | 43/15 | −1.945 [−2.993, −0.832] | 0.016 |
| T-cell r2 | 16 | 37/15 | −2.121 [−3.697, −0.579] | 0.102 |

This sensitivity retained negative estimates for much of the T-cell pressure curve. Only T-cell r1 cap 4 and T-cell r2 cap 8 survived the family-specific all-20 Holm correction; T-cell r1 cap 8 was borderline (Holm-p=0.061), and both cap-16 results remained nominally negative but not multiplicity-corrected significant. Reduced retained pool counts, especially in N50, limit precision. These results are heterogeneity/sensitivity evidence, not evidence that poly-U causally mediates optimization.

## BoN versus score-matched candidate features

At cap 16, each BoN selection was matched to the nearest different candidate by CNN proxy mean, with SHA256 sequence-ID tie-breaking. Entries are BoN-minus-matched paired contrasts with nominal pool-cluster 95% intervals.

| Context | GC | uAUG | Longest poly-U | Support | Proxy mean |
|---|---:|---:|---:|---:|---:|
| N50 | +0.003 [−0.015, +0.021] | −0.038 [−0.100, 0.000] | +0.175 [−0.188, +0.538] | −0.001 [−0.005, +0.002] | +0.017 [+0.003, +0.032] |
| HepG2 | −0.017 [−0.034, 0.000] | +0.038 [−0.050, +0.113] | +0.188 [−0.050, +0.438] | −0.009 [−0.061, +0.047] | +0.018 [+0.007, +0.029] |
| T-cell r1 | −0.002 [−0.017, +0.013] | +0.013 [−0.038, +0.075] | −0.163 [−0.500, +0.138] | −0.040 [−0.077, −0.009] | +0.032 [+0.018, +0.050] |
| T-cell r2 | −0.004 [−0.019, +0.011] | −0.025 [−0.113, +0.063] | −0.150 [−0.400, +0.100] | −0.009 [−0.037, +0.018] | +0.022 [+0.010, +0.035] |

GC, uAUG, and poly-U contrasts were small and their intervals included zero. T-cell r1 BoN selections had lower existing support than score-matched candidates. Proxy means were close by construction but not identical. These descriptive contrasts do not establish an optimizer-specific biological distortion.

## Conclusion

The strict-H1 unadjusted T-cell BoN improvements are sensitive to adjustment for the four predeclared selected-sequence features. The data do **not** support a claim that the full cap2–16 effect persists independently of GC, uAUG, poly-U, and train-support composition. The clearest remaining evidence is restricted to nominal T-cell r1 cap4/8 adjusted estimates and selected poly-U-excluded strata, with only two family-stratum comparisons surviving Holm correction. All other null, attenuated, and sign-reversed results are retained. No causal mechanism, universal proxy benefit/harm, or new blind validation is claimed.

## Reproducibility and validation

- Frozen protocol SHA256: `316db9f855f7a43a75a17b9907e98882e85acd8f60c5ad31b260e604fb477817`.
- Round 1: 45,764 raw checks passed over 196 hashed input files, 4,096 candidates, and 1,600 BoN/Random pairs; GC, uAUG (`ATG` substring), poly-U, support, measured MRL joins, selection identity, and regret were checked.
- Round 2: a separately written implementation recomputed 620 primary, family, multiplicity, and score-matched fields with maximum absolute difference `4.44e-16` and no mismatches.
- Round 3: final tests plus remote/local SHA256 verification are recorded in `verification/round3_delivery.json` and `verification/round3_local_delivery.json`.
- Remote workspace: `/home/proxy-function-gap-paper1-sequence`; artifact: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_sequence_ablation_v1`.
- Full machine-readable results: `analysis/results/primary_paired.tsv`, `sequence_family_sensitivity.tsv`, `score_matched_features.tsv`, `pool_effects.tsv`, `selection_feature_contrasts.tsv`, and `summary.json`.
- Initial test/analysis failures were retained in logs and the remote failed-output directory. They concerned synthetic test fixtures, Holm test expectations, and deterministic cap-1 rank handling, not post-hoc scientific choices. Code was not pushed.
