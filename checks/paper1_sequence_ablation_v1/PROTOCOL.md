# Paper 1 sequence mechanism ablation v1 — frozen protocol

## Evidence boundary

Retrospective/exploratory mechanism sensitivity only. All four Castillo outer contexts and their historical outer results were inspected before this analysis. This is not a new blind test, prospective confirmation, or causal identification. No model is trained, calibrated, tuned, or rerun. No feature, cap, context, or subset may be selected after seeing mechanism results. Unfavorable and null results are retained.

## Frozen inputs

- Pools: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/derived/castillo_fixed_pools_v3_strict_h1_retry1`.
- Analysis: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_v3_strict_h1_retry1`.
- Contexts: `n50_hek293t_r1`, `defined_hepg2_r1`, `defined_tcell_r1`, `defined_tcell_r2`.
- Per context: 16 pools, 64 unique candidates per pool, caps 1/2/4/8/16, seeds 101/202/303/404/505.
- Candidate fields: `sequence_id`, `utr50`, `gc`, `uAUG`, `longest_poly_u`, `support`, `proxy_mean`. `support` is the existing train-support field and is used unchanged; no new distance definition is introduced.
- Selection fields: `pool_id`, `proposal_cap`, `seed`, `strategy`, `selected_id`, `selected_index`.
- Measured outcome: evaluator `measured_rl`. Regret is checked as pool maximum measured MRL minus selected measured MRL.

Raw sequence strings and sequence-level measured MRL remain remote. Local delivery contains code, hashes, aggregate/pool-level results, and validation logs, not raw sequence strings or sequence-level measured labels.

## Blind sequence-feature threshold

The only sequence-family cutoff is the longest-poly-U upper quartile. It is computed once from all 4,096 strict-H1 candidates pooled across contexts, using only sequence features and no measured MRL, selection outcome, regret, or mechanism result. With sorted values `x`, the cutoff is `x[ceil(0.75*n)-1]` (the nearest-rank 75th percentile); the upper-quartile family is `longest_poly_u >= cutoff`. The numeric cutoff, candidate count, and feature-only source hashes are frozen in `frozen_analysis.json` before any mechanism estimate is computed. uAUG uses its existing binary indicator without a threshold.

## Primary paired analysis

The observational unit is a pool × seed × cap BoN/Random pair. The outcome is `BoN regret − Random regret`; negative favors BoN. Feature contrasts are BoN minus Random for continuous GC fraction, binary uAUG, longest poly-U length (T=U), and existing support.

For each context and cap 2/4/8/16, fit by ordinary least squares:

`delta_regret = pool-specific intercept + beta_gc*delta_gc + beta_uAUG*delta_uAUG + beta_polyU*delta_polyU + beta_support*delta_support + error`.

All 16 pool indicators are included and there is no global intercept. The adjusted effect is the arithmetic mean of the 16 pool intercepts, i.e. the estimated paired effect at zero feature contrasts. This is an associational adjustment, not a causal mediation estimand. Cap 1 is retained with exactly zero unadjusted/adjusted effect and degenerate interval; its p-value is 1.

The unadjusted effect is the mean of per-pool seed-averaged paired regret effects. Uncertainty uses nonparametric pool-cluster bootstrap with 10,000 replicates and seed 20261125. For adjusted fits, all seed rows from a sampled pool move together and each draw receives its own fixed-effect indicator. Percentile 95% intervals use linear quantile interpolation. Two-sided bootstrap p-values are `min(1, 2*min((1+#(b<=0))/(B+1), (1+#(b>=0))/(B+1)))`.

All 20 context × cap results are reported. Holm and Bonferroni corrections are applied separately to unadjusted and adjusted p-values over all 20 comparisons (cap-1 p=1 included). The primary interpretation emphasizes adjusted cap 2–16 effects and both T-cell replicates; no claim rests on an isolated nominal result.

## Frozen sequence-family sensitivities

Separately for (1) uAUG and (2) upper-quartile longest poly-U, exclude a BoN/Random pair if either selected sequence belongs to the target family. There is no reranking, replacement, re-optimization, or change to candidate pools. For each context and cap, report retained pairs/pools and the unadjusted paired effect with pool-cluster bootstrap using the same replicate count, seed sequence, interval, and p-value definitions. Each sensitivity family receives its own all-20 Holm and Bonferroni corrections. Empty estimates are reported as missing, never dropped silently.

## Score-matched candidate features

At cap 16, reconstruct the existing score match for every BoN selection: among the other 63 candidates in the same pool, choose the candidate minimizing absolute CNN `proxy_mean` distance; break exact ties by ascending `sha256(sequence_id)` hexadecimal digest. Compare BoN versus matched candidate GC, uAUG, longest poly-U, support, and proxy mean. Report paired mean contrasts and nominal pool-cluster 95% intervals. These are descriptive secondary mechanism contrasts, not newly selected features or confirmatory tests.

## Validation and delivery

Three real rounds are required: (1) raw field/schema/sequence/label/regret validation and tests; (2) separately implemented independent statistical recomputation and tests; (3) final tests plus remote/local manifest and SHA256 verification. Any mismatch is a failure and must be resolved without changing the frozen scientific choices. Code is not pushed.
