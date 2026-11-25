# Paper 1 strict-H1 equal-budget Random(N) sensitivity — frozen retrospective protocol

Frozen before equal-budget selection or estimation on 2026-11-25.

## Evidence boundary and prohibited changes

This is a retrospective sensitivity analysis of all four previously inspected strict-H1 Castillo outer contexts. It is not a new blind test, prospective confirmation, or causal identification. No model is trained, calibrated, tuned, or run; no proposal, prediction, pool, or measured label is regenerated. No result-dependent change to contexts, pools, seeds, caps, random streams, estimands, bootstrap, or multiplicity families is allowed. Null, unfavorable, and sign-reversed results are retained. Existing strict-H1 and sequence-ablation outputs are read-only and are never overwritten.

## Frozen inputs and reconstruction

- Pools: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/derived/castillo_fixed_pools_v3_strict_h1_retry1`.
- Frozen selections, CNN predictions, and evaluations: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_v3_strict_h1_retry1`.
- Original aggregate: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_v3_strict_h1_retry1/strict_h1_paired_summary.json`.
- Contexts, in fixed order: `n50_hek293t_r1`, `defined_hepg2_r1`, `defined_tcell_r1`, `defined_tcell_r2`.
- Every context contributes all 16 pools of 64 candidates. Caps are 1/2/4/8/16 and original paired seeds are 101/202/303/404/505.
- The existing learner `q_value=1/64` and candidate order define uniform proposal probabilities. Reconstruct the original nested proposal prefix with the frozen `_draw_index` algorithm and `random.Random(seed)`: N successive `random()` draws, cumulative probabilities in original candidate order, and the last positive-probability index as floating-point fallback. Prefixes are nested by N and sampled with replacement.
- BoN(N) chooses the first occurrence of the maximum frozen CNN `proxy_mean` in that exact prefix. Random(1) is the first prefix item. All 1,600 original Random/BoN pairs (3,200 selection records) must match selected indices, selected IDs, and evaluation regrets exactly before any new estimate is accepted.
- Measured outcome is the existing evaluator `measured_rl`; regret is pool maximum measured MRL minus selected measured MRL. Raw sequence strings and sequence-level measured labels remain remote.

## Equal-budget Random(N)

For each context, pool, original seed, and cap N, choose exactly one position uniformly from the same nested prefix of length N. This includes duplicate proposals as separate prefix positions, matching the proposal-generating process. No reranking, replacement, rejection, or label-informed choice is allowed.

The position draw uses NumPy `Generator(PCG64(SeedSequence(entropy=20261125, spawn_key=(context_index, pool_index, seed_index, cap_index))))`, with zero-based indices in the frozen orders above, followed by exactly one `integers(0, N)` call. These record-specific child streams are independent of the original MT19937 proposal stream and of all bootstrap streams. At N=1 the sole position is zero, so Random(1) must reproduce the original Random ID and regret.

## Estimands and reporting

For every pool, average the five paired seed-level regrets before cross-pool estimation. Report mean BoN(N), Random(N), and Random(1) regrets and these effects:

1. Original pressure contrast: `BoN(N) - Random(1)`; negative favors BoN.
2. Equal-budget proxy-ranking value: `BoN(N) - Random(N)`; negative favors proxy ranking at the same candidate budget.
3. Candidate-budget effect: `Random(N) - Random(1)`; negative means lower regret under the Random(N) realization.

Identity `BoN(N)-Random(1) = [BoN(N)-Random(N)] + [Random(N)-Random(1)]` must hold per seed, pool, and aggregate. The third contrast is a paired randomization realization, not proof that increasing N changes the identical uniform proposal distribution; its expectation over the independent position draw is zero for every N.

Report both T-cell replicates and every cap in N50 and HepG2, including negative, null, non-monotone, and unfavorable findings. Do not describe Random(N) as the old fixed one-candidate baseline.

## Cluster bootstrap and frozen all-20 multiplicity

The outer unit is the complete pool. For each context, cap, and contrast, resample 16 pool indices with replacement for 10,000 replicates using NumPy PCG64 and a matrix of indices shaped `(10000,16)`; compute replicate means and linear-interpolation percentile intervals. Original-contrast draws reproduce strict-H1 seed `20261124 + 100*context_index + cap_index`. Equal-budget and candidate-budget draws use `20261125 + contrast_offset + 100*context_index + cap_index`, with contrast offsets 0 and 10000 respectively. Nominal 95% and all-20 Bonferroni 99.75% intervals use percentiles 2.5/97.5 and 0.125/99.875 from the same draws.

Two-sided bootstrap p-values are `min(1, 2*min((1+#(b<=0))/(B+1), (1+#(b>=0))/(B+1)))`. If all pool effects are exactly zero, set effect/interval to zero and p=1 without consuming bootstrap draws. Apply Holm and Bonferroni separately within each contrast's complete 20 context-by-cap family, including all four deterministic cap-1 zeros with p=1. Holm is enforced monotonically in sorted p-value order; Bonferroni p is `min(1,20*p)`. Do not drop cap1, merge the three families, add outcomes, or switch families based on significance. Original strict-H1 intervals and claims remain unchanged.

## Execution, validation, and delivery

All new computation is CPU-only statistics and record reconstruction; no CUDA or model execution is permitted. Three real rounds are required: (1) raw schema/prefix/ID/regret validation and tests; (2) separately implemented independent statistical reconstruction and exact output comparison; (3) final tests plus remote/local manifest and SHA256 verification. Failed commands, tests, and validations are retained in timestamped or round-specific logs. Any scientific mismatch stops delivery until resolved without changing frozen choices.

New remote code is rooted at `/home/proxy-function-gap-paper1-budget`; new artifacts are rooted at `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_budget_matched_v1`. Local delivery is `deliverables/reports/source_data/paper1_budget_matched_v1/` plus `deliverables/reports/docs/PAPER1_BUDGET_MATCHED_V1.md`. Local files contain code, hashes, pool-level aggregates, summary results, and logs, but no sequence strings or sequence-level measured labels. Full selected-ID/regret audit tables remain remote.
