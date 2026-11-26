# T-cell error diagnosis v1 — frozen diagnostic protocol

Date: 2026-11-26.

## Evidence boundary and prohibitions

This is a retrospective diagnostic and sensitivity analysis of the previously inspected strict-H1 `defined_tcell_r1` and `defined_tcell_r2` outer contexts. It is not a new blind test, prospective confirmation, causal identification, or repair validation. Existing pools, predictions, selections, evaluations, labels, and historical artifacts are read-only.

No model is trained, calibrated, tuned, or run. Target T-cell measured labels are used only by the evaluator for descriptive error/selection decomposition after all rules in this protocol are fixed. They do not enter features, proposal generation, ranking, model choice, thresholds for a deployable rule, or hyperparameter selection. Both replicates, both architectures, every cap, null results, unfavorable results, and sign reversals are retained. No repair is selected in this work package.

## Frozen inputs

- Pools and labels: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/derived/castillo_fixed_pools_v3_strict_h1_retry1/{context}`.
- CNN predictions: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_v3_strict_h1_retry1/{context}/cnn_scores`.
- Frozen 4-mer MLP predictions: the corresponding `cross_architecture/*_kmer.json` files.
- Original selections and evaluations: corresponding `selection` and `evaluation` directories.
- Raw replicate CSVs are the files named by each pool manifest. Their only whole-file uses are schema, exact-sequence duplicate, and shared-sequence replicate-variability audits; all candidate-level primary estimates use strict-H1 candidates only.

Each context contributes all 16 pools of 64 distinct candidates (1,024 rows), seeds 101/202/303/404/505, and caps 1/2/4/8/16. Candidate sequence identity is the exact 50-nt sequence. Sequence strings and candidate-level measured labels remain remote; local exports contain aggregates only.

## Pointwise and ranking diagnostics

For CNN and MLP separately, `error = prediction - measured_rl`. Report n, prediction and MRL means/SDs, mean and median error, MAE, RMSE, Pearson, Spearman, and mean within-pool Spearman. Cross-architecture descriptive summaries include prediction/error correlations, error-sign agreement, top16 overlap, and cap16 BoN selected-identity agreement. Decompose MSE exactly into `mean(error)^2 + var(error)` (population variance). Also report MAE after subtracting the global median error and after subtracting each pool's median error. These are oracle explanatory counterfactuals, not legal calibration or repair; any constant offset leaves every within-pool ranking and regret unchanged.

Top-score tails are the deterministic highest 1/2/4/8/16 candidates in each complete 64-candidate pool, ordered by descending frozen prediction, then sequence ID, then original row position. For each tail report selected-top1 realized MRL/regret, mean absolute error/bias, Spearman when defined, predicted-order pair inversion fraction, and whether another tail member has higher measured MRL than the predicted winner. Exact identity:

`full-pool regret = (full-pool oracle MRL - tail oracle MRL) + (tail oracle MRL - predicted-winner MRL)`.

The first term is tail opportunity/oracle loss and the second is within-tail ranking loss. Pool/oracle summaries retain the full-pool maximum, SD, range, and between-pool variability. No oracle or measured label is used to construct a tail.

## BoN, Random(N), and proposal decomposition

Reconstruct original nested proposal prefixes exactly with `random.Random(seed)`, uniform `q=1/64`, cumulative probabilities in original candidate order, and last-positive-index floating-point fallback. BoN takes the first occurrence of the maximum frozen prediction in the prefix. Random(1) is position zero. Random(N) uses one uniform position from the same prefix with NumPy PCG64 SeedSequence entropy 20261125 and spawn key `(context_index, pool_index, seed_index, cap_index)`, exactly reproducing the earlier equal-budget protocol. Context indices are r1=0 and r2=1 within this analysis; absolute context-index offsets do not affect the uniform-position estimand but original four-context indices 2 and 3 are used for byte-compatible reconstruction.

For CNN, every original selected index/ID and evaluation regret must match before new results are accepted. MLP BoN is a fully reported counterfactual on the identical prefixes, not a selected repair. For every model/cap report BoN, Random(N), Random(1) regret and realized MRL, BoN−Random(N), BoN−Random(1), and Random(N)−Random(1). Negative regret differences favor the first strategy. On the same prefix:

`regret = (full-pool oracle - prefix oracle) + (prefix oracle - selected MRL)`;

the opportunity term cancels in BoN−Random(N). Also report prefix-oracle loss and within-prefix ranking loss for BoN and Random(N). Pool-level averages first average the five seeds. Leave-one-pool-out cap16 effects assess pool influence without changing any selection.

## Feature, outlier, duplicate, and replicate audits

Features are already frozen or label-free sequence properties: GC fraction, uAUG, longest poly-U, and existing train support. GC and support use four label-free within-context quartiles; poly-U strata are 0, 1, 2, and >=3 (the historical label-free cutoff); uAUG uses false/true. For all candidates and each pool's top16 tail, report n, bias, MAE, measured/predicted means, and top16 representation. These are descriptive strata only: no error regression, label-trained correction, or stratum-specific reranking is fitted.

Absolute-error outliers use the fixed Tukey rule `abs(error) > Q3 + 1.5*(Q3-Q1)` within each context/model. Report count, MAE contribution, MSE contribution, and cap16 BoN involvement. Any outlier removal is prohibited as a repair here. Exact duplicate counts are reported within pools, across pools, across replicates, and in each raw replicate file. Shared strict-H1 sequences and, separately, all shared raw sequences yield paired r2−r1 MRL descriptive statistics, correlations, error correlations, and error-sign agreement. Raw shared-sequence results are auxiliary and distribution-distinct, not substitutes for strict-H1 estimates.

## Uncertainty and multiplicity

Primary pool-level effects and tail summaries use 10,000 complete-pool bootstrap replicates, NumPy PCG64, percentile 95% intervals, and deterministic seed `20261126 + 100*context_index + 10*model_index + cap_index + contrast_offset`; model indices CNN=0/MLP=1 and contrast offsets are BoN−Random(N)=0, BoN−Random(1)=1000, Random(N)−Random(1)=2000, MLP−CNN BoN=3000. Resample a `(10000,16)` index matrix and compute replicate means of pool-averaged values. Two-sided sign p is `min(1,2*min((1+#(b<=0))/(B+1),(1+#(b>=0))/(B+1)))`; structural all-zero effects have zero interval and p=1 without draws. Holm and Bonferroni corrections retain each model/contrast's complete 10-row context-cap family, including cap1. Intervals and p-values are retrospective descriptive sensitivity quantities, not confirmatory claims.

Pointwise global estimates and feature/outlier/replicate summaries are descriptive; their absence of an interval or p-value is not evidence of no effect. No new hypotheses are promoted based on nominal significance.

## Execution and delivery

New remote code: `/home/proxy-function-gap-tcell-diagnosis`. New artifacts: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_error_diagnosis_v1`. All computation is CPU-only. Three real rounds are mandatory: (1) raw hashes/schema/identity tests and original analysis; (2) separately implemented reconstruction and numerical comparison; (3) final tests, export/report arithmetic, and remote/local SHA256 verification. Failures are logged and retained.

Local aggregate delivery: `deliverables/reports/source_data/tcell_error_diagnosis_v1/` and root `TCELL_ERROR_DIAGNOSIS_V1.md`. Remote-only candidate/selection tables are not exported. Repair hypotheses may only specify mechanisms and legal future development/auxiliary data sources; any future threshold or correction must be frozen and developed without these target T-cell labels, then evaluated on new independent data.
