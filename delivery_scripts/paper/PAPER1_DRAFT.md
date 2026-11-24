---
title: "Contextual Boundaries of Proxy Selection Pressure in Biological Sequence Design"
subtitle: "Fixed-pool evidence from 5′ UTR optimization"
author: "Anonymous authors"
date: ""
lang: en
---

### Abstract

Computational proxies are used to train, sample, and rank biological sequences, but their scores need not remain reliable after optimization pressure changes the candidate distribution. This study examines fixed-pool best-of-N selection in 5′ untranslated region (5′ UTR) design, holding candidate pools fixed while varying proxy-selection pressure. Historical measurements from Sample et al. and Castillo-Hair et al. are evaluated using a frozen convolutional ensemble trained only on Sample data. A strict-H1 reconstruction excluding exact and Hamming-1 conflicts shows reproducible but context-dependent gains under fixed-pool pressure: nominal paired BoN-minus-Random regret intervals are negative at caps 2–16 in both defined-end T-cell contexts, and all eight remain negative under a conservative Bonferroni sensitivity across all 20 comparisons, with monotone improvement through cap 16 in replicate 1 and best mean performance at cap 8 in replicate 2. The boundary is equally clear: HepG2 has its lowest mean regret at cap 8 but its interval crosses zero under both all-20 and within-context multiplicity corrections, so this result is exploratory only; N50 shows no sustained improvement, and the earlier Sample-designed benefit remains a retrospective, training-lineage-specific result. Within-pool score matching yields no consistent optimizer-specific error increment: all four absolute-error-difference intervals include zero, but this null result does not establish high-score extrapolation as the mechanism. CNN mean absolute error is lower than that of a separately trained 4-mer multilayer perceptron in every context, yet absolute errors remain strongly correlated (0.878–0.962); architectural change therefore does not provide independent functional validation. Gaussian perturbations of normalized proxy outputs do not produce monotone degradation. A verifier-guided selection pilot and subsequent Q recalibration do not establish a selection advantage. The strict-H1 analysis is a retrospective protocol-sensitivity analysis of previously inspected outer contexts, not a new blind test or prospective wet-lab validation. These results support context-specific proxy contracts rather than a universal beneficial or harmful direction of optimization pressure.

**Keywords:** proxy–function gap; biological sequence design; 5′ UTR; model-based optimization; winner’s curse; uncertainty; external validation

## 1. Introduction

Biological sequence optimization replaces a difficult-to-measure functional endpoint $Y(x,c)$ with a computational proxy $P(x,c_0)$, where $c$ denotes deployment context. Proxy models are useful because they provide inexpensive scores for large sequence libraries. However, the act of selecting high-scoring candidates changes the distribution on which the proxy must remain valid. This creates a selection problem rather than only a pointwise regression problem: the best candidate according to the proxy can be suboptimal according to the historical functional assay.

Prior work has documented reward hacking, offline model-based optimization failures, and uncertainty-aware sequence design [1]–[5]. These studies establish that optimization can exploit proxy errors, but they do not imply a single direction or mechanism across biological contexts. In particular, four questions remain open: (i) whether fixed-pool pressure has a reproducible direction across studies and assays; (ii) whether an optimizer-specific increment persists after matching proxy scores; (iii) whether model disagreement certifies functional reliability; and (iv) whether Gaussian proxy-output noise is necessarily amplified by best-of-N selection.

This study addresses these questions using frozen models and historical experimental labels. Candidate pools are fixed before proxy scoring, so selection pressure changes without changing the pool-generating process. The main endpoint is measured-MRL (mean ribosome load) regret, not random-split correlation or a proxy score. A conservative reading is required: retrospective measurements provide an independent endpoint for a frozen model, but they do not constitute new wet-lab validation of unmeasured sequences.

The contributions are:

1. A strict-H1 fixed-pool reconstruction across four Castillo-Hair contexts, using same-seed Random/BoN pairing, actual model-forward accounting, and pool-cluster uncertainty, with Sample results retained as historical context.
2. Within-pool score matching to assess optimizer-specific error increments without treating a null result as proof of high-score extrapolation.
3. Same- and cross-architecture common-proxy analysis using a frozen convolutional neural network (CNN) ensemble and a separately trained 4-mer multilayer perceptron (MLP).
4. Alternative-pool sensitivity and normalized-output noise experiments that preserve all negative, non-monotone, and context-boundary results.
5. A preregistered verifier-guided self-consistency (VGSC)-inspired pilot and subsequent Q recalibration whose negative outcomes restrict the method narrative.

## 2. Related work

Reward hacking describes systematic divergence between a proxy and an intended objective under optimization [1]. Offline model-based optimization studies further show that optimizers can find designs with overestimated proxy values outside the data distribution [2], [3]. Conservative objective models and source-distribution constraints reduce, but do not eliminate, this problem [2], [4]. Deep ensembles provide scalable uncertainty estimates, yet ensembles trained on the same data can share errors [5], [6].

Biological sequence design has used conservative optimization, distribution-constrained optimization, and diverse sampling [4], [7], [8]. Sample et al. measured large 5′ UTR libraries and demonstrated both deep-learning-guided design and later prediction–measurement separation in high-predicted regions [9]. Castillo-Hair et al. supplied measurements across cell types and assay contexts [10]. Leppek et al. showed that translation, stability, early expression, and total expression are not interchangeable endpoints [11]. Work on model transfer cautions against treating different reporter systems and endogenous mRNA as one deployment context [12].

Inference-time optimization can also exploit reward-model errors [13]. VGSC uses verifier-guided stochastic decoding under imperfect generators and verifiers [14]. Active learning and selective prediction provide related label-acquisition and abstention mechanisms [15]. The present study does not claim novelty for best-of-N, ensembles, or abstention. Its focus is the evidentiary boundary of proxy-selection pressure and common failure under fixed-pool historical evaluation.

## 3. Methods

### 3.1 Fixed-pool regret

For pool $C_t$ in context $c$, let $P_i$ be the frozen proxy mean for candidate $x_i$, and let $Y_i$ be its measured endpoint. A selection strategy $\pi$ chooses $x_t^\pi$. Regret is

$$
R_t^\pi = \max_{i\in C_t}Y_i - Y_{x_t^\pi}.
$$

Lower regret is better. Abstention is reported separately with coverage and is never assigned an artificial regret. Proxy scores are not used as the final functional endpoint. Candidate pool size is fixed within each context to avoid pool-size confounding.

### 3.2 Model and data lineage

The main proxy is a fresh three-member 50-nucleotide CNN regression ensemble trained on Sample random-library data. Three seeds are trained to development-loss plateau without a maximum epoch limit. Member predictions are inverse-transformed using member-specific train-only target scalers. An earlier normalized-output inference artifact and all downstream results were invalidated and excluded.

Castillo-Hair GSE232927 supplies the external contexts. N50 HEK293T, defined-end HepG2, and two defined-end T-cell contexts each contain 16 pools of 64 unique 50-nucleotide sequences with finite measured MRL. N25 is excluded because the frozen model requires 50-nucleotide input. The strict-H1 sensitivity analysis uses newly reconstructed pools under `castillo_fixed_pools_v3_strict_h1_retry1`: exact and Hamming-1 conflicts are excluded within/across pools and against the combined Sample train/dev/calibration reference. Two independent deterministic builds, learner/label/manifest hashes, and raw/reference/pool hashes were checked for every context. Hamming-2 counts were a predeclared diagnostic only, not an exclusion or pass gate. The N50 learner hashes are unchanged from the earlier build; the other three changed, so their earlier numerical estimates are retained only as historical pre-repair results and are not mixed with strict-H1 estimates. Labels are unavailable to the Learner until selections are frozen.

### 3.3 Selection protocols

Random is always a single-candidate baseline. Best-of-N (BoN) uses proposal caps $\{1,2,4,8,16\}$ and common seeds $\{101,202,303,404,505\}$. The strict-H1 pressure effect at each cap is BoN regret minus same-seed Random regret, averaged within each pool; negative values favor BoN. Because the BoN candidate budget changes while Random remains a single-candidate baseline, the pressure curve is not an equal-candidate-budget comparison. This paired estimand replaces the cap-16-minus-cap-1 contrast for the strict-H1 external analysis. Every score executes the three frozen members on CUDA with no CPU fallback; precomputed predictions are not counted as new forwards. The Evaluator reads historical labels only after Learner artifacts are validated. All strict-H1 intervals are pool-cluster percentile bootstrap intervals with 10,000 replicates and the frozen base seed 20261124 plus deterministic context/cap/metric offsets. Reported intervals are nominal 95% CIs unless explicitly labeled otherwise. Conservative multiplicity sensitivity uses the same bootstrap draws with Bonferroni-adjusted percentiles for all 20 context–cap comparisons (99.75% intervals) and separately within each five-cap context (99% intervals); it is not a new confirmatory analysis. All contexts were previously inspected, so the reconstruction is retrospective protocol sensitivity, not a new blind test.

A separate FP-REJ/FP-TILT pilot uses a verifier $V=1-r$, where $r$ estimates proxy overestimation probability, and a pool-level risk model $Q$ for abstention. The dev-frozen FP-REJ configuration is $\alpha=20$ and cap 16. The pilot is included only as a falsification result, not as evidence of VGSC superiority.

### 3.4 Score matching and common-proxy analysis

For every cap-16 BoN selection, the Evaluator selects a different sequence with the nearest proxy mean in the same pool, using a deterministic hash tie-break. The absolute-error difference is reported with pool-cluster bootstrap. This comparison tests for an additional error increment at similar proxy means; because it controls only the proxy mean, a null result does not identify high-score extrapolation as the causal mechanism.

Common-proxy analysis first examines the CNN ensemble spread. It then trains a 4-mer MLP with 259 sequence features on the same Sample train/dev labels. The MLP seeds converge at epochs 108, 124, and 177, with best epochs 103, 119, and 172. Absolute-error correlations between CNN and MLP predictions quantify error concordance across architectures without attributing its cause.

### 3.5 Alternative pools and output noise

An alternative N50 pool series uses a new deterministic namespace and the same protocol. Proxy-output noise is added to each normalized CNN output before inverse transformation, with $\sigma\in\{0,0.1,0.25,0.5,1.0\}$. This is not probability-logit noise. Sigma-zero records must exactly match the original grid by strategy, cap, and seed.

### 3.6 Reproducibility

Raw files and derived artifacts are stored under `\path{/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/}`{=latex}. Source code is under `/home/proxy-function-gap`; the strict-H1 source commit is `ecaccfafd12d81258b8cd0648b48175398b6d98c`. Each evaluation uses independent verification rounds for file hashes, label isolation, selection identity, and regret. Strict-H1 pressure, score matching, common-error metrics, and the aggregate paired summary each passed three recomputation rounds; three CUDA self-checks and the 264-test suite passed. Statistical intervals resample complete pools rather than treating seeds or repeated measurements as independent biological evidence. The multiplicity sensitivity JSON and its deterministic generating script accompany the aggregate summary and recalculate every nominal and Bonferroni-adjusted pressure interval from the same pool-level effects. The first failed NPZ-preflight attempt and the corrected retry logs are retained; no result-dependent gate, endpoint, model, seed, or cap was changed.

## 4. Results

### 4.1 Strict-H1 fixed-pool pressure gives reproducible but context-dependent gains

Table 1 and Figure 2 report the strict-H1 retrospective reconstruction. The main phenomenon is the concordant benefit in both T-cell contexts. T-cell r1 decreases monotonically from 4.799 at cap 1 to 3.555 at cap 16; its paired BoN-minus-Random effect at cap 16 is −1.243 with nominal 95% CI [−1.710, −0.769]. T-cell r2 improves through cap 8 and partially reverses at cap 16, with a cap-16 effect of −1.355 [−2.083, −0.653]. Every nominal paired interval at caps 2–16 excludes zero in both contexts. All eight also exclude zero under the conservative all-20 Bonferroni sensitivity; the adjusted cap-16 intervals are [−1.950, −0.559] and [−2.456, −0.326], respectively. Thus the earlier statement that all four Castillo contexts fail to reproduce improvement is not valid for this strict-H1 reconstruction.

**Table 1. Strict-H1 fixed-pool BoN measured-MRL regret**

| Context | cap 1 | cap 2 | cap 4 | cap 8 | cap 16 |
|---|---:|---:|---:|---:|---:|
| N50 HEK293T | 6.422 | 6.341 | 6.617 | 6.705 | 6.519 |
| Defined HepG2 r1 | 4.463 | 4.332 | 4.042 | 3.559 | 3.794 |
| Defined T-cell r1 | 4.799 | 4.122 | 3.642 | 3.599 | 3.555 |
| Defined T-cell r2 | 6.398 | 5.421 | 5.254 | 4.889 | 5.043 |

The boundaries prevent a universal beneficial-pressure claim. HepG2 reaches its minimum at cap 8, where the paired effect is −0.904 with nominal 95% CI [−1.708, −0.081], but partially reverses at cap 16 to −0.670 [−1.628, +0.306]. The cap-8 interval crosses zero under both the all-20 Bonferroni sensitivity [−2.098, +0.400] and the within-context five-comparison sensitivity [−1.927, +0.150], so the cap-8 improvement is exploratory only. N50 is non-monotone, and every N50 nominal paired interval includes zero. Sample-designed pools retain their earlier improvement from 1.905 to 0.681, but this is a historical, training-lineage-specific result; Sample random has only three eligible pools and no sustained improvement. These Sample analyses use the earlier cap-based contrast and are not pooled with the strict-H1 same-seed external estimates. The two T-cell contexts are consistent within one study and reconstruction protocol, not independent prospective biological replications.

![](../figures/Figure2_strict_h1_selection_pressure.pdf){width=100%}

**Figure 2 | Strict-H1 fixed-pool pressure gives reproducible but context-dependent gains.** **a–d,** Within-pool paired BoN-minus-same-seed-Random measured-MRL regret across proposal caps; negative values favor BoN. Random is always a single-candidate baseline, so the cap curve is not an equal-candidate-budget comparison. Ribbons are nominal pool-cluster bootstrap 95% intervals over 16 pools per context. **e,** Paired effects at cap 16 (filled points) and cap 8 (open diamonds), with nominal 95% intervals. All eight T-cell cap 2–16 intervals remain negative under the all-20 Bonferroni sensitivity. HepG2 cap 8 is exploratory because both multiplicity-adjusted intervals include zero; all N50 nominal intervals include zero. All results are retrospective strict-H1 protocol sensitivity from previously inspected outer contexts, not a new blind test; H2 was diagnostic only. [Source data: pressure](../figures/source_data/strict_h1_figure2_paired_pressure.csv), [pool effects](../figures/source_data/strict_h1_figure2_pool_effects.csv), and [multiplicity sensitivity](../reports/source_data/paper1_strict_h1_v3/analysis/strict_h1_multiplicity_sensitivity.json).

### 4.2 Score matching does not support an optimizer-specific increment

Table 2 and Figure 3 report absolute errors for BoN selections and their within-pool score matches under the same strict-H1 reconstruction. All intervals include zero and directions differ across contexts.

**Table 2. Strict-H1 score-matched absolute-error differences**

| Context | BoN | Matched | Difference | Pool-cluster nominal 95% interval |
|---|---:|---:|---:|---:|
| N50 HEK293T | 2.881 | 2.707 | +0.174 | [−0.444, +0.712] |
| Defined HepG2 r1 | 3.277 | 2.951 | +0.326 | [−0.796, +1.484] |
| Defined T-cell r1 | 1.615 | 1.918 | −0.303 | [−0.897, +0.148] |
| Defined T-cell r2 | 1.566 | 1.663 | −0.097 | [−0.495, +0.326] |

The data therefore do not identify a stable optimizer-specific error increment under this matching. This null result is not proof that high-score extrapolation causes the observed errors: matching on proxy mean leaves other sequence features and higher-order distribution shifts uncontrolled. Nor does it show that optimization is harmless. Because proxy-score matching alone cannot localize the source of failure, we next ask whether errors are shared within and across model architectures.

![](../figures/Figure3_strict_h1_score_matching.pdf){width=100%}

**Figure 3 | Strict-H1 score matching does not identify a stable optimizer-specific error increment.** **a,** BoN-minus-score-matched absolute-error differences at cap 16 in four Castillo-Hair contexts (16 pools per context); positive values indicate higher BoN error. Points are observed differences and bars are nominal pool-cluster bootstrap 95% intervals; all four intervals include zero. **b,** Mean absolute errors for BoN selections and nearest-proxy-score matches from the same pools. Matching controls proxy mean only, is descriptive, and neither removes all distribution shift nor establishes high-score extrapolation as the causal mechanism. These are retrospective protocol-sensitivity results. [Source data: summary](../figures/source_data/strict_h1_figure3_score_matching.csv) and [pool effects](../figures/source_data/strict_h1_figure3_pool_effects.csv).

### 4.3 Common proxy risk crosses model architectures

The strict-H1 cross-architecture analysis asks whether changing model family removes the common-error signal. CNN mean absolute error is lower than 4-mer MLP error in every context, and all four paired CNN-minus-MLP intervals exclude zero. Nevertheless, absolute errors remain strongly correlated (0.878–0.962; Table 3 and Figure 4), so lower average error does not imply independent biological validation.

**Table 3. Strict-H1 absolute-error concordance across architectures**

| Context | CNN MAE | 4-mer MAE | CNN−MLP [nominal 95% CI] | Correlation [nominal 95% CI] |
|---|---:|---:|---:|---:|
| N50 HEK293T | 2.805 | 2.852 | −0.047 [−0.077, −0.020] | 0.962 [0.956, 0.967] |
| Defined HepG2 r1 | 2.919 | 3.029 | −0.110 [−0.152, −0.065] | 0.953 [0.946, 0.959] |
| Defined T-cell r1 | 1.567 | 1.821 | −0.254 [−0.297, −0.209] | 0.878 [0.859, 0.897] |
| Defined T-cell r2 | 1.544 | 1.739 | −0.195 [−0.231, −0.160] | 0.933 [0.919, 0.944] |

These correlations show that absolute errors remained strongly coupled despite architectural change. Because the CNN and MLP share Sample training data and endpoint definitions, shared lineage is a plausible contributor, but the analysis neither establishes that it causally dominates nor excludes other common factors. Additional same-data models should therefore not be treated as independent functional validation. We next test whether the N50 boundary and a simple noise-amplification account persist under alternative pools and output perturbations.

![](../figures/Figure4_strict_h1_common_proxy_error.pdf){width=100%}

**Figure 4 | Strict-H1 common proxy errors persist across model architectures.** **a,** Means of 16 within-pool Pearson correlations between CNN-ensemble and 4-mer MLP absolute errors, with nominal pool-cluster bootstrap 95% intervals; faint points show pool-level estimates. **b,** Paired CNN-minus-MLP mean absolute error; negative values favor CNN and all nominal intervals exclude zero. Both models share Sample training data and endpoint definitions, so concordance does not causally apportion the shared errors and is not independent functional validation. This retrospective analysis uses aggregate pool-level source data, not candidate-level errors from the historical pools. [Source data: summary](../figures/source_data/strict_h1_figure4_cross_architecture_summary.csv) and [pool effects](../figures/source_data/strict_h1_figure4_pool_effects.csv).

### 4.4 Alternative pools and output noise preserve the N50 boundary

The alternative N50 pool series gives regrets 6.521, 5.914, 5.625, 5.618, and 5.726 at caps 1, 2, 4, 8, and 16. Like the original series, it is non-monotone and reverses after cap 4–8. In these two series, the absence of sustained improvement is therefore not specific to one deterministic pool draw.

At cap 16, regrets for $\sigma=0,0.1,0.25,0.5,1.0$ are 6.519, 6.018, 6.143, 6.205, and 6.526 (Extended Data Fig. 1). Sigma-zero exactly reproduces the original grid. Moderate noise can improve selection, while the largest noise returns close to baseline. The hypothesis that BoN necessarily amplifies Gaussian output noise is not supported in this normalized-output protocol. These observations are descriptive and are not used to propose noise injection as a method. The sensitivity results constrain mechanism claims but do not validate a correction; we therefore report the preregistered VGSC-inspired pilot as a falsification test, keeping its outer gate separate from mechanism discovery and its random-pool analysis secondary.

![](../figures/ExtendedData_proxy_noise.pdf){width=100%}

**Extended Data Fig. 1 | Proxy-output noise is not monotonically amplified by best-of-N.** **a,** Mean measured-MRL regret in N50 HEK293T (16 pools) after Gaussian perturbation of normalized CNN outputs at caps 1, 4, and 16; shaded bands are pool-cluster bootstrap 95% intervals. **b,** Within-pool paired regret differences from sigma zero at caps 4 and 16; negative values indicate lower regret after perturbation, and bars are pool-cluster bootstrap 95% intervals. Cap 1 is invariant to output noise in this protocol. These protocol-sensitivity results are exploratory and do not recommend noise injection as a correction. [Source data: context summary](../figures/source_data/extended_data_noise.csv) and [pool-level effects](../figures/source_data/extended_data_noise_pool.csv).

### 4.5 VGSC-inspired pilot and Q recalibration do not establish an advantage

In the original designed-outer pilot, FP-REJ before Q has mean regret 1.024, compared with 0.681 for BoN at the same cap. The difference is +0.343 with a pool-cluster 95% interval of [+0.225, +0.467] (Extended Data Fig. 2). The original Q configuration rejected every selection, giving zero coverage and an undefined post-Q regret. Three eligible random pools gave a small FP-REJ+Q advantage of −0.055 with an interval of [−1.400, +1.051], insufficient to support a general advantage.

A subsequent development/calibration-only Q repair recalibrated probabilities and froze a 0.99 threshold without using outer results. Calibration log loss improved from 0.326896 to 0.068580, but V+Q regret was 3.4041 versus 3.2983 for V-only, with coverage 0.9375. In both exploratory retrospective outer contexts, recalibrated Q triggered no rejection: V+Q and V-only regrets were identical while Q added head and full-pool feature costs. FP-TILT retained its full support set and exceeded the common 48-CNN main budget, so its results are not equal-cost evidence of superiority. Thus neither the original pilot nor the Q repair supports a selection advantage; no outer result was used to retune the threshold, $\alpha$, cap, features, or model type.

![](../figures/ExtendedData_VGSC_paired_difference.pdf){width=100%}

**Extended Data Fig. 2 | The original verifier-guided fixed-pool pilot did not pass its gate.** Paired VGSC-minus-BoN measured-MRL regret differences at cap 16; positive values favor BoN and bars are pool-cluster bootstrap 95% intervals. This historical pilot uses FP-REJ before Q in the designed comparison because the original Q abstained on all 16 pools; the random comparison uses FP-REJ+Q in three eligible pools and its interval crosses zero. Subsequent recalibration is reported in the main text and is not substituted into this historical figure. CNN member budgets are matched at cap 16, while verifier-model and Q costs are additional. All outer results are retrospective protocol sensitivity, not a new blind test. [Source data](../figures/source_data/extended_data_vgsc.csv).

## 5. Discussion

The strict-H1 reconstruction supports a conditional proxy contract with genuine but context-dependent benefits, rather than a universal account of proxy failure. The two T-cell contexts reproduce negative paired effects across caps 2–16, including under the conservative all-20 multiplicity sensitivity, while the nominal HepG2 cap-8 benefit is exploratory and N50 has no sustained improvement. This heterogeneity is a result to preserve, not a nuisance to average away. Score matching does not identify a stable optimizer-specific increment, while cross-architecture error correlations show that errors remained coupled despite model-family change. Neither analysis proves high-score extrapolation as the mechanism or causally apportions shared errors to training data. Alternative pools and the negative noise experiment further constrain the mechanism narrative.

These findings should not be interpreted as proof that proxies are useless or universally reliable. Sample-designed and T-cell pools show that a frozen proxy can remain useful under increased selection pressure in specified contexts. The HepG2 reversal and N50 boundary show why proxy validity must be declared by library, cell, assay, and training lineage, and then checked against an independent endpoint. Repeated deterministic recomputation supports numerical reproducibility, not prospective biological replication.

The negative VGSC pilot and Q repair are consequential. Stochastic verifier guidance and pool-level abstention do not automatically solve proxy failure, and better calibrated Q probabilities did not translate into better selection regret. Reusing the same outer results to tune the threshold would invalidate the evaluation and was not done.

## 6. Limitations

This study is retrospective historical replay. It does not include new synthesis, cell assays, animal studies, or prospective wet-lab validation. Sample outer pools have already supported the pilot analysis and are secondary in the pressure analysis; all previously inspected outer results, including the strict-H1 reconstruction, are retrospective or protocol-sensitivity evidence, not new blind tests. The Bonferroni analysis is a conservative sensitivity check, not a new confirmatory analysis. Castillo contexts are external to Sample training but remain fixed subsets of one study, not independent-study meta-analysis. T-cell replicates and deterministic pool seeds are not independent biological replications.

The strict-H1 pools pass exact/H1 isolation within/across pools and against the declared Sample train/dev/calibration reference. This does not remove cell-type, library, assay, or unmeasured lineage shift. H2 was retained as a diagnostic rather than silently promoted to a new gate: within/cross-pool H2 pairs were 0 for N50 and HepG2, 2 for T-cell r1, and 0 for T-cell r2; cross-Sample H2 pairs were 0, 6, 48, and 44, respectively. The three changed learner-pool contexts therefore replace earlier numerical estimates only for this sensitivity analysis; old estimates are not selectively hidden or combined with the new ones.

The alternative MLP shares training data and endpoint definitions with the CNN, so the analysis cannot separate their contributions from architecture-independent sequence or assay effects; high error correlation alone does not establish causal dominance by shared data. Score matching controls proxy mean but not all sequence features or higher-order distribution shifts, and its null intervals do not prove high-score extrapolation. Sequence-family analyses are exploratory. The noise protocol uses normalized-output Gaussian perturbations and does not represent all verifier misspecification. Finally, the full paper has not undergone external peer review.

## 7. Conclusion

Strict-H1 fixed-pool reconstruction shows reproducible but context-dependent benefits from proxy-selection pressure: both T-cell contexts improve, with negative cap 2–16 intervals retained under the all-20 Bonferroni sensitivity, whereas the HepG2 cap-8 result remains exploratory and N50 shows no sustained gain. Score matching does not identify a stable optimizer-specific increment, but this null result does not prove high-score extrapolation as the mechanism. Proxy errors remain correlated across architectures without establishing shared training data as the causally dominant explanation. These retrospective sensitivity results support context-specific proxy contracts, not universal improvement or failure. Future work should prioritize independent-study replication, prospective blind evaluation, and mechanism-specific proxy contracts rather than additional same-data model ensembles.

## Data and code availability

Code is maintained under `\path{/home/proxy-function-gap}`{=latex} and `\path{/home/proxy-function-gap-data}`{=latex}. Data, weights, and large artifacts are under `\path{/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/}`{=latex}.

- Primary new evidence: `PAPER1_STRICT_H1_RESULT.md` and aggregate source data under `paper1_strict_h1_v3/`.
- Historical external, mechanism, cross-architecture, and noise result documents remain available in the report directory.
- Q repair: `q_actual_cost_recalibration_20261124.md`.
- Strict-H1 figure scripts, source CSV files, multiplicity sensitivity JSON/script, and three-round validation accompany the deliverables.

Release upload status is recorded separately in `release_manifest.json`; this local document update does not itself claim an upload.

## References

[1] J. Skalse, N. H. R. Howe, D. Krasheninnikov, and D. Krueger, “Defining and characterizing reward hacking,” in Proc. NeurIPS, 2022.

[2] B. Trabucco, A. Kumar, X. Geng, and S. Levine, “Conservative objective models for effective offline model-based optimization,” in Proc. ICML, 2021.

[3] B. Trabucco, X. Geng, A. Kumar, and S. Levine, “Design-Bench: Benchmarks for data-driven offline model-based optimization,” in Proc. ICML, 2022.

[4] M. Yao, M. Zeng, H. Bastani, and C. Zhang, “Generative adversarial model-based optimization via source critic regularization,” in Proc. NeurIPS, 2024.

[5] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” in Proc. NeurIPS, 2017.

[6] E. N. Weinstein, A. Amin, J. Frazer, and D. S. Marks, “Non-identifiability and the blessings of misspecification in models of molecular fitness,” in Proc. NeurIPS, 2022.

[7] A. J. Reddy, X. Geng, M. Hersch, and S. Levine, “Designing cell-type-specific promoter sequences using conservative model-based optimization,” in Proc. NeurIPS, 2024.

[8] M. Jain, E. Bengio, A. Hernandez-Garcia, J. Rector, and Y. Bengio, “Biological sequence design with GFlowNets,” in Proc. ICML, 2022.

[9] P. J. Sample, B. Wang, D. W. Reid, V. Presnyak, I. J. McFadyen, D. R. Morris, and G. Seelig, “Human 5′ UTR design and variant effect prediction from a massively parallel translation assay,” Nature Biotechnology, vol. 37, pp. 803–809, 2019. doi: 10.1038/s41587-019-0164-5.

[10] S. Castillo-Hair, S. Fedak, P. S. Wang, et al., “Optimizing 5′UTRs for mRNA-delivered gene editing using deep learning,” Nature Communications, vol. 15, 5284, 2024. doi: 10.1038/s41467-024-49508-2.

[11] K. Leppek, G. W. Byeon, W. Kladwang, et al., “Combinatorial optimization of mRNA structure, stability, and translation for RNA-based therapeutics,” Nature Communications, vol. 13, 1536, 2022. doi: 10.1038/s41467-022-28776-w.

[12] N. Schlusser, A. González, M. Pandey, et al., “Current limitations in predicting mRNA translation with deep learning models,” Genome Biology, vol. 25, 227, 2024. doi: 10.1186/s13059-024-03369-6.

[13] M. Khalaf, R. Anil, A. Beirami, et al., “Inference-time reward hacking in large language models,” in Proc. NeurIPS, 2025.

[14] E. Fadeeva, X. Li, M. Sachan, and M. Panov, “Verifier-guided self-consistency: Test-time reasoning with imperfect generators and verifiers,” arXiv:2611.06606, 2026.

[15] J. Chen, J. Yoon, S. Ebrahimi, and C. Zhang, “ASPEST: Bridging the gap between active learning and selective prediction,” arXiv:2304.03870, 2023.
