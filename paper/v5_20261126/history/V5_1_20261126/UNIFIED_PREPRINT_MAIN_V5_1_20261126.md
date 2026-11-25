---
title: "Proxy ranking helps at equal budget—but only within a compositional and deployment contract"
subtitle: "Context-specific 5′ UTR selection gains, their attenuation after sequence adjustment, and independently gated limits of control and repair"
author: "Anonymous authors"
date: "2026-11-26"
lang: en
---

> **Draft status.** Structurally complete integrated preprint, version 5.1. V5.1 corrects bibliographic metadata and adds persistent source links and citations only; no scientific result or evidence boundary is changed. “Anonymous authors” is a deliberate pre-submission placeholder, not a technical error; author names and affiliations must be completed before submission. The recommended submission architecture remains one main article plus Supplementary Study S2 (limitations and falsification of risk control) and Supplementary Study S3 (retrospective label-repair diagnostics). V5 makes equal-candidate-budget proxy-ranking value the primary positive result, then retains its attenuation after sequence-composition adjustment and family sensitivity analyses. Every new analysis is a retrospective sensitivity analysis on previously inspected outer contexts, not a new blind test, prospective confirmation, or causal proof. PERSIST-seq remains a completed small retrospective full-MLP-contract evaluation without detectable transfer. Candidate-level uncertainty abstention failed its frozen calibration gate, and outer scoring was hard-locked and not performed. Outer label-efficiency gains remain unevaluated. V4 and all earlier versions are preserved; no new wet-lab experiment, model training, inference, or unblinding was performed for V5 integration.

## Abstract

Computational proxies can rank biological sequences usefully without supplying a context-free design guarantee. In retrospective strict-H1 5′ untranslated-region (5′ UTR) pools, we first isolated proxy-ranking value at equal candidate budget by comparing best-of-N (BoN) with Random(N) on identical proposal prefixes. In T-cell r1, BoN-minus-Random(N) measured mean ribosome load (MRL) regret effects were −0.799949, −1.448756, −1.579340, and −1.705760 at caps 2, 4, 8, and 16; all four survived the complete all-20 Holm family (each adjusted p=0.004000, including structural cap-1 zeros). T-cell r2 was negative at every cap 2–16, but only cap 4 survived Holm (−1.014876 [−1.544490, −0.465832]; adjusted p=0.006399). This is the primary positive result: proxy ranking has measurable value in specific contexts at equal candidate budget. Its boundary is equally clear. HepG2 cap 2 survived Holm, cap 4 was nominally negative but Holm p=0.055994, and N50 HEK293T had no nominally supported equal-budget effect. Moreover, adjustment for GC, uAUG, longest poly-U, and training support attenuated the original T-cell gains: only T-cell r1 caps 4 and 8 retained nominal negative adjusted intervals, and no adjusted effect survived all-20 Holm. uAUG-excluded effects were substantially smaller, whereas selected poly-U-excluded strata retained some negative effects. These retrospective sensitivities establish composition dependence, not sequence causation. Cross-architecture absolute errors remained correlated at 0.878–0.962, and frozen or exploratory cross-endpoint analyses showed that MRL rankings were not interchangeable with translation efficiency, stability, or reporter expression. A completed small retrospective PERSIST-seq evaluation of the full variable-length 4-mer MLP contract used 59 constructs spanning 49 unique 5′ UTRs; BoN-minus-Random regret was +0.104723 [−0.100964, 0.212706], p=0.245175, with Spearman −0.008447. It showed no detectable transfer, was not blind, and neither directly contradicts T-cell CNN gains nor proves universal failure; a source-confirmed tie-rule audit and SHA256 alternative left the primary effect, interval, and p value unchanged. Risk control and label repair require independent gates: tested selectors and pool-level Q did not establish decision benefit, candidate-level abstention failed frozen calibration (reduction −0.005420 [−0.509271, 0.628623]) with outer scoring locked, and label-repair results remain development/calibration diagnostics. All new analyses reuse historical outer data and are not prospective or causal confirmation.

**Keywords:** proxy–function gap; biological sequence design; 5′ UTR; selection pressure; regret; uncertainty; abstention; active learning; external validation

## Graphical abstract

**Graphical abstract | Evidence chain from proxy-pressure phenomena to constrained decision rules.** The retained V3.1 map links retrospective pressure and mechanism analyses, frozen candidate-risk calibration failure, and development/calibration label-repair diagnostics. V5 adds the equal-budget ranking result and its composition-sensitive boundary in Figs. 14 and 15 without altering this historical figure. Schlusser TE is a frozen one-time cross-endpoint evaluation and Leppek is exploratory; candidate-risk outer scoring was not performed, and Paper 3 outer labels remain unread. This summary does not constitute independent validation or introduce new results. Source: `../figures/v3/Figure11_v3_evidence_chain.pdf`.

## 1. Introduction

Biological sequence design commonly replaces an expensive functional measurement \(Y(x,c)\) with a computational proxy \(P(x,c_0)\), where \(x\) is a sequence, \(c_0\) is the training context, and \(c\) is the deployment context. The proxy can be a learned predictor, an ensemble score, or a verifier used to rank candidates. The resulting decision is not evaluated by average regression error alone. A design campaign usually submits one or a few top-ranked sequences, making functional regret—the difference between the best measured outcome in the candidate pool and that of the selected sequence—the relevant endpoint.

Selection creates two distinct challenges. First, optimization pressure changes which candidates are chosen, even when the candidate pool itself is fixed. Second, deployment may change the cell, construct, assay, or functional endpoint. These challenges should not be collapsed into a single statement that “distribution shift causes failure.” Increased pressure can improve real function when the proxy retains ranking information, and an accurate model for one endpoint need not rank another endpoint correctly. Conversely, agreement among models trained on the same data is not independent biological validation.

Prior work establishes related failures and remedies, including reward hacking, offline model-based optimization, conservative objective models, distribution-constrained optimization, deep ensembles, and selective prediction [1–8]. Experimental 5′ UTR studies provide a particularly useful testbed because large sequence libraries and multiple measured endpoints are available [9–12]. The unresolved issue is not whether proxy failure can occur, but how to move from a heterogeneous phenomenon to mechanism tests and then to remedies without claiming more than the evidence supports.

We therefore pose a chain of questions:

1. **Equal-budget value:** Does proxy ranking improve measured function over Random(N) on the same N-candidate prefix?
2. **Composition boundary:** Do those gains persist after adjustment for selected-sequence composition and within sequence-family sensitivities?
3. **Mechanism:** If pressure effects differ, is there an optimizer-specific error increment beyond proxy-score-matched candidates?
4. **Error structure:** Does changing architecture provide independent error evidence?
5. **Deployment contract:** Are rankings preserved when the functional endpoint or complete model–construct contract changes?
6. **Risk control:** Can alternative selectors, pool-level risk prediction, or uncertainty-based abstention reduce realized regret without using deployment labels?
7. **Repair:** When historical labels are revealed, is there headroom for improvement, and can any acquisition strategy realize it efficiently?

Each question is motivated by the previous result, but no remedy is treated as a logical consequence of a preceding failure. In particular, failure of a selector does not prove that risk is predictable, and the existence of full-label headroom does not prove that a practical acquisition rule will achieve it on untouched data.

The study makes five contributions. First, an equal-candidate-budget Random(N) control isolates proxy-ranking value: all four T-cell r1 cap 2–16 effects and T-cell r2 cap 4 survive all-20 Holm sensitivity. Second, sequence fixed-effect adjustment and uAUG/poly-U family sensitivities expose substantial attenuation and heterogeneity, preventing a composition-independent or causal interpretation. Third, score matching, cross-architecture error concordance, two cross-endpoint studies, and a completed PERSIST-seq independent-study, same-endpoint retrospective evaluation define a context-specific proxy contract; these Paper 1 phenomenon and mechanism results remain the strongest current article. PERSIST makes the full training-lineage/architecture/construct contract explicit rather than treating T-cell gains as portable. Fourth, it preserves negative evidence for verifier-guided selection, conservative/diversity reranking, pool-level Q-based risk discrimination, and development-frozen candidate-level abstention. Fifth, it delineates label repair as a retrospective diagnostic requiring an independent gate rather than a completed deployment gain. This one-main-article/two-supplement structure presents S2 as limitations and falsification and S3 as diagnosis, rather than presenting three unequal evidence states as equally established method contributions.

## 2. Related work

Reward hacking describes optimization against a proxy that diverges from the intended objective [1]. Offline model-based optimization similarly warns that optimizers can exploit overestimated model outputs outside the training distribution [2,3]. Conservative objective models and source-critic regularization address this issue through conservative training or distribution constraints [2,4], while deep ensembles provide scalable predictive uncertainty [5]. Work on molecular-fitness models further shows that misspecification and non-identifiability complicate the interpretation of model disagreement [6]. Biological design studies have combined conservative optimization, uncertainty, and diversity [7], and GFlowNets provide a distinct generative design framework [8].

Sample et al. measured random and designed 5′ UTR libraries and demonstrated deep-learning-guided design as well as prediction–measurement separation in high-predicted regions [9]. Castillo-Hair et al. extended measured 5′ UTR assays across cell types and gene-editing contexts [10]. Leppek et al. measured combinations of RNA structure, stability, ribosome loading, and expression [11]. Schlusser et al. documented limitations in transferring translation models across reporter and endogenous systems [12]. These studies motivate separate treatment of cell, construct, assay, and endpoint rather than a generic “external validation” label.

Inference-time optimization can also exploit reward-model errors [13]. Verifier-guided self-consistency (VGSC) studies stochastic decoding with imperfect verifiers [14]; our use is a fixed-pool falsification pilot, not a claim to reproduce its reasoning-task results. ASPEST connects active learning and selective prediction [15], motivating the distinction between abstention and label acquisition. We do not claim novelty for best-of-N, ensembles, lower confidence bounds, expected improvement, or abstention. The contribution is their evaluation against measured functional regret under explicit evidence and isolation boundaries.

## 3. Results

### 3.1 Equal-candidate-budget controls isolate context-specific proxy-ranking value

We first separated the value of proxy ranking from the number of candidates seen. The strict-H1 reconstruction contained 16 pools of 64 sequences in each of four Castillo-Hair contexts, with exact and Hamming-1 isolation from the Sample train/development/calibration reference. For every pool, original seed, and cap, Random(N) uniformly selected one position from the exact nested proposal prefix used by BoN(N), using an independent record-specific PCG64 child stream. The primary contrast was BoN(N) regret minus same-prefix Random(N) regret; negative values favor proxy ranking at equal candidate budget. This is a retrospective sensitivity analysis of previously inspected outer contexts, not a new blind test.

All four T-cell r1 effects at caps 2–16 were negative and survived the complete all-20 Holm family, including cap-1 structural zeros (Table 1; Fig. 14). Their effects were −0.799949, −1.448756, −1.579340, and −1.705760, each with Holm p=0.004000. T-cell r2 was negative at all four caps, but only cap 4 survived Holm: −1.014876 [−1.544490, −0.465832], Holm p=0.006399. Caps 2, 8, and 16 remained nominal negative results without multiplicity-robust support (Holm p=0.177582, 0.222178, and 0.064994).

**Table 1 | Equal-budget BoN(N)-minus-Random(N) measured-MRL regret.**

| Context | Cap | Effect | Nominal 95% CI | All-20 Holm p |
|---|---:|---:|---:|---:|
| N50 HEK293T | 2 | −0.326743 | [−1.157441, +0.449406] | 1.000000 |
| N50 HEK293T | 4 | −0.419171 | [−1.060154, +0.192942] | 1.000000 |
| N50 HEK293T | 8 | +0.186082 | [−1.059415, +1.397352] | 1.000000 |
| N50 HEK293T | 16 | +0.135939 | [−0.989306, +1.135124] | 1.000000 |
| Defined HepG2 r1 | 2 | −0.715625 | [−1.143750, −0.293750] | **0.006399** |
| Defined HepG2 r1 | 4 | −0.959098 | [−1.578833, −0.298963] | 0.055994 |
| Defined HepG2 r1 | 8 | −0.643750 | [−1.334375, +0.043750] | 0.588541 |
| Defined HepG2 r1 | 16 | −1.086886 | [−2.084375, −0.118105] | 0.303970 |
| Defined T-cell r1 | 2 | −0.799949 | [−1.201008, −0.449730] | **0.004000** |
| Defined T-cell r1 | 4 | −1.448756 | [−1.809149, −1.094514] | **0.004000** |
| Defined T-cell r1 | 8 | −1.579340 | [−1.975375, −1.183081] | **0.004000** |
| Defined T-cell r1 | 16 | −1.705760 | [−2.163445, −1.211914] | **0.004000** |
| Defined T-cell r2 | 2 | −0.680824 | [−1.248022, −0.135771] | 0.177582 |
| Defined T-cell r2 | 4 | −1.014876 | [−1.544490, −0.465832] | **0.006399** |
| Defined T-cell r2 | 8 | −0.940873 | [−1.738304, −0.151800] | 0.222178 |
| Defined T-cell r2 | 16 | −1.086941 | [−1.833685, −0.318589] | 0.064994 |

HepG2 cap 2 also survived Holm, but cap 4 was only nominal (Holm p=0.055994), cap 8 included zero, and cap 16 did not survive correction. N50 had no nominally supported equal-budget effect and was directionally unfavorable at caps 8 and 16. No Random(N)-minus-Random(1) contrast survived its separate all-20 Holm family. Because each prefix position has the same uniform proposal distribution, those nonzero paired realizations are not evidence for a systematic candidate-count effect.

The original BoN(N)-minus-Random(1) pressure contrast remains unchanged and is reported as secondary historical context (Fig. 1). Both T-cell contexts improved at caps 2–16, and all eight all-20 Bonferroni intervals remained negative. T-cell r1 mean regret decreased from 4.799 at cap 1 to 3.555 at cap 16; T-cell r2 improved through cap 8 and partially reversed at cap 16. HepG2 cap 8 was nominal (−0.904 [−1.708, −0.081]) but exploratory after multiplicity adjustment, and all N50 nominal intervals included zero. Earlier Sample-designed regret decreased from 1.905 to 0.681, whereas three eligible Sample random pools showed no sustained gain. These different historical contrasts were not pooled.

**Inference and next question.** Equal-budget comparisons establish specific-context proxy-ranking value, not a universal advantage or independent prospective replication. The next question is whether this value survives sequence-composition adjustment and family sensitivity analyses.

### 3.2 Sequence adjustment attenuates the gains and defines a composition-sensitive boundary

We next asked whether the original strict-H1 BoN-minus-Random(1) pressure gains persisted after adjustment for four predeclared selected-sequence features: continuous GC fraction, uAUG indicator, longest contiguous poly-U, and existing training support. For each context and cap, ordinary least squares included all 16 pool indicators and the four BoN-minus-Random feature contrasts; the adjusted effect was the mean pool intercept at zero feature contrast. This observational adjustment does not identify mediation or sequence causation.

The unadjusted T-cell gains were not preserved in full (Table 2; Fig. 15). T-cell r1 adjusted effects at caps 2, 4, 8, and 16 were −0.215, −0.486, −0.484, and −0.569. Only caps 4 and 8 retained nominal negative adjusted intervals ([−0.941, −0.028] and [−0.855, −0.041]); neither survived all-20 Holm (p=0.662 and 0.642). T-cell r2 adjusted effects were −0.135, +0.043, −0.576, and −0.329, with every interval including zero and no adjusted effect surviving Holm. HepG2 cap 8 attenuated from −0.904 to −0.192 [−1.036, +0.914]. N50 remained without a sustained benefit, and sign reversals—including HepG2 cap 4—were retained rather than interpreted as new benefits or harms.

**Table 2 | Original and GC/uAUG/poly-U/support-adjusted strict-H1 pressure effects.**

| Context | Cap | Unadjusted effect [95% CI] | Adjusted effect [95% CI] | Adjusted all-20 Holm p |
|---|---:|---:|---:|---:|
| Defined T-cell r1 | 2 | −0.677 [−0.944, −0.390] | −0.215 [−0.481, +0.094] | 1.000 |
| Defined T-cell r1 | 4 | −1.156 [−1.634, −0.702] | −0.486 [−0.941, −0.028] | 0.662 |
| Defined T-cell r1 | 8 | −1.199 [−1.653, −0.733] | −0.484 [−0.855, −0.041] | 0.642 |
| Defined T-cell r1 | 16 | −1.243 [−1.697, −0.773] | −0.569 [−1.296, +0.098] | 1.000 |
| Defined T-cell r2 | 2 | −0.977 [−1.614, −0.360] | −0.135 [−0.593, +0.460] | 1.000 |
| Defined T-cell r2 | 4 | −1.144 [−1.892, −0.383] | +0.043 [−0.850, +0.962] | 1.000 |
| Defined T-cell r2 | 8 | −1.510 [−2.134, −0.870] | −0.576 [−1.787, +0.696] | 1.000 |
| Defined T-cell r2 | 16 | −1.355 [−2.085, −0.657] | −0.329 [−1.582, +0.863] | 1.000 |

Family sensitivities reinforced, rather than concealed, the boundary. Excluding every pair containing uAUG substantially reduced both T-cell effects: no comparison survived its all-20 Holm family, and all T-cell intervals at caps 4–16 included zero. Excluding pairs in the blind upper-quartile poly-U family (`longest_poly_u >= 3`, frozen from candidate features only) retained negative estimates for much of the T-cell curve. Only T-cell r1 cap 4 (−0.950 [−1.577, −0.363]) and T-cell r2 cap 8 (−1.945 [−2.993, −0.832]) survived that family's all-20 Holm correction (both p=0.016); cap-16 results were nominal but not corrected-significant. Reduced retained pool counts, especially in N50, limit precision. These are heterogeneity and sensitivity findings, not evidence that uAUG or poly-U causally mediates optimization.

**Inference and next question.** Equal-budget ranking value and composition-sensitive attenuation can coexist: the first isolates a decision benefit under the original contract, and the second limits its interpretation and portability. I therefore test optimizer-specific and shared error structure without treating adjustment as a causal decomposition.

### 3.3 Score matching does not identify an optimizer-specific error increment

For each cap-16 BoN selection, we selected a different sequence from the same pool with the nearest proxy mean. If optimized candidates had consistently larger absolute errors than these score-matched candidates, this would support an increment associated with the optimized identity beyond the matched score. Instead, all four pool-cluster intervals included zero and the directions differed (Table 3; Fig. 2).

**Table 3 | Strict-H1 score-matched absolute-error differences.**

| Context | BoN MAE | Matched MAE | BoN − matched | Nominal 95% interval |
|---|---:|---:|---:|---:|
| N50 HEK293T | 2.881 | 2.707 | +0.174 | [−0.444, +0.712] |
| Defined HepG2 r1 | 3.277 | 2.951 | +0.326 | [−0.796, +1.484] |
| Defined T-cell r1 | 1.615 | 1.918 | −0.303 | [−0.897, +0.148] |
| Defined T-cell r2 | 1.566 | 1.663 | −0.097 | [−0.495, +0.326] |

This null result does not prove that high-score extrapolation is the causal mechanism. Matching controls only proxy mean and leaves sequence composition and higher-order distribution differences uncontrolled. It also does not show that optimization is harmless. The supported conclusion is narrower: this protocol did not detect a stable optimizer-specific increment. Because score matching alone could not localize shared model error, we next asked whether changing architecture removes that error structure.

### 3.4 Proxy errors remain strongly coupled across architectures

We compared the frozen CNN ensemble with a separately trained 4-mer MLP using the same Sample train/development labels. CNN mean absolute error was lower in every context, and all paired CNN-minus-MLP intervals excluded zero (Table 4; Fig. 3). Nevertheless, within-pool absolute-error correlations ranged from 0.878 to 0.962, with all nominal intervals far from zero.

**Table 4 | Strict-H1 absolute-error concordance across architectures.**

| Context | CNN MAE | 4-mer MAE | CNN − MLP [95% interval] | Error correlation [95% interval] |
|---|---:|---:|---:|---:|
| N50 HEK293T | 2.805 | 2.852 | −0.047 [−0.077, −0.020] | 0.962 [0.956, 0.967] |
| Defined HepG2 r1 | 2.919 | 3.029 | −0.110 [−0.152, −0.065] | 0.953 [0.946, 0.959] |
| Defined T-cell r1 | 1.567 | 1.821 | −0.254 [−0.297, −0.209] | 0.878 [0.859, 0.897] |
| Defined T-cell r2 | 1.544 | 1.739 | −0.195 [−0.231, −0.160] | 0.933 [0.919, 0.944] |

Lower average error and independent error evidence are therefore different properties. Because both architectures shared training labels and endpoint definitions, shared lineage is a plausible contributor, but these correlations do not causally apportion error to training data, architecture, sequence features, or assay effects. Additional same-data models should not be treated as independent functional validation.

The normalized-output noise experiment further constrained a simple mechanism account. At N50 cap 16, regrets for Gaussian noise standard deviations 0, 0.1, 0.25, 0.5, and 1.0 were 6.519, 6.018, 6.143, 6.205, and 6.526. There was no monotone degradation. This exploratory protocol does not recommend noise injection and does not represent all verifier misspecification; it only rejects inevitable monotone amplification in the tested setting.

**Inference and next question.** Architecture changes and same-data voting cannot certify functional correctness. However, all analyses so far concern MRL. A proxy contract must also specify the intended endpoint, leading to the cross-endpoint analyses below.

### 3.5 Cross-endpoint analyses show that rankings are not interchangeable

The Schlusser HepG2 TE evaluation was frozen before label access and unblinded once after three Learner rounds. It is confirmatory within its declared cross-endpoint protocol, but not a new wet-lab experiment, same-endpoint MRL replication, or independent-study meta-analysis. The deployment endpoint was endogenous mRNA TE derived from published RNA-seq/Ribo-seq, whereas both proxies were trained to predict MRL.

At cap 16, BoN-CNN mean TE regret was 4864.01. The paired BoN-CNN-minus-Random regret difference was 0.626608, with gene-cluster 95% interval [−0.828459, 9.99825] and transcript-cluster interval [−0.892444, 9.96536] (Fig. 4a). Thus, no evidence of BoN-CNN improvement in TE was observed. The point direction favored Random, but both intervals included zero, so a precise Random advantage is also not claimed. Descriptive CNN MRL–TE Spearman correlation was 0.072073. TE was extremely right-skewed (median 0.686604; maximum 4864.903), making absolute regret dominated by the within-pool maximum. The paired contrast removes the shared oracle but remains uncertain.

Leppek provided an independent, exploratory endpoint audit without model fitting or inferential testing. Among 24 HBB-Nluc CDS constructs, in-solution half-life and 6 h Nluc/Fluc had Pearson correlation −0.192 and Spearman correlation −0.307. Of 276 unordered construct pairs, 171/276 (62.0%) were discordant, and only 1/6 top-ranked constructs overlapped (Fig. 4b). In 191 pooled constructs with both measurements, MRL and in-solution half-life had Pearson correlation −0.346. The CDS and pooled construct sets were not joined as if they provided paired MRL–expression measurements, and `Predicted expression` was never treated as measured protein expression.

Together, these analyses support endpoint-specific proxy contracts. Schlusser supplies a frozen one-time test of an MRL selector deployed to TE; Leppek describes stability–expression non-equivalence in small, defined construct sets. Neither establishes a universal quantitative failure rate or a biological causal mechanism.

### 3.6 The complete PERSIST-seq MLP contract shows no detectable transfer

A separate retrospective external evaluation in the PERSIST-seq study of Leppek et al. [11] used measured ribosome load as the same MRL endpoint but an independent study and different variable-length constructs. The frozen `UTR (5') variants` cohort contained 59 eligible constructs spanning 49 unique 5′ UTRs. Exact and equal-length Hamming-1 audits against the combined Sample train/development/calibration reference found zero conflicts, so no construct was excluded. Labels had previously been inspected: this was neither a blind test nor prospective validation.

The three existing 4-mer MLP members and their training scalers were frozen. Native full-length 5′ UTRs were encoded without truncation, N-padding, recalibration, or new training; deterministic float32 ensemble inference ran on CUDA. With all 59 constructs fixed as the candidate pool, 200 paired algorithm seeds, and cap 16, BoN-minus-Random measured-MRL regret was **+0.104723** (95% UTR-cluster bootstrap interval **[−0.100964, 0.212706]**; two-sided sign **p=0.245175**). Positive values favor Random. Prediction–measurement Spearman correlation was **−0.008447** [−0.326701, 0.315406] (Fig. 13). Thus, this completed small retrospective replication showed no detectable transfer of ranking signal and did not support a BoN benefit; the interval crosses zero and does not establish ineffectiveness.

This result does not directly contradict the T-cell pressure gains, which used different constructs, study contexts, and the fixed-length CNN selector. Instead, it sharpens the proxy contract: training lineage, architecture, and construct must be specified together with endpoint and context, and a benefit observed under one contract cannot be transported to another without evaluation. One small, uncertain external result is not proof of universal proxy failure.

A source-code audit confirmed the frozen tie rule: prediction descending, then ascending `Sequence ID`, then row position. A retrospective SHA256 alternative tie rule changed neither the five order-sensitive original selections nor the primary effect, 95% interval, or sign p value. This sensitivity supports numerical robustness to the tested tie rule without upgrading the evidence state.

**Inference and next question.** The full PERSIST MLP contract supplies no detectable transfer, not a direct refutation of the T-cell CNN result. Any risk-control or repair proposal therefore requires its own gate and cannot inherit validity from either the equal-budget gain or the external null.

### 3.7 Alternative selectors and pool-level Q prediction do not solve the decision problem

The next question was whether decision rules could control the observed risk without deployment labels. We evaluated two families: verifier-guided selection and conservative/uncertainty/diversity reranking. All outer results in this section are retrospective sensitivity analyses on previously viewed pools.

In the designed-outer VGSC-inspired pilot, FP-REJ without Q had mean regret 1.024 versus 0.681 for BoN at cap 16. The paired difference was +0.343 [0.225, 0.467], favoring BoN (Supplementary Fig. S2.1). The original Q configuration rejected all 16 pools, giving zero coverage and undefined post-Q regret. Three eligible random pools yielded a small FP-REJ+Q difference of −0.055 [−1.400, 1.051], insufficient to support an advantage. FP-TILT did not provide equal-cost superiority evidence because it scored the full support and exceeded the common CNN budget.

The baseline comparison used 48 CNN member item-forwards for every designed cap-16 method. BoN regret was 0.672377. COM-style and diversity-LCB regrets were 0.661818 and 0.640713, but their paired differences from BoN included zero: −0.010559 [−0.045744, 0.017583] and −0.031664 [−0.098743, 0.030134]. Development selection chose zero penalties for LCB and distance, making them identical to BoN. In the three-pool random-outer sensitivity analysis, COM-style selection was worse than BoN by +1.811064 [0.123119, 2.660000]. The COM comparator is an adaptation, not the official COM UTR-ResNet or official weights.

Selector failure does not imply that risk prediction must succeed. We tested this separately using the pool-level Q head. Recalibration improved calibration log loss from 0.327 to 0.069, but probability calibration did not translate into regret discrimination. On 16 held-out calibration pools, Q versus FP-REJ regret had Spearman ρ = −0.353 [−0.795, 0.193], Q versus BoN regret had ρ = −0.200 [−0.718, 0.401], and top-half BoN-regret AUC was 0.297 [0.094, 0.727]. Rejecting the 25% or 50% highest-Q pools increased rather than decreased accepted regret relative to full coverage. Q therefore did not discriminate realized pool-level regret in this evaluation.

### 3.8 Candidate-level uncertainty abstention fails its frozen calibration gate

A separate analysis used ensemble standard deviation to abstain on high-uncertainty BoN candidates rather than to rerank them. The earlier designed-outer curve was constructed by flattening pool-by-seed selected decisions and computing empirical quantiles on those same viewed outer data. Accepted regret was 0.458902 at 10% empirical coverage versus 0.672377 at full coverage (Supplementary Study S2), but these outer empirical quantiles supplied no numeric deployable threshold and do not constitute blind evidence.

The frozen v1 protocol selected a numeric threshold using development only. Development selected **q = 0.50** from {0.50, 0.75, 0.90}, yielding threshold **0.104026897** (full precision 0.10402689717871444). Its point-estimate selection rule did not imply a development benefit; all three development intervals included zero. Calibration retained that threshold and yielded **coverage 0.425** (34/80 decisions, 15/16 pools), accepted mean regret **2.866883**, and full-BoN mean regret **2.861463**. The primary reduction, full-BoN minus accepted regret, was **−0.005420**, with 95% pool-cluster interval **[−0.509271, 0.628623]**. The point estimate was negative and the interval included zero, so both effect gates failed.

Because calibration did not pass, the outer-scoring invocation was rejected by a hard lock. Outer was not scored, no new outer result was generated, and no alternative threshold was tested. The retrospective signal therefore did not pass the frozen calibration gate. We make no claim that abstention improves regret, and the historical outer curve cannot reverse this result or serve as blind evidence.

### 3.9 Label-repair headroom is representation dependent; acquisition efficacy remains diagnostic

The final question was whether revealing historical labels could change the deployed top-1 choice. Supplementary Study S3 used a common Bayesian linear last layer with three frozen representations: summary statistics (`summary4`), normalized 4-mer frequencies (`kmer4`), and CNN hidden features (`cnn64`). All 48 zero-label/full-query gates were evaluated before interpreting acquisition curves. These are development/calibration diagnostics; permanent outer labels were neither read nor scored.

Full-query headroom was present for 5/16 `kmer4` pools, 1/16 `cnn64` pools, and 0/16 `summary4` pools (Fig. 6a). The largest `kmer4` headroom was 3.686 MRL units. This establishes that top-1 repair is possible for some representation–pool combinations under the diagnostic full-information setting, not that labels will generally help or that an acquisition rule is efficient.

On the V2-frozen five-pool `kmer4` curve cohort, expected improvement (EI) at 16 labels had pooled regret 1.220084, compared with random 3.497016, maximum variance 2.644595, and regret-analytic 2.131689 (Fig. 6b). At 32 labels, maximum variance was best (2.068558), while regret-analytic was 2.131689. Thus, no complex acquisition strategy dominated across budgets. The V4 analytic approximation converged numerically at 384/384 acquisition steps, but it assumes a diagonal independent fantasy posterior; this convergence supports numerical reproducibility, not biological or outer effectiveness. The `summary4` curve remained flat even with all query labels, and individual `kmer4` pools retained non-monotone and unfavorable outcomes.

**Inference and next question.** The positive diagnostic is representation-dependent repair headroom, with simple EI sufficient to realize gains in some fixed-cohort budgets. The unsupported claim would be general label-efficient repair. A future outer evaluation is gated on a frozen acquisition protocol and an independent risk-control result, rather than on these calibration curves alone.

## 4. Discussion

This study replaces a binary account of proxies as “reliable” or “unreliable” with an explicit decision contract. At equal candidate budget, T-cell r1 showed Holm-robust ranking gains at every cap 2–16, while T-cell r2 retained corrected support only at cap 4. HepG2 cap 2 was also corrected-positive, but the remaining HepG2 and N50 results prevent a context-free claim. The original fixed-pool pressure results remain concordant historical evidence, but the Random(N) contrast—not candidate-count expansion—is the primary isolation of proxy-ranking value.

The composition results are a substantive boundary, not a footnote. Adjustment for GC, uAUG, poly-U, and support attenuated every original T-cell effect; no adjusted comparison survived all-20 Holm. The contrast between broad attenuation after uAUG exclusion and retained negative estimates in selected poly-U-excluded strata demonstrates sensitivity to the analyzed subset and feature definition. These analyses cannot show that any sequence feature caused, mediated, or refuted proxy-ranking value. The contract must therefore name candidate composition, training lineage, library, cell, construct, assay, endpoint, selection rule, and evidence state.

The completed PERSIST-seq evaluation adds an independent-study, same-endpoint boundary rather than a direct refutation of those gains. Its 59 constructs, 49 UTR clusters, different study and constructs, and frozen variable-length 4-mer MLP define a different proxy contract from the fixed-length CNN T-cell analyses. BoN-minus-Random regret was +0.104723 [−0.100964, 0.212706] (p=0.245175), with Spearman −0.008447. This small retrospective analysis therefore shows no detectable transfer, while its uncertainty and historical label contact preclude claims of universal failure or blinded confirmation. Source inspection established prediction-descending/`Sequence ID`/row-order tie handling; the SHA256 alternative left the effect, interval, and p value unchanged. Training lineage, architecture, construct, and deterministic selection details are necessary components of the contract, not optional implementation details.

The mechanistic sequence is deliberately conservative. Score matching did not detect a stable optimizer-specific increment; it did not prove high-score extrapolation. Strong cross-architecture error correlations showed that architectural diversity did not yield independent error evidence; they did not causally identify shared training data as the dominant source. Cross-endpoint results showed that MRL, TE, stability, and reporter expression cannot be assumed interchangeable; they did not estimate a universal transfer failure rate. Negative normalized-output noise results further argue against a single, inevitable noise-amplification mechanism.

These results motivate decision-layer tests without guaranteeing their success. The verifier-guided pilot and conservative/diversity baselines did not establish a general selection advantage. Pool-level Q probabilities could be calibrated while remaining uninformative about realized regret, illustrating that calibration and decision discrimination are distinct. Candidate-level uncertainty abstention also failed its frozen calibration gate: development selected q = 0.50 and threshold 0.104026897, but calibration reduction was −0.005420 [−0.509271, 0.628623] at coverage 0.425. Outer scoring was hard-locked and not performed. The historical outer empirical-coverage curve is retrospective sensitivity evidence only and cannot be converted into a blinded result.

The label-repair analysis likewise separates possibility from realized efficiency. Some representations allow full-query labels to change top-1 regret, whereas `summary4` does not. Simple EI can outperform other strategies at selected budgets, but regret-analytic acquisition does not dominate and outer performance is unknown. S2's failed gate removes the evidentiary basis for claiming a validated sequential risk-control-and-repair system; S3 remains a standalone diagnostic.

For thesis and publication organization, the strongest current article is the Paper 1 phenomenon/mechanism study reported here, anchored by reproducible T-cell pressure gains, cross-architecture error concordance, and endpoint non-equivalence. S2 is a self-contained limitations-and-falsification supplement; S3 is a retrospective repair-diagnostic supplement. Neither supplement is presented as an independently validated positive method paper. The negative results remain useful boundaries and are not promoted into positive claims.

## 5. Limitations

**Retrospective evidence and repeated outer use.** Sample and Castillo outer pools had been inspected before the strict-H1 reconstruction and selector/baseline sensitivity analyses. They are not new blind tests. The Schlusser TE evaluation was frozen and unblinded once within its own protocol, but it evaluates a single cross-endpoint study. The Leppek CDS endpoint audit is exploratory and descriptive. PERSIST-seq labels had also been inspected historically, so its completed same-endpoint external evaluation is retrospective, not blind or prospective. None of these analyses constitutes prospective wet-lab validation of new sequences.

**Independence and external replication.** Castillo contexts are external to Sample training but belong to one study. T-cell contexts and deterministic pool builds are not independent prospective biological replications. A small independent-study, same-endpoint PERSIST-seq MRL evaluation has now been completed and did not show transferable ranking signal: BoN-minus-Random regret was +0.104723 [−0.100964, 0.212706], p=0.245175, and Spearman was −0.008447. Its different constructs and variable-length MLP prevent direct transport or direct contradiction of the T-cell CNN findings; its 49 clusters and interval crossing zero do not prove absence of an effect. Gene, transcript, UTR, or pool bootstrap clusters quantify within-study sampling uncertainty and do not create additional independent studies.

**Selection budgets and retrospective multiplicity.** The historical strict-H1 pressure curve compares BoN(N) with single-proposal Random(1) and is not equal-budget; V5 therefore prioritizes the new BoN(N)-minus-Random(N) sensitivity. Random(N) uses independent randomized positions on the same prefix and equal candidate count, but repeated algorithmic draws are not independent biological replication. Holm and Bonferroni corrections are complete all-20 retrospective sensitivity families including structural cap-1 zeros, not prospective confirmatory testing. Baseline comparisons at cap 16 match CNN member item-forwards, but verifier, Q, feature-generation, and training costs can differ. FP-TILT results are not equal-cost evidence. Development-selected zero LCB and distance penalties make those comparisons identical to BoN and their permutations trivial.

**Mechanism and composition identification.** Score matching controls proxy mean only. The sequence adjustment is an observational ordinary-least-squares sensitivity with pool fixed effects and four selected-sequence contrasts; it is not a randomized feature intervention, mediation analysis, or proof that composition explains the gains. Family exclusions discard complete selected pairs rather than rerank candidates, reduce retained pools and precision, and define sensitivity strata rather than causal effects. Cross-architecture models share training labels and endpoints, and error concordance cannot apportion causality. Gaussian normalized-output noise is only one perturbation family.

**Endpoint interpretation.** MRL and TE have different units and biological meanings; cross-endpoint absolute error was not computed. TE is not direct protein output, and extreme right skew makes absolute TE regret difficult to interpret. Identical UTR50 sequences can map to transcripts with different TE values that a fixed 50-nucleotide model cannot distinguish. Leppek CDS and pooled constructs are separate sets, and predicted expression is not a measured endpoint.

**PERSIST external-evaluation scope.** The analysis covers one small 5′ UTR variant set (59 constructs, 49 UTR clusters) from one independent study, with variable-length native inputs and the existing 4-mer MLP. It is not a blind test, a direct replication of the T-cell CNN contract, a reads/FASTQ-level reproduction, or evidence about other cells, constructs, endpoints, or architectures. Non-significance is not proof of no effect, and repeated algorithm seeds are not independent biological samples.

**Comparator scope.** The conservative comparator is a COM-style adaptation rather than official COM UTR-ResNet reproduction. Generative methods such as T3PO-mRNA and mRNAutilus operate on a different evaluation surface and were not compared to fixed-pool reranking [16,17]. No conclusion is made about de novo generation quality.

**Risk-control and repair status.** The historical candidate-level uncertainty curve used empirical quantiles on previously viewed outer data and is not blind evidence. The frozen development/calibration protocol selected q = 0.50 and threshold 0.104026897, then failed at calibration (coverage 0.425; reduction −0.005420 [−0.509271, 0.628623]); outer scoring was hard-locked and not performed. Q calibration improvement is not regret prediction. Paper3 V4 uses development/calibration only; full-query headroom is a diagnostic upper information setting, not an unbiased deployable estimate. The regret-analytic target uses a diagonal independent fantasy-posterior approximation. Numerical convergence and three-run identity do not establish generalization.

**Scope and review.** Conclusions are limited to historical 5′ UTR/mRNA datasets and the tested models. They do not establish in vivo, animal, or clinical efficacy; do not authenticate unmeasured designed sequences; and do not show that uncertainty can replace experiments. The manuscript has not undergone external peer review.

## 6. Methods

### 6.1 Data roles and evidence taxonomy

**Table 5 | Data roles and permitted conclusions.**

| Dataset | Role | Evidence state | Not licensed |
|---|---|---|---|
| Sample 2019 GSE114002 | Proxy training; designed/random fixed-pool historical context | Retrospective; training lineage | Prospective design validation or step-trajectory replay |
| Castillo-Hair 2024 GSE232927 | N50 HEK293T, defined HepG2, two defined T-cell contexts | Strict-H1 retrospective protocol sensitivity; V5 equal-budget Random(N) and sequence-composition sensitivities reuse the same historical outer contexts | New blind test, prospective replication, independent-study meta-analysis, or sequence-mechanism causation |
| Schlusser 2024 HepG2 | Endogenous TE deployment of MRL proxies | Frozen one-time cross-endpoint evaluation | Same-endpoint MRL replication or direct protein measurement |
| Leppek 2022 GSE173083 CDS/pooled constructs | Stability, MRL, and reporter-expression endpoint audit | Exploratory descriptive | Generalizable significance, causal mechanism, or MRL–expression pairing across construct sets |
| Leppek 2022 PERSIST-seq GSE173083 `UTR (5') variants` | Independent-study, same-endpoint MRL evaluation of frozen variable-length 4-mer MLP and BoN | Completed small retrospective external evaluation; 59 constructs/49 UTRs; no transferable ranking signal | Blind/prospective validation, direct T-cell CNN refutation, universal failure, or a BoN benefit |
| Sample development/calibration pools | V/Q analysis, frozen candidate-risk gate, and Paper3 replay | Retrospective diagnostics; candidate-risk calibration failed; no Paper2 outer score or Paper3 outer access | Blind abstention benefit or outer-validated label efficiency |

No eligible complete historical step-by-step measured optimization trajectories were available. The pressure analysis therefore uses fixed-pool BoN and does not claim trajectory replay. Raw and derived artifacts remain under the frozen project storage; local manuscript verification uses aggregate source data and, for Leppek/PERSIST-seq, construct-level measured fields.

We use three evidence labels throughout: **confirmed within a stated retrospective or one-time protocol** for numerical phenomena supported by the corresponding frozen analysis; **retrospective diagnostic or external evaluation** for mechanism, selector, repair, and PERSIST analyses that do not constitute untouched blind validation; and **frozen-gate failure or unevaluated deployment** for candidate-level uncertainty abstention and label efficiency, respectively. Candidate-risk calibration failed and outer was not scored; label-efficiency outer evaluation has not been performed. “Confirmed” never means prospectively wet-lab validated. Extended Data Fig. 2 summarizes governance controls and the evidence map at the V3.1 freeze; Fig. 13 adds the completed PERSIST same-endpoint evaluation. Cross-endpoint OOD and the PERSIST result answer different deployment questions.

### 6.2 Functional regret and selection pressure

For candidate pool \(C_t\), proxy score \(P_i\), measured endpoint \(Y_i\), and selection strategy \(\pi\), regret is

\[
R_t^\pi = \max_{i\in C_t}Y_i - Y_{x_t^\pi}.
\]

Lower regret is better. BoN maximizes the ensemble-mean proxy among the first \(N\) proposals, with caps \(\{1,2,4,8,16\}\) and seeds \(\{101,202,303,404,505\}\). Historical Random(1) uses the first proposal; original strict-H1 effects are BoN(N) regret minus same-seed Random(1) regret averaged within each pool. V5's primary equal-budget contrast instead uses Random(N) as specified below. Candidate pool size is fixed within each context. Abstention is reported separately with coverage and is never assigned an artificial regret value.

### 6.3 Equal-budget Random(N) PCG64 sensitivity

The budget-matched analysis reused all 16 pools per context, all five original seeds, and caps 1/2/4/8/16 without model training, tuning, or inference. Random(N) uniformly selected one position from the exact same nested proposal prefix used by BoN(N). A NumPy `SeedSequence(20261125)` spawned independent record-specific PCG64 child streams keyed by `(context_index, pool_index, seed_index, cap_index)`; each child made one `integers(0, N)` draw. Cap 1 was therefore a structural identity. Original Random/BoN selected IDs, indices, and regrets were reconstructed before interpretation; local manuscript verification independently recomputes aggregate effects from the de-identified pool table and checks the existing raw/independent/delivery rounds.

The three contrast families were BoN(N)−Random(1), BoN(N)−Random(N), and Random(N)−Random(1). Each used 10,000 complete-pool cluster percentile bootstrap replicates, with new base seed 20261125 and deterministic offsets; the original family retained its base seed 20261124. Holm and Bonferroni p values were computed separately over each complete all-20 context–cap family, explicitly including cap-1 structural zeros with p=1. Nominal intervals were 95%; the all-20 Bonferroni 99.75% intervals were retained in machine-readable results. Bootstrap p values are retrospective sensitivity quantities. The primary positive claim is restricted to the five equal-budget comparisons with Holm p<0.05: all four T-cell r1 cap 2–16 rows and T-cell r2 cap 4. HepG2 cap 2 is retained as an additional context-specific positive boundary, not collapsed into the T-cell claim.

### 6.4 Sequence fixed effects and family sensitivity

The sequence-ablation sensitivity reused all 4,096 candidates and 1,600 original BoN/Random(1) pairs. The four predeclared covariates were continuous GC fraction, the existing uAUG indicator, longest contiguous poly-U length (T=U), and the existing train-support value unchanged. For caps 2–16, ordinary least squares separately within each context and cap included all 16 pool indicators plus the four BoN-minus-Random feature contrasts. The adjusted effect was the mean pool intercept, representing the estimated paired effect at zero feature contrast; it was not interpreted as a causal or mediated effect. Cap 1 was retained as a structural zero, yielding complete all-20 Holm and Bonferroni families. Pool-cluster percentile bootstrap used 10,000 replicates and seed 20261125.

The longest-poly-U upper-quartile cutoff was frozen from all candidate sequence features without MRL, selection, regret, or mechanism results. The nearest-rank 75th percentile was 3; the family was `longest_poly_u >= 3` and contained 1,591/4,096 candidates because of ties. Family sensitivities removed a complete BoN/Random pair when either selected sequence was uAUG-positive or belonged to that poly-U family. No candidate was reranked, replaced, or re-optimized. Each family retained all 20 comparisons including cap 1 and used its own all-20 corrections. All attenuated, null, sign-reversed, reduced-pool, and corrected-significant rows were retained. The analysis is retrospective and exploratory; it establishes composition sensitivity rather than sequence causation.

### 6.5 Frozen CNN and alternative architecture

The primary proxy is a three-member 50-nucleotide CNN regression ensemble trained on Sample random-library data with seeds 11, 23, and 37. Training used batch size 1024, Adam learning rate 0.001, minimum development-loss change 0.0001, patience 5, and no maximum-epoch stopping rule. Best epochs were 62, 50, and 77. Member outputs were inverse transformed with member-specific train-only target scalers. The earlier normalized-output ensemble-v1 lineage and downstream artifacts were invalidated and excluded.

The alternative proxy was a three-member 4-mer-frequency MLP with 259 sequence features, trained on the same Sample train/development labels with train-only target scaling. Seeds 101, 202, and 303 converged at epochs 108, 124, and 177, with best epochs 103, 119, and 172. This model is an architecture sensitivity analysis, not a new main predictor or independent biological validator.

All original CNN/MLP training and reported model inference used explicit CUDA with no CPU fallback. The present manuscript restructuring and independent statistical verification used CPU only and launched no model training or inference.

### 6.6 PERSIST-seq independent-study, same-endpoint external evaluation

The frozen primary cohort was the author-provided PERSIST-seq/GSE173083 `UTR (5') variants` group. Eligibility required a unique Sequence ID, an ACGT 5′ UTR of at least four nucleotides, and finite measured `Ribosome load`; no filtering by labels or predictions was allowed. All 59 constructs and 49 unique UTRs passed. Combined Sample train/development/calibration references were audited for exact and equal-length Hamming-1 conflicts; any conflict would exclude the complete UTR, but both conflict counts were zero. The processed author XLSX was used without FASTQ/reads-level reproduction. Historical label inspection makes this a retrospective external evaluation, not a blind test.

Predictions used the existing three-member 4-mer MLP with frozen weights and training scalers. Each native full-length 5′ UTR was represented by 256 normalized 4-mer frequencies, GC, uAUG, and longest poly-U/50 (259 features), with arithmetic averaging across members. No model was trained, no 50-nucleotide CNN was substituted, and no truncation, N-padding, or recalibration was performed. Inference was deterministic float32 on `cuda:0` with TF32 disabled and no CPU fallback; proposals and statistical analyses used CPU.

The full 59-construct set remained fixed as the candidate pool. For each of 200 PCG64 seeds (20261125–20261324), Random selected the first permutation candidate and BoN selected the highest-predicted candidate among the first 16. Source inspection confirmed that maximum-prediction ties were resolved by ascending `Sequence ID` and then row position, equivalently sorting `(-prediction, Sequence ID, row position)`; this was also the frozen protocol rule, not stable-first selection. Regret was the pool maximum measured MRL minus selected measured MRL; the primary effect was same-seed BoN minus Random, so negative values favor BoN. The 95% percentile interval used 10,000 bootstrap resamples of 49 complete UTR clusters (seed 20261126), rebuilding proposals and oracles within each resample. The two-sided bootstrap sign p value followed the frozen formula. A significant BoN benefit required an interval entirely below zero and p<0.05; neither the point estimate nor the interval supported that rule. The observed +0.104723 [−0.100964, 0.212706], p=0.245175, and Spearman −0.008447 were retained without changing subset, seeds, cap, endpoint, or model.

A retrospective tie audit found nine original seeds with exact maximum-prediction ties, five of which were sensitive to stable-first ordering. The alternative rule, fixed before inspecting effect direction, sorted tied candidates by ascending SHA256 of the UTF-8 original `Sequence ID`, then `Sequence ID`, then row position. It changed none of the five original selected effects. Bootstrap updates were restricted to those five seeds; 79 bootstrap seed selections and 78 replicate-level subeffects changed, with maximum single-cell absolute change 0.0011278032, but the 200-seed primary effect, 95% UTR-cluster interval, and two-sided sign p value were exactly unchanged. Frozen v1 remains primary, v1.1 is a retrospective sensitivity, and no version was selected by direction. The audit used CPU only, performed no training or inference, and left the original scientific inputs and results unchanged.

### 6.7 Strict-H1 pool reconstruction and isolation

Each Castillo context contained 16 pools of 64 unique 50-nucleotide sequences with finite measured MRL. N25 was excluded because the frozen model requires 50-nucleotide inputs. Strict-H1 pools (`castillo_fixed_pools_v3_strict_h1_retry1`) excluded exact and Hamming-1 conflicts within/across pools and against the combined Sample train/development/calibration reference. Two deterministic builds and learner, label, manifest, raw, reference, and pool hashes were checked. Hamming-2 counts were a predeclared diagnostic, not an exclusion gate. The Learner had no measured labels; the Evaluator read labels only after selection artifacts were frozen.

All strict-H1 intervals used 10,000 complete-pool bootstrap replicates with base seed 20261124 and deterministic offsets. Nominal intervals were 95% percentile intervals. Multiplicity sensitivity used the same bootstrap draws for all-20 Bonferroni 99.75% intervals and within-context five-comparison 99% intervals. These adjusted intervals are conservative sensitivity analyses, not new confirmatory tests.

### 6.8 Score matching, common error, and noise sensitivity

For each cap-16 BoN selection, a different sequence with the nearest proxy mean in the same pool was selected with deterministic hash tie-breaking. The endpoint was optimized-minus-matched mean absolute error. Cross-architecture analyses computed CNN-minus-MLP mean absolute error and within-pool Pearson correlation of candidate absolute errors, followed by pool-cluster summaries. High correlation was interpreted as non-independence, not as a causal decomposition.

Alternative N50 pools used a separate deterministic namespace and the same protocol. Noise was added independently to normalized CNN member outputs before member-specific inverse transformation, with \(\sigma\in\{0,0.1,0.25,0.5,1.0\}\). Sigma-zero rows were required to reproduce the original grid by strategy, cap, and seed. This was not probability-logit noise.

### 6.9 Schlusser cross-endpoint evaluation

The frozen Learner contained 6,340 rows, 6,309 genes, and 6,340 transcripts. CNN and 4-mer proxies predicted MRL; deployment labels were published HepG2 endogenous TE. One hundred paired algorithmic seeds sampled 16 proposals with replacement; caps used nested prefixes. Random chose the first proposal, BoN maximized the corresponding MRL ensemble mean, and LCB-CNN used mean minus one sample standard deviation. The primary comparison was cap-16 BoN-CNN versus Random.

Regret used the maximum TE within the frozen 6,340-row candidate set. Gene-cluster bootstrap (2,000 replicates, seed 20261124) was primary and transcript-cluster bootstrap (seed 20261125) was sensitivity. Each resample rebuilt candidate pools, proposals, selections, and the within-sample oracle. Three complete CUDA Learner rounds preceded one Evaluator unblinding. No cap, model, baseline, or statistic was changed after results. MRL and TE were never subtracted as absolute error. Descriptive Spearman correlation did not replace regret.

### 6.10 Leppek endpoint audit

The CDS analysis contained 24 HBB-Nluc constructs with complete in-solution half-life, 6 h Nluc/Fluc, and 24 h Nluc/Fluc. The pooled analysis contained 233 constructs with MRL and polysome measures; in-cell and in-solution half-life were available for 203 and 191 constructs, respectively. Complete pairs were used without imputation. `Predicted expression` remained a predicted field and was excluded from measured expression.

Pearson and Spearman correlations were descriptive, with no p values or confidence intervals. For 24 CDS constructs, all 276 unordered pairs were classified as concordant or discordant. Top-6 overlap was fixed a priori at 25% of 24. The analysis did not fit models, access outer results, or join CDS expression to pooled MRL as paired measurements.

### 6.11 Verifier-guided and baseline selectors

The V head estimated the probability that ensemble mean exceeded measured MRL by a development-frozen margin; \(V=1-r\). FP-REJ used development-frozen \(\alpha=20\) and cap 16. Q was a pool-level high-regret probability used only for abstention, not ranking. Thresholds and calibration were development/calibration procedures. The original outer pilot remained a falsification result and was not used to retune \(\alpha\), cap, threshold, features, or model class.

The additional selectors were BoN, ensemble LCB, train-distance penalty, COM-style conservative ensemble, and diversity-LCB. LCB and distance penalties minimized development mean top-1 regret; diversity-LCB minimized development regret@4. At designed cap 16, each method used 48 CNN member item-forwards. COM-style training used 50-step discrete adversarial negatives and development-based checkpointing, but differed from official COM in architecture, targets, split, and fixed-pool evaluation. Thirty-two deterministic within-proposal permutations were retained as negative controls.

### 6.12 Q discrimination and uncertainty risk–coverage

Q analysis used 32 development pools (in-sample for the Q calibrator) and 16 honest calibration holdout pools. Spearman and Pearson associations were computed between Q probability and realized BoN or FP-REJ regret, with 10,000 pool-cluster bootstrap replicates. High-regret AUC used top-half BoN regret. Q-based abstention at 25% and 50% was compared with full coverage.

Candidate-level risk–coverage historically ranked BoN candidates by ensemble standard deviation and removed high-risk selected decisions at empirical coverage fractions. Those thresholds were descriptive quantiles from the same viewed outer role, not deployment or blind thresholds. The 0.459 value at 10% coverage and 0.672 full-coverage value are retrospective sensitivity results only. Frozen v1 instead used development selected-decision risks to choose q = 0.50 and threshold 0.10402689717871444; acceptance required ensemble standard deviation no greater than that threshold. Calibration was evaluated once with the threshold unchanged. The primary endpoint was full-BoN mean regret minus accepted mean regret, with a 10,000-replicate pool-cluster percentile bootstrap (seed 20261125). Calibration coverage was 0.425, accepted regret 2.866883 versus full-BoN 2.861463, and reduction −0.005420 [−0.509271, 0.628623]. Positive point reduction and a lower interval bound above zero were both required; neither occurred. The hard lock therefore prevented outer scoring, and no outer metrics were generated.

### 6.13 Bayesian label-repair replay

All acquisition strategies shared a Bayesian linear last layer on a frozen representation \(\phi(x)\):

\[
y=\phi(x)^Tw+\epsilon,\qquad \epsilon\sim N(0,\sigma^2).
\]

Feature/target scaling and hyperparameters used development data only. Four-fold development-pool-blocked cross-validation selected prior variance 0.1 and noise variance 0.16 for all three representations. Deployment selected posterior-mean top-1. The 48 gates compared zero-label and full-query regret. Curves were restricted to the six V2-frozen representation/pool cases, including five `kmer4` pools and one `summary4` pool, with budgets 0, 4, 8, 16, 32, and 64.

Strategies were random (three seeds), maximum latent posterior variance, diversity, analytic expected improvement, top-proxy, and regret-analytic acquisition. Random was averaged within pool before cross-pool aggregation. All strategies used the same posterior updates, budgets, and Hamming≤8 within-batch exclusion interface.

V4 regret-analytic acquisition used Gauss–Hermite quadrature for hypothetical labels and normal-max values with fixed 64 label nodes and 32/64/128 value nodes; final values used (64,128) and convergence threshold \(10^{-5}\). It approximated the fantasy posterior as diagonal independent across candidates. Strict convergence and top-choice stability were numerical diagnostics, not evidence of outer acquisition efficacy. Permanent outer labels and metrics remained untouched.

### 6.14 Reproducibility and manuscript verification

The strict-H1 source commit was `ecaccfafd12d81258b8cd0648b48175398b6d98c`; fresh CNN training commit was `fd5e673785612a34a0f6e334761e190d681449dc`; COM-style training commit was `5c9f2547efad919ffcaf4b00e05e07e70ad3493e`; Schlusser used its frozen configuration/model hashes; Paper3 V4 final implementation was frozen at `b2644f8`. Historical failed and invalidated artifacts were retained and excluded from scientific conclusions.

The accompanying `verify_unified_preprint_v5_20261126.py` independently recomputes strict-H1 means from 320 pool rows, all 60 equal-budget family effects from the 320-row budget pool table, and all 20 unadjusted sequence effects from the sequence pool table. It cross-checks every adjusted sequence estimate and all 40 family-sensitivity rows against TSV/JSON, independently recomputes all primary and family all-20 Holm adjustments, and validates the existing sequence raw/independent/delivery audits, including the separate full-implementation recomputation of 620 fields with maximum absolute difference 4.44e-16. It also recomputes score-matching and cross-architecture summaries, Q Spearman statistics, Leppek correlations/rank reversals, and Paper3 V4 gate/curve aggregates. For PERSIST-seq, it recomputes the primary effect and secondary descriptive statistics from 59 construct-level predictions/labels and 200 per-seed records, checks the source-confirmed tie rule and exact v1/v1.1 primary invariance, validates existing three-round PERSIST and tie-audit checks, and enforces the no-detectable-transfer boundary. For Paper2 frozen risk v1, it checks the development lock, failed calibration gate, and outer hard-lock/no-score evidence. Three separately executed read-only CPU rounds write distinct V5 JSON/log outputs; the release builder independently checks source claims, PDF page text/rendering/fonts/overflow, and byte-identical clean rebuilds. No remote access, unblinding, training, GPU inference, outer scoring, or modification of figures/PPT/source_data was performed for V5 integration.

## Figure legends

**Figure 1 | Strict-H1 fixed-pool pressure gives context-dependent gains.** Panels a–d show within-pool paired BoN-minus-same-seed-Random measured-MRL regret across proposal caps; negative values favor BoN. Random is always a single-candidate baseline, so the cap curve is not an equal-candidate-budget comparison. Ribbons are nominal pool-cluster bootstrap 95% intervals over 16 pools per context. Panel e shows paired effects at cap 16 and cap 8; panel f summarizes all-20 Bonferroni sensitivity. All eight T-cell cap 2–16 intervals remain negative under that adjustment. HepG2 cap 8 is exploratory because both adjusted intervals include zero; all N50 nominal intervals include zero. These are retrospective strict-H1 protocol-sensitivity results from previously inspected outer contexts, not a new blind test. Source: `../figures/Figure2_v2_selection_pressure.pdf`.

**Figure 2 | Score matching does not identify a stable optimizer-specific error increment.** Panel a shows cap-16 BoN-minus-score-matched absolute-error differences in four contexts; positive values indicate higher BoN error. Points are observed differences and bars are nominal pool-cluster 95% intervals. Panel b shows the distribution of pool-level paired differences. Matching controls proxy mean only and does not establish high-score extrapolation as a causal mechanism. Source: `../figures/Figure3_v2_score_matching.pdf`.

**Figure 3 | Common proxy errors persist across model architectures.** Panel a shows means of 16 within-pool Pearson correlations between CNN-ensemble and 4-mer MLP absolute errors with nominal pool-cluster intervals. Panel b compares paired CNN and 4-mer MLP MAE by context; negative paired differences favor CNN. Both models share Sample training labels and endpoint definitions, so concordance is not independent functional validation and does not causally apportion error. Source: `../figures/Figure4_v2_common_proxy_error.pdf`.

**Figure 4 | Endpoint mismatch limits transfer of MRL rankings.** (a) Schlusser HepG2 TE distribution and cap-16 paired regret differences. All primary gene-cluster method intervals include zero; MRL–TE correlation is descriptive. This is a frozen one-time cross-endpoint evaluation, not same-endpoint MRL replication. (b) Leppek paired correlations and CDS-24 rank reversals. Half-life versus 6 h expression gives 171/276 discordant pairs and top-6 overlap 1/6. Leppek results are exploratory descriptions without significance testing. Sources: `../figures/Figure9_v2_schlusser_te.pdf` and `../figures/Figure10_v2_leppek_endpoints.pdf`.

**Figure 5 | Alternative selectors do not establish a measured-MRL regret advantage.** Panel a shows five cap-16 selectors at matched 48 CNN member item-forwards; panel b shows paired differences versus BoN. COM-style and diversity-LCB intervals include zero, and LCB/distance equal BoN under development-selected zero penalties. Results are retrospective sensitivity on previously viewed designed outer pools. Source: `../figures/Figure7_v2_paper2_baselines.pdf`.

**Figure 6 | Label-repair headroom is representation dependent and acquisition benefits remain diagnostic.** Panel a shows all 48 zero-label/full-query gates across three representations and 16 calibration pools. Panel b shows fixed-cohort `kmer4` pooled regret across six acquisition strategies. EI at 16 labels is descriptively lower than random, maximum variance, and regret-analytic acquisition, but no strategy dominates across budgets or pools. Permanent outer labels were not read or scored. Source: `../figures/Figure8_v2_paper3_headroom.pdf`.

**Figure 13 | PERSIST-seq independent-study MRL evaluation shows no detectable transfer.** The frozen external cohort contains 59 constructs spanning 49 unique 5′ UTRs with zero exact or equal-length Hamming-1 conflicts against Sample train/development/calibration. Native variable-length sequences were scored by the existing frozen three-member 4-mer MLP on CUDA. BoN-minus-Random measured-MRL regret is +0.104723 with 95% UTR-cluster interval [−0.100964, 0.212706] and two-sided sign p=0.245175; positive values favor Random. Prediction–measurement Spearman correlation is −0.008447 [−0.326701, 0.315406]. Source-confirmed prediction-descending/`Sequence ID`/row-order ties and the retrospective SHA256 alternative give the same primary effect, interval, and p value. This is a completed small retrospective external evaluation, not a blind test, direct T-cell CNN refutation, universal failure proof, or BoN benefit. Source: `../figures/v3/Figure13_v3_persist_external.pdf`.

**Figure 14 | Equal candidate budget reveals context-specific proxy-ranking value.** BoN(N)-minus-Random(N) measured-MRL regret is shown for identical nested proposal prefixes across four strict-H1 contexts and caps 1–16; negative values favor proxy ranking. Random(N) uses record-specific PCG64 child streams to select uniformly from the same prefix. Nominal intervals use 10,000 complete-pool bootstrap replicates. The complete all-20 Holm family includes cap-1 structural zeros. Every T-cell r1 cap 2–16 effect survives Holm, whereas T-cell r2 survives only at cap 4. HepG2 cap 2 survives, cap 4 is nominal with Holm p=0.055994, and N50 has no nominally supported effect. The historical BoN(N)-minus-Random(1) contrast and Random(N)-minus-Random(1) realizations are retained separately. All results are retrospective sensitivities on previously inspected outer contexts, not new blind tests. Source: `../figures/v3/Figure14_v3_equal_budget_value.pdf`.

**Figure 15 | Sequence-composition adjustment and family exclusions attenuate pressure gains.** Original strict-H1 BoN-minus-Random(1) effects are compared with estimates adjusted for GC, uAUG, longest poly-U, and training support using context–cap models with pool fixed effects. Only T-cell r1 caps 4 and 8 retain nominal negative adjusted intervals, and no adjusted effect survives the complete all-20 Holm family. Family panels retain complete-pair exclusions for uAUG and the blind upper-quartile poly-U family (`longest_poly_u >= 3`), without reranking candidates. No uAUG-excluded comparison survives Holm; only T-cell r1 cap 4 and T-cell r2 cap 8 survive the poly-U-excluded family correction. All null, attenuated, sign-reversed, and reduced-pool results are retained. These retrospective sensitivities establish composition dependence and heterogeneity, not sequence causation. Source: `../figures/v3/Figure15_v3_sequence_composition_sensitivity.pdf`.

**Extended Data Figure 1 | Proxy-output noise is not monotonically amplified by BoN.** Mean measured-MRL regret and paired differences from sigma zero are shown for N50 HEK293T after normalized-output Gaussian perturbations. Cap 1 is invariant. These exploratory results do not recommend noise injection as a correction. Source: `../figures/ExtendedData_proxy_noise.pdf`.

**Extended Data Figure 2 | Data governance defines auditable OOD boundaries.** Panel a summarizes frozen study roles, exact/Hamming-1 sequence isolation, Hamming-2 as a predeclared diagnostic rather than a pass gate, the Learner–Evaluator firewall, and locked evaluations. Panel b is the evidence map at the V3.1 freeze; V4 subsequently completed the small retrospective PERSIST-seq independent-study, same-endpoint MRL evaluation shown in Fig. 13, which did not show transferable ranking signal. Retrospective strict-H1 and PERSIST contexts are not new blind tests, and governance supports provenance and isolation rather than automatic biological generalization. Source: `../figures/v3/Figure12_v3_data_governance_ood.pdf`.

## Data and code availability

Code and large artifacts are maintained in the frozen project locations documented in the source protocols. Local manuscript evidence is under `deliverables/reports/source_data/`; the versioned manuscript, independent CPU verification script, source hashes, and three-round claim audit accompany this draft. Primary evidence documents are `PAPER1_STRICT_H1_RESULT.md`, `PAPER1_BUDGET_MATCHED_V1.md`, `PAPER1_SEQUENCE_ABLATION_V1.md`, `PERSIST_MRL_EXTERNAL_V1.md`, `PERSIST_TIE_AUDIT_V1_1.md`, `SCHLUSSER2024_HEPG2_TE_RESULT.md`, `PAPER2_RISK_PREDICTION_20261124.md`, `PAPER2_BASELINES_RESULT.md`, `PAPER2_RISK_FROZEN_V1.md`, and `paper3_headroom_v4_20261124.md`. Unless otherwise stated, these documents are under `deliverables/reports/docs/`; `paper3_headroom_v4_20261124.md` is under `deliverables/reports/`. V4 Markdown/PDFs, builder, verifier, story, ledger, verification outputs, and manifest remain unchanged alongside the V5 release. No upload, GitHub push, public repository, or public release DOI is claimed by this local draft; the locations above describe the accompanying local evidence package rather than public availability.

## References

[1] J. Skalse, N. H. R. Howe, D. Krasheninnikov, and D. Krueger, “Defining and characterizing reward hacking,” in *Advances in Neural Information Processing Systems 35*, 2022. <https://doi.org/10.52202/068431-0687>.

[2] B. Trabucco, A. Kumar, X. Geng, and S. Levine, “Conservative objective models for effective offline model-based optimization,” in *Proceedings of the 38th International Conference on Machine Learning*, 2021. <https://proceedings.mlr.press/v139/trabucco21a.html>.

[3] B. Trabucco, X. Geng, A. Kumar, and S. Levine, “Design-Bench: Benchmarks for data-driven offline model-based optimization,” in *Proceedings of the 39th International Conference on Machine Learning*, 2022. <https://proceedings.mlr.press/v162/trabucco22a.html>.

[4] M. S. Yao, Y. Zeng, H. Bastani, et al., “Generative adversarial model-based optimization via source critic regularization,” in *Advances in Neural Information Processing Systems 37*, 2024. <https://doi.org/10.52202/079017-1397>.

[5] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” in *Advances in Neural Information Processing Systems 30*, 2017. <https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html>.

[6] E. N. Weinstein, A. Amin, J. Frazer, and D. S. Marks, “Non-identifiability and the blessings of misspecification in models of molecular fitness,” in *Advances in Neural Information Processing Systems 35*, 2022. <https://doi.org/10.52202/068431-0396>.

[7] A. J. Reddy, X. Geng, M. H. Herschl, et al., “Designing cell-type-specific promoter sequences using conservative model-based optimization,” in *Advances in Neural Information Processing Systems 37*, 2024. <https://doi.org/10.52202/079017-2954>.

[8] M. Jain, E. Bengio, A. Hernandez-Garcia, et al., “Biological sequence design with GFlowNets,” in *Proceedings of the 39th International Conference on Machine Learning*, 2022. <https://proceedings.mlr.press/v162/jain22a.html>.

[9] P. J. Sample, B. Wang, D. W. Reid, V. Presnyak, I. J. McFadyen, D. R. Morris, and G. Seelig, “Human 5′ UTR design and variant effect prediction from a massively parallel translation assay,” *Nature Biotechnology*, vol. 37, pp. 803–809, 2019. <https://doi.org/10.1038/s41587-019-0164-5>.

[10] S. Castillo-Hair, S. Fedak, B. Wang, et al., “Optimizing 5′UTRs for mRNA-delivered gene editing using deep learning,” *Nature Communications*, vol. 15, 5284, 2024. <https://doi.org/10.1038/s41467-024-49508-2>.

[11] K. Leppek, G. W. Byeon, W. Kladwang, et al., “Combinatorial optimization of mRNA structure, stability, and translation for RNA-based therapeutics,” *Nature Communications*, vol. 13, 1536, 2022. <https://doi.org/10.1038/s41467-022-28776-w>.

[12] N. Schlusser, A. González, M. Pandey, et al., “Current limitations in predicting mRNA translation with deep learning models,” *Genome Biology*, vol. 25, 227, 2024. <https://doi.org/10.1186/s13059-024-03369-6>.

[13] H. Khalaf, C. Mayrink Verdun, A. Oesterling, H. Lakkaraju, and F. Calmon, “Inference-time reward hacking in large language models,” in *Advances in Neural Information Processing Systems 38*, 2025. <https://doi.org/10.52202/085713-2064>.

[14] E. Fadeeva, X. Li, M. Sachan, and M. Panov, “Verifier-guided self-consistency: Test-time reasoning with imperfect generators and verifiers,” arXiv:2611.06606, 2026. <https://arxiv.org/abs/2611.06606>.

[15] J. Chen, J. Yoon, S. Ebrahimi, S. Arik, S. Jha, and T. Pfister, “ASPEST: Bridging the gap between active learning and selective prediction,” arXiv:2304.03870, 2023. <https://arxiv.org/abs/2304.03870>.

[16] “Full-length mRNA design with reward-guided masked diffusion model fine-tuning (T3PO-mRNA),” OpenReview submission KTCInSlPpL. <https://openreview.net/forum?id=KTCInSlPpL>.

[17] S. Patel, S. Tang, Y. Kim, et al., “mRNAutilus: Multi-objective-guided discrete generation of mRNA with optimized therapeutic properties,” arXiv:2605.31296, 2026. <https://arxiv.org/abs/2605.31296>.
