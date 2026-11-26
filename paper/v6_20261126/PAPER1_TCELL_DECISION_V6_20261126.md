---
title: "Equal-budget CNN ranking reduces functional regret in T-cell 5′ UTR selection"
subtitle: "Useful global ranking coexists with high pointwise error, while top-tail reversals define the remaining decision bottleneck"
author: "Anonymous authors"
date: "2026-11-26"
lang: en
---

> **Draft status.** Independent Paper 1 main-text reconstruction, version 6. “Anonymous authors” is a deliberate pre-submission placeholder. This version centers the T-cell 5′ untranslated-region (5′ UTR) decision problem and measured mean ribosome load (MRL), rather than a general proxy contract. All experimental analyses are retrospective analyses of historical data with strict sequence/grouped holdouts as specified below. None is a new prospective blind test, an independent wet-lab replication, or evidence of causation. V6 performs manuscript integration and read-only statistical verification only; it does not retrain models, reopen target evaluations, or alter source data, figures, or presentation files.

## Abstract

Selecting a 5′ UTR by a predicted translation proxy requires useful ordering of the few highest-scoring candidates, not merely low average prediction error. We evaluated this distinction using measured MRL in two defined-end T-cell repeats. On identical N-candidate proposal prefixes, best-of-N (BoN) selection by a frozen Sample-trained CNN reduced regret relative to Random(N) at every cap 2–16 in both repeats. In repeat 1, effects were −0.799949, −1.448756, −1.579340, and −1.705760 MRL at caps 2, 4, 8, and 16, all surviving the complete all-20 Holm sensitivity family. Repeat 2 was negative at all four caps, with cap 4 surviving that family (−1.014876 [−1.544490, −0.465832]; Holm p=0.006399). This equal-budget gain coexisted with high pointwise error (MAE 1.567 and 1.544 MRL) and moderate global Spearman correlation (0.591 and 0.529). Tail decomposition localized the remaining bottleneck: within the full-pool predicted top 16, Spearman correlation fell to 0.206 and 0.144, the predicted winner was not the measured tail optimum in 15/16 pools per repeat, and within-tail ranking loss was 2.480/0.836 versus 2.117/1.953 MRL for ranking/opportunity components. Target-context adaptation with 50,000 labels markedly reduced MAE but did not improve cap16 regret. A subsequent ListNet objective produced only small, gate-failing point improvements on a new component-disjoint test. Direct HepG2 anchoring significantly worsened combined T-cell cap16 regret (+0.428897 [+0.245460, +0.616474]). These retrospective results support measured functional regret—not MAE—as the primary endpoint for T-cell 5′ UTR model development, while preserving the demonstrated value of the original frozen CNN.

**Keywords:** T-cell; 5′ UTR; mean ribosome load; best-of-N; functional regret; top-tail ranking; decision-focused evaluation; target adaptation; ListNet

## 1. Introduction

A predictive model for biological sequence design is usually evaluated by average error, but used to choose one or a few candidates. These are different tasks. Mean absolute error (MAE) treats every sequence equally, whereas best-of-N (BoN) selection depends on the identities and ordering of candidates near the top of the predicted score distribution. A model can therefore be miscalibrated or inaccurate on average and still improve a real functional choice; conversely, a substantial reduction in average error need not improve the selected sequence.

The T-cell 5′ UTR setting makes this distinction experimentally concrete. Sample et al. established large-scale 5′ UTR assays and CNN-guided sequence design [1], and Castillo-Hair et al. measured defined-end libraries across cell contexts, including two T-cell repeats [2]. Here, the functional endpoint is measured mean ribosome load (MRL), and the decision endpoint is regret: the difference between the best measured MRL in a fixed candidate pool and the MRL of the sequence selected by a decision rule. Lower regret is better. This framing is related to decision-focused learning and to warnings that optimization against a learned objective can exploit proxy errors [3–8], but the present question is narrower and testable: does a frozen Sample-trained CNN improve T-cell choices at an equal candidate budget, and where does its remaining decision loss arise?

Four findings organize the study. First, same-prefix Random(N) controls isolate a positive equal-budget contribution of CNN ranking in both T-cell repeats. Second, joint analysis of pointwise error, global correlation, and top-tail decompositions shows that high MAE and useful global ranking coexist; the unresolved bottleneck is ordering within the high-score tail, not absence of ranking information. Third, supervised target adaptation and a full-pool ListNet objective show that regression improvement and decision improvement can decouple. Fourth, a same-sequence HepG2 experimental anchor demonstrates that even genuine cross-cell functional measurements cannot be assumed to repair T-cell tail choices.

The contribution is therefore decision-focused rather than a claim of universal proxy validity or a catalogue of failed repairs. The frozen CNN already provides measurable T-cell value under equal candidate budgets. Tail analysis explains why further model development should be accepted only when it improves measured functional regret under shared proposals, with MAE and correlation retained as complementary diagnostics. Auxiliary residual, pairwise-ranking, and multi-context models are reported separately as development attempts stopped before any T-cell evaluation; they are not mislabeled as T-cell failures or as validated improvements.

## 2. Results

### 2.1 Frozen CNN ranking lowers T-cell regret at equal candidate budget

We compared BoN(N) with Random(N) on exactly the same nested proposal prefixes. For each pool, proposal seed, and cap \(N\), BoN selected the candidate with the largest frozen CNN ensemble score among the first \(N\) proposals; Random(N) uniformly selected one position from that same prefix. Both methods therefore saw the same candidate count and proposal opportunities. The paired endpoint was BoN regret minus Random(N) regret, with negative values favoring CNN ranking. These strict-H1 analyses reused previously inspected historical outer pools and are retrospective sensitivity analyses, not new blind tests.

CNN ranking reduced mean measured-MRL regret at every cap 2–16 in both T-cell repeats (Table 1; Fig. 14). In repeat 1, the effects ranged from −0.799949 at cap 2 to −1.705760 at cap 16, and all four survived the complete all-20 context–cap Holm family, including structural cap-1 zeros (each adjusted p=0.004000). In repeat 2, all four point estimates and nominal intervals favored BoN, but only cap 4 survived the all-20 Holm family: −1.014876 [−1.544490, −0.465832], adjusted p=0.006399. The other repeat-2 results remain nominal positive findings without multiplicity-robust support in that family; they are not described as corrected-significant.

**Table 1 | Equal-budget BoN(N)-minus-Random(N) measured-MRL regret on identical proposal prefixes.**

| T-cell repeat | Cap | BoN regret | Random(N) regret | Effect [nominal 95% CI] | All-20 Holm p |
|---|---:|---:|---:|---:|---:|
| r1 | 2 | 4.121807 | 4.921756 | −0.799949 [−1.201008, −0.449730] | **0.004000** |
| r1 | 4 | 3.642443 | 5.091199 | −1.448756 [−1.809149, −1.094514] | **0.004000** |
| r1 | 8 | 3.599316 | 5.178656 | −1.579340 [−1.975375, −1.183081] | **0.004000** |
| r1 | 16 | 3.555181 | 5.260940 | −1.705760 [−2.163445, −1.211914] | **0.004000** |
| r2 | 2 | 5.421317 | 6.102141 | −0.680824 [−1.248022, −0.135771] | 0.177582 |
| r2 | 4 | 5.254170 | 6.269046 | −1.014876 [−1.544490, −0.465832] | **0.006399** |
| r2 | 8 | 4.888535 | 5.829409 | −0.940873 [−1.738304, −0.151800] | 0.222178 |
| r2 | 16 | 5.043377 | 6.130317 | −1.086941 [−1.833685, −0.318589] | 0.064994 |

Cap-1 contrasts are structural identities and are included in the correction family. No Random(N)-minus-Random(1) contrast survived its separate all-20 Holm family, consistent with randomization realizations rather than a systematic change in candidate quality from increasing N. The older BoN(N)-minus-Random(1) pressure contrast is not substituted for this equal-budget analysis. Together, these results establish a specific positive phenomenon: on historical, sequence-isolated T-cell pools, the frozen CNN extracts useful functional choices from the same candidates offered to a random selector.

### 2.2 High pointwise error coexists with useful global ranking, but top-tail ordering is the bottleneck

The equal-budget benefit does not imply low pointwise error. Across the 1,024 strict-H1 candidates in each repeat, CNN MAE was 1.567 and 1.544 MRL, with systematic positive bias of +1.064 and +0.407 MRL (Table 2; Fig. 16). Nevertheless, global Spearman correlations were 0.591 and 0.529, and mean within-pool Spearman correlations were 0.584 and 0.515. Thus, substantial absolute error and useful global ordering are simultaneously observed.

An oracle median-shift diagnostic separates calibration from ranking. Subtracting the global median error would reduce MAE to 1.108 and 1.476 MRL, but a constant shift cannot change within-pool ranks, BoN selections, or regret. This diagnostic explains part of the pointwise error; it is not a deployable calibration result because it uses target labels, and it cannot repair a selection rule by itself.

**Table 2 | Pointwise error and global ordering for the frozen CNN on strict-H1 T-cell candidates.**

| Repeat | Bias | MAE | RMSE | Global Spearman | Mean within-pool Spearman | Oracle median-shift MAE |
|---|---:|---:|---:|---:|---:|---:|
| r1 | +1.064393 | 1.567454 | 1.971066 | 0.591052 | 0.583869 | 1.108358 |
| r2 | +0.407480 | 1.543969 | 2.308732 | 0.529222 | 0.515055 | 1.475718 |

To localize the decision loss, we decomposed regret for the first candidate in the full-pool predicted top \(k\):

\[
R_{\mathrm{top}1@k}
=\underbrace{Y_{\mathrm{pool\ oracle}}-Y_{\mathrm{tail\ oracle}}}_{\text{tail opportunity loss}}
+\underbrace{Y_{\mathrm{tail\ oracle}}-Y_{\mathrm{predicted\ winner}}}_{\text{within-tail ranking loss}}.
\]

At \(k=16\), global ordering weakened sharply within the tail. Top-16 Spearman correlation was 0.206 in r1 and 0.144 in r2, and predicted-order pair inversion rates were 43.1% and 45.0%. In both repeats, another top-16 candidate had higher measured MRL than the predicted winner in 15/16 pools. The full-pool top-1 regret of 3.316 MRL in r1 comprised 2.480 within-tail ranking loss and 0.836 tail opportunity loss; in r2, 4.071 comprised 2.117 and 1.953, respectively (Table 3). Within-tail ranking therefore accounted for 74.8% and 52.0% of this full-pool top-1 regret.

**Table 3 | Full-pool predicted-top-16 regret decomposition and ordering diagnostics.**

| Repeat | Top-1 regret | Within-tail ranking loss | Tail opportunity loss | Top-16 Spearman | Pair inversions | Winner overturned |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 3.315826 | 2.479998 | 0.835829 | 0.206058 | 43.1% | 15/16 pools |
| r2 | 4.070571 | 2.117221 | 1.953350 | 0.143984 | 45.0% | 15/16 pools |

The same distinction holds for the actual BoN16 proposal prefixes. BoN16 regret was \(3.555=1.720+1.835\) MRL in r1 and \(5.043=2.088+2.955\) MRL in r2, where the terms are prefix opportunity loss and within-prefix ranking loss. In a BoN-minus-Random(N) paired contrast, the shared prefix opportunity loss cancels exactly; the observed equal-budget gain must arise from different choices among the same reachable candidates. Full-pool top-16 analysis uses all 64 existing scores and is a diagnostic, not an equal-budget alternative to BoN16.

Outlier diagnostics further show why a single average-error summary is insufficient. Tukey absolute-error outliers comprised 6.05% and 9.96% of candidates but contributed 41.5% and 66.5% of MSE in r1 and r2. They accounted for 5/80 and 11/80 BoN16 selection events, respectively, indicating repeat heterogeneity rather than a universal outlier-enrichment rule. No measured outlier was removed or used to rerank candidates. The supported conclusion is not that the CNN is globally useless, nor that calibration is irrelevant in general; it is that the remaining T-cell selection loss is directly visible in top-tail reversals and must be evaluated there.

### 2.3 Target supervision reduces MAE without improving cap16 selection

We next tested whether target-context supervision could repair the tail. The v1 adaptation experiment used exact/Hamming-1 connected components to allocate 50,000 train, 10,000 development, and 1,024 test sequences, with the test arranged as 16 repeat-linked pools of 64. Target r1/r2 labels were explicitly used in train and development; this is target-context retrospective grouped holdout evidence, not zero-label transfer or an independent study. The primary model fine-tuned each frozen Sample CNN member with MSE plus a dynamic top-16 pairwise logistic term. Model selection used the worse of the two development-repeat cap16 regrets.

The selected 50k model reduced MAE from 1.352159 to 0.914506 in r1 and from 1.062273 to 1.018716 in r2, with RMSE and Spearman also improving (Table 4; Fig. 17). Yet cap16 regret increased by +0.038929 and +0.012551 MRL, respectively. The combined adapted-minus-original effect was +0.025740 [−0.072019, +0.140994], and all ten repeat–cap Holm-adjusted p values were 1. The frozen minimum absolute improvement was 0.131566 MRL; the observed direction was instead unfavorable. Thus, the primary adaptation gate failed on repeat direction, interval, and magnitude.

**Table 4 | Fifty-thousand-label target adaptation: pointwise gains do not yield cap16 decision gains.**

| Repeat | Original MAE | Adapted MAE | MAE change | Original cap16 regret | Adapted cap16 regret | Adapted − original [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 1.352159 | 0.914506 | −0.437653 | 2.389657 | 2.428586 | +0.038929 [−0.064193, +0.146964] |
| r2 | 1.062273 | 1.018716 | −0.043556 | 2.873002 | 2.885553 | +0.012551 [−0.143780, +0.187218] |

The MSE-only arm made exactly the same cap selections as the selected joint model at every cap, so the small selected pairwise weight produced no additional realized decision benefit. Tail pair accuracy changed in opposite directions across repeats (−0.005211 and +0.015625). A nested 20k run had small favorable point estimates, but it was not the primary budget, fell below the required magnitude, and was followed by reversal at 50k; it was not selected post hoc as a success. Likewise, mixed from-scratch results did not rescue the primary claim. These results falsify the specific proposition that the selected 50k joint adaptation materially improves cap16 T-cell selection, not every possible use of target supervision.

### 2.4 ListNet gives only small, gate-failing improvements on a new grouped test

The v2 experiment changed the objective rather than the network: full-64 ListNet cross-entropy between measured-MRL and predicted softmax distributions, plus MSE, replaced the v1 dynamic pairwise term [9]. Eighty-one runs spanning learning rate, temperature, and MSE weight all converged. The selected model used development mean-replicate cap16 regret. A new test-v2 excluded every exact/H1 component touching v1 train/dev/test and Sample train/dev/calibration, yielding another 16 component-disjoint pools of 64. Predictions for all four arms were frozen before the single test-v2 label evaluation. The primary estimand was per-sequence mean-replicate MRL regret; repeat-specific results were secondary.

At cap16, ListNet regret was 1.729031 versus 1.766447 for the original CNN and 1.736616 for v1 main (Table 5; Fig. 17). The effects were small and favorable in point estimate, −0.037415 [−0.183204, +0.102494] and −0.007585 [−0.028698, +0.012474], but both intervals crossed zero, both absolute improvements were below the required 0.1 MRL, and both Holm-10 p values were 1. The joint co-primary intersection–union gate therefore failed. These are not reported as significant or substantive improvements.

**Table 5 | Full-64 ListNet+MSE on component-disjoint test-v2: mean-replicate cap16 results.**

| Comparator | Comparator regret | ListNet regret | ListNet − comparator [unadjusted 95% CI] | Required reduction | Holm-10 p | Gate |
|---|---:|---:|---:|---:|---:|---|
| Original CNN | 1.766447 | 1.729031 | −0.037415 [−0.183204, +0.102494] | 0.1 | 1.0 | FAIL |
| v1 main | 1.736616 | 1.729031 | −0.007585 [−0.028698, +0.012474] | 0.1 | 1.0 | FAIL |

Secondary repeat analysis also prevents selective reporting. Relative to v1 main at cap16, ListNet changed regret by +0.020946 in r1 and −0.036115 in r2; the favorable r2 secondary interval does not cancel the r1 reversal or change the failed mean-replicate primary gate. Relative to the original CNN, the mean-replicate MAE fell by 0.306915 while regret fell by only 0.037415. Relative to v1 main, MAE instead increased by 0.002104 while the regret point estimate decreased by 0.007585. Top-tail pair accuracy improved modestly (+0.018750 versus original and +0.011458 versus v1 main), but these local ordering changes were too small in realized MRL to meet the decision endpoint.

The v1 and v2 absolute regret values are not compared across their different test pools. The joint conclusion across experiments is stronger than “all methods fail”: the original CNN has demonstrated ranking value, and specific adaptations do not add enough validated tail benefit. Average regression metrics and even local pair accuracy cannot substitute for the measured decision outcome.

### 2.5 A same-sequence HepG2 functional anchor worsens T-cell cap16 choices

A direct experimental cross-cell anchor provided an independent way to test whether global functional information is sufficient for tail selection. For 1,024 T-cell sequences with measured HepG2 MRL, we ranked the same proposal prefixes by HepG2 MRL instead of CNN score. HepG2 measurements are genuine functional data, not T-cell labels; however, the same sequences were measured in both cells, so this is paired cross-cell retrospective evidence rather than sequence-independent validation.

HepG2 anchoring increased cap16 T-cell regret by +0.463090 in r1 and +0.394705 in r2. The repeat-linked combined effect was +0.428897 [+0.245460, +0.616474], two-sided bootstrap p=0.00019998, where positive values indicate worse performance than CNN. This combined interval is wholly in the harmful direction. In the separate ten-test repeat–cap Holm family, r2 cap16 survived (p=0.017998), whereas r1 cap16 was nominal but did not survive (p=0.053995); both repeat point estimates are retained without overstating their individual correction status. A frozen 50/50 within-pool rank fusion also failed to improve the combined cap16 result (+0.154527 [−0.004346, +0.326043]).

The reversal is especially informative because HepG2 global Spearman correlation with T-cell MRL was comparable to or slightly higher than CNN correlation: 0.730001 versus 0.722091 in r1 and 0.750157 versus 0.724564 in r2. Yet CNN and HepG2 top-16 sets had mean Jaccard overlap only 0.470432, and their full-pool winners differed in 16/16 pools in both repeats. At cap16, 61/80 pool–seed selections changed per repeat; harmful changes outnumbered beneficial changes 39/21 in r1 and 39/22 in r2. Shared proposal headroom was identical by construction, while within-prefix oracle opportunity loss increased from 1.075351 to 1.538440 in r1 and from 1.302036 to 1.696741 in r2. The loss therefore arose from choosing differently among the same reachable candidates, not from unequal proposal opportunities. A simple cross-cell functional substitute is not a validated solution to the T-cell tail problem.

### 2.6 Auxiliary residual, pairwise, and multi-context models stop before target evaluation

Three development programs tested whether non-T-cell supervision could earn access to a T-cell evaluation. Each used frozen source-inner selection and destination auxiliary grouped tests, with fail-closed target gating (Table 6). None passed both required auxiliary directions. Consequently, no corrected T-cell predictions were generated, no T-cell MRL was read, and target evaluation count remained zero for these programs. Their results are development evidence only, not T-cell successes or T-cell failures.

**Table 6 | Auxiliary development gates; effect is corrected/new-model regret minus frozen-CNN regret.**

| Development program | Direction | Destination auxiliary test effect [95% CI] | Gate and target status |
|---|---|---|---|
| Shared residual ridge | N50→HepG2 | +0.000343 (interval not used for v1 gate) | FAIL; T-cell not evaluated |
| Shared residual ridge | HepG2→N50 | +0.004638 (interval not used for v1 gate) | FAIL; T-cell not evaluated |
| Convex tail pairwise logistic | N50→HepG2 | −0.006543 [−0.014204, +0.001233] | FAIL: CI crosses zero; T-cell not evaluated |
| Convex tail pairwise logistic | HepG2→N50 | +0.021187 [−0.007718, +0.050096] | FAIL; T-cell not evaluated |
| Multi-context tail backbone | N50→HepG2 | +0.044629 [+0.021840, +0.068346] | FAIL; T-cell not evaluated |
| Multi-context tail backbone | HepG2→N50 | +0.028050 [−0.049121, +0.104614] | FAIL; T-cell not evaluated |

The residual models showed that small development MSE and regret gains did not transfer across auxiliary contexts. The pairwise objective produced a small favorable interval-crossing estimate in one direction and an unfavorable estimate in the other. The multi-context backbone was unfavorable in both, significantly so in N50→HepG2. These results do not prove that residual learning, ranking losses, or multi-context training are generally ineffective. They show that the tested frozen implementations did not meet the prerequisite for target evaluation, and they must not be invoked as evidence that T-cell adaptation itself was attempted and failed.

## 3. Discussion

The central result is a positive, quantitative dissociation between average prediction quality and decision quality. The original frozen Sample CNN had substantial T-cell pointwise error, yet same-prefix controls showed real measured-MRL benefits at equal candidate budgets in both repeats. This rules out two oversimplified interpretations: high MAE does not imply that a proxy has no selection value, and a positive BoN contrast does not imply that the highest-scoring candidate is reliably optimal.

Tail decomposition supplies a practical explanation of where model development should look next. Global calibration shifts can reduce MAE without changing a single selection. In contrast, predicted winners were overturned by another tail candidate in 15/16 pools per repeat, and within-tail ranking loss was a large component of regret. At the same time, tail opportunity loss remained nonzero, especially in r2. Future work should therefore distinguish better ordering of available high-score candidates from expanding proposal coverage; a same-prefix reranker can address only the former.

The adaptation experiments show why this distinction must be enforced empirically. Fifty-thousand target labels produced markedly lower MAE but no cap16 gain. ListNet introduced small favorable point estimates and pair-accuracy changes on a fresh component-disjoint test, but neither statistical support nor the prespecified biological magnitude was sufficient. The direct HepG2 anchor provides a complementary experimental falsification: even true same-sequence functional measurements from another cell can preserve global correlation while degrading T-cell top-tail choices. Auxiliary gated studies reinforce the need for transfer evidence before target scoring without being misrepresented as additional target failures.

For T-cell 5′ UTR model development, we therefore recommend measured functional regret as the primary acceptance endpoint, accompanied by selected MRL, a same-prefix Random(N) control, and complete reporting across frozen caps and repeats. MAE, RMSE, Spearman correlation, and pair accuracy remain useful diagnostics but should not serve as surrogates for the final selection decision. The minimum meaningful regret reduction and the evaluation budget should be fixed before target labels are accessed. Improvements on development data, oracle reranking headroom, and individual favorable repeats should not override a failed primary gate.

This recommendation does not imply that MAE is irrelevant, that every ranking objective will fail, or that the frozen CNN is universally superior. Model misspecification and shared training lineage can still limit error interpretation and conservative design [12–13], and MRL should not be treated as interchangeable with other molecular or translation endpoints [14–15]. The present study identifies a validated baseline benefit and a precise unresolved problem. A future method should be considered an advance only when it improves measured T-cell regret under a new, appropriately isolated evaluation while preserving the equal-budget advantage already demonstrated here.

## 4. Limitations

**Retrospective evidence and repeated historical use.** All analyses use historical datasets and are retrospective. The strict-H1 outer pools had been inspected before the V6 integration and before several sensitivity analyses. The v1 and v2 target-adaptation tests were frozen and evaluated once within their own protocols, but remain grouped holdouts reconstructed from historical terminal data, not prospective blind tests or independent wet-lab studies. Repeated algorithmic proposals and bootstrap replicates do not create additional biological replications.

**Limited effective sample and repeat dependence.** Each principal T-cell evaluation contains 16 pools. The two repeats share underlying sequence libraries, and the paired cross-cell study measures the same sequences in HepG2 and T cells. Pool-cluster and repeat-linked intervals quantify sampling uncertainty within these constructs; they do not establish independent-study replication. Non-significance is not evidence of exact equivalence or absence of an effect.

**Different evaluation pools and estimands.** Strict-H1, target-adaptation v1, ListNet test-v2, and the paired HepG2 anchor use separately constructed pools and should not be compared by absolute regret across experiments. The v1 primary success required both repeats to improve; the v2 primary endpoint was per-sequence mean-replicate regret with repeat-specific results secondary. These estimands are reported separately rather than harmonized post hoc.

**Retrospective multiplicity and uncertainty.** Holm families and percentile bootstrap intervals are prespecified within their source analyses but remain retrospective sensitivity quantities in the historical-study context. The all-20 equal-budget family includes cap-1 structural zeros. The v2 co-primary intervals are unadjusted 95% intervals used in an intersection–union gate, not Holm simultaneous intervals. Secondary repeat and pair-accuracy results cannot reverse a failed primary gate.

**Diagnostic rather than causal decomposition.** Top-tail opportunity/ranking decomposition is an accounting identity, not a causal mediation analysis. Oracle median shifts and oracle reranking use measured labels to describe error structure and available headroom; neither is a deployable intervention. Feature, outlier, and repeat comparisons do not identify sequence mechanisms or experimental causes.

**Scoring and proposal budgets.** Same-prefix comparisons equalize candidate opportunities. Full-pool top-16 diagnostics use all 64 existing scores and are not deployable BoN16 results. In ListNet v2, every arm scored all 64 test sequences with all three members before offline prefix-restricted selection; cap16 is therefore a proposal/selection budget, not a claim that only 16 model forwards occurred. Proposal draws can repeat positions. Training and development-search budgets differed between ListNet v2 and v1, despite common architecture, target labels, deployment seeds, and test scoring.

**Target supervision and transport.** The adaptation studies intentionally use T-cell train/development labels and are not zero-shot transfer. Only the tested 50k joint objective, grid, ListNet+MSE objective, and frozen selection rules are evaluated. Small favorable 20k, from-scratch, individual-repeat, or pair-accuracy results do not validate alternative post hoc choices. The HepG2 result rejects direct same-sequence anchoring and the fixed 50/50 fusion, not all possible cross-cell models.

**Auxiliary-gated scope.** Residual, pairwise, and multi-context programs did not pass their auxiliary gates and never generated a T-cell evaluation. They provide no direct evidence about corrected T-cell regret and are not counted as T-cell failures. Their conclusions are restricted to the tested auxiliary transfer directions and implementations.

**Functional and external scope.** MRL is not synonymous with translation efficiency, protein output, in vivo efficacy, or clinical benefit. Conclusions are limited to historical 50-nt 5′ UTR libraries, the tested cells and constructs, and the evaluated CNN/MLP families. No new sequences were synthesized or experimentally validated. The manuscript has not undergone external peer review.

## 5. Methods

### 5.1 Study design, evidence roles, and terminology

**Table 7 | Data roles and permitted interpretation.**

| Analysis | Data role and isolation | Evidence state | Not licensed |
|---|---|---|---|
| Equal-budget strict-H1 | Sixteen 64-sequence pools per T-cell repeat; exact/H1 isolation within/across pools and from Sample train/dev/calibration | Retrospective historical-outer sensitivity | New blind test, prospective replication, universal benefit |
| Error and tail diagnosis | Same strict-H1 candidates, frozen CNN/MLP scores and measured MRL | Retrospective descriptive/oracle diagnosis | Deployable calibration, causal mechanism, realized repair |
| Target adaptation v1 | Component-disjoint 50k train/10k dev/1,024 test; 16 repeat-linked test pools | Target-context retrospective grouped holdout; one evaluation | Zero-label transfer, independent study, new blind test |
| ListNet v2 | New 1,024-sequence component-disjoint test excluding all v1 roles and Sample reference | Target-context retrospective grouped holdout; one evaluation | External validation, cross-pool absolute comparison with v1 |
| HepG2 anchor | Same 1,024 sequences measured in HepG2 and two T-cell repeats; 16 pools | Paired cross-cell retrospective evidence | Sequence-independent validation or causal cell mechanism |
| Residual/pairwise/multi-context | Source-inner development and destination auxiliary grouped tests | Development gate evidence; target count zero | Any corrected T-cell success or failure claim |

Throughout, “effect” denotes a paired numerical contrast, not a causal effect. Negative BoN-minus-Random, adapted-minus-original, ListNet-minus-comparator, or corrected-minus-frozen values favor the first-named method; the HepG2 anchor contrast is also anchor minus CNN, so positive values are harmful. Retrospective/grouped holdout terminology is used regardless of whether an individual source protocol was frozen before its single evaluation.

### 5.2 Source data and strict-H1 reconstruction

Source measurements came from Sample et al. (GSE114002) [1] and Castillo-Hair et al. (GSE232927) [2]. Strict-H1 used defined-end T-cell r1/r2, with 16 deterministic pools of 64 unique 50-nt sequences per repeat. Pools excluded exact and Hamming-1 conflicts within and across pools and against the combined Sample train/development/calibration reference. Two deterministic pool builds and associated hashes were checked in the source study. The Learner did not receive measured MRL; the Evaluator joined labels after selection artifacts were frozen. V6 reused de-identified aggregate exports and did not rebuild candidate pools or access sequence-level labels.

### 5.3 Frozen Sample CNN and alternative MLP

The primary predictor was a three-member 50-nt CNN regression ensemble trained on Sample random-library data with seeds 11, 23, and 37. Member-specific target scalers were fit on training data only. Training used batch size 1024, Adam learning rate 0.001, development-based best checkpoints, minimum development-loss change 0.0001, patience 5, and no scientific maximum-epoch stopping rule. Best epochs were 62, 50, and 77. Scores were inverse transformed and averaged arithmetically. The 4-mer MLP used only for architecture/error diagnostics had three members and 259 sequence features; it shared Sample training labels and was not independent biological validation. V6 performed no model training or inference.

### 5.4 Equal-budget selection and functional regret

For candidate pool \(C_t\), measured MRL \(Y_i\), and strategy \(\pi\), regret was

\[
R_t^\pi=\max_{i\in C_t}Y_i-Y_{x_t^\pi}.
\]

BoN maximized the frozen CNN ensemble mean among the first \(N\) proposals, with \(N\in\{1,2,4,8,16\}\) and proposal seeds 101, 202, 303, 404, and 505. Random(N) selected uniformly from the exact same nested prefix using a record-specific PCG64 child stream spawned from `SeedSequence(20261125)` and keyed by context, pool, proposal seed, and cap. Random(1) used the first proposal. Five proposal realizations were averaged within each pool before pool-level inference. Cap 1 was a structural identity for BoN and Random(N).

### 5.5 Equal-budget inference and multiplicity

The primary contrast was BoN(N) minus Random(N) regret. Nominal 95% intervals used 10,000 complete-pool cluster percentile bootstrap replicates with seed 20261125 and deterministic offsets [10]. Holm and Bonferroni adjustments were computed over the complete 4 contexts × 5 caps family, including cap-1 structural zeros with p=1 [11]. The V6 primary T-cell claim uses that all-20 Holm family. A separate family evaluated Random(N) minus Random(1). These procedures quantify retrospective sensitivity and do not transform historical outer pools into a prospective confirmatory trial.

### 5.6 Pointwise, global-rank, and tail diagnostics

MAE, RMSE, signed bias, global Spearman correlation, and mean within-pool Spearman correlation were computed over 1,024 candidates per repeat. The oracle global median shift subtracted the measured median prediction error solely as a diagnostic. Absolute-error outliers were defined within repeat and model by the Tukey rule \(|e|>Q_3+1.5\,\mathrm{IQR}\); no outlier was removed for selection.

Full-pool predicted top-\(k\) tails used frozen prediction descending order, sequence ID, and original row position, with \(k\in\{1,2,4,8,16\}\). The decomposition in Section 2.2 separates tail opportunity loss from within-tail ranking loss. Pair inversion rates excluded measured ties as specified in the source analysis. For actual BoN16 prefixes, the analogous identity was

\[
R_{\mathrm{BoN16}}=
(Y_{\mathrm{pool\ oracle}}-Y_{\mathrm{prefix\ oracle}})
+(Y_{\mathrm{prefix\ oracle}}-Y_{\mathrm{BoN16}}).
\]

The first term is identical for methods sharing a prefix and cancels in paired comparisons. Full-pool and prefix decompositions are reported separately. Intervals for diagnostic pool summaries used 10,000 complete-pool bootstrap replicates; no target labels were used to fit a correction in this diagnostic.

### 5.7 Target-context adaptation v1

All 314,563 exact-shared valid 50-nt sequences across T-cell r1/r2 were used to construct exact/H1 connected components before role allocation. Components were assigned intact to 50,000 train, 10,000 development, and 1,024 test sequences. Test had 16 pools of 64, with no exact/H1 edges across roles and no exact/H1 contact with Sample train/dev/calibration. Both repeats used identical sequences, pool membership, and proposal prefixes. Train labels were the per-sequence arithmetic mean of r1/r2 MRL after splitting. Nested label budgets were 1k, 5k, 20k, and 50k.

The main model initialized seeds 11/23/37 from corresponding frozen Sample CNN members. Its objective was MSE plus \(\lambda\) times dynamic top-16 pairwise logistic loss over all 120 unordered pairs in the current predicted tail; measured ties had zero pair weight. The grid crossed learning rates \(10^{-5},10^{-4},10^{-3}\) with \(\lambda=0.01,0.1,1\). MSE-only and from-scratch controls used prespecified grids. Development selection minimized the maximum of r1 and r2 cap16 regret, with frozen tie-breaking by mean-repeat regret, combined loss, smaller learning rate, and smaller \(\lambda\). The selected main model used learning rate \(10^{-5}\) and \(\lambda=0.01\). All 72 final runs converged under the dual-plateau rule on RTX 5090 CUDA; three seed members were equally averaged.

All methods shared nested proposal prefixes and full-pool repeat-specific oracles. The primary contrast was 50k main minus original CNN at cap16. Success required improvement in both repeats, a repeat-linked pool-block 95% interval entirely below zero for the combined contrast, and absolute reduction at least \(\max(0.1,5\%\times2.631329)=0.131566\) MRL. The interval used 10,000 repeat-linked pool bootstrap replicates, seed 20261126. Ten repeat–cap tests formed the secondary Holm family. Test labels were read once after all runs, predictions, and choices were frozen.

### 5.8 Full-64 ListNet adaptation v2

ListNet v2 reused v1 train/development roles and the FreshSampleCNN architecture. For group-standardized predictions \(p\) and true mean-repeat MRL \(y\), the listwise term was

\[
\mathcal L_{\mathrm{ListNet}}
=-\sum_i \operatorname{softmax}(y/\tau)_i\,
\log\operatorname{softmax}(p/\tau)_i,
\]

and the total objective was \(\mathcal L_{\mathrm{ListNet}}+w\,\mathrm{MSE}\). MSE retained the final incomplete training pool; ListNet used complete 64-sequence pools. The grid crossed learning rates \(10^{-5},10^{-4},10^{-3}\), temperatures \(0.5,1,2\), and MSE weights \(0.1,1,10\), totaling 27 configurations × 3 seeds = 81 runs. Development mean-repeat cap16 regret selected the model; the selected configuration was learning rate \(10^{-4}\), \(\tau=0.5\), and \(w=10\). No test-v2 quantity selected hyperparameters, checkpoints, seeds, caps, or weights.

Test-v2 first removed every component touching v1 train/dev/test, then every component with exact/H1 contact to Sample train/dev/calibration. From 39,222 eligible components, 1,004 components containing 1,024 sequences were frozen by hash order into 16 complete-component pools. There were zero exact/H1 conflicts against v1 roles or Sample and zero cross-test-pool components. Original CNN, v1 main, v1 MSE-only, and ListNet each scored all 1,024 sequences with three members before labels were read. All arms used identical nested prefixes for offline selection.

The primary endpoint used per-sequence \((r1+r2)/2\) MRL. Joint success required ListNet to beat both original CNN and v1 main at cap16, with each repeat-linked pool-block 95% interval below zero and reduction at least \(\max(0.1,5\%\times\mathrm{comparator\ regret})\). The primary inference was a co-primary intersection–union test with unadjusted intervals. A secondary 2 comparators × 5 caps Holm family was descriptive after primary failure. The interval used 10,000 replicates and seed 20261126. Repeat-specific, top-tail, pair-accuracy, MAE, RMSE, bias, and Spearman results were secondary. No post-evaluation protocol changes were made.

### 5.9 Paired HepG2 experimental anchor

The anchor cohort contained 1,024 unique 50-nt sequences in 16 pools, selected from 48,748 eligible sequences with valid paired HepG2 and T-cell measurements after Sample exact/H1 exclusion and within/across-pool exact/H1 isolation. HepG2 MRL was used directly as a ranking score; no residual model was fit. CNN, HepG2 anchor, a fixed 50/50 within-pool rank fusion, and Random(N) used identical proposal prefixes and first-occurrence tie handling. Regret always used measured T-cell MRL against the full 64-sequence pool oracle.

The primary contrast was HepG2-anchor BoN minus Sample-CNN BoN at cap16. The combined interval resampled 16 repeat-linked pools 10,000 times with both repeats and all five proposal seeds held together, seed 20261126. Success required both repeats to improve, a combined interval below zero, and reduction at least 0.120412 MRL. Ten repeat–cap anchor-versus-CNN tests formed the Holm family. Secondary analyses reported global and within-pool Spearman correlation, top-tail overlap and inversions, selection turnover, and the identity

\[
R=\text{oracle opportunity loss}+\text{proposal headroom}.
\]

Proposal headroom was identical across methods. T-cell MRL was evaluated once after freezing; no anchor weight was changed afterward.

### 5.10 Auxiliary-only development gates

The shared residual program fit ridge corrections to \(p,p^2\), GC, uAUG, longest poly-U, and an intercept using N50 or HepG2 source data after strict exclusion of T-cell and Sample exact/H1 sequences. The pairwise program optimized a convex logistic correction within frozen CNN top-16 tails using source train pairs and source-inner development selection. The multi-context program trained the same-capacity CNN on expanded Sample contexts with MSE plus dynamic top-16 pairwise loss. Each program evaluated a single selected model in both N50→HepG2 and HepG2→N50 destination auxiliary grouped tests. Both directions had to show corrected-minus-frozen cap16 regret below zero with the required interval gate before any T-cell evaluation. At least one direction failed in every program, so target scoring was hard-blocked and target count was zero. No target result is imputed for these programs.

### 5.11 V6 integration and independent verification

V6 added no wet-lab experiment, model run, or target unblinding. Round 1 independently reloaded aggregate TSV/JSON evidence, recomputed arithmetic identities and claim values, and checked every primary numeric statement against an independent claim–evidence ledger. Round 2 audited title/abstract emphasis, evidence-state language, correction families, test-set separation, gate semantics, figure references, and prohibited success/causal/blind wording. Round 3 built the PDF from the delivered Markdown in a clean temporary output directory, checked page text and rendering, fonts, overflow, required figures, and byte-identical deterministic rebuilds. Verification outputs and the ledger are versioned separately under `deliverables/reports/docs/`.

## Figure legends

**Figure 14 | Equal candidate budget reveals T-cell proxy-ranking value.** BoN(N)-minus-Random(N) measured-MRL regret is shown for identical nested proposal prefixes across four strict-H1 contexts and caps 1–16; negative values favor proxy ranking. Random(N) uses record-specific PCG64 child streams to select uniformly from the same prefix. Nominal intervals use 10,000 complete-pool bootstrap replicates. The complete all-20 Holm family includes cap-1 structural zeros. Every T-cell r1 cap 2–16 effect survives Holm, whereas T-cell r2 survives only at cap 4. HepG2 cap 2 also survives and N50 has no nominally supported effect, defining the broader context boundary without changing the T-cell result. All results are retrospective sensitivities on previously inspected outer contexts, not new blind tests. Source: `../figures/v3/Figure14_v3_equal_budget_value.pdf`.

**Figure 16 | T-cell pointwise error, useful global ranking, and top-tail decision loss coexist.** The diagnostic panels summarize frozen-CNN MAE/bias and global versus top-16 Spearman correlation, full-pool top-16 regret decomposition into tail opportunity and within-tail ranking losses, and actual BoN16 prefix decomposition. CNN MAE is 1.567/1.544 MRL and global Spearman is 0.591/0.529 in r1/r2, whereas top-16 Spearman falls to 0.206/0.144. The predicted full-pool top-16 winner is overturned in 15/16 pools per repeat. Full-pool and prefix analyses use different candidate coverage and are not interchangeable; oracle components describe headroom rather than realized repair. All analyses are retrospective strict-H1 diagnostics. Source: `../figures/v4/Figure16_v4_tcell_global_value_tail_bottleneck.pdf`.

**Figure 17 | Regression and cross-cell improvements do not establish better T-cell tail selection.** Panels summarize the separate v1 and v2 target-context grouped holdouts and the paired HepG2-anchor experiment. In v1, 50k adaptation lowers MAE in both repeats but increases cap16 regret; the combined effect is +0.025740 [−0.072019, +0.140994]. In test-v2, ListNet gives small favorable mean-repeat point estimates versus original CNN and v1 main, but both intervals cross zero, both reductions are below 0.1 MRL, and the co-primary gate fails. Direct HepG2 anchoring worsens combined cap16 T-cell regret by +0.428897 [+0.245460, +0.616474], despite comparable or higher global Spearman correlation. Absolute regret is not compared across different test pools. All experiments are retrospective grouped/paired holdouts, not new blind tests or causal demonstrations. Source: `../figures/v4/Figure17_v4_tcell_regression_decision_decoupling.pdf`.

## Data and code availability

Primary raw measurements are publicly available from Sample et al., GSE114002 [1], and Castillo-Hair et al., GSE232927 [2]. The V6 manuscript uses only de-identified aggregate result tables and frozen verification evidence under `deliverables/reports/source_data/`: `paper1_budget_matched_v1/`, `tcell_error_diagnosis_v1/`, `tcell_target_adaptation_v1/`, `tcell_listnet_adaptation_v2/`, `tcell_cross_cell_anchor_v1/`, `tcell_residual_bstrict_v1/`, `tcell_rank_bstrict_v2/`, and `tcell_multicontext_backbone_v1/`. These exports contain no sequence-level labels or predictions. Scientific source reports are `PAPER1_BUDGET_MATCHED_V1.md` and `TCELL_RESIDUAL_BSTRICT_V1.md` under `deliverables/reports/docs/`, and `TCELL_ERROR_DIAGNOSIS_V1.md`, `TCELL_TARGET_ADAPTATION_V1.md`, `TCELL_LISTNET_ADAPTATION_V2.md`, `TCELL_CROSS_CELL_ANCHOR_V1.md`, `TCELL_RANK_BSTRICT_V2.md`, and `TCELL_MULTICONTEXT_BACKBONE_V1.md` in the project root.

The independent V6 claim–evidence ledger, verification script, and three-round verification outputs are under `deliverables/reports/docs/`; the versioned Markdown, PDF builder, release manifest, and PDF are under `deliverables/paper/`. Full sequence-level artifacts and model weights remain in the frozen remote project storage documented by the source reports. Earlier public code is archived at <https://github.com/Cunyu-Liu/proxy-function-gap/tree/paper23-gates-v1>; that code-only archive does not contain the complete V6 evidence package, raw labels, weights, or the newest T-cell analysis code. No new V6 public release or DOI is claimed. Figure 14 is reused unchanged, and Figures 16/17 are cited from the separately versioned figure delivery; no figure, PPT, or source-data file was modified during V6 manuscript integration.

## References

[1] P. J. Sample, B. Wang, D. W. Reid, V. Presnyak, I. J. McFadyen, D. R. Morris, and G. Seelig, “Human 5′ UTR design and variant effect prediction from a massively parallel translation assay,” *Nature Biotechnology*, vol. 37, pp. 803–809, 2019. <https://doi.org/10.1038/s41587-019-0164-5>.

[2] S. Castillo-Hair, S. Fedak, B. Wang, et al., “Optimizing 5′UTRs for mRNA-delivered gene editing using deep learning,” *Nature Communications*, vol. 15, 5284, 2024. <https://doi.org/10.1038/s41467-024-49508-2>.

[3] J. Skalse, N. H. R. Howe, D. Krasheninnikov, and D. Krueger, “Defining and characterizing reward hacking,” in *Advances in Neural Information Processing Systems 35*, 2022. <https://doi.org/10.52202/068431-0687>.

[4] B. Trabucco, A. Kumar, X. Geng, and S. Levine, “Conservative objective models for effective offline model-based optimization,” in *Proceedings of the 38th International Conference on Machine Learning*, 2021. <https://proceedings.mlr.press/v139/trabucco21a.html>.

[5] B. Trabucco, X. Geng, A. Kumar, and S. Levine, “Design-Bench: Benchmarks for data-driven offline model-based optimization,” in *Proceedings of the 39th International Conference on Machine Learning*, 2022. <https://proceedings.mlr.press/v162/trabucco22a.html>.

[6] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” in *Advances in Neural Information Processing Systems 30*, 2017. <https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html>.

[7] P. Donti, B. Amos, and J. Z. Kolter, “Task-based end-to-end model learning in stochastic optimization,” in *Advances in Neural Information Processing Systems 30*, 2017. <https://proceedings.neurips.cc/paper/2017/hash/3fc2c60b5782d9e7e7b93a9e9b7a468f-Abstract.html>.

[8] B. Wilder, B. Dilkina, and M. Tambe, “Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization,” in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 33, pp. 1658–1665, 2019. <https://doi.org/10.1609/aaai.v33i01.33011658>.

[9] Z. Cao, T. Qin, T.-Y. Liu, M.-F. Tsai, and H. Li, “Learning to rank: From pairwise approach to listwise approach,” in *Proceedings of the 24th International Conference on Machine Learning*, 2007, pp. 129–136. <https://doi.org/10.1145/1273496.1273513>.

[10] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York: Chapman & Hall, 1993. <https://doi.org/10.1007/978-1-4899-4541-9>.

[11] S. Holm, “A simple sequentially rejective multiple test procedure,” *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979. <https://www.jstor.org/stable/4615733>.

[12] E. N. Weinstein, A. Amin, J. Frazer, and D. S. Marks, “Non-identifiability and the blessings of misspecification in models of molecular fitness,” in *Advances in Neural Information Processing Systems 35*, 2022. <https://doi.org/10.52202/068431-0396>.

[13] A. J. Reddy, X. Geng, M. H. Herschl, et al., “Designing cell-type-specific promoter sequences using conservative model-based optimization,” in *Advances in Neural Information Processing Systems 37*, 2024. <https://doi.org/10.52202/079017-2954>.

[14] K. Leppek, G. W. Byeon, W. Kladwang, et al., “Combinatorial optimization of mRNA structure, stability, and translation for RNA-based therapeutics,” *Nature Communications*, vol. 13, 1536, 2022. <https://doi.org/10.1038/s41467-022-28776-w>.

[15] N. Schlusser, A. González, M. Pandey, et al., “Current limitations in predicting mRNA translation with deep learning models,” *Genome Biology*, vol. 25, 227, 2024. <https://doi.org/10.1186/s13059-024-03369-6>.
