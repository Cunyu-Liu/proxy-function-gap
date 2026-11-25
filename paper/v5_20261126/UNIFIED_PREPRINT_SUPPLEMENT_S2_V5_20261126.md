---
title: "Supplementary Study S2: Limits of selector-based and risk-based proxy control"
subtitle: "Verifier guidance, conservative reranking, pool-level Q prediction, and frozen calibration failure of uncertainty abstention"
date: "2026-11-26"
version: "v5"
---

## Scope and evidence status

This supplement corresponds to the original Paper 2 research line. It is a study of limitations and falsification, not an independently validated positive method paper. The selector and Q-head analyses contain completed retrospective falsification results. Candidate-level ensemble-uncertainty abstention now also has a completed frozen development/calibration evaluation: the retrospective signal did not pass the frozen calibration gate. The outer-scoring invocation was rejected by a hard lock, so no outer result exists for this version. Historical outer empirical-coverage results remain retrospective sensitivity evidence, not blind evidence.

## S2.1 Question, hypotheses, and decision chain

The V5 main article establishes equal-budget proxy-ranking value in specific T-cell settings, then retains its attenuation after GC/uAUG/poly-U/support adjustment, shared cross-architecture errors, endpoint mismatch, and no detectable transfer of the complete PERSIST MLP contract. Those findings motivate, but do not prove, a decision-layer remedy. Neither equal-budget gains nor external nulls can substitute for an independent risk-control gate. I separate three hypotheses:

- **H2.1 — selector hypothesis:** A verifier-guided, conservative, uncertainty-penalized, distance-penalized, or diversity-aware selector can reduce measured-MRL regret relative to BoN at a declared compute budget.
- **H2.2 — pool-risk hypothesis:** A pool-level Q head can discriminate realized regret and improve accepted regret by rejecting high-risk pools.
- **H2.3 — candidate-risk hypothesis:** Ensemble standard deviation can rank candidate-level risk such that abstaining on high-uncertainty selections reduces accepted regret under a development-frozen numeric threshold.

H2.1 and H2.2 were not supported in the reported settings. H2.3 failed its frozen calibration gate: the point estimate was negative and its interval included zero. Failure of H2.1 does not logically entail success of H2.2 or H2.3; the completed H2.3 gate now prevents a positive Paper 2 deployment claim.

## S2.2 Verifier-guided fixed-pool pilot

### Protocol

The V head estimated proxy-overestimation probability and defined \(V=1-r\). FP-REJ used development-frozen \(\alpha=20\) and cap 16. Q was a pool-level abstention probability and did not alter candidate order. Development contained 32 disjoint pools, calibration 16 pools, and selection used five paired seeds. The Learner froze selections with `regret=null`; the Evaluator subsequently read historical measured MRL.

The designed outer set contained 16 pools. Four random outer pools were planned, but pool 003 was excluded as a whole because two candidates lacked measured MRL; no imputation or partial-pool scoring was used. Three eligible random pools remained secondary evidence.

### Result and inference

| Context | Method | Pools | Coverage | Mean regret | Interpretation |
|---|---|---:|---:|---:|---|
| Designed | FP-REJ without Q | 16 | 1.0 | 1.024271 | Worse than BoN |
| Designed | BoN cap 16 | 16 | 1.0 | 0.681114 | Reference |
| Designed | FP-REJ+Q | 16 | 0.0 | Undefined | Q rejected every selection |
| Random eligible | FP-REJ+Q | 3 | 1.0 | 6.827722 | Secondary only |
| Random eligible | BoN cap 16 | 3 | 1.0 | 6.882963 | Secondary reference |

Designed FP-REJ-minus-BoN regret was **+0.343157**, pool-cluster 95% interval **[0.225072, 0.467098]**; positive values favor BoN. Random FP-REJ+Q-minus-BoN was **−0.055241**, interval **[−1.400000, 1.051111]**. The small three-pool result does not reverse the designed gate failure.

FP-TILT retained its full support set and exceeded the common 48-CNN-member main budget; its results are not equal-cost evidence of superiority. Subsequent Q recalibration did not trigger rejection in the exploratory outer contexts, so V+Q and V-only selections were identical while Q added model and feature costs. No outer result was used to retune threshold, \(\alpha\), cap, features, or model class.

**Conclusion:** H2.1 is not supported for the tested verifier-guided selector. The result is a specific falsification, not a proof that all verifier-guided methods must fail.

## S2.3 Conservative, uncertainty, distance, and diversity baselines

### Development-only configuration

The comparator family included BoN, ensemble LCB, train-distance penalty, COM-style conservative ensemble, and diversity-LCB. LCB and distance minimized development mean top-1 regret; diversity-LCB minimized development regret@4. Development selected zero penalties for LCB and distance, making both identical to BoN. This also makes their feature-permutation negative controls trivial.

The COM comparator is an independently implemented COM-style adaptation, not the official COM UTR-ResNet experiment or official weights. It used the same CNN family, 50-step discrete adversarial negatives, development checkpointing, and three CUDA workers. At designed cap 16, every selector used 48 CNN member item-forwards. Training and extra feature costs were recorded separately.

### Designed outer retrospective result

| Method | Mean regret | Regret@4 | CNN member forwards | Top-4 maximum measured MRL |
|---|---:|---:|---:|---:|
| BoN | 0.672377 | 0.375235 | 48 | 7.220927 |
| LCB | 0.672377 | 0.375235 | 48 | 7.220927 |
| Distance | 0.672377 | 0.375235 | 48 | 7.220927 |
| COM-style | 0.661818 | 0.381116 | 48 | 7.215046 |
| Diversity-LCB | 0.640713 | 0.378499 | 48 | 7.217662 |

Negative paired differences favor the method.

| Comparison | Mean difference | Pool-cluster 95% interval |
|---|---:|---:|
| LCB − BoN | 0.000000 | [0.000000, 0.000000] |
| Distance − BoN | 0.000000 | [0.000000, 0.000000] |
| COM-style − BoN | −0.010559 | [−0.045744, 0.017583] |
| Diversity-LCB − BoN | −0.031664 | [−0.098743, 0.030134] |

COM-style and diversity-LCB do not demonstrate a top-1 regret advantage because their intervals include zero. Their numerical point estimates must not be described as established improvement.

### Random outer retrospective result

Only three eligible pools were available. BoN regret was 6.088421, COM-style 7.899485, and diversity-LCB 6.772796. COM-style-minus-BoN was **+1.811064 [0.123119, 2.660000]**, favoring BoN. Diversity-LCB-minus-BoN was **+0.684375 [−0.573597, 2.250072]**, with an interval crossing zero. This small secondary analysis argues against a context-general COM-style benefit but is not independent-study replication.

**Conclusion:** The evaluated rerankers did not establish a context-general regret advantage. This is not evidence that all possible conservative or uncertainty-aware designs are ineffective.

## S2.4 Pool-level Q calibration does not imply regret discrimination

### Analysis roles

Q recalibration used frozen scores without new training or inference. Development contained 32 pools and was in-sample for the Q calibrator. Calibration contained 16 honest holdout pools. Pool-cluster bootstrap used 10,000 replicates and seed 20261124. No outer labels were read for this analysis.

### Calibration and discrimination

Calibration log loss improved from 0.327 to 0.069. This is a positive probability-calibration result only. The realized-regret associations were:

| Role | n | Q vs FP-REJ regret | Q vs BoN regret |
|---|---:|---:|---:|
| Development, in-sample | 32 | ρ = −0.005 [−0.376, 0.373] | ρ = −0.011 [−0.368, 0.347] |
| Calibration, held out | 16 | ρ = −0.353 [−0.795, 0.193] | ρ = −0.200 [−0.718, 0.401] |

Calibration top-half BoN-regret AUC was **0.297 [0.094, 0.727]**, with an interval crossing 0.5. All Spearman intervals included zero. The negative calibration point direction was not reinterpreted as a useful inverse risk score because it was uncertain and contrary to the intended direction.

Q-based rejection did not reduce accepted regret:

| Calibration abstention | Accepted regret | Random-abstention expectation | Reduction |
|---:|---:|---:|---:|
| 0% | 2.899691 | 2.899691 | 0.000000 |
| 25% | 2.958708 | 2.899691 | −0.059017 |
| 50% | 3.069908 | 2.899691 | −0.170216 |

**Conclusion:** H2.2 is not supported. Better log loss does not establish ranking of realized decision loss. Q-based pool rejection is not carried forward as a positive method.

## S2.5 Candidate-level ensemble-uncertainty abstention fails the frozen calibration gate

The baseline artifacts separately evaluated candidate-level abstention using ensemble standard deviation. Unlike LCB reranking, this rule leaves the selected BoN candidate unchanged among accepted cases and removes high-uncertainty selections. The historical designed-outer analysis used empirical quantiles computed from those same viewed outer decisions; it did not store a numeric deployable threshold and therefore is not blind evidence.

| Historical empirical coverage | Accepted count | Accepted regret | Evidence status |
|---:|---:|---:|---|
| 0.10 | 8 | **0.458902** | Retrospective outer empirical quantile |
| 0.20 | 16 | 0.498983 | Retrospective outer empirical quantile |
| 0.50 | 40 | 0.672155 | Retrospective outer empirical quantile |
| 1.00 | 80 | **0.672377** | Full coverage |

The frozen v1 protocol instead defined one selected-decision-level threshold on development only. The candidate was the BoN cap-16 top-1 decision for each pool and paired seed, not a pool-level risk aggregate or within-pool filtering of every candidate. Development selected from q ∈ {0.50, 0.75, 0.90} by maximum full-BoN-minus-accepted regret. It selected **q = 0.50**, giving numeric threshold **0.104026897** (full precision 0.10402689717871444), development coverage 0.500, accepted regret 2.425647, and reduction 0.009992 [−0.290836, 0.319825]. All three development intervals included zero; selection by the preregistered point-estimate rule was not itself evidence of benefit.

Calibration was evaluated once without reselecting q or the threshold. Of 80 decisions, 34 were accepted across 15/16 pools, giving **coverage 0.425**. Accepted mean regret was **2.866883**, versus **2.861463** for full BoN. The primary full-BoN-minus-accepted reduction was **−0.005420**, with 95% pool-cluster interval **[−0.509271, 0.628623]** (10,000 replicates, seed 20261125). Both effect gates failed: the point estimate was not positive and the interval did not exclude zero. Minimum accepted-decision and accepted-pool counts passed, but could not rescue the effect gate.

The outer-scoring invocation was rejected by `PermissionError` because calibration had not authorized it. Thus `outer_scoring_authorized=false`, `outer_scored=false`, and no outer result or new outer decision file was produced. No alternative threshold was tried after failure.

**Conclusion:** The retrospective signal did not pass the frozen calibration gate. H2.3 is not supported under this rule and protocol. The old outer curve cannot be used to reverse that conclusion or as blind evidence. This version does not provide a positive Paper 2 risk-control result and does not negate every possible future pool-level risk model; any different risk definition or deployment rule requires a new version and independent evaluation roles.

## S2.6 Sensitivity results retained from the pressure study

An alternative N50 pool series gave regrets 6.521, 5.914, 5.625, 5.618, and 5.726 at caps 1, 2, 4, 8, and 16. Like the original strict-H1 series, it was non-monotone and reversed after intermediate caps. Normalized-output Gaussian noise at N50 cap 16 gave regrets 6.519, 6.018, 6.143, 6.205, and 6.526 for \(\sigma=0,0.1,0.25,0.5,1.0\). The absence of monotone degradation does not justify noise injection as a correction.

## S2.7 Limitations

- Selector and historical risk–coverage outer results use previously viewed pools and are retrospective, not new blind tests.
- The old outer curve computed empirical quantiles from the same evaluated outer role and stored no numeric deployable threshold; it cannot substitute for the frozen calibration evaluation.
- The random outer analysis has only three eligible pools.
- COM-style is not an official COM reproduction; conclusions apply to this adaptation.
- Matching CNN member forwards does not equalize verifier, Q, training, or feature-generation costs.
- Development-selected zero LCB/distance penalties limit what can be learned from those comparisons.
- Q development associations are in-sample for the calibrator; only the 16 calibration pools are honest holdout for that analysis.
- Calibration quality, discrimination, and downstream decision benefit are distinct; only the first improved for Q.
- The frozen candidate-risk rule applies to one BoN-selected decision per pool and seed, not pool-level aggregation or filtering every candidate within a pool.
- Development selected q=0.50 by a point-estimate rule while all development intervals included zero; calibration then failed both effect gates.
- The calibration failure hard-locked outer scoring. No outer score exists for this version, and no conclusion is made about other risk definitions without a new protocol and independent evaluation roles.
- No conclusion is made about generative VGSC, prefix verifiers, other active selective-prediction methods, or wet-lab efficacy.

## S2.8 Supplementary figure legends

**Supplementary Figure S2.1 | Verifier-guided fixed-pool pilot fails its designed gate.** Paired FP-REJ-minus-BoN regret at cap 16; positive values favor BoN. The designed comparison uses FP-REJ before Q because original Q abstained on all pools. The random comparison uses three eligible pools and its interval includes zero. Source: `../figures/ExtendedData_VGSC_paired_difference.pdf`.

**Supplementary Figure S2.2 | Conservative and diversity selectors do not establish a designed-outer advantage.** Five methods are compared at 48 CNN member item-forwards, with paired differences and pool-cluster intervals. COM-style and diversity-LCB intervals include zero; LCB and distance equal BoN. Source: `../figures/Figure7_v2_paper2_baselines.pdf`.

**Supplementary Figure S2.3 | Normalized-output noise does not cause monotone BoN degradation.** N50 regret and paired differences are shown across noise levels. This is a sensitivity analysis, not a proposed correction. Source: `../figures/ExtendedData_proxy_noise.pdf`.

## S2.9 Verification

The V5 integrated CPU verification script checks the FP-REJ effect and interval, five-method 48-forward budget, designed/random paired intervals, retrospective evidence flags, Q Spearman statistics from 48 pool rows, negative Q-abstention reductions, and the historical 0.458902/0.672377 empirical-coverage values. It independently checks the frozen development quantile and threshold, calibration coverage/regret/reduction/interval, failed effect gates, and outer hard-lock/no-score flags against `paper2_risk_frozen_v1/frozen_protocol.json` and the refusal log. Three actual V5 verification rounds and clean PDF rebuild checks are retained. No new training, inference, remote access, unblinding, outer scoring, or modification of figures/PPT/source_data was performed for this supplement.
