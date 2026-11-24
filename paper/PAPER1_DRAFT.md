# Contextual Boundaries of Proxy Selection Pressure in Biological Sequence Design

## Fixed-pool evidence from 5′ UTR optimization

**Anonymous authors**

### Abstract

Computational proxies are used to train, sample, and rank biological sequences, but their scores need not remain reliable after optimization pressure changes the candidate distribution. This study examines fixed-pool best-of-N selection in 5′ untranslated region (5′ UTR) design, holding the candidate-generating process fixed while varying proxy-selection pressure. Historical measurements from Sample et al. and Castillo-Hair et al. are evaluated using a frozen convolutional ensemble trained only on Sample data. Sample-designed pools show monotone regret reduction from 1.905 to 0.681 as the proposal cap increases from 1 to 16, whereas four external Castillo contexts exhibit non-monotone behavior or later reversals. Alternative N50 pools also reverse after cap 4–8. Within-pool score matching yields no consistent optimizer-specific error increment: all four absolute-error-difference intervals include zero. In contrast, absolute errors are strongly correlated between the convolutional ensemble and an independently trained 4-mer multilayer perceptron (0.931–0.965), indicating shared proxy failure across architectures. Gaussian perturbations of normalized proxy outputs do not produce monotone degradation. These results support context-dependent high-score extrapolation risk and common-proxy failure rather than a universal direction of optimization failure. The complete evidence is retrospective historical replay, not prospective wet-lab validation.

**Keywords:** proxy–function gap; biological sequence design; 5′ UTR; model-based optimization; winner’s curse; uncertainty; external validation

## 1. Introduction

Biological sequence optimization replaces a difficult-to-measure functional endpoint \(Y(x,c)\) with a computational proxy \(P(x,c_0)\), where \(c\) denotes deployment context. Proxy models are useful because they provide inexpensive scores for large sequence libraries. However, the act of selecting high-scoring candidates changes the distribution on which the proxy must remain valid. This creates a selection problem rather than only a pointwise regression problem: the best candidate according to the proxy can be suboptimal according to the historical functional assay.

Prior work has documented reward hacking, offline model-based optimization failures, and uncertainty-aware sequence design [1]–[5]. These studies establish that optimization can exploit proxy errors, but they do not imply a single direction or mechanism across biological contexts. In particular, four questions remain open: (i) whether fixed-pool pressure has a reproducible direction across studies and assays; (ii) whether an optimizer-specific increment persists after matching proxy scores; (iii) whether model disagreement certifies functional reliability; and (iv) whether Gaussian proxy-output noise is necessarily amplified by best-of-N selection.

This study addresses these questions using frozen models and historical experimental labels. Candidate pools are fixed before proxy scoring, so selection pressure changes without changing the pool-generating process. The main endpoint is measured-MRL (mean ribosome load) regret, not random-split correlation or a proxy score. A conservative reading is required: retrospective measurements provide an independent endpoint for a frozen model, but they do not constitute new wet-lab validation of unmeasured sequences.

The contributions are:

1. A fixed-pool evaluation of selection pressure across Sample 2019 and four Castillo-Hair contexts, with actual model-forward accounting and pool-cluster uncertainty.
2. Within-pool score matching to separate high-score extrapolation from optimizer-specific distortion.
3. Same- and cross-architecture common-proxy analysis using a frozen convolutional neural network (CNN) ensemble and a separately trained 4-mer multilayer perceptron (MLP).
4. Alternative-pool sensitivity and normalized-output noise experiments that preserve all negative and non-monotone results.

## 2. Related work

Reward hacking describes systematic divergence between a proxy and an intended objective under optimization [1]. Offline model-based optimization studies further show that optimizers can find designs with overestimated proxy values outside the data distribution [2], [3]. Conservative objective models and source-distribution constraints reduce, but do not eliminate, this problem [2], [4]. Deep ensembles provide scalable uncertainty estimates, yet ensembles trained on the same data can share errors [5], [6].

Biological sequence design has used conservative optimization, distribution-constrained optimization, and diverse sampling [4], [7], [8]. Sample et al. measured large 5′ UTR libraries and demonstrated both deep-learning-guided design and later prediction–measurement separation in high-predicted regions [9]. Castillo-Hair et al. supplied measurements across cell types and assay contexts [10]. Leppek et al. showed that translation, stability, early expression, and total expression are not interchangeable endpoints [11]. Work on model transfer cautions against treating different reporter systems and endogenous mRNA as one deployment context [12].

Inference-time optimization can also exploit reward-model errors [13]. VGSC uses verifier-guided stochastic decoding under imperfect generators and verifiers [14]. Active learning and selective prediction provide related label-acquisition and abstention mechanisms [15]. The present study does not claim novelty for best-of-N, ensembles, or abstention. Its focus is the evidentiary boundary of proxy-selection pressure and common failure under fixed-pool historical evaluation.

## 3. Methods

### 3.1 Fixed-pool regret

For pool \(C_t\) in context \(c\), let \(P_i\) be the frozen proxy mean for candidate \(x_i\), and let \(Y_i\) be its measured endpoint. A selection strategy \(\pi\) chooses \(x_t^\pi\). Regret is

\[
R_t^\pi = \max_{i\in C_t}Y_i - Y_{x_t^\pi}.
\]

Lower regret is better. Abstention is reported separately with coverage and is never assigned an artificial regret. Proxy scores are not used as the final functional endpoint. Candidate pool size is fixed within each context to avoid pool-size confounding.

### 3.2 Model and data lineage

The main proxy is a fresh three-member 50-nucleotide CNN regression ensemble trained on Sample random-library data. Three seeds are trained to development-loss plateau without a maximum epoch limit. Member predictions are inverse-transformed using member-specific train-only target scalers. An earlier normalized-output inference artifact and all downstream results were invalidated and excluded.

Castillo-Hair GSE232927 supplies the external contexts. N50 HEK293T is the main external test. Defined-end HepG2 and T-cell replicates are secondary contexts. All sequences have 50 nucleotides and finite measured MRL. N25 is excluded because the frozen model requires 50-nucleotide input. Exact Sample train/dev/calibration overlaps and Hamming-1 conflicts are excluded. Candidate pools are deterministic, and a second build must produce identical hashes. Labels are unavailable to the Learner until selections are frozen.

### 3.3 Selection protocols

Random and best-of-N (BoN) use proposal caps \(\{1,2,4,8,16\}\) and seeds \(\{101,202,303,404,505\}\). Each context has 16 pools of 64 unique sequences. Every score executes the three frozen members on CUDA; precomputed predictions are not counted as new forwards. The Evaluator reads historical labels only after Learner artifacts are validated.

### 3.4 Score matching and common-proxy analysis

For every cap-16 BoN selection, the Evaluator selects a different sequence with the nearest proxy mean in the same pool, using a deterministic hash tie-break. The absolute-error difference is reported with pool-cluster bootstrap. This tests whether optimized candidates have additional error beyond same-score high-tail extrapolation.

Common-proxy analysis first examines the CNN ensemble spread. It then trains a 4-mer MLP with 259 sequence features on the same Sample train/dev labels. The MLP seeds converge at epochs 108, 124, and 177, with best epochs 103, 119, and 172. Absolute-error correlations between CNN and MLP predictions quantify whether architectural diversity removes shared failure.

### 3.5 Alternative pools and output noise

An alternative N50 pool series uses a new deterministic namespace and the same protocol. Proxy-output noise is added to each normalized CNN output before inverse transformation, with \(\sigma\in\{0,0.1,0.25,0.5,1.0\}\). This is not probability-logit noise. Sigma-zero records must exactly match the original grid by strategy, cap, and seed.

### 3.6 Reproducibility

Raw files and derived artifacts are stored under `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/`. Source code is under `/home/proxy-function-gap`. Each evaluation uses independent verification rounds for file hashes, label isolation, selection identity, and regret. Statistical intervals resample complete pools rather than treating seeds or repeated measurements as independent biological evidence.

## 4. Results

### 4.1 Fixed-pool pressure is context-dependent

Table 1 reports mean measured-MRL regret. Sample-designed pools improve monotonically, but Sample-random and the external contexts do not. This rejects both context-free claims that more proxy pressure necessarily helps and that it necessarily harms.

**Table 1. Fixed-pool BoN regret across contexts**

| Context | cap 1 | cap 2 | cap 4 | cap 8 | cap 16 |
|---|---:|---:|---:|---:|---:|
| Sample designed | 1.905 | 1.528 | 1.161 | 0.770 | 0.681 |
| Sample random | 6.552 | 6.057 | 6.760 | 6.507 | 6.883 |
| N50 HEK293T | 6.422 | 6.341 | 6.617 | 6.705 | 6.519 |
| Defined HepG2 r1 | 4.135 | 3.941 | 3.738 | 3.741 | 3.803 |
| Defined T-cell r1 | 4.739 | 4.787 | 4.807 | 4.747 | 4.352 |
| Defined T-cell r2 | 6.563 | 6.276 | 5.850 | 6.209 | 5.928 |

Sample-designed cap-16 regret is 0.681 with a pool-cluster 95% interval of [0.541, 0.833]. The external curves show later plateaus or reversals rather than sustained improvement. Defined HepG2 reaches its minimum near cap 4, whereas the two T-cell replicates show different intermediate behavior. These differences preclude a single universal pressure–regret curve.

### 4.2 Alternative N50 pools also fail to show sustained improvement

The alternative N50 pool series gives regrets 6.521, 5.914, 5.625, 5.618, and 5.726 at caps 1, 2, 4, 8, and 16. Like the original series, it is non-monotone and reverses after cap 4–8. The absence of sustained improvement is therefore not an artifact of one deterministic pool draw.

### 4.3 Score matching does not support an optimizer-specific increment

Table 2 reports absolute errors for BoN selections and their within-pool score matches. All intervals include zero and directions differ across contexts.

**Table 2. Score-matched absolute-error differences**

| Context | BoN | Matched | Difference | Pool-cluster 95% interval |
|---|---:|---:|---:|---:|
| N50 HEK293T | 2.881 | 2.707 | +0.174 | [−0.462, 0.721] |
| Defined HepG2 r1 | 3.234 | 3.134 | +0.101 | [−0.581, 0.778] |
| Defined T-cell r1 | 2.389 | 3.094 | −0.705 | [−1.493, 0.024] |
| Defined T-cell r2 | 2.512 | 2.741 | −0.229 | [−1.196, 0.809] |

The evidence is therefore more consistent with high-score extrapolation risk than with an additional optimizer-specific error mechanism. It does not show that optimization is harmless; it shows that the data do not identify a distinct optimizer increment after score matching.

### 4.4 Common proxy risk crosses model architectures

Low CNN disagreement does not reliably identify high-error regions. More importantly, CNN and 4-mer MLP absolute errors remain strongly correlated across every external context (Table 3).

**Table 3. Absolute-error correlation across architectures**

| Context | CNN mean absolute error | 4-mer mean absolute error | Correlation |
|---|---:|---:|---:|
| N50 HEK293T | 2.805 | 2.852 | 0.962 |
| Defined HepG2 r1 | 2.907 | 3.015 | 0.953 |
| Defined T-cell r1 | 2.076 | 2.223 | 0.931 |
| Defined T-cell r2 | 2.568 | 2.690 | 0.965 |

Shared training data and endpoint definitions therefore dominate over architectural diversity. Additional same-data models should not be treated as independent functional validation.

### 4.5 Proxy-output noise is not necessarily amplified

At cap 16, regrets for \(\sigma=0,0.1,0.25,0.5,1.0\) are 6.519, 6.018, 6.143, 6.205, and 6.526. Sigma-zero exactly reproduces the original grid. Moderate noise can improve selection, while the largest noise returns close to baseline. The hypothesis that BoN necessarily amplifies Gaussian output noise is not supported. These observations are descriptive and are not used to propose noise injection as a method.

## 5. Discussion

The results support a conditional proxy contract rather than a universal account of proxy failure. Fixed-pool pressure can improve measured selection in one context and stall or reverse in another. Score matching does not identify a stable optimizer-specific increment, while cross-architecture error correlations demonstrate that shared errors can dominate over model diversity. Alternative pools and the negative noise experiment further constrain the mechanism narrative.

These findings should not be interpreted as proof that proxies are useless. Sample-designed pools show that a frozen proxy can remain useful under increased selection pressure. Nor do they justify a claim that optimization always damages function. Instead, the results show that proxy validity must be declared by library, cell, assay, and training lineage, and then checked against an independent endpoint.

## 6. Limitations

This study is retrospective historical replay. It does not include new synthesis, cell assays, animal studies, or prospective wet-lab validation. Sample outer pools have already supported the pilot analysis and are secondary in the pressure analysis. Castillo contexts are external to Sample training but remain fixed subsets of one study, not independent-study meta-analysis. T-cell replicates and deterministic pool seeds are not independent biological replications.

The alternative MLP shares training data and endpoint definitions with the CNN, so high error correlation is expected to be partly data-driven and does not identify every possible source of common failure. Score matching controls proxy mean but not all sequence features or higher-order distribution shifts. Sequence-family analyses are exploratory. The noise protocol uses normalized-output Gaussian perturbations and does not represent all verifier misspecification. Finally, the full paper has not undergone external peer review.

## 7. Conclusion

Fixed-pool historical evaluation shows that proxy-selection pressure has context-dependent effects. High-score extrapolation risk is a more consistent explanation than a stable optimizer-specific increment, and common proxy errors persist across architectures. Alternative pools and output-noise results reject simple monotonic failure narratives. Future work should prioritize independent-study replication, prospective blind evaluation, and mechanism-specific proxy contracts rather than additional same-data model ensembles.

## Data and code availability

Source code and result documents are available at https://github.com/Cunyu-Liu/proxy-function-gap. The analysis environment maintains code under `/home/proxy-function-gap` and `/home/proxy-function-gap-data`; large data, weights, and artifacts are under `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/`. Result documents include `PAPER1_EXTERNAL_RESULT.md`, `PAPER1_MECHANISM_RESULT.md`, `PAPER1_CROSS_ARCH_RESULT.md`, and `PAPER1_SENSITIVITY_NOISE_RESULT.md`.

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
