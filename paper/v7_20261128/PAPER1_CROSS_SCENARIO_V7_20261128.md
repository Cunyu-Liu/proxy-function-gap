---
title: "Decision-focused evaluation reveals task-dependent value of sequence proxies across biological design"
subtitle: "Equal-budget functional regret separates useful ranking from regression quality in mRNA, protein, and AAV design"
author: "Anonymous authors"
date: "2026-11-28"
lang: en
---

> **Draft status.** Cross-scenario integration, version 7. All analyses use historical public data and are retrospective. Grouped holdouts and frozen historical evaluations are not prospective blind tests, independent wet-lab replications, or clinical validation. This version integrates the T-cell mechanism study with GB1, GFP, AAV9 production, TEM-1, and non-T-cell mRNA scenarios. It does not reopen any frozen evaluation or alter source data.

## Abstract

Computational proxies are widely used to rank biological sequences, yet a model that predicts an assay well on average need not select the best functional candidate. We evaluated this distinction using measured functional regret—the loss relative to the best measured candidate in a fixed pool—under equal-budget best-of-N (BoN) and RandomN controls. In two defined-end T-cell 5′ UTR repeats, a frozen Sample-trained CNN reduced measured mean ribosome load (MRL) regret at every cap 2–16; repeat 1 survived the complete all-20 Holm family at all four caps, and repeat 2 survived at cap 4. Tail decomposition localized the remaining bottleneck to ordering within the high-score tail, and target-context adaptation reduced MAE without improving cap16 regret. The decision-focused pattern extended to three additional high-value tasks: GB1 IgG-binding lnW (cap16 BoN−RandomN −0.160894, 95% CI [−0.282853, −0.050515]), GFP brightness (additive model −0.2800 [−0.3095, −0.2521]), and AAV9 production (−4.206431 [−4.564111, −3.931098]). However, benefits were not universal: TEM-1 Amp→Amp transfers were often favorable but Amp→cefotaxime intervals all crossed zero; mRNA stability selection was significantly worse than random; and gene-editing indel, HEK293T high-cap MRL, Leppek MRL, and endogenous TE showed no robust corrected benefit. These results support a task-dependent proxy contract: functional regret, equal-budget controls, and explicit endpoint semantics should govern model acceptance, while MAE and global correlation remain complementary diagnostics.

**Keywords:** sequence design; functional regret; best-of-N; decision-focused evaluation; 5′ UTR; GB1; GFP; AAV; TEM-1; proxy validity

## 1. Introduction

Biological sequence design commonly replaces an expensive functional measurement with a computational proxy. The proxy may be a regression model, an ensemble score, or an experimentally derived fitness from a related condition. The decision is usually not to predict every sequence accurately, but to choose one or a few candidates from a large pool. These are different objectives. A model can have substantial average error and still improve the selected candidate; conversely, a model can reduce MAE or increase global correlation without improving the highest-scoring choices.

This distinction matters across therapeutic and engineering applications. mRNA design requires translation, stability, and editing outcomes that are not interchangeable. Protein engineering often uses deep mutational scanning (DMS) fitness as a proxy for binding, brightness, or resistance. AAV capsid design uses production, packaging, or tropism enrichments that do not directly measure clinical efficacy. In each case, the relevant question is whether the proxy improves the measured functional decision under a fair candidate budget.

We therefore organized the study around a single decision-focused endpoint: functional regret. For a fixed candidate pool with measured endpoint \(Y\), a selector chooses candidate \(x^\pi\), and regret is \(R^\pi=\max_{x\in C}Y(x)-Y(x^\pi)\). Lower regret is better. BoN selects the highest proxy score among \(N\) proposals; RandomN selects uniformly from the same candidate budget. The paired contrast \(R^{\mathrm{BoN}}-R^{\mathrm{RandomN}}\) isolates the value of proxy ranking from candidate opportunity.

The study has three aims. First, establish a concrete positive mechanism case in T-cell 5′ UTR MRL selection. Second, test whether the same decision-focused pattern appears in other high-value sequence design tasks. Third, identify boundaries where proxy ranking does not improve measured function, so that the conclusion is not overstated as universal proxy validity.

## 2. Results

### 2.1 T-cell MRL: equal-budget ranking value coexists with high pointwise error

In the strict-H1 T-cell reconstruction, BoN and RandomN used identical nested proposal prefixes. CNN ranking reduced measured-MRL regret at every cap 2–16 in both repeats (Fig. 14). In repeat 1, the cap16 effect was −1.705760 MRL [−2.163445, −1.211914], and all four nontrivial caps survived the complete all-20 Holm family (adjusted p=0.004000). In repeat 2, all four point estimates and nominal intervals favored BoN, but only cap 4 survived the full family (−1.014876 [−1.544490, −0.465832]; adjusted p=0.006399). The cap16 repeat-2 estimate remained favorable but was not multiplicity-robust in that family.

This positive decision result coexisted with high pointwise error. CNN MAE was 1.567 and 1.544 MRL, while global Spearman correlations were 0.591 and 0.529. Within the full-pool predicted top 16, Spearman fell to 0.206 and 0.144, and the predicted winner was not the measured tail optimum in 15/16 pools per repeat. At cap16, within-tail ranking loss accounted for 2.480 of 3.316 MRL regret in repeat 1 and 2.117 of 4.071 in repeat 2 (Fig. 16). Thus, the remaining bottleneck was not absence of global ranking information, but unreliable ordering among the highest-scoring candidates.

### 2.2 T-cell adaptation: regression improvement did not imply decision improvement

We next tested whether target-context supervision could repair the tail. A 50,000-label grouped holdout fine-tuned the frozen CNN with MSE plus a dynamic top-16 pairwise term. The selected model reduced MAE from 1.352159 to 0.914506 in repeat 1 and from 1.062273 to 1.018716 in repeat 2, yet cap16 regret increased by +0.038929 and +0.012551 MRL. The combined effect was +0.025740 [−0.072019, +0.140994], and all ten repeat–cap Holm-adjusted p values were 1. The MSE-only arm made exactly the same cap selections as the joint model, indicating that the selected pairwise weight produced no additional realized decision benefit.

A subsequent full-64 ListNet objective was evaluated on a new component-disjoint test. ListNet regret was 1.729031 versus 1.766447 for the original CNN and 1.736616 for the 50k model. The effects were small and favorable in point estimate, −0.037415 [−0.183204, +0.102494] and −0.007585 [−0.028698, +0.012474], but both intervals crossed zero and both improvements were below the prespecified 0.1 MRL threshold. The joint co-primary gate failed (Fig. 17).

A same-sequence HepG2 experimental anchor provided a complementary falsification. Although HepG2 global Spearman with T-cell MRL was comparable to or slightly higher than CNN correlation, direct HepG2 ranking increased combined cap16 T-cell regret by +0.428897 [+0.245460, +0.616474]. Harmful selection changes outnumbered beneficial changes in both repeats. Global functional information from another cell therefore did not automatically repair T-cell tail choices.

### 2.3 GB1 binding: additive proxy ranking improves measured lnW regret

We extended the same decision-focused evaluation to GB1 IgG-Fc binding using the Olson DMS landscape. The endpoint was experimentally derived lnW, not Kd or direct binding free energy. A one-hot additive ridge with shared substitution effects was trained under position-blocked isolation and selected only on development pools. On 55 permanent position-pair test pools, cap16 BoN−RandomN regret was −0.160894 [−0.282853, −0.050515] (Fig. 18). All four budgets favored BoN, but 31/55 pools improved and 24/55 worsened at cap16, preserving substantial heterogeneity. MAE was 2.159 and global Spearman 0.348, again showing that moderate pointwise quality can coexist with useful functional selection.

### 2.4 GFP brightness: low-order supervision supports high-order selection despite correlation decay

In the Sarkisyan GFP landscape, models trained on 1–2 mutations were evaluated on ≥5-mutation permanent pools. The CNN ensemble achieved cap16 BoN−RandomN regret −0.260830 [−0.301866, −0.218031], while a simpler additive ridge achieved −0.2800 [−0.3095, −0.2521] and lower BoN regret at every budget (Fig. 18). Thus, the decision benefit did not require a complex CNN.

Pointwise ranking quality nevertheless decayed with mutation order: CNN Spearman was 0.582 at order 5, 0.076 at order 8, and −0.020 at order 9 (Fig. 19). The fixed-pool selection benefit therefore cannot be inferred from, or replaced by, high-order global correlation. Simple additive baselines are essential controls.

### 2.5 AAV9 production: strong average decision benefit without extreme-best recovery

In the Fit4Function AAV9 production archive, strict exclusion of cross-library exact and Hamming-1 overlaps yielded 63,941 test variants. A sequence CNN trained only on modeling data achieved cap16 BoN−RandomN regret −4.206431 [−4.564111, −3.931098] on 3,996 disjoint pools (Fig. 18). Cap4 and cap64 were also favorable. The endpoint was nuclease-resistant viral-genome production enrichment, not tropism, infectivity, transgene expression, or clinical efficacy.

Global Spearman was high (0.821), but true top1% recall was only 18.75%, and BoN regret remained nonzero (Fig. 19). Average decision improvement and recovery of the extreme best variants are therefore distinct claims. The result supports production-oriented reranking, not a general gene-therapy efficacy conclusion.

### 2.6 TEM-1 resistance: proxy value depends on the deployment condition

Using Stiffler TEM-1 data, one antibiotic condition's experimentally derived relative resistance served as a proxy for another condition's measured endpoint. Across Amp→Amp directions, BoN usually reduced regret: at cap16, 18/20 point estimates favored BoN and 12/20 intervals were wholly below zero. The overall cap16 descriptive effect was −0.159379 [−0.207874, −0.113933], but this pooled description must not hide direction heterogeneity.

In contrast, all 15 Amp→cefotaxime direction–budget intervals crossed zero, and effects were near zero (Fig. 18). Global Amp→Amp Spearman median was 0.312, falling to 0.132 in the proxy top decile; Amp→cefotaxime global Spearman median was −0.053. The same proxy can therefore have selection value within a drug family but no resolved value across drugs. The endpoint is experimentally derived relative resistance, not direct enzyme kinetics, MIC, or clinical drug response.

### 2.7 Non-T-cell mRNA scenarios: stability, editing, and translation boundaries

The mRNA extension preserved endpoint-specific regret. HEK293T N50 high-cap MRL selection showed no sustained benefit; HepG2 defined-end cap16 was nominally favorable but not Holm-robust. Castillo gene-editing indel had 0/4 corrected-positive contexts at cap16, with both favorable and unfavorable point estimates. mRNA stability was significantly worse under proxy ranking: cap16 BoN−RandomN integrated-abundance regret was +0.2838 [+0.0824, +0.4435]. The 27 experimental sequences were the candidate unit; eight spike-ins were calibration controls, and six time points were not biological replicates.

Historical Leppek MRL and Schlusser endogenous TE results were integrated without re-unblinding. Leppek BoN−Random1 regret was +0.104723 [−0.100964, +0.212706], and Schlusser TE was +0.626608 [−0.828459, +9.998249]. These protocols used Random1 rather than the newer RandomN and are not pooled with them. Together, the mRNA scenarios show that translation, stability, editing, and endogenous TE cannot be treated as interchangeable proxy targets.

## 3. Discussion

The central contribution is a decision-focused evaluation principle supported across multiple biological design tasks. Proxy ranking can provide real equal-budget functional value in T-cell MRL, GB1 binding, GFP brightness, and AAV9 production. Yet the same evaluation reveals clear boundaries: TEM-1 cross-drug transfer, mRNA stability, gene-editing indel, and endogenous TE do not show robust corrected benefit, and stability is significantly harmful. The appropriate conclusion is not that proxies are universally useful or universally broken, but that proxy value is a function of task, endpoint, training lineage, and deployment condition.

Three practical lessons follow. First, functional regret should be the primary acceptance endpoint for sequence-selection models. MAE, RMSE, and global correlation are useful diagnostics, but T-cell adaptation, GFP order decay, and AAV top-tail recall show that they do not determine the selected candidate's measured function. Second, equal-budget RandomN and simple additive baselines are necessary. GFP additive outperformed CNN, and GB1 additive ridge was sufficient; complexity is not automatically valuable. Third, endpoint semantics must be explicit. lnW is not Kd, production is not tropism, brightness is not pure expression, and relative resistance is not direct enzyme kinetics.

The T-cell mechanism case explains why this matters. High global ranking can coexist with high MAE, while the remaining decision loss concentrates in the high-score tail. A model development program that accepts improvements based only on average error can therefore appear successful while leaving the actual design decision unchanged or worse. The cross-scenario results show that this is not a T-cell-specific curiosity.

This study does not establish prospective clinical or wet-lab efficacy. All data are historical, and grouped holdouts are not independent studies. The evidence supports a methodological standard for evaluating sequence proxies, not a universal design algorithm. Future methods should be accepted only when they improve measured functional regret under a new, appropriately isolated evaluation while preserving the equal-budget advantage demonstrated here.

## 4. Limitations

**Retrospective evidence.** All analyses use historical public data. Strict-H1, grouped holdouts, and frozen historical evaluations are not prospective blind tests or independent wet-lab replications. Repeated algorithmic seeds and bootstrap replicates do not create additional biological studies.

**Endpoint and unit heterogeneity.** MRL, lnW, log brightness, production enrichment, indel, integrated abundance, TE, and relative resistance have different units and semantics. We report task-specific effects and do not compute a cross-task pooled effect size. Relative regret reductions in figures are descriptive normalizations, not meta-analytic estimates.

**RandomN implementations differ.** GB1 and TEM-1 use exact uniform-selection expectations; GFP uses fixed random repeats; AAV uses same-pool random draws; T-cell uses shared nested prefixes. All answer equal-budget random selection, but Monte Carlo variance handling differs.

**Limited effective samples.** T-cell and GFP use 16 pools; GB1 uses 55 position-pair pools; TEM-1 uses 52 test positions; indel and stability use 48 and 27 UTRs. Cluster intervals describe within-library uncertainty, not cross-study generalization.

**No clinical claim.** AAV production is not tropism, infectivity, transgene expression, or clinical efficacy. TEM-1 relative resistance is not direct enzyme activity or patient response. GFP brightness is not pure expression. GB1 lnW is not Kd.

## 5. Methods

### 5.1 Functional regret and equal-budget controls

For candidate pool \(C\), measured endpoint \(Y\), and selector \(\pi\), regret is \(R^\pi=\max_{x\in C}Y(x)-Y(x^\pi)\). BoN selects the highest proxy score among \(N\) proposals. RandomN selects uniformly from the same candidate budget. The primary paired effect is \(R^{\mathrm{BoN}}-R^{\mathrm{RandomN}}\); negative values favor proxy ranking. Cap1 is a structural identity when both methods see the same first candidate.

### 5.2 T-cell data and adaptation

T-cell analyses used Castillo-Hair defined-end repeats and Sample-trained frozen CNNs. Strict-H1 pools excluded exact and Hamming-1 conflicts. Equal-budget RandomN used PCG64 position draws from the same nested prefix. Target adaptation used 50,000 train, 10,000 development, and 1,024 test sequences with component-disjoint isolation; ListNet used a new component-disjoint test excluding all prior roles. Bootstrap units were repeat-linked pools.

### 5.3 GB1, GFP, AAV, and TEM-1

GB1 used ProteinGym Olson lnW with position-blocked train/dev/test and an additive substitution ridge. GFP used Sarkisyan brightness with 1–2 mutation training and ≥5 mutation permanent pools. AAV9 used Fit4Function production with exact/Hamming-1 cross-library exclusion and three CUDA-trained CNN seeds. TEM-1 used Stiffler MaveDB relative resistance across antibiotic conditions, with position-cluster bootstrap and no model fitting.

### 5.4 mRNA extension

HEK293T/HepG2 MRL reused frozen historical equal-budget results. Castillo indel used 48 UTRs and four cell–target contexts; stability used 27 experimental sequences with spike-in calibration. Leppek and Schlusser were integrated as historical results without re-unblinding. Predicted fields were never substituted for measured endpoints.

### 5.5 Statistics

Each scenario used its prespecified cluster unit: pools, position-pair pools, positions, or UTRs. Bootstrap replicates were 10,000 where reported. T-cell used complete all-20 Holm families including cap1 zeros. TEM-1 direction intervals were diagnostic and not multiplicity-corrected. Cross-scenario synthesis compares direction and evidence status, not raw effect sizes across units.

## 6. Figure captions

**Figure 14 | Equal-budget T-cell proxy ranking value.** BoN(N)−RandomN(N) measured-MRL regret across caps and repeats, with complete all-20 Holm status. Source: `deliverables/figures/v3/Figure14_v3_equal_budget_value.pdf`.

**Figure 16 | Useful global ranking and top-tail bottleneck in T cells.** Pointwise error, global Spearman, equal-budget benefit, and top16 regret decomposition. Source: `deliverables/figures/v4/Figure16_v4_tcell_global_value_tail_bottleneck.pdf`.

**Figure 17 | Regression–decision decoupling in T-cell adaptation.** MAE reduction without cap16 regret improvement, ListNet gate failure, and HepG2 anchor reversal. Source: `deliverables/figures/v4/Figure17_v4_tcell_regression_decision_decoupling.pdf`.

**Figure 18 | Cross-scenario functional value and boundaries.** Descriptive within-task relative regret reduction for positive tasks and null/harmful boundaries; original units and CIs remain task-specific. Source: `deliverables/figures/v5/Figure18_v5_cross_scenario_functional_value.pdf`.

**Figure 19 | Regression metrics do not substitute for decisions.** T-cell MAE–regret dissociation, GFP order-dependent correlation decay, and AAV high global correlation with low extreme-best recall. Source: `deliverables/figures/v5/Figure19_v5_regression_decision_dissociation.pdf`.

## 7. Data and code availability

Analysis and build code are available at <https://github.com/Cunyu-Liu/proxy-function-gap/tree/paper23-gates-v1>. Frozen evidence bundles and scenario reports are listed in the local delivery index. Raw public datasets remain subject to their original licenses; no new clinical or wet-lab data were generated.

## References

[1] Sample PJ, Wang B, Reid DW, et al. Human 5′ UTR design and variant effect prediction from a massively parallel translation assay. *Nature Biotechnology*. 2019;37:803–809. <https://doi.org/10.1038/s41587-019-0164-5>

[2] Castillo-Hair S, Fedak S, Wang B, et al. Optimizing 5′ UTRs for mRNA-delivered gene editing using deep learning. *Nature Communications*. 2024;15:5284. <https://doi.org/10.1038/s41467-024-49508-2>

[3] Leppek K, Byeon GW, Kladwang W, et al. Combinatorial optimization of mRNA structure, stability, and translation for RNA-based therapeutics. *Nature Communications*. 2022;13:1536. <https://doi.org/10.1038/s41467-022-28776-w>

[4] Schlusser N, González A, Pandey M, Zavolan M. Current limitations in predicting mRNA translation with deep learning models. *Genome Biology*. 2024;25:227. <https://doi.org/10.1186/s13059-024-03369-6>

[5] Olson CA, Wu NC, Sun R. A comprehensive biophysical description of pairwise epistasis throughout an entire protein domain. *Current Biology*. 2014;24:2643–2651. <https://doi.org/10.1016/j.cub.2014.09.072>

[6] Sarkisyan KS, Bolotin DA, Meer MV, et al. Local fitness landscape of the green fluorescent protein. *eLife*. 2016;5:e14128. <https://doi.org/10.7554/eLife.14128>

[7] Bryant DH, Bashir A, Sinai S, et al. Deep diversification of an AAV capsid protein by machine learning. *Nature Communications*. 2021;12:3328. <https://doi.org/10.1038/s41467-021-23690-1>

[8] Stiffler MA, Hekstra DR, Ranganathan R. Evolvability as a function of purifying selection in TEM-1 β-lactamase. *Cell*. 2015;160:882–892. <https://doi.org/10.1016/j.cell.2015.01.035>

[9] Notin P, Dias M, Frazer J, et al. ProteinGym: Large-scale benchmarks for protein fitness prediction and design. *Advances in Neural Information Processing Systems*. 2023;36.

[10] Trabucco B, Kumar A, Geng X, Levine S. Conservative objective models for effective offline model-based optimization. *International Conference on Machine Learning*. 2021.

[11] Trabucco B, Geng X, Kumar A, Levine S. Design-Bench: Benchmarks for data-driven offline model-based optimization. *International Conference on Machine Learning*. 2022.

[12] Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems*. 2017;30.

[13] Reddy AJ, Geng X, Herschl MH, et al. Designing cell-type-specific promoter sequences using conservative model-based optimization. *Advances in Neural Information Processing Systems*. 2024;37.

[14] Chen J, Yoon J, Ebrahimi S, Arik S, Jha S, Pfister T. ASPEST: Bridging the gap between active learning and selective prediction. arXiv:2304.03870. 2023. <https://arxiv.org/abs/2304.03870>

[15] Fadeeva E, Li X, Sachan M, Panov M. Verifier-guided self-consistency: Test-time reasoning with imperfect generators and verifiers. arXiv:2611.06606. 2026. <https://arxiv.org/abs/2611.06606>