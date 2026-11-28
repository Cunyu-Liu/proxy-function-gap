---
title: "Decision value of sequence proxies tracks high-score tail reliability across biological design tasks"
subtitle: "Equal-budget functional regret reveals when global prediction quality is insufficient for molecular selection"
author: "Anonymous authors"
date: "2026-11-28"
lang: en
---

> **Draft status.** Tail-reliability integration, version 8. All analyses are retrospective historical-data studies. Grouped holdouts and frozen historical evaluations are not prospective blind tests, independent wet-lab replications, or clinical validation. This version unifies T-cell, GB1, GFP, AAV9 production, TEM-1, and non-T-cell mRNA scenarios around a common decision-focused question.

## Abstract

Computational proxies are used to select biological sequences, yet average prediction quality need not imply useful selection. We asked when proxy ranking improves measured functional decisions across molecular design tasks. Using fixed candidate pools and equal-budget best-of-N (BoN) versus RandomN controls, we evaluated functional regret in T-cell 5′ UTR translation, GB1 binding, GFP brightness, AAV9 production, TEM-1 resistance, and mRNA stability/editing contexts. A frozen CNN reduced T-cell MRL regret at equal budget, but tail decomposition showed that remaining loss concentrated within the high-score tail; target adaptation reduced MAE without improving cap16 regret. Across tasks, decision benefit tracked tail reliability rather than global accuracy alone: GB1 lnW (cap16 BoN−RandomN −0.160894, 95% CI [−0.282853, −0.050515]), GFP brightness (additive −0.2800 [−0.3095, −0.2521]), and AAV9 production (−4.206431 [−4.564111, −3.931098]) showed positive equal-budget value, while TEM-1 cross-drug transfer, mRNA stability, gene-editing indel, and endogenous TE showed null or harmful effects. A cross-scenario tail audit showed that positive tasks retained useful tail enrichment, whereas failures had weak, negative, or endpoint-mismatched tail ordering. These results support a decision-focused acceptance standard: sequence proxies should be judged by measured functional regret under equal-budget controls, with MAE and global correlation retained as diagnostics rather than substitutes.

**Keywords:** sequence design; functional regret; high-score tail; best-of-N; decision-focused evaluation; proxy reliability; mRNA; protein; AAV

## 1. Introduction

A predictive model for biological sequence design is usually trained to minimize average error, but deployed to choose one or a few candidates. This creates a mismatch between the training objective and the decision objective. A model can be inaccurate on average yet still rank useful candidates above random; conversely, it can improve MAE or global correlation while leaving the selected candidate unchanged or worse. The relevant question is therefore not only whether a proxy predicts an assay, but whether it improves the measured functional decision under a fair candidate budget.

This question is shared across therapeutic and engineering applications. mRNA design requires translation, stability, and editing outcomes that are not interchangeable. Protein engineering uses DMS fitness as a proxy for binding, brightness, or resistance. AAV capsid design uses production or packaging enrichments that do not directly measure clinical efficacy. Although the molecules and assays differ, the decision structure is the same: a proxy ranks a fixed pool, and the experimenter submits a small number of candidates.

We organized the study around functional regret. For a pool \(C\) with measured endpoint \(Y\), a selector \(\pi\) chooses \(x^\pi\), and regret is \(R^\pi=\max_{x\in C}Y(x)-Y(x^\pi)\). BoN selects the highest proxy score among \(N\) proposals; RandomN selects uniformly from the same budget. The paired contrast isolates proxy ranking value from candidate opportunity.

The paper has three connected aims. First, establish a concrete positive decision result and localize its remaining bottleneck in T-cell MRL selection. Second, test whether the same decision-focused pattern extends to other high-value tasks. Third, identify a unifying diagnostic: whether the high-score tail preserves functional ordering. The resulting claim is not that proxies are universally useful, but that their decision value is task-dependent and must be accepted through measured regret rather than regression metrics alone.

## 2. Results

### 2.1 T-cell MRL: equal-budget value coexists with high pointwise error

In the strict-H1 T-cell reconstruction, BoN and RandomN used identical nested proposal prefixes. CNN ranking reduced measured-MRL regret at every cap 2–16 in both repeats (Fig. 14). Repeat 1 cap16 effect was −1.705760 MRL [−2.163445, −1.211914], and all four nontrivial caps survived the complete all-20 Holm family. Repeat 2 was favorable at all four caps in point estimate and nominal interval, but only cap 4 survived the full family (−1.014876 [−1.544490, −0.465832]).

This decision benefit coexisted with high pointwise error: MAE 1.567/1.544 MRL and global Spearman 0.591/0.529. Within the predicted top 16, Spearman fell to 0.206/0.144, and the predicted winner was not the measured tail optimum in 15/16 pools per repeat. Within-tail ranking loss accounted for 2.480 of 3.316 MRL regret in repeat 1 and 2.117 of 4.071 in repeat 2 (Fig. 16). The remaining bottleneck was therefore high-score tail ordering, not absence of global ranking information.

### 2.2 T-cell adaptation: regression improvement did not repair the decision

A 50,000-label grouped holdout fine-tuned the frozen CNN with MSE plus a dynamic top-16 pairwise term. MAE decreased from 1.352159 to 0.914506 in repeat 1 and from 1.062273 to 1.018716 in repeat 2, yet cap16 regret increased by +0.038929 and +0.012551 MRL. The combined effect was +0.025740 [−0.072019, +0.140994], and all ten repeat–cap Holm-adjusted p values were 1. The MSE-only arm made the same cap selections as the joint model.

A full-64 ListNet objective on a new component-disjoint test gave only small, gate-failing point improvements: −0.037415 [−0.183204, +0.102494] versus original CNN and −0.007585 [−0.028698, +0.012474] versus the 50k model (Fig. 17). A same-sequence HepG2 anchor also failed: despite comparable or slightly higher global correlation, direct HepG2 ranking increased combined cap16 T-cell regret by +0.428897 [+0.245460, +0.616474]. These experiments show why average error and cross-cell global correlation cannot substitute for target-tail decision quality.

### 2.3 Cross-scenario tail audit: decision benefit tracks tail reliability

To connect the scenarios mechanistically, we assembled a unified tail audit across 11 task/context rows (Fig. 20). The audit compares global Spearman, a task-specific tail reliability metric, and equal-budget relative regret reduction. It does not pool raw effect sizes across endpoint units.

Positive decision tasks retained useful tail information. GB1 had moderate global Spearman (0.348) but true top-decile recall 0.142 and cap16 relative regret reduction 30.1%. GFP additive ranking had global Spearman 0.460, true top5% lift 10.47 at B16, and relative regret reduction 88.3%. AAV9 production had high global Spearman (0.821) but only 18.75% true top1% recall; nevertheless, average cap16 decision benefit was large (65.5% relative reduction). T-cell repeats showed moderate global correlation but substantial equal-budget benefit, with remaining loss localized to the tail.

Boundary tasks showed weak or negative tail reliability. TEM-1 Amp→Amp had median global Spearman 0.312 but source top10% Spearman 0.132; Amp→cefotaxime had near-zero global and tail correlation and no resolved selection benefit. mRNA stability had strongly negative global Spearman (−0.766) and zero top-decile recall, with significantly harmful selection. Castillo indel contexts had low top-decile recall and mixed/null corrected effects. HEK293T high-cap MRL and endogenous TE likewise lacked robust benefit.

The unifying pattern is therefore not “higher global accuracy always helps.” It is: **proxy ranking helps when the high-score tail retains enough functional ordering to improve the selected candidate; it fails when tail ordering collapses or the endpoint contract changes.** This is a diagnostic association across tasks, not a causal intervention proof.

### 2.4 GB1 binding: simple additive proxy improves measured lnW regret

GB1 used Olson IgG-Fc binding lnW, not Kd. A one-hot additive ridge with shared substitution effects was trained under position-blocked isolation. On 55 permanent position-pair pools, cap16 BoN−RandomN regret was −0.160894 [−0.282853, −0.050515] (Fig. 18). All budgets favored BoN, but 31/55 pools improved and 24/55 worsened at cap16. MAE was 2.159 and global Spearman 0.348, reinforcing that moderate pointwise quality can coexist with useful selection.

### 2.5 GFP brightness: low-order supervision supports high-order selection

GFP models trained on 1–2 mutations were evaluated on ≥5-mutation permanent pools. CNN cap16 BoN−RandomN was −0.260830 [−0.301866, −0.218031], while additive ridge achieved −0.2800 [−0.3095, −0.2521] and lower BoN regret at every budget (Fig. 18). Complexity was not necessary. CNN Spearman decayed from 0.582 at order 5 to −0.020 at order 9 (Fig. 19), yet fixed-pool selection remained beneficial. High-order global correlation therefore cannot replace the decision endpoint.

### 2.6 AAV9 production: strong average benefit without extreme-best recovery

AAV9 Fit4Function production used strict cross-library exact/Hamming-1 exclusion. Cap16 BoN−RandomN regret was −4.206431 [−4.564111, −3.931098] on 3,996 disjoint pools (Fig. 18). Production is nuclease-resistant viral-genome enrichment, not tropism, infectivity, transgene expression, or clinical efficacy. Global Spearman was 0.821, but true top1% recall was only 18.75% (Fig. 19). Average decision improvement and extreme-best recovery are distinct claims.

### 2.7 TEM-1 and mRNA boundaries: endpoint and condition contracts matter

TEM-1 Amp→Amp transfers were often favorable, but Amp→cefotaxime intervals all crossed zero (Fig. 18). The endpoint is experimentally derived relative resistance, not direct enzyme kinetics or clinical response. In mRNA, HEK293T high-cap MRL lacked sustained benefit; HepG2 cap16 was nominal but not Holm-robust; indel had 0/4 corrected-positive contexts; stability was significantly worse (+0.2838 [+0.0824, +0.4435]). Historical Leppek MRL and Schlusser TE also showed no robust benefit. These boundaries prevent a universal proxy-validity claim.

## 3. Discussion

The study supports a decision-focused standard for sequence proxy evaluation. The same principle appears across mRNA, protein, and AAV tasks: measured functional regret under equal-budget controls reveals value that MAE and global correlation can miss, and also reveals failures that average metrics can hide. The T-cell deep dive explains the mechanism: global ranking can be useful while the high-score tail remains unreliable; improving average error does not necessarily repair that tail.

The cross-scenario tail audit gives the paper a unifying thread. Positive tasks are not simply those with the highest global correlation; they are tasks where the proxy tail retains enough functional ordering to improve selection. Boundary tasks show weak, negative, or endpoint-mismatched tail ordering. This is a diagnostic pattern, not proof that tail reliability causally determines benefit in every task. Future method work should therefore target tail decision quality directly and be accepted only when measured regret improves on a new isolated evaluation.

Simple baselines are essential. GFP additive outperformed CNN, and GB1 additive ridge was sufficient. Complex models are not automatically valuable. Endpoint semantics are equally essential: lnW is not Kd, production is not tropism, brightness is not pure expression, and relative resistance is not direct enzyme kinetics.

The work does not establish prospective clinical or wet-lab efficacy. All data are historical, and grouped holdouts are not independent studies. The contribution is an evaluation principle and a cross-task evidence map, not a universal design algorithm.

## 4. Limitations

All analyses are retrospective. Endpoint units differ, and no cross-task pooled effect size is computed. RandomN implementations differ across tasks. Cluster intervals describe within-library uncertainty. T-cell and GFP use 16 pools; GB1 uses 55 position-pair pools; TEM-1 uses 52 test positions; indel/stability use 48/27 UTRs. AAV production is not clinical efficacy. The tail audit is associational, not a causal intervention. Author, affiliation, license, and target repository metadata remain pre-submission placeholders.

## 5. Methods

Functional regret and equal-budget controls follow Section 1. T-cell used Castillo-Hair defined-end repeats, Sample-trained frozen CNNs, strict-H1 isolation, PCG64 RandomN, and repeat-linked pool bootstrap. GB1 used ProteinGym Olson lnW with position-blocked additive ridge. GFP used Sarkisyan brightness with low-order training and high-order permanent pools. AAV9 used Fit4Function production with exact/H1 exclusion and three CUDA CNN seeds. TEM-1 used Stiffler MaveDB relative resistance with position-cluster bootstrap. mRNA extension reused frozen historical MRL results and computed indel/stability regret with endpoint-specific units. The cross-scenario tail audit assembled task-specific global and tail metrics from frozen artifacts without pooling raw effects.

## 6. Figure captions

**Figure 14 | Equal-budget T-cell proxy ranking value.** Source: `deliverables/figures/v3/Figure14_v3_equal_budget_value.pdf`.

**Figure 16 | Useful global ranking and top-tail bottleneck in T cells.** Source: `deliverables/figures/v4/Figure16_v4_tcell_global_value_tail_bottleneck.pdf`.

**Figure 17 | Regression–decision decoupling in T-cell adaptation.** Source: `deliverables/figures/v4/Figure17_v4_tcell_regression_decision_decoupling.pdf`.

**Figure 18 | Cross-scenario functional value and boundaries.** Source: `deliverables/figures/v5/Figure18_v5_cross_scenario_functional_value.pdf`.

**Figure 19 | Regression metrics do not substitute for decisions.** Source: `deliverables/figures/v5/Figure19_v5_regression_decision_dissociation.pdf`.

**Figure 20 | Decision value map across tasks.** Global Spearman versus descriptive relative regret reduction, colored by task family and marked by evidence status. Source: `deliverables/figures/v6/Figure20_v6_decision_value_map.pdf`.

**Figure 21 | Unified decision-focused protocol and governance.** Shared decision structure, endpoint-specific semantics, leakage controls, and evidence boundaries. Source: `deliverables/figures/v6/Figure21_v6_unified_protocol_governance.pdf`.

## 7. Data and code availability

Analysis and build code are available at <https://github.com/Cunyu-Liu/proxy-function-gap/tree/paper23-gates-v1>. Frozen evidence bundles and scenario reports are listed in the local delivery index. Raw public datasets remain subject to their original licenses.

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