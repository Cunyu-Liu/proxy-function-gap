---
title: "Decision value of sequence proxies tracks high-score tail reliability across biological design tasks"
subtitle: "Equal-budget functional regret reveals when global prediction quality is insufficient for molecular selection"
author: "Anonymous authors"
date: "2026-11-28"
lang: en
---

> **Draft status.** Tail-reliability integration, version 10. All analyses are retrospective historical-data studies. Grouped holdouts and frozen historical evaluations are not prospective blind tests, independent wet-lab replications, or clinical validation. This version reorganizes the results into a connected question-to-evidence chain, adds a pre-registered cross-scenario test of a unified tail-reliability metric, redesigns the main figure set, and records an independent-data qualification audit (Cao 2021 endogenous translation efficiency; Ogden 2019 AAV2 tropism) that was access-barred and therefore not evaluated. `DRAFT_NOT_SUBMITTED`.

## Abstract

Computational proxies are used to select biological sequences, yet average prediction quality need not imply useful selection. We asked when proxy ranking improves measured functional decisions across molecular design tasks, and what single diagnostic explains the pattern. Using fixed candidate pools and equal-budget best-of-N (BoN) versus RandomN controls, we evaluated functional regret in T-cell 5′ UTR translation, GB1 binding, GFP brightness, AAV9 production, TEM-1 resistance, and mRNA stability/editing contexts. A frozen CNN reduced T-cell MRL regret at equal budget, but tail decomposition localized the remaining loss to ordering within the high-score tail; neither regression-side repair (target adaptation, ListNet) nor a pre-registered selection-side intervention (UPTS) repaired that loss, the latter failing its auxiliary gate before any target evaluation. Across tasks, a unified top-decile tail-recall metric was strongly associated with equal-budget decision benefit (Spearman ρ = 0.88, scenario-permutation p = 0.0008, leave-one-out ρ 0.83–0.95), numerically exceeding global correlation (ρ = 0.77); the difference interval touched zero, so superiority over global correlation is suggestive rather than confirmed. Positive tasks (GB1 lnW, GFP brightness, AAV9 production) retained useful tail enrichment, whereas TEM-1 cross-drug transfer, mRNA stability, gene-editing indel, and endogenous TE showed null or harmful effects with weak or negative tail reliability. These results support a decision-focused acceptance standard: sequence proxies should be judged by measured functional regret under equal-budget controls, with tail reliability as a cross-task diagnostic and MAE/global correlation retained as diagnostics rather than substitutes.

**Keywords:** sequence design; functional regret; high-score tail; best-of-N; decision-focused evaluation; proxy reliability; mRNA; protein; AAV

## 1. Introduction

A predictive model for biological sequence design is usually trained to minimize average error, but deployed to choose one or a few candidates. This creates a mismatch between the training objective and the decision objective: a model can be inaccurate on average yet still rank useful candidates above random, and it can improve MAE or global correlation while leaving the selected candidate unchanged or worse. The relevant question is therefore not whether a proxy predicts an assay, but whether it improves the measured functional decision under a fair candidate budget.

This question is shared across therapeutic and engineering applications. mRNA design requires translation, stability, and editing outcomes that are not interchangeable; protein engineering uses DMS fitness as a proxy for binding, brightness, or resistance; AAV capsid design uses production enrichments that do not directly measure clinical efficacy. Although the molecules and assays differ, the decision structure is identical: a proxy ranks a fixed pool, and the experimenter submits a small number of candidates.

We organize the study around functional regret. For a pool \(C\) with measured endpoint \(Y\), a selector \(\pi\) chooses \(x^\pi\), and regret is \(R^\pi=\max_{x\in C}Y(x)-Y(x^\pi)\). BoN selects the highest proxy score among \(N\) proposals; RandomN selects uniformly from the same budget. The paired contrast isolates proxy ranking value from candidate opportunity.

The paper follows a connected chain of questions, each answered by a specific experiment and motivating the next. (1) Does proxy ranking improve measured decisions at equal budget? (2) Where does the remaining loss live? (3) Does repairing regression repair the decision? (4) Does repairing the selection rule repair the decision? (5) Across tasks, when does proxy ranking have value? (6) What single diagnostic explains value and failure? (7) Does the pattern extend to an independent study? The resulting claim is not that proxies are universally useful, but that their decision value is task-dependent, tracks high-score tail reliability, and must be accepted through measured regret rather than regression metrics alone.

## 2. Results

### 2.1 Does proxy ranking improve measured decisions at equal budget?

In the strict-H1 T-cell reconstruction, BoN and RandomN used identical nested proposal prefixes. CNN ranking reduced measured-MRL regret at every cap 2–16 in both repeats (Fig. 1). Repeat 1 cap16 effect was −1.705760 MRL [−2.163445, −1.211914], and all four nontrivial caps survived the complete all-20 Holm family. Repeat 2 was favorable at all four caps in point estimate and nominal interval, but only cap 4 survived the full family (−1.014876 [−1.544490, −0.465832]); cap16 (−1.086941 [−1.833685, −0.318589]) is reported as nominal, not upgraded.

This decision benefit coexisted with high pointwise error: MAE 1.567/1.544 MRL and global Spearman 0.591/0.529. A useful global ranking signal therefore exists, but average error remains large. Where, then, does the remaining selection loss come from?

### 2.2 Where does the remaining loss live?

Within the predicted top 16, Spearman fell to 0.206/0.144, and the predicted winner was not the measured tail optimum in 15/16 pools per repeat. Decomposing cap16 regret into opportunity loss (the true tail optimum lying outside the proposal prefix) and within-tail ranking loss (choosing wrongly inside the prefix) showed that within-tail ranking accounted for 2.480 of 3.316 MRL regret in repeat 1 and 2.117 of 4.071 in repeat 2 (Fig. 2). The bottleneck was therefore ordering inside the high-score tail, not absence of global ranking information and not candidate opportunity alone. If the loss is inside the tail, can improving the proxy's regression quality repair it?

### 2.3 Does repairing regression repair the decision?

A 50,000-label grouped holdout fine-tuned the frozen CNN with MSE plus a dynamic top-16 pairwise term. MAE decreased from 1.352159 to 0.914506 in repeat 1 and from 1.062273 to 1.018716 in repeat 2, yet cap16 regret increased by +0.038929 and +0.012551 MRL; the combined effect was +0.025740 [−0.072019, +0.140994], and all ten repeat–cap Holm-adjusted p values were 1 (Fig. 3). The MSE-only arm made the same cap selections as the joint model. A full-64 ListNet objective on a new component-disjoint test gave only small, gate-failing point improvements: −0.037415 [−0.183204, +0.102494] versus original CNN and −0.007585 [−0.028698, +0.012474] versus the 50k model. A same-sequence HepG2 anchor also failed: despite comparable global correlation, direct HepG2 ranking increased combined cap16 T-cell regret by +0.428897 [+0.245460, +0.616474]. Regression-side and cross-cell anchor repairs thus did not repair the tail decision. If regression cannot repair it, can the selection rule itself, using the proxy's own uncertainty, repair it?

### 2.4 Does repairing the selection rule repair the decision?

We pre-registered a selection-side intervention (UPTS v1) that left the pool, the proxy scores, and the proposal prefix unchanged and only altered the within-prefix choice rule: select the first occurrence of the maximum of \(s_\kappa = \text{proxy\_mean} - \kappa\cdot\text{proxy\_std}\), where proxy_std is the three-member ensemble sample standard deviation. Because the prefix is fixed, opportunity loss is unchanged by construction; because scores are frozen, global MAE and Spearman are unchanged by construction. Any effect is therefore purely within-tail ranking — a clean intervention on the diagnosed bottleneck.

The penalty κ was selected on auxiliary dev only (N50↔HepG2 leave-one-context-out), with a pre-registered dual-direction gate requiring cap16 regret improvement on both auxiliary tests before any T-cell evaluation. The κ=0 rule reproduced the frozen BoN baseline exactly. The gate failed: N50→HepG2 improved by +0.012174 MRL (κ=0.5), but HepG2→N50 worsened by −0.029558 MRL (κ=4), so the experiment stopped without reading T-cell labels (Fig. 4). A pre-registered diagnostic explains the failure: within the auxiliary top 16, ensemble disagreement carried almost no overestimation signal (Spearman of proxy_std with signed error 0.0036/0.0092; with absolute error 0.0047/0.0172). Members share training data and architecture, so their errors are highly correlated and disagreement reflects numerical noise rather than functional overestimation.

This is the eighth pre-registered repair direction falsified at or before its gate, and the first selection-side one. Together with the training-side failures, it indicates that the residual within-tail loss is not repairable by reweighting the frozen proxy's internal uncertainty or by auxiliary-shared corrections; it would require information outside the current proxy. We retain this as a mechanism boundary, not a method contribution. The T-cell deep dive thus establishes a positive decision result, localizes its bottleneck, and shows that neither regression nor selection reweighting repairs it. Does this decision-focused pattern generalize beyond T cells?

### 2.5 Across tasks, when does proxy ranking have value?

We applied the same equal-budget regret protocol to five additional task families, keeping endpoint-specific units and cluster bootstrap (Fig. 5).

**GB1 binding.** Using Olson IgG-Fc binding lnW (not Kd), a one-hot additive ridge with shared substitution effects was trained under position-blocked isolation. On 55 permanent position-pair pools, cap16 BoN−RandomN regret was −0.160894 [−0.282853, −0.050515]. All budgets favored BoN, but 31/55 pools improved and 24/55 worsened at cap16; MAE was 2.159 and global Spearman 0.348. Moderate pointwise quality coexisted with useful selection.

**GFP brightness.** Models trained on 1–2 mutations were evaluated on ≥5-mutation permanent pools. CNN cap16 BoN−RandomN was −0.260830 [−0.301866, −0.218031], while additive ridge achieved −0.2800 [−0.3095, −0.2521] and lower BoN regret at every budget. Complexity was not necessary. CNN Spearman decayed from 0.582 at order 5 to −0.020 at order 9, yet fixed-pool selection remained beneficial: high-order global correlation cannot replace the decision endpoint.

**AAV9 production.** Fit4Function production (nuclease-resistant viral-genome enrichment, not tropism, infectivity, transgene expression, or clinical efficacy) used strict cross-library exact/Hamming-1 exclusion. Cap16 BoN−RandomN regret was −4.206431 [−4.564111, −3.931098] on 3,996 disjoint pools. Global Spearman was 0.821, but true top1% recall was only 18.75%: average decision improvement and extreme-best recovery are distinct claims.

**TEM-1 and mRNA boundaries.** TEM-1 Amp→Amp transfers were often favorable, but Amp→cefotaxime intervals all crossed zero (15/15); the endpoint is experimentally derived relative resistance, not direct enzyme kinetics or clinical response. In mRNA, HEK293T high-cap MRL lacked sustained benefit; HepG2 cap16 was nominal but not Holm-robust (p=0.303970); indel had 0/4 corrected-positive contexts; stability was significantly worse (+0.2838 [+0.0824, +0.4435]). Historical Leppek MRL and Schlusser TE also showed no robust benefit. These boundaries prevent a universal proxy-validity claim. What single diagnostic separates the positive tasks from the failures?

### 2.6 What single diagnostic explains value and failure?

The scenarios above used heterogeneous tail metrics. To test the unifying hypothesis quantitatively, we recomputed a single tail-reliability definition on every scenario's frozen evaluation population: the top-decile recall \(|\text{predicted top-}k \cap \text{true top-}k|/k\), with \(k=\max(3,\lceil 0.1N\rceil)\), alongside within-tail Spearman and global Spearman (Fig. 6). The computation reproduced the known AAV top1% recall (0.1875) exactly, anchoring correctness.

A pre-registered association test (10 scenarios with a defined relative regret reduction; Castillo indel excluded) found that unified tail recall was strongly associated with equal-budget decision benefit: Spearman ρ = 0.8788, scenario-permutation p = 0.00076, and leave-one-out ρ ranging 0.8333–0.9500. Global Spearman was also associated (ρ = 0.7697, p = 0.00627) but numerically weaker; the bootstrap interval for the difference ρ_tail − ρ_global was [0.0000, 0.4616], touching zero, so the pre-registered superiority rule was not met and the comparative claim is retained as descriptive rather than confirmed. Secondary within-tail Spearman predictors were weaker (predicted-tail ρ = 0.4286, true-tail ρ = 0.5714, n = 7), while top1% recall remained strong (ρ = 0.8524): for fixed-pool selection, whether the proxy tail *contains* the true elite tracks benefit more than fine ordering within the tail.

The unifying pattern is therefore: **proxy ranking helps when the high-score tail retains enough functional enrichment to improve the selected candidate; it fails when tail enrichment collapses or the endpoint contract changes.** This is a statistically supported cross-task association, not a causal intervention proof, and the small scenario count limits power. Positive tasks are not simply those with the highest global correlation; boundary tasks show weak, negative, or endpoint-mismatched tail reliability. Does this pattern extend beyond the training study family?

### 2.7 Does the pattern extend to an independent study?

All results above are retrospective analyses of historical public data, largely within the Sample/Castillo training family. We therefore qualified two independent-study candidates. Cao 2021 (zzz2010/5UTR_Optimizer) provides endogenous 5′ UTR translation efficiency (TE, Ribo-seq/RNA-seq derived) rather than the synthetic-library protein production initially assumed; the endpoint is cross-endpoint relative to the MRL-trained proxy, sequences require a Gencode FASTA join, and the 14–16 MB files were not reliably downloadable at the available cluster bandwidth. Ogden 2019 (GSE139657) provides in vivo AAV2 liver tropism, but the NCBI supplementary download was access-barred. Neither candidate was evaluated; both are recorded as qualified future work with corrected endpoint semantics. The independent-study soft spot therefore remains an explicit limitation, not a resolved claim.

## 3. Discussion

The study supports a decision-focused standard for sequence proxy evaluation, now organized as a connected chain: a positive equal-budget result, a localized tail bottleneck, falsified regression-side and selection-side repairs, a cross-task evidence map, and a quantitative tail-reliability diagnostic. The same principle appears across mRNA, protein, and AAV tasks: measured functional regret under equal-budget controls reveals value that MAE and global correlation can miss, and also reveals failures that average metrics can hide.

The cross-scenario association test strengthens the unifying thread. A unified top-decile tail-recall metric is strongly associated with decision benefit (ρ = 0.88, permutation p = 0.0008, leave-one-out robust), numerically exceeding global correlation. Because the difference interval touches zero, we do not claim confirmed superiority over global correlation; the dissociation experiments (adaptation, ListNet, anchor) carry the stronger evidence that average error and cross-cell global correlation cannot substitute for target-tail decision quality. The UPTS falsification sharpens the mechanism: holding scores and prefixes fixed isolates within-tail ranking, and its failure shows the bottleneck is not an artifact of the selection rule applied to frozen scores. The accumulated gate failures argue that, within the current data assets, the strongest contribution is the evaluation principle and the cross-task evidence map rather than a deployable design algorithm.

Simple baselines are essential. GFP additive outperformed CNN, and GB1 additive ridge was sufficient. Endpoint semantics are equally essential: lnW is not Kd, production is not tropism, brightness is not pure expression, and relative resistance is not direct enzyme kinetics.

The work does not establish prospective clinical or wet-lab efficacy. All data are historical, and grouped holdouts are not independent studies. The independent-data audit identified and qualified candidate surfaces (Cao 2021 endogenous TE; Ogden 2019 AAV2 tropism) but evaluated none; they remain future work. The contribution is an evaluation principle, a quantitatively supported cross-task diagnostic, and an evidence map with explicit boundaries — not a universal design algorithm.

## 4. Limitations

All analyses are retrospective. Endpoint units differ, and no cross-task pooled raw effect size is computed; the cross-scenario association uses unit-free relative regret reduction. The association has only 10 scenario rows, limiting power; its intervals describe scenario-sampling uncertainty, not biological replication, and it is associational rather than causal. RandomN implementations differ across tasks. Cluster intervals describe within-library uncertainty. T-cell and GFP use 16 pools; GB1 uses 55 position-pair pools; TEM-1 uses 52 test positions; indel/stability use 48/27 UTRs. AAV production is not clinical efficacy. The UPTS result is retrospective auxiliary-gated method development, not an independent blind confirmation.

We did not compare against official COM or generative model-based optimization methods on an equivalent fixed-pool measured-regret surface. COM exists only as an honestly labeled independent adaptation; Design-Bench scores candidates with a learned oracle and its UTR task is the Sample data used here for training; GABO/aSCR constrains generative trajectories; T3PO-mRNA and mRNAutilus generate pool-external sequences; Reddy et al. design promoters with wet-lab validation. Claims are therefore limited to the fixed-pool equal-budget selection surface. No prospective or truly independent same-endpoint evaluation was performed; independent candidates were qualified but access-barred and not evaluated. A pure one-hot ridge for T-cell and a deployed CNN+MLP ensemble baseline were not separately run (the 4-mer MLP cross-architecture baseline and protein additive ridge cover the simple-baseline family). Author, affiliation, license, and target repository metadata remain pre-submission placeholders.

## 5. Methods

Functional regret and equal-budget controls follow Section 1. T-cell used Castillo-Hair defined-end repeats, Sample-trained frozen CNNs, strict-H1 isolation, PCG64 RandomN, and repeat-linked pool bootstrap. GB1 used ProteinGym Olson lnW with position-blocked additive ridge (λ=10.0 dev-selected, five seeds). GFP used Sarkisyan brightness with low-order training and high-order permanent pools (additive and CNN, dev-selected). AAV9 used Fit4Function production with exact/H1 exclusion and three CUDA CNN seeds (dev MSE early stopping, converged by frozen rule). TEM-1 used Stiffler MaveDB relative resistance with position-cluster bootstrap (source condition fitness as proxy, no fitting). mRNA extension reused frozen historical MRL results and computed indel/stability regret with endpoint-specific units. Hyperparameters were selected on train/dev only; each frozen test was evaluated once.

The unified tail audit (v2) recomputed top-decile recall and within-tail Spearman on each scenario's frozen evaluation population from per-candidate scores and labels (T-cell learner/labels/CNN scores; GB1/GFP/AAV prediction NPZ plus labels; mRNA stored frozen global top-decile metrics; TEM-1 raw scores under the frozen position split). The pre-registered association test used 200,000 scenario-level permutations (seed 20261128) and 10,000 scenario bootstrap replicates (seed 20261129); the protocol was hashed before computation. UPTS v1 was a CPU-only statistics experiment with 64 SHA256-frozen inputs, a pre-registered κ grid [0, 0.5, 1, 2, 4, 8, 16], auxiliary-dev selection, and a dual-direction gate. Three verification rounds covered hash/identity checks, independent pure-Python recomputation, and local↔remote delivery hash consistency for each new analysis. The independent-data audit recorded source URLs, endpoint semantics, and access barriers without evaluation.

## 6. Figure captions

**Figure 1 | Equal-budget T-cell proxy ranking value.** (a) cap16 functional regret under RandomN versus BoN for both repeats; (b) BoN−RandomN effect across caps 2–16 with 95% cluster intervals (filled markers survive all-20 Holm). Retrospective grouped holdout. Source: `deliverables/figures/v8/Figure1_v8_equal_budget_value.pdf`.

**Figure 2 | The high-score tail bottleneck.** (a) Global versus predicted-top-16 Spearman; (b) cap16 regret decomposed into opportunity loss and within-tail ranking loss (percent within-tail annotated). Source: `deliverables/figures/v8/Figure2_v8_tail_bottleneck.pdf`.

**Figure 3 | Regression–decision dissociation.** (a) MAE falls after 50k adaptation; (b) combined cap16 regret effects of adaptation, ListNet, and HepG2 anchor (positive = worse; anchor harmful). Source: `deliverables/figures/v8/Figure3_v8_regression_decision_dissociation.pdf`.

**Figure 4 | A pre-registered selection-side intervention fails its auxiliary gate.** (a) Auxiliary-dev κ selection curves; (b) dual-direction test gate (HepG2→N50 FAIL); (c) ensemble disagreement carries near-zero overestimation signal. T-cell target not evaluated. Source: `deliverables/figures/v8/Figure4_v8_upts_falsification.pdf`.

**Figure 5 | Cross-scenario functional value and boundaries.** Equal-budget cap16 BoN−RandomN regret faceted by endpoint unit (no cross-unit pooling); color/marker encode evidence status. TEM-1 rows lack cluster intervals. Source: `deliverables/figures/v8/Figure5_v8_cross_scenario_forest.pdf`.

**Figure 6 | Unified tail-reliability diagnostic.** (a) Global Spearman versus relative regret reduction; (b) unified top-decile tail recall versus relative regret reduction (ρ = 0.88, permutation p = 0.0008); (c) leave-one-scenario-out robustness. Associational, retrospective. Source: `deliverables/figures/v8/Figure6_v8_unified_tail_diagnostic.pdf`.

**ED Figure 1 | Label distributions across scenarios.** Per-scenario endpoint distribution (IQR box, median line, q5–q95 whisker, mean diamond; skew annotated), illustrating heterogeneity and extreme right skew (AAV production, TEM-1 cefotaxime). Source: `deliverables/figures/v8/ED_Figure1_v8_label_distributions.pdf`.

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

[15] Yao MS, Zeng Y, Bastani H, Gardner J, Gee JC, Bastani O. Generative adversarial model-based optimization via source critic regularization. *Advances in Neural Information Processing Systems*. 2024;37.

[16] Cao J, Novoa EM, Zhang Z, et al. High-throughput 5′ UTR engineering for enhanced protein production in non-viral gene therapies. *Nature Communications*. 2021;12:4138. <https://doi.org/10.1038/s41467-021-24436-7> (code repository data are endogenous TE; synthetic-library protein production requires GSE176581 reprocessing).

[17] Ogden PJ, Kelsic ED, Sinai S, Church GM. Comprehensive AAV capsid fitness landscape reveals a viral gene and enables machine-guided design. *Science*. 2019;366:1139–1143. <https://doi.org/10.1126/science.aaw6463>

[18] Full-length mRNA Design with Reward-Guided Masked Diffusion Model Fine-Tuning. OpenReview KTCInSlPpL (authors/year pending verification; access blocked at audit time).

[19] mRNAutilus: Multi-Objective-Guided Discrete Generation of mRNA with Optimized Therapeutic Properties. arXiv:2605.31296. 2026. <https://arxiv.org/abs/2605.31296>