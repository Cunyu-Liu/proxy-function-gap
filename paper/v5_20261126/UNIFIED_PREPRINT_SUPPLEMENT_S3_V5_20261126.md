---
title: "Supplementary Study S3: Representation-dependent historical-label repair headroom"
subtitle: "Development/calibration replay with simple and regret-oriented acquisition"
date: "2026-11-26"
version: "v5"
---

## Scope and evidence status

This supplement corresponds to the original Paper 3 research line and uses the latest V4 analytic acquisition results. All 48 zero-label/full-query gates and all six V2-frozen curve cases are retrospective development/calibration diagnostics. Permanent outer labels were not read and outer metrics were not computed. This is **not a new blind test**. The results identify representation-dependent repair headroom and descriptive acquisition behavior; they **do not establish label efficiency**, outer generalization, prospective biological improvement, or superiority of a complex acquisition function. S2's frozen calibration failure removes the current evidentiary basis for a validated sequential “detect risk, then repair” system; S3 therefore remains a standalone diagnostic supplement.

## S3.1 Question, hypotheses, and decision chain

The preceding results establish specific-context equal-budget ranking value but also composition-sensitive attenuation, shared proxy errors, endpoint mismatch, and no detectable transfer under the complete PERSIST MLP contract. Tested selectors and pool-level Q prediction do not solve the decision problem. This motivates a separately gated repair question rather than implying that active learning must work or inheriting validity from Paper 1:

- **H3.1 — repairability:** Can revealing historical labels change the posterior-mean top-1 choice enough to reduce measured-MRL regret?
- **H3.2 — representation dependence:** Does repair headroom differ across summary, sequence-composition, and CNN-hidden representations?
- **H3.3 — acquisition:** Can a regret-oriented rule outperform random, maximum-variance, diversity, EI, or top-proxy acquisition at equal label budgets?

H3.1 is supported only in the diagnostic full-query setting for a subset of representation–pool combinations. H3.2 is supported descriptively in the 48 calibration gates. H3.3 is not supported as a general advantage: simple EI or maximum variance is preferable at different budgets, and regret-analytic acquisition does not dominate. No hypothesis is promoted to outer-validated repair.

## S3.2 Common Bayesian replay model

For frozen representation \(\phi(x)\), the shared label model was

\[
y=\phi(x)^Tw+\epsilon,\qquad \epsilon\sim N(0,\sigma^2).
\]

All strategies used the same representation, posterior updates, deployment rule, and query budgets. Deployment always selected posterior-mean top-1. Feature/target scaling used development data only. Four-fold development-pool-blocked cross-validation selected prior variance 0.1 and noise variance 0.16 for `summary4`, `kmer4`, and `cnn64`.

Representations were:

- `summary4`: proxy mean, standard deviation, range, and training-support summary;
- `kmer4`: normalized 4-mer frequencies of the 50-nucleotide sequence;
- `cnn64`: mean and sample standard deviation of three frozen CNN members’ penultimate features, retaining development-nonconstant columns.

Development combined `dev_000`–`dev_031` (2,048 labeled unique examples). Query pools were `calibration_000`–`calibration_015`, each with 64 candidates. Full-query revelation is a diagnostic information upper setting, not a deployable or unbiased outer performance estimate.

## S3.3 All 48 zero-label/full-query gates

Headroom was defined as zero-label regret minus full-query regret. A strictly positive value passed the diagnostic gate. All 48 cases were evaluated and retained; no pool was added because it produced a favorable V4 result.

**Table S3.1 | Zero-label regret → full-query regret (headroom), in measured-MRL units.**

| Query pool | summary4 | kmer4 | cnn64 |
|---|---|---|---|
| calibration_000 | 4.739 → 4.739 (0.000) | 5.371 → 5.026 (**+0.345**) | 4.739 → 4.739 (0.000) |
| calibration_001 | 3.686 → 3.686 (0.000) | 3.686 → 0.000 (**+3.686**) | 1.288 → 1.288 (0.000) |
| calibration_002 | 4.903 → 4.903 (0.000) | 5.522 → 4.903 (**+0.619**) | 4.429 → 4.429 (0.000) |
| calibration_003 | 1.533 → 1.533 (0.000) | 0.677 → 0.677 (0.000) | 6.005 → 0.070 (**+5.935**) |
| calibration_004 | 3.074 → 3.074 (0.000) | 0.000 → 0.000 (0.000) | 1.655 → 1.655 (0.000) |
| calibration_005 | 1.207 → 1.207 (0.000) | 1.969 → 1.969 (0.000) | 2.226 → 2.226 (0.000) |
| calibration_006 | 3.024 → 3.024 (0.000) | 4.544 → 4.544 (0.000) | 2.234 → 2.234 (0.000) |
| calibration_007 | 2.840 → 2.840 (0.000) | 0.000 → 0.000 (0.000) | 2.840 → 2.840 (0.000) |
| calibration_008 | 1.652 → 1.652 (0.000) | 1.699 → 1.699 (0.000) | 2.179 → 2.179 (0.000) |
| calibration_009 | 3.373 → 3.373 (0.000) | 2.757 → 2.757 (0.000) | 2.825 → 2.825 (0.000) |
| calibration_010 | 1.285 → 1.285 (0.000) | 2.290 → 2.290 (0.000) | 1.285 → 1.285 (0.000) |
| calibration_011 | 1.667 → 1.667 (0.000) | 4.944 → 4.944 (0.000) | 4.089 → 4.089 (0.000) |
| calibration_012 | 5.332 → 5.332 (0.000) | 6.105 → 6.105 (0.000) | 3.845 → 3.845 (0.000) |
| calibration_013 | 2.404 → 2.404 (0.000) | 1.365 → 0.729 (**+0.636**) | 2.803 → 2.803 (0.000) |
| calibration_014 | 0.000 → 0.000 (0.000) | 2.589 → 0.000 (**+2.589**) | 2.262 → 2.262 (0.000) |
| calibration_015 | 1.271 → 1.271 (0.000) | 2.837 → 2.837 (0.000) | 3.422 → 3.422 (0.000) |

| Representation | Positive gates | Mean headroom | Median | Range | Mean zero regret | Mean full-query regret |
|---|---:|---:|---:|---:|---:|---:|
| summary4 | 0/16 | 0.000000 | 0.000 | 0.000–0.000 | 2.624554 | 2.624554 |
| kmer4 | 5/16 | 0.492195 | 0.000 | 0.000–3.686 | 2.897205 | 2.405010 |
| cnn64 | 1/16 | 0.370929 | 0.000 | 0.000–5.935 | 3.007687 | 2.636758 |

The 6/48 positive gates are not evidence that 12.5% is a general biological repair rate. They describe these fixed calibration pools, representations, posterior model, and gate. The `cnn64/calibration_003` positive gate was not added to the curve cohort because the six curve cases had been frozen in V2. Avoiding this post-result addition prevents favorable cohort selection.

**Inference:** Full-query repair is possible but strongly representation and pool dependent. In particular, `summary4` cannot change top-1 regret even with all query labels. This result directs attention to representation/model misspecification rather than to acquisition numerical precision alone.

## S3.4 Acquisition strategies and fixed curve cohort

Six strategies were compared:

1. **Random:** three frozen seeds, averaged within each pool before cross-pool pooling.
2. **Maximum variance:** maximum latent posterior variance, excluding observation noise.
3. **Diversity:** maximum minimum standardized feature distance to development, revealed query, and current-batch selections.
4. **Expected improvement (EI):** analytic latent Gaussian improvement relative to the current full-query posterior-mean maximum.
5. **Top-proxy:** frozen ensemble proxy mean in descending order.
6. **Regret-analytic:** negative expected best-minus-selected regret after a hypothetical label update under the V4 diagonal independent fantasy-posterior approximation.

Budgets were 0, 4, 8, 16, 32, and 64. Every strategy used the same posterior and label budget. Within each batch of four, the same Hamming≤8 exclusion interface was applied sequentially; the fixed pools had minimum pairwise Hamming distance 19, so the rule was not expected to drive differences.

The curve cohort remained `summary4/calibration_001` and `kmer4/calibration_{000,001,002,013,014}` regardless of V4 gate outcomes. This fixed cohort conditions the pooled curves and must not be interpreted as an average over all 16 calibration pools.

## S3.5 Pooled acquisition curves

### kmer4: five V2-frozen pools

| Budget | Random | Max-variance | Diversity | EI | Top-proxy | Regret-analytic |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.706713 | 3.706713 | 3.706713 | 3.706713 | 3.706713 | 3.706713 |
| 4 | 3.623058 | 2.837233 | 2.451708 | 2.200743 | 2.776780 | 2.200743 |
| 8 | 3.623058 | 3.195900 | 2.618210 | 2.200743 | 2.776780 | 2.131689 |
| 16 | 3.497016 | 2.644595 | 2.786586 | **1.220084** | 2.580565 | 2.131689 |
| 32 | 3.082822 | **2.068558** | 2.367245 | 2.131689 | 2.517434 | 2.131689 |
| 64 | 2.131689 | 2.131689 | 2.131689 | 2.131689 | 2.131689 | 2.131689 |

EI at 16 labels was descriptively lower than random by 2.276932 and lower than regret-analytic by 0.911605. Regret-analytic was jointly lowest with EI at 4 labels, uniquely lowest at 8 labels, worse than EI at 16 labels, and worse than maximum variance at 32 labels. No complex strategy dominated across budgets. At full query, strategies shared the same information and converged to the same regret by construction.

### summary4: one V2-frozen pool

Every strategy and every budget had regret 3.686476. Full-query information did not change the selection. This complete negative result is retained rather than omitted because it lacks an acquisition curve signal.

## S3.6 Pool-level counterexamples and non-monotonicity

Pooled means do not remove unfavorable individual outcomes:

- `kmer4/calibration_000`: EI remained at zero-label regret 5.371352 through 16 labels, whereas regret-analytic reached 5.026081 at 8 labels. Full-query regret remained 5.026081, not zero.
- `kmer4/calibration_001`: maximum variance and EI reached zero regret at 4 labels, but diversity reached only 1.493438 at 8 labels and remained there through 32 labels.
- `kmer4/calibration_002`: at 16 labels, EI achieved zero regret whereas regret-analytic was 4.903296; maximum variance was 7.147677, worse than zero-label regret 5.522315. This is a direct counterexample to general regret-analytic or maximum-variance superiority.
- `kmer4/calibration_013`: regret declined to 0.729066, but no strategy reached zero at intermediate budgets.
- `kmer4/calibration_014`: maximum variance remained at zero-label regret 2.588550 through 16 labels, whereas EI reached zero at 4 labels.

These examples show why the supported statement is “some fixed-cohort budgets favor simple EI,” not “active learning reliably repairs proxy failure.”

## S3.7 V4 numerical convergence and its approximation boundary

V3 regret acquisition still contained posterior Monte Carlo error: only 122/384 acquisition steps met its strict convergence criterion. V4 replaced posterior and hypothetical-label sampling with Gauss–Hermite quadrature and analytic normal-max evaluation under a diagonal independent fantasy-posterior approximation.

V4 used fixed 64 label nodes and 32/64/128 value nodes, final level (64,128), and convergence threshold \(10^{-5}\). All **384/384** steps converged and the final-choice stability was 100%. The median and maximum last-level absolute score differences were \(7.544930\times10^{-11}\) and \(1.146502\times10^{-9}\).

Independent numerical checks passed: three-candidate exhaustive scalar integration had maximum absolute error \(7.633\times10^{-17}\), and the two-candidate closed-form normal-max check had error \(1.735\times10^{-17}\). High-sample CUDA Monte Carlo comparisons met their frozen error rules.

This numerical improvement did not create acquisition dominance. V3 regret-Q appeared best at 8 and 16 labels, whereas V4 was best only at 8 and lost to EI at 16. Thus, some V3 ranking was affected by posterior Monte Carlo noise. Moreover, V4’s diagonal independent approximation is not the exact original correlated joint-posterior multivariate-max objective. Quadrature convergence establishes numerical reproducibility of the approximate target, not correctness of that approximation for every decision, biological validity, or outer label efficiency.

## S3.8 Limitations

- All results are retrospective development/calibration diagnostics; permanent outer labels and metrics remain untouched.
- Full-query headroom is an information upper setting, not a deployable acquisition result.
- Positive gates are conditional on fixed pools, representations, Bayesian linear model, hyperparameters, and a strict top-1 regret criterion.
- The curve cohort was fixed by V2 and contains only five `kmer4` pools plus one `summary4` pool. It is not an all-pool average and has no new `cnn64` curve.
- Random has three seeds; other strategies have one deterministic run. No outer biological inference or independent-study interval is available.
- Non-monotone and unfavorable individual-pool results preclude general acquisition superiority claims.
- Regret-analytic acquisition uses a diagonal independent fantasy-posterior approximation. Numerical convergence does not remove representation/model misspecification or top-1 discontinuities.
- The Bayesian linear last layer is only one repair model. Results do not identify whether alternative nonlinear adaptation would help.
- No new sequences were synthesized or experimentally measured, and no wet-lab or clinical claim is made.

## S3.9 Gated next experiment

A Paper 3 outer experiment should not be justified by V4 numerical convergence or selected calibration budgets. Before any outer access, the protocol must freeze the representation, initial development labels, acquisition set (including random, maximum variance, and EI), budgets, posterior model, deployment selector, near-duplicate handling, and pool-cluster endpoints. The evaluation must use untouched outer labels and preserve all budgets and pools. If simple EI or maximum variance is not beaten, the simple baseline remains the conclusion; if no strategy reduces outer regret, repair is recorded as unsuccessful in that setting.

S2's candidate-risk rule failed its frozen calibration gate, and its outer scoring was hard-locked and not performed. There is therefore no validated “detect risk, then repair” sequence to carry into an outer experiment. S3 remains a standalone retrospective repairability diagnosis. Any future combined system would require a separately justified new risk-control protocol and independent evaluation roles; a failed S2 gate must not be bypassed to justify Paper 3 outer access.

## S3.10 Supplementary figure legend

**Supplementary Figure S3.1 | Repair headroom and acquisition curves are representation dependent.** (a) All 48 zero/full-query gates across `summary4`, `kmer4`, and `cnn64`; positive gates are 0/16, 5/16, and 1/16. (b) Fixed-cohort `kmer4` pooled regret across six acquisition strategies and budgets. EI is descriptively best at 16 labels, maximum variance at 32, and regret-analytic does not dominate. These are development/calibration diagnostics with no permanent outer evaluation. Source: `../figures/Figure8_v2_paper3_headroom.pdf`.

## S3.11 Verification

The V5 integrated CPU verification script recomputes all 48 gate arithmetic values, positive-gate counts and representation summaries, and all 72 pooled curve cells from 288 curve records. It checks the EI@16/random/max-variance/regret-analytic values, maximum-variance advantage at 32 labels, 384/384 convergence, diagonal-approximation flag, three identical scientific hashes, and false outer-access flags. Three actual V5 verification rounds and clean PDF rebuild checks are retained. No new model training, acquisition run, remote access, unblinding, or modification of figures/PPT/source_data was performed for this supplement.
