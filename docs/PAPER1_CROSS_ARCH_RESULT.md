# Paper 1 cross-architecture common-proxy audit

## Alternative proxy

A 4-mer-frequency MLP with 259 features was trained on the same Sample train/dev labels as the fresh CNN, with train-only target scaling and no epoch limit. Seeds 101/202/303 converged at epochs 108/124/177, with best epochs 103/119/172. Three independent CUDA rounds recomputed full train/dev losses from each best checkpoint and verified scaler and model hashes. This is an alternative architecture for mechanism analysis, not a new main model.

## External error correlation

| Context | CNN mean absolute error | 4-mer mean absolute error | Absolute-error correlation |
|---|---:|---:|---:|
| N50 HEK293T | 2.805 | 2.852 | 0.962 |
| Defined HepG2 r1 | 2.907 | 3.015 | 0.953 |
| Defined T-cell r1 | 2.076 | 2.223 | 0.931 |
| Defined T-cell r2 | 2.568 | 2.690 | 0.965 |

All four contexts show very high error correlation across architectures. Shared training data and endpoint limitations therefore dominate over model family: changing architecture does not automatically provide independent evidence or eliminate common proxy risk.

Low architecture-gap/high-error joint counts were 60, 65, 54 and 61 out of 1024 candidates respectively. These descriptive strata do not form a universal safety certificate. Every context passed three independent recomputation rounds for both architectures, architecture gaps, correlations and disagreement strata.

## Interpretation

The result strengthens the common-proxy finding from the same-architecture audit. High-score extrapolation risk is shared across architectures, so model voting alone cannot authenticate biological function. The remaining Paper 1 work is synthesis and sensitivity analysis, not adding more same-data models.
