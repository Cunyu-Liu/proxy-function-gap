# Paper 1 mechanism: score matching and common-proxy audit

## Protocol

All four Castillo contexts used actual three-member CUDA inference for every candidate (192 member forwards per 64-sequence pool). For each cap-16 BoN selection, the Evaluator selected the nearest different sequence by proxy mean within the same pool, with deterministic hash tie-breaking. Absolute-error differences use pool-cluster bootstrap. Sequence-family summaries are descriptive quartiles, not newly invented biological thresholds. All contexts passed three independent score-match verification rounds.

## Score-matched absolute error

| Context | BoN mean absolute error | Score-matched mean absolute error | Difference | Pool-cluster 95% interval |
|---|---:|---:|---:|---:|
| N50 HEK293T | 2.881 | 2.707 | +0.174 | [−0.462, 0.721] |
| Defined HepG2 r1 | 3.234 | 3.134 | +0.101 | [−0.581, 0.778] |
| Defined T-cell r1 | 2.389 | 3.094 | −0.705 | [−1.493, 0.024] |
| Defined T-cell r2 | 2.512 | 2.741 | −0.229 | [−1.196, 0.809] |

Every interval includes zero and directions are not consistent across replicates. The data therefore do not support an optimizer-specific error increment beyond same-score high-tail extrapolation. The appropriate conclusion is high-score extrapolation risk, not proof of optimizer-specific distortion.

## Common proxy and sequence signals

- N50: 67/1024 candidates were jointly in the lowest-spread quartile and highest-error quartile; low-spread mean absolute error 2.741 versus 2.826 for other candidates. Low disagreement does not identify the highest-error region.
- HepG2: 75/1024 joint candidates; low-spread mean error 3.095 versus 2.845. This is a context-specific warning signal, not universal validation.
- T-cell r1/r2: 60/1024 and 59/1024 joint candidates respectively. Low-spread and other-spread errors are similar within each replicate.
- uAUG, GC, poly-U and support summaries are retained in each context's `summary.json`; they are exploratory mechanisms and are not used to tune the outer protocol.

## Conclusion and remaining work

Fixed-pool external results support context-dependent pressure effects and high-score extrapolation risk. They do not support a universal optimizer-specific effect or a universal low-disagreement safety certificate. This satisfies the current score-matched and common-proxy checks, but the full Paper 1 package still requires a unified report and any remaining sensitivity analyses before preprint submission.
