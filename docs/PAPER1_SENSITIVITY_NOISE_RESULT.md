# Paper 1 sensitivity and proxy-output noise

## Alternative candidate pools

An independently selected N50 pool series used the same raw data and a new deterministic namespace. Double-build hashes, Learner label isolation and Hamming-1 checks passed; evaluation passed three independent rounds.

| N50 series | cap 1 | cap 2 | cap 4 | cap 8 | cap 16 |
|---|---:|---:|---:|---:|---:|
| Original | 6.422 | 6.341 | 6.617 | 6.705 | 6.519 |
| Alternative | 6.521 | 5.914 | 5.625 | 5.618 | 5.726 |

Both series are non-monotone. The alternative series reaches its minimum near cap 4–8 and reverses at cap 16. The lack of sustained improvement is therefore not an artifact of one candidate-pool draw.

## Proxy-output noise

Noise was added to each normalized CNN output before member-specific inverse transformation. This is not V/Q probability-logit noise. Sigma=0 selections and regrets exactly match the original grid by strategy/cap/seed. All 16 pools passed three independent evaluation rounds.

| Sigma | cap 1 regret | cap 4 regret | cap 16 regret |
|---:|---:|---:|---:|
| 0 | 6.422 | 6.617 | 6.519 |
| 0.1 | 6.422 | 6.260 | 6.018 |
| 0.25 | 6.422 | 6.133 | 6.143 |
| 0.5 | 6.422 | 6.088 | 6.205 |
| 1.0 | 6.422 | 6.248 | 6.526 |

Noise does not monotonically degrade BoN in this context. Moderate perturbations can improve selections, consistent with exploration or high-tail noise diversification, while sigma=1 returns close to baseline. These are descriptive results and do not justify adding noise as a new method. The hypothesis that BoN necessarily amplifies Gaussian proxy-output noise is not supported here.

## Unified interpretation

Paper 1 now has fixed-pool pressure curves, alternative-pool sensitivity, external contexts, score matching, same- and cross-architecture common-proxy checks, and a negative noise result. The evidence supports conditional high-score extrapolation risk and shared proxy errors. It does not support VGSC superiority, universal optimizer-specific distortion or inevitable noise amplification.
