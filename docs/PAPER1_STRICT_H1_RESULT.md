# Paper 1 strict-H1 reconstruction result

## Status and provenance

This is a retrospective reconstruction and protocol-sensitivity analysis of previously inspected Castillo outer contexts, not a new blind test. No gate, endpoint, model, seed or cap was changed after seeing results. Valid outputs are under `castillo_fixed_pools_v3_strict_h1_retry1` and `paper1_v3_strict_h1_retry1`; source commit is `ecaccfafd12d81258b8cd0648b48175398b6d98c`.

The first analysis attempt used an NPZ without `utr50` and failed before CUDA inference or scientific evaluation. Its directory and logs were retained. The corrected retry added a sequence-key preflight and used the same Sample train NPZ as the earlier protocol. Related tests passed three times, three CUDA self-checks passed, and the complete suite passed 264 tests. Frozen CNN and 4-mer MLP inference metadata report `cuda:0` and `cpu_fallback=false` throughout.

## Double builds and isolation

All four contexts completed two independent deterministic builds with 16 pools of 64 sequences. Pool exact/H1 isolation, cross-Sample-train/dev/calibration exact/H1 isolation, and learner/label/manifest/raw/reference/pool-hash checks all passed. Evaluation, score matching, common-error metrics and the aggregate paired summary each passed three recomputation rounds.

| Context | v2/v3 learner hashes | Within/cross-pool H2 pairs | Cross-Sample H2 pairs |
|---|---|---:|---:|
| N50 HEK293T | Identical | 0 | 0 |
| Defined HepG2 r1 | Different | 0 | 6 |
| Defined T-cell r1 | Different | 2 | 48 |
| Defined T-cell r2 | Different | 0 | 44 |

H2 was a predeclared diagnostic only, not an exclusion or pass gate. The three changed contexts explain why their earlier numerical results must be replaced for the strict-H1 sensitivity analysis.

## Paired BoN pressure

Effects are BoN regret minus same-seed Random regret; negative favors BoN. Intervals are 95% pool-cluster bootstrap intervals over per-pool paired effects (10,000 replicates).

| Context | Cap | Random regret | BoN regret | Paired effect | 95% CI |
|---|---:|---:|---:|---:|---|
| N50 | 1 | 6.422 | 6.422 | 0.000 | [0.000, 0.000] |
| N50 | 2 | 6.422 | 6.341 | −0.081 | [−0.731, 0.535] |
| N50 | 4 | 6.422 | 6.617 | +0.195 | [−0.508, 0.861] |
| N50 | 8 | 6.422 | 6.705 | +0.283 | [−0.627, 1.254] |
| N50 | 16 | 6.422 | 6.519 | +0.097 | [−1.039, 1.244] |
| HepG2 | 1 | 4.464 | 4.464 | 0.000 | [0.000, 0.000] |
| HepG2 | 2 | 4.464 | 4.332 | −0.132 | [−0.728, 0.483] |
| HepG2 | 4 | 4.464 | 4.042 | −0.422 | [−1.019, 0.156] |
| HepG2 | 8 | 4.464 | 3.559 | −0.904 | [−1.708, −0.081] |
| HepG2 | 16 | 4.464 | 3.794 | −0.670 | [−1.628, 0.306] |
| T-cell r1 | 1 | 4.799 | 4.799 | 0.000 | [0.000, 0.000] |
| T-cell r1 | 2 | 4.799 | 4.122 | −0.677 | [−0.944, −0.388] |
| T-cell r1 | 4 | 4.799 | 3.642 | −1.156 | [−1.630, −0.709] |
| T-cell r1 | 8 | 4.799 | 3.599 | −1.199 | [−1.643, −0.738] |
| T-cell r1 | 16 | 4.799 | 3.555 | −1.243 | [−1.710, −0.769] |
| T-cell r2 | 1 | 6.398 | 6.398 | 0.000 | [0.000, 0.000] |
| T-cell r2 | 2 | 6.398 | 5.421 | −0.977 | [−1.612, −0.335] |
| T-cell r2 | 4 | 6.398 | 5.254 | −1.144 | [−1.898, −0.385] |
| T-cell r2 | 8 | 6.398 | 4.889 | −1.510 | [−2.132, −0.868] |
| T-cell r2 | 16 | 6.398 | 5.043 | −1.355 | [−2.083, −0.653] |

N50 remains non-monotone with no sustained improvement. HepG2 improves most at cap 8 and partially reverses at cap 16. T-cell r1 decreases monotonically through cap 16; T-cell r2 improves through cap 8 and slightly reverses at cap 16. These are context-specific retrospective effects, not prospective confirmation.

## Score matching and common error

Score-matched effects are optimized-minus-matched absolute error at cap 16. Common-error effects are CNN-minus-MLP mean absolute error.

| Context | Optimized MAE | Matched MAE | Difference [95% CI] | CNN MAE | 4-mer MAE | CNN−MLP [95% CI] | Error correlation [95% CI] |
|---|---:|---:|---|---:|---:|---|---|
| N50 | 2.881 | 2.707 | +0.175 [−0.444, 0.712] | 2.805 | 2.852 | −0.047 [−0.077, −0.020] | 0.962 [0.956, 0.967] |
| HepG2 | 3.277 | 2.952 | +0.326 [−0.796, 1.484] | 2.919 | 3.029 | −0.110 [−0.153, −0.065] | 0.953 [0.946, 0.959] |
| T-cell r1 | 1.615 | 1.918 | −0.303 [−0.897, 0.149] | 1.568 | 1.821 | −0.254 [−0.297, −0.209] | 0.878 [0.859, 0.897] |
| T-cell r2 | 1.566 | 1.663 | −0.097 [−0.496, 0.326] | 1.544 | 1.739 | −0.195 [−0.231, −0.160] | 0.933 [0.919, 0.944] |

Every score-matched interval includes zero and directions are not consistent. Thus the strict-H1 sensitivity analysis still does not support an optimizer-specific error increment beyond score-matched extrapolation. CNN mean absolute error is lower than 4-mer MLP error in every context, with all paired intervals excluding zero. Nevertheless, absolute-error correlations remain high (0.878–0.953), so changing architecture does not remove shared proxy error.

## Which earlier conclusions change

- **Retained:** no single pressure direction describes all contexts; N50 shows no sustained BoN improvement; score matching does not establish an optimizer-specific error increment; cross-architecture errors remain strongly correlated.
- **Changed:** the earlier statement that all four Castillo contexts fail to reproduce improvement is not valid under strict-H1 reconstruction. T-cell r1 now improves monotonically with negative paired intervals at caps 2–16, T-cell r2 has negative intervals at caps 2–16 with best mean at cap 8, and HepG2 has a negative interval at cap 8. Earlier numerical pressure, score-match and error estimates for the three changed pools are historical pre-repair results, not the strict-H1 sensitivity estimates.
- **Qualified:** high cross-architecture correlation remains, but T-cell r1 is lower than in the old pools. CNN's lower MAE does not imply independent biological validation because both models share training data and retain correlated errors.

No universal beneficial or harmful proxy-pressure claim is made. All outer findings remain retrospective/protocol-sensitivity evidence. Local source data contain only pool-level aggregates, configuration, hashes and validation logs, not sequence strings or sequence-level measured labels.
