# Paper 2 missing baselines: verified result

## Evidence boundary

This is a COM-style conservative-objective adaptation, not the official COM UTR-ResNet or official weights. Outer pools were already viewed: every outer result here is retrospective sensitivity evidence, not a new blind test. No outer value was used for training or hyperparameter selection. Calibration was not used for selection. The earlier FP-REJ/Q falsification result is unchanged.

## COM convergence

All reported COM seeds used actual parallel CUDA workers, 50-step discrete adversarial negatives and the dual train/dev patience rule. No maximum-epoch stopping rule was imposed.

| Seed | Epochs | Best epoch | Best train MSE | Best dev MSE | Best alpha | Seconds | Peak allocated bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 88 | 73 | 0.466557 | 0.671121 | 0.000000 | 4142.06 | 974468096 |
| 23 | 82 | 61 | 0.480084 | 0.674638 | 0.000000 | 3754.42 | 974522880 |
| 37 | 81 | 47 | 0.482091 | 0.671891 | 0.000003 | 3742.41 | 974468096 |

Training code commit: `5c9f2547efad919ffcaf4b00e05e07e70ad3493e`. Config SHA256: `a6797903f7c286a627750fe8bc1c3b34bcb7b4053f1e6caef28600b1cb92d53c`. Histories, checkpoints, per-epoch hashes and three independent CUDA reload/recompute reports are stored with the artifacts.

Three aborted implementations are preserved under `invalid_com_training_dual_gradient`, `invalid_com_training_positive_gradient`, and `invalid_com_training_particle_scale`; none contributes a model, selection or scientific conclusion.

## Dev-only frozen choices

| Method | Lambda | Gamma | Dev mean regret | Dev regret@4 |
|---|---:|---:|---:|---:|
| bon | 0.000000 | 0.000000 | 2.435639 | 1.517411 |
| com | 0.000000 | 0.000000 | 2.390833 | 1.577731 |
| distance | 0.000000 | 0.000000 | 2.435639 | 1.517411 |
| diversity_lcb | 2.000000 | 0.500000 | 2.581618 | 1.513300 |
| lcb | 0.000000 | 0.000000 | 2.435639 | 1.517411 |

LCB and distance minimize dev top-1 regret; diversity-LCB minimizes dev regret@4. All configurations, including unfavorable ones, remain in the frozen JSON.

## Designed outer retrospective sensitivity (16 pools)

| Method | Mean regret | Regret@4 | CNN member forwards | Top-4 max measured MRL |
|---|---:|---:|---:|---:|
| bon | 0.672377 | 0.375235 | 48 | 7.220927 |
| lcb | 0.672377 | 0.375235 | 48 | 7.220927 |
| distance | 0.672377 | 0.375235 | 48 | 7.220927 |
| com | 0.661818 | 0.381116 | 48 | 7.215046 |
| diversity_lcb | 0.640713 | 0.378499 | 48 | 7.217662 |

Negative paired differences favor the method; positive differences favor the reference.

| Method | Reference | Mean difference | Pool-cluster bootstrap 95% CI |
|---|---|---:|---|
| lcb | bon | 0.000000 | [0.000000, 0.000000] |
| distance | bon | 0.000000 | [0.000000, 0.000000] |
| distance | lcb | 0.000000 | [0.000000, 0.000000] |
| com | bon | -0.010559 | [-0.045744, 0.017583] |
| com | lcb | -0.010559 | [-0.045744, 0.017583] |
| diversity_lcb | bon | -0.031664 | [-0.098743, 0.030134] |
| diversity_lcb | lcb | -0.031664 | [-0.098743, 0.030134] |

## Random outer retrospective sensitivity (3 eligible pools; secondary)

| Method | Mean regret | Regret@4 | CNN member forwards | Top-4 max measured MRL |
|---|---:|---:|---:|---:|
| bon | 6.088421 | 4.212646 | 48 | 7.787354 |
| lcb | 6.088421 | 4.212646 | 48 | 7.787354 |
| distance | 6.088421 | 4.212646 | 48 | 7.787354 |
| com | 7.899485 | 4.063283 | 48 | 7.936717 |
| diversity_lcb | 6.772796 | 4.246592 | 48 | 7.753408 |

| Method | Reference | Mean difference | Pool-cluster bootstrap 95% CI |
|---|---|---:|---|
| lcb | bon | 0.000000 | [0.000000, 0.000000] |
| distance | bon | 0.000000 | [0.000000, 0.000000] |
| distance | lcb | 0.000000 | [0.000000, 0.000000] |
| com | bon | 1.811064 | [0.123119, 2.660000] |
| com | lcb | 1.811064 | [0.123119, 2.660000] |
| diversity_lcb | bon | 0.684375 | [-0.573597, 2.250072] |
| diversity_lcb | lcb | 0.684375 | [-0.573597, 2.250072] |

## Negative controls and risk–coverage

| Context | Method | Observed regret | Permuted regret | Permutation 95% interval |
|---|---|---:|---:|---|
| designed | bon | 0.672377 | 1.950522 | [1.674914, 2.212984] |
| designed | lcb | 0.672377 | 0.672377 | [0.672377, 0.672377] |
| designed | distance | 0.672377 | 0.672377 | [0.672377, 0.672377] |
| designed | com | 0.661818 | 2.004269 | [1.750332, 2.282709] |
| designed | diversity_lcb | 0.640713 | 0.692927 | [0.629473, 0.776676] |
| random | bon | 6.088421 | 7.311621 | [6.328536, 8.454018] |
| random | lcb | 6.088421 | 6.088421 | [6.088421, 6.088421] |
| random | distance | 6.088421 | 6.088421 | [6.088421, 6.088421] |
| random | com | 7.899485 | 6.992836 | [5.774064, 8.011122] |
| random | diversity_lcb | 6.772796 | 6.572556 | [5.785602, 7.352968] |

Thirty-two deterministic within-proposal permutations were retained for every method. Full risk–coverage rows, regret–cost curves, top-k values for k=1/2/4/8/16, per-member costs and all paired intervals are in the summary JSON files; the table above is not a selected subset of favorable configurations.

## Conclusion

These are baseline and sensitivity results only. No claim of biological improvement, prospective validation, independent-study replication or official COM reproduction is made. Any unfavorable or interval-crossing result remains evidence against a method advantage; the preserved FP-REJ/Q gate failure is not reversed.

## Verification

Three complete software rounds passed (291 tests plus 106 subtests each). COM checkpoints and every outer evaluation were independently checked in three rounds, including CUDA reload/dev-loss recomputation, raw regret/top-k recomputation, record completeness, frozen hyperparameters, permutation counts and source hashes.
