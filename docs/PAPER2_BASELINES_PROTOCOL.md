# Paper 2 missing baselines: frozen protocol

## Scope and evidence boundary

This run adds only the conservative-objective, train-distance penalty, ensemble LCB and diversity-LCB baselines. It does not modify Leppek/Schlusser or Paper 3 files, and it does not reverse the preserved FP-REJ/Q falsification result. All outer analyses are retrospective sensitivity analyses on previously viewed pools, not new blind tests or a basis for further tuning.

The source for COM is Trabucco et al. (2021), arXiv:2107.06882, and the authors' `design-baselines` repository at commit `785dbcfa58107bfcc426257a1c2e69d7f71c3c27`. The implementation below is an independent PyTorch adaptation of the core COM mechanism, **not the official COM UTR-ResNet run and not official weights**.

## COM adaptation

- Same `FreshSampleCNN`, train/dev/calibration pack, train-only target scaler, seeds 11/23/37, Adam model learning rate 0.001, batch size 1024 and dual train/dev patience rule as the fresh CNN comparator.
- Every training batch starts from the observed one-hot sequence and performs 50 gradient-ascent steps in the authors' three-log-ratio discrete relaxation, with zero reference logit, `soft_interpolation=0.6` (the official UTR dataset default) and `particle_lr=2*sqrt(200)`. The relaxed negative is detached before the conservative model update.
- The optimized loss is normalized MSE plus `alpha*(mean(f(x_negative))-mean(f(x_positive)))`. A separate Adam optimizer updates `log_alpha` at 0.01 using `alpha*(2-conservatism)` and `alpha=softplus(log_alpha)`, initialized from 0.1. These are the official discrete-task `tau=2`, alpha initialization and alpha learning rate.
- No maximum epoch or small fixed-step stopping rule is imposed. Every epoch stores online COM/regression/conservatism/alpha and full train/dev normalized MSE. The best-dev checkpoint is restored. Three independent spawned CUDA workers share the RTX 5090; CPU fallback is forbidden.

Differences from the official experiment are explicit: CNN rather than UTR-ResNet, measured MRL rather than oracle-relabelled design-bench targets, this repository's split and convergence protocol, and fixed-pool reranking rather than de novo design generation. Consequently results are labelled `COM-style adaptation` throughout.

## Dev-only selection

All methods use the same proposal draws, five paired seeds and caps 1/2/4/8/16. Every scored proposal invokes all three corresponding CNN members on CUDA. Per-method deployment cost is `3*cap` member item-forwards; model training cost is reported separately and not hidden in selection cost.

Predeclared dev grids at cap 16:

- LCB: `mu-lambda*sigma`, lambda in `{0, 0.5, 1, 2, 4}`; minimize dev mean top-1 regret.
- Distance: `mu-lambda*(1-support)`, where support is one minus normalized minimum train Hamming distance; lambda in `{0, 0.5, 1, 2, 4, 8}`; minimize dev mean top-1 regret.
- Diversity LCB: sequential `mu-lambda*sigma+gamma*minimum_normalized_Hamming_to_already_selected`; the same lambda grid and gamma in `{0, 0.5, 1, 2, 4}`; minimize dev regret@4.
- BoN uses ensemble mean and COM uses the three-member conservative ensemble mean. No outer metric enters any choice. Ties use the smaller lambda then gamma. Calibration is reserved for calibration only and is not a hyperparameter-selection role.

## Retrospective evaluation

- Report regret versus actual member-forward cost, risk–coverage, measured-MRL top-k (`k=1/2/4/8/16`), and pool-cluster paired bootstrap 95% intervals against BoN and LCB at cap 16. Negative differences favor the new baseline.
- Risk–coverage uses train distance for the distance baseline and ensemble standard deviation otherwise. Empirical outer coverage thresholds are descriptive only, never deployment thresholds.
- Thirty-two deterministic within-proposal permutations form negative controls at cap 16: shuffle ensemble mean for BoN, COM mean for COM, ensemble standard deviation for LCB/diversity-LCB, and train distance for the distance penalty. Actual and all permuted results are retained.
- No pool, seed, configuration or result is removed because it is unfavorable. Random outer pools remain secondary and are not pooled with designed pools.
