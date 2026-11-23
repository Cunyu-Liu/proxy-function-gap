# Paper 1 fixed-pool pressure: external context results

## Scope and status

Historical-trajectory replay is not claimed: original split-v2 reports no eligible complete step trajectories, and fresh outer contains no step sequences. This stage evaluates fixed-pool best-of-N using a frozen fresh Sample ensemble on independent Castillo-Hair GSE232927 data. It is a verified external context result, not the complete Paper 1 evidence package.

## Data and protocol

- Main external context: random-end HEK293T N50 r1; 2,238,030 finite unique 50nt rows, no exact Sample train/dev/calibration overlap.
- Context controls: defined-end HepG2 r1 and T-cell r1/r2. Sample exact overlaps are excluded. N25 is excluded because all rows are 25nt.
- Each context has 16 deterministic pools of 64 unique sequences, with Hamming-1 exclusion. Double-build hashes, no-label Learner views and cross-pool near-duplicate checks passed.
- Random/BoN caps `{1,2,4,8,16}`, seeds `{101,202,303,404,505}`. Actual three-member CUDA inference; no Castillo labels used for training, calibration or pool ordering.
- Every context passed three independent evaluation rounds. All label values are historical measurements, never proxy predictions.

## Mean measured-MRL regret

| Context | cap 1 | cap 2 | cap 4 | cap 8 | cap 16 | Pattern |
|---|---:|---:|---:|---:|---:|---|
| Sample designed, secondary pilot analysis | 1.905 | 1.528 | 1.161 | 0.770 | 0.681 | Monotone decrease |
| Sample random, 3 eligible pools | 6.552 | 6.057 | 6.760 | 6.507 | 6.883 | Non-monotone |
| Castillo N50 HEK293T | 6.422 | 6.341 | 6.617 | 6.705 | 6.519 | Non-monotone; no sustained improvement |
| Castillo defined HepG2 r1 | 4.135 | 3.941 | 3.738 | 3.741 | 3.803 | Best near cap 4, then reverses |
| Castillo defined T-cell r1 | 4.739 | 4.787 | 4.807 | 4.747 | 4.352 | Late decrease only |
| Castillo defined T-cell r2 | 6.563 | 6.276 | 5.850 | 6.209 | 5.928 | Non-monotone; best near cap 4 |

Pool-cluster bootstrap intervals are stored in each `bon_pressure.json`; small random/replicate differences are not interpreted as universal effects.

## Interpretation

The fixed-pool pressure effect is context-dependent. Sample designed improves monotonically, but four independent Castillo contexts do not reproduce that pattern. This rejects a context-free claim that increasing proxy pressure necessarily improves or necessarily damages measured function. The result supports a conditional proxy-contract narrative, not VGSC superiority.

## Evidence limitations and next gates

This stage supplies fixed-pool best-of-N, external context checks and negative/non-monotone controls. It does not yet complete score-matched controls, common-proxy ablation, poly-U/GC/uAUG mechanism analyses or the complete Paper 1 package. Do not claim a completed preprint.

Next: freeze mechanism analyses, compare score-matched random versus optimized candidates, test sequence-family sensitivity and retain all negative results. Raw files, double builds, selection/evaluation manifests and independent validation reports remain under `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/`.
