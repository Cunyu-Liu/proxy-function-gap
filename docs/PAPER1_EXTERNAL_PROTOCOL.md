# Paper 1 external fixed-pool protocol

## Why this protocol

The original split-v2 explicitly reports no eligible complete step trajectories after training/scaler/control exclusions. Fresh outer contains no step sequences. Therefore historical-trajectory replay is not claimed. Paper 1 uses fixed-pool best-of-N and context controls instead.

## Data roles

- Main external test: Castillo-Hair GSE232927 random-end HEK293T N50 r1.
- Context controls: defined-end HepG2 r1 and T-cell r1/r2; each context is analyzed separately and never pooled into one endpoint.
- Random-end N25 is excluded because every sequence is 25nt and the frozen model requires fixed 50nt. No padding or retraining is used to manufacture compatibility.
- Sample 2019 pilot outer is secondary development evidence only, not independent confirmation.

Raw files and SHA256 are pinned under `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/raw/castillo_hair_gse232927/`.

## Learner and split

The frozen fresh Sample CNN ensemble is used zero-shot. No Castillo MRL is used for training, calibration, pool selection or configuration tuning. Exact sequences occurring in Sample train/dev/calibration are excluded before pool creation. Finite-label availability is checked by the Evaluator; label values are not used for deterministic pool ordering. Sequence, Hamming-1 and complete-pool conflicts are blocked.

Each context uses 16 pools of 64 unique valid 50nt sequences where available. The proposal grid is cap `{1,2,4,8,16}` with seeds `{101,202,303,404,505}`. Only Random and BoN are run. UTR50 validity is mandatory; no new GC/uAUG/poly-U threshold is invented.

## Evaluation

The Learner writes selected sequence IDs and actual CUDA forward counts with `regret=null`. The Evaluator then reads the corresponding historical MRL vault. Primary endpoint is measured-MRL regret at each cap with pool-cluster bootstrap intervals. Random is the no-proxy-pressure control. Cross-cell and replicate comparisons are secondary and context-specific.

A pressure effect is not assumed in either direction. Monotone decrease, increase and non-monotone behavior are all reported. No result from Sample pilot outer or Castillo development is used to change the frozen grid. Missing labels exclude a complete pool; they are never imputed or removed one sequence at a time.
