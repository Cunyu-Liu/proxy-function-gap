# Paper 1 proxy-output noise protocol

Noise is added to each frozen CNN member's normalized regression output before the member-specific train-only inverse transform. It is not described as V/Q probability-logit noise and does not alter measured MRL. Sigma grid `{0,0.1,0.25,0.5,1.0}` uses deterministic per-selection seeds. Only Random and BoN caps `{1,4,16}` are compared. The primary endpoint is measured-MRL regret and its degradation relative to sigma=0 under the same proposals.

N50 is the preregistered external stress context. Pool definitions, model checkpoints and evaluation code are frozen before running. No noise level is selected to improve a method. All results, including identical sensitivity across strategies, remain in the report.
