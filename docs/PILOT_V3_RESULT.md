# VGSC fixed-pool pilot v3: verified result and gate decision

## Decision

**Do not enter the main method line.** Preserve FP-REJ/FP-TILT as baselines and retain the budget/noise protocols. Q is not usable in the designed outer context. This is a completed falsification pilot, not a completed preprint or biological-function claim.

## Verified protocol

- Valid prediction lineage: ensemble-v2, measured-MRL units after member-specific train-only inverse transform.
- Development: 32 disjoint pools; calibration: 16 disjoint pools; five paired seeds.
- Frozen deployment selector: FP-REJ, alpha=20, proposal cap=16. Selection used only dev regret.
- Production selection executed actual CUDA forwards for three CNN members and V/Q models. Actual member, verifier, V and full-pool Q-feature costs are recorded separately.
- Outer Learner produced selections with `regret=null`; the separate Evaluator read historical MRL only after Learner validation.
- Designed: 16 outer pools. Random: pools 000–002 were complete; pool 003 was excluded as a whole because two candidates lacked measured MRL. No imputation or partial-pool scoring was used.
- V and Q each passed three CUDA reload/recompute rounds. Both outer evaluation sets passed three independent regret/provenance rounds.

## Results

| Context | Method | Outer pools | Coverage | Mean regret | Interpretation |
|---|---|---:|---:|---:|---|
| Designed | FP-REJ without Q | 16 | 1.0 | 1.0242708233 | Worse than BoN |
| Designed | BoN cap16 | 16 | 1.0 | 0.6811140309 | Reference |
| Designed | FP-REJ+Q | 16 | 0.0 | Not defined | Q rejected every selection |
| Random, eligible | FP-REJ+Q | 3 | 1.0 | 6.8277223619 | Secondary only |
| Random, eligible | BoN cap16 | 3 | 1.0 | 6.8829629630 | Secondary only |

Designed FP-REJ minus BoN: **+0.3431567924**, pool-cluster bootstrap 95% interval **[0.2250717297, 0.4670981804]**; positive values favor BoN. Random FP-REJ+Q minus BoN: **−0.0552406011**, interval **[−1.4000000000, 1.0511111111]**. Three random pools are not independent-study replication and do not establish a general advantage.

## Invalidated and aborted artifacts

- ensemble-v1 and all pools/V derived from it: invalid for MRL science because target inverse transformation was missing.
- production_v3_q and retry1: invalid because only one seed per pool was serialized; retry2 contains 160 dev and 80 calibration selections.
- selection/dev_v3: aborted output-path collision; retry1 is the valid 32-pool grid.
- outer_designed_v1 and outer_random_v1 initial learner directories: failed verification-path reads; retry1 directories are valid Learner outputs.
- evaluation/outer_random_v1: no valid scores produced; the complete three-pool evaluation is in outer_random_v1_eligible.

All failures remain on `/teams` with logs. No invalid artifact was used for the reported results.

## Gate interpretation

The main designed context fails both method requirements: FP-REJ does not beat BoN and Q has zero coverage. The small random difference is secondary and uncertain. No further threshold, alpha or pool tuning is permitted on these outer results. The next research stage returns to Paper 1 phenomenon, negative controls, noise protocol and genuinely independent replication.

## Reproducibility

- Summaries: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/evaluation/outer_designed_v1/summary.json` and `outer_random_v1_eligible/summary.json`.
- Deck: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/ppt/proxy_function_gap_vgsc_pilot_20261124.pptx`.
- Protocol: `docs/E1_OUTER_PROTOCOL.md`.
- GitHub upload is pending because neither repository has a configured remote and `gh` is unavailable on the server. Local commits remain in `/home`.
