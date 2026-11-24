# Ensemble, V, Q and fixed-pool selection protocol

## Current boundary

Only train/dev/calibration inference and dev/calibration selection are executed on production data. No outer labels or metrics are accessed and E1 remains disabled. The verified prediction lineage is ensemble-v2 in measured-MRL units. Ensemble-v1 and all downstream artifacts derived from it are invalidated because they omitted the train-only inverse target transform.

## Fresh three-member inference

`EnsembleInferenceConfig` names the 11/23/37 best checkpoints and expected checkpoint/model hashes. Each member independently validates pack/order/source/code/model identity and executes all train/dev/calibration rows on explicit CUDA. Raw model outputs are inverse-transformed with that member's train-only target scaler. Outputs contain no labels: row_id, each member prediction, ensemble mean, sample std (`ddof=1`), and range. Per-member actual item-forward, batch count, GPU/dtype, model hash, scaler and peak memory are recorded. There is no CPU fallback or distillation.

## Production pool manifests

The production adapter joins one frozen fixed pool to the inference role by unique row ID. It verifies the fixed-pool final delivery, split/pool hashes, complete source rows, q values, Learner label-source permission, inference lineage, and train-only support. Prediction NPZ files may contain the full role; pool rows are selected as a strict ID subset and never positionally aligned. Support is `1 - minimum Hamming distance to any fresh train UTR50 / 50` and is recomputed during loading. Production manifests missing any provenance fields fail closed.

The disjoint multi-pool v2 delivery contains 32 dev, 16 calibration, 4 outer-random and 16 outer-designed pools. Pools are disjoint by sequence, near component, allocation unit and trajectory, but remain same-study units rather than independent external studies.

## V head

Target: `b=1` when `ensemble_mean - measured > delta`; `r=P(b=1)` and `V=1-r`.

- Delta, feature set, and logistic-vs-MLP model are selected only by dev grouped OOF log loss.
- Fold assignment is deterministic round-robin over sorted group IDs. Fold scalers use only fold-train rows.
- Each iterative optimization uses CUDA and squared-hinge margin loss, with full train/dev eval each epoch, min_delta/patience, and dev absolute-best plateau reset. Finite margins can produce exact zero loss; OOF model selection still uses log loss. No max step is allowed.
- Calibration rows are used only for final Platt probability calibration using true logistic log-loss. They cannot select delta, features, or model type.
- Fold states, OOF predictions, candidate scores, best epochs, model hashes, input-pool provenance and file hashes are saved.

## Budgeted fixed-pool selection

Strategies: Random, BoN, LCB, WSC identity-weighted vote, greedy verifier-reward argmax, FP-REJ, and analytic FP-TILT. V-only is the main guided family; Q is not a ranking input.

Alpha grid: `{0,1,5,10,20}`. Proposal-cap grid: `{1,2,4,8,16}`. Five paired seeds: `{101,202,303,404,505}`. Every strategy uses the same seed for each alpha/cap pair.

Production selection executes the three frozen CNN members and V fold/Platt models on CUDA for every score. Repeated proposals are not cached and precomputed NPZ lookup is not counted as a production forward. Proxy/verifier member, V and Q item-forwards are recorded separately. Random costs zero. FP-TILT scores the full support; FP-REJ can stop early. Identical N does not imply identical actual cost. LCB is `ensemble_mean - lambda * ensemble_std`.

The common primary CNN budget is BoN cap16: 48 member item-forwards. V and Q costs are additional separate columns, not hidden inside that budget. Full-support FP-TILT is reported at its actual cost and marked over budget when it exceeds 48; it is never truncated to force an equal-cost comparison. `production-grid --actual-cost-budget` is a separate execution-truncation control and is not a substitute for per-strategy cost reporting.

## Q head

Q is trained only after all selection manifests are hash-verified and the combined record set is frozen. It uses pool-level proxy/support/cost features and the frozen selection's dev regret. It is a high-regret probability used only for abstention/risk coverage.

Calibration fits the final Platt logistic probability calibrator first. That calibrator is then applied to grouped dev OOF predictions, and threshold selection uses calibrated-OOF risk-coverage rows with the configured minimum coverage. Both raw and calibrated OOF probabilities are saved, and the manifest declares `threshold_domain=platt_calibrated_probability`. Calibration cannot select Q features, model type or threshold. Q never changes candidate order, FP-TILT weights, or FP-REJ acceptance weights. Q can only add abstention; it cannot restore a selector-abstained record. All selection and pool source hashes are saved.

The 2026-11-24 dev-only recalibration found that the legacy probability grid ending at 0.9 had zero coverage after Platt calibration (minimum calibrated dev OOF 0.96494). Before any outer rerun, the candidate grid was extended only with stricter high-probability thresholds 0.95, 0.975 and 0.99. The minimum coverage remained 0.5, 1.0 was not added, and the original dev risk-coverage objective remained unchanged. This feasibility amendment is recorded as dev-only protocol history, not as a new blind or outer-driven choice.

## Executable entries

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/home/proxy-function-gap-sample-runtime/.deps:src:gate0:runtime/src
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
$PY -m proxy_function_gap.pool_cli ensemble-inference --config configs/ensemble_inference_v2.json
$PY -m proxy_function_gap.pool_cli build-production-pool --source-pool-manifest <fixed_pool.json> --pool-id dev_000 --prediction-manifest <ensemble_v2/prediction_manifest.json> --output-dir <output>
$PY -m proxy_function_gap.pool_cli pool-preflight --manifest <pool_manifest.json>
$PY -m proxy_function_gap.pool_cli production-grid --manifest <pool_manifest.json> --inference-config configs/ensemble_inference_v2.json --v-artifacts <v_artifacts> --output <selection.json> [--actual-cost-budget 48]
$PY -m proxy_function_gap.pool_cli q-decisions --manifest <dev_or_calibration_pool_manifest.json> --selection <selection.json> --q-artifacts <q_artifacts> --output <q_decisions.json>
$PY -m proxy_function_gap.pool_cli synthetic-grid --manifest <synthetic_pool_manifest.json> --output <selection.json>
$PY -m proxy_function_gap.pool_cli fit-v --dev <dev_pool_a.json ...> --calibration <calibration_pool_a.json ...> --output-dir <v_artifacts>
$PY -m proxy_function_gap.pool_cli fit-q --selection <selection_a.json ...> --dev <dev_pool_a.json ...> --calibration <calibration_pool_a.json ...> --output-dir <q_artifacts>
```

## Remaining dependencies

Q calibration and the final frozen configuration are required before E1. No outer execution is authorized by these entry points. Development proxy, OOF, calibration or selection metrics are not final scientific conclusions.
