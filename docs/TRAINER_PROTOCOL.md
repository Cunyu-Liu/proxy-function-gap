# Fresh CUDA proxy trainer protocol

## Scope

The trainer is an independent fresh Sample-like 50nt CNN regression implementation. It does not load official weights or copy official GPL code. Formal training is not launched during development. Synthetic smoke is software validation only and always records `scientific_result=false`.

## Production pack and preflight

A production data pack consists of an NPZ and `data_manifest.json`. The NPZ contains exactly `x_role`, `y_role`, and `row_id_role` for train/dev/calibration. Inputs are finite one-hot `(n,4,50)` arrays, labels are finite vectors, and row IDs are non-empty unique strings.

The manifest must pin:

- `pack`, `split_manifest`, and `learner_allowlist` by path and SHA-256.
- `split_version=sample2019-split-v2`, matching `split_hash`, and `model_lineage=original_evolution`.
- `role_sources`: each role's fresh TSV path, SHA-256, and `row_id_column` (default `row_id`). Label allowlist paths must equal these TSV paths; no outer role is allowed.
- `row_order_sha256`: each role's canonical order hash, defined as SHA-256 of the sorted unique row IDs joined by newline, including the trailing newline.

Preflight requires exact equality of NPZ and fresh TSV row-ID sets and unique counts. Different physical row order is allowed only when the canonical sorted-ID hash is identical. Pack counts must match the manifest and may not exceed split-v2 role counts. Source hashes are verified independently. Worker loading repeats pack and per-role order-hash checks.

Formal parallel training requires `synthetic=false` and `training_authorized=true`. The CLI also requires `train --execute-training`. Preflight success alone does not authorize training. Synthetic uses the same schema with `synthetic=true` and `training_authorized=false` and can run only through the explicitly named smoke engine.

## Model, scaler, and CUDA

`FreshSampleCNN` uses channels-first 4x50 input, two Conv1d/ReLU blocks, pooling, and a small regression head. Each seed initializes a fresh model. Target mean and population standard deviation are fitted only on train labels; constant train targets use scale 1 and record raw_std 0.

Every worker requires CUDA, a valid device index, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and an explicit per-process memory fraction before model/data allocation. There is no CPU training fallback. At least three unique seeds and one fraction per seed are required, with fractions summing to at most 1. Spawn workers share the specified single GPU. Actual GPU, dtype, deterministic/TF32 state, and `max_memory_allocated` are recorded; utilization is never fabricated.

## Eval-based convergence and per-epoch identity

There is no maximum epoch or fixed-step limit. Each epoch records online batch-training loss separately as `train_online_loss`. After optimization, the model is switched to `eval()` and the same frozen weights compute full-train and full-dev normalized MSE: `train_eval_loss` and `dev_eval_loss`. Only these two eval losses drive convergence.

Significant improvement requires a decrease greater than `min_delta` from the corresponding absolute best. Plateau resets for significant train/dev improvement and **always resets for a dev absolute best**, even when the improvement is smaller than `min_delta`. Thus a continuing sequence of small absolute dev improvements cannot stop training. Only simultaneous lack of significant improvement without dev absolute best increments plateau, and only reaching `patience` stops training.

Every history record includes train_online/train_eval/dev_eval losses, best dev/epoch, plateau state, pack SHA-256, per-role canonical order hashes, effective-config SHA-256, model-state SHA-256, and both `shuffle_seed=seed+epoch` and `shuffle_epoch=epoch`. Batch order therefore remains auditable and is preserved by resume.

## Checkpoint and strict resume

`latest.pt` is atomically updated every epoch with `status=running`; `best.pt` is atomically updated on every dev absolute best. At convergence, latest is rewritten with `status=completed`. A completed checkpoint is rejected for resume. A best checkpoint remains a running snapshot captured at its epoch and may be used when its identity matches.

Every checkpoint contains model/optimizer state, true initial model hash from the original run, model-definition hash, trainer code-file identity, effective-config hash, train-only scaler, plateau/history, best model state, pack/source/split/allowlist hashes, and per-role order hashes. `initial_model_hash` is never replaced by a fresh initialization hash during resume.

Resume configuration must name every seed's checkpoint and its expected SHA256:

```json
{
  "resume_from": {
    "11": {"path": "/teams/.../seed_11/best.pt", "sha256": "<64-lowercase-hex>"},
    "23": {"path": "/teams/.../seed_23/best.pt", "sha256": "<64-lowercase-hex>"},
    "37": {"path": "/teams/.../seed_37/best.pt", "sha256": "<64-lowercase-hex>"}
  }
}
```

Before loading, the actual checkpoint SHA256 must match. Resume then rejects completed status or any mismatch in seed, code identity, model definition, effective config, scaler, pack, source, split, allowlist, or row-order identity. All state and history are restored and written to a new run directory. Final model weights are restored from best.

## Production adapter and formal-run provenance

`trainer.production.prepare_production_pack` accepts only an authorized fresh child v2 selected build. It verifies the root child/integration/verification/final-validation chain, parent pack/split/raw manifests, learner allowlist, per-role production NPZ and fresh TSV hashes, sequence hashes, role fields, and exact physical row order before writing a unified training NPZ and manifest. Outer NPZ/TSV files are never read by the adapter.

The formal engine additionally requires a clean main Git repository. Each summary records the actual code commit and a frozen-config SHA256 covering data/output paths, seeds, per-process fractions, device, dtype, batch size, learning rate, min_delta, patience, and resume references. Parallel workers are monitored as spawn processes: any exception or abnormal exit terminates and joins every other worker immediately. Partial history/checkpoints are preserved but are never marked converged.

## Commands

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/home/proxy-function-gap-sample-runtime/.deps:src:gate0:runtime/src
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
$PY -m proxy_function_gap.trainer.cli preflight --config configs/trainer.example.json
$PY -m proxy_function_gap.trainer.cli cuda-self-check --config configs/trainer.example.json
# Only the explicitly synthetic, unauthorized three-seed smoke:
$PY -m proxy_function_gap.trainer.cli synthetic-train-smoke --config <synthetic-config.json>
# Formal training remains unauthorized in this integration:
$PY -m proxy_function_gap.trainer.cli train --config configs/trainer.example.json --execute-training
```

`make-synthetic-pack` creates a tiny NPZ plus role TSVs and manifest. It cannot be used by the formal engine.

## Verification status

The preceding integration passed 175 tests in three rounds. The independent-review repair adds production row-ID/source/order checks, eval-based stopping, absolute-best behavior, strict resume identities, completed-checkpoint rejection, and three-seed synthetic training/resume coverage. Three new full pytest rounds passed 190, 191, and 191 tests respectively, each with three-process CUDA self-checks and source/data audits. The CLI three-seed training and expected-SHA256 resume smoke also passed. These are software checks, not scientific results. Formal training remains unauthorized and was not started.
