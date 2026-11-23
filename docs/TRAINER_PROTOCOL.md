# Fresh CUDA proxy trainer protocol

## Scope

The trainer is an independent fresh Sample-like 50nt CNN regression implementation. It does not load official weights or copy official GPL code. Formal training is not launched during integration. Synthetic smoke checks are software checks only and always record `scientific_result=false`.

## Roles, permissions, and hashes

`data_manifest.json` must pin the pack, split-v2 manifest, and Learner allowlist by path and SHA-256. Preflight requires:

- `split_version=sample2019-split-v2`, matching `split_hash`, and `model_lineage=original_evolution`.
- split-v2 role counts for train/dev/calibration, with outer metrics disabled.
- Learner feature and label allowlists containing exactly the required development roles and no outer role; dev/calibration parameter origin is allowed.
- Exactly x/y arrays for train/dev/calibration. Inputs are finite one-hot `(n,4,50)` arrays and labels are finite vectors with matching counts.
- Actual pack SHA-256 before preflight and again in every worker before use.

`training_authorized=true` and `synthetic=false` are both required by the parallel formal training engine. The CLI additionally requires `train --execute-training`. Preflight success alone does not authorize training.

## Model and train-only scaler

`FreshSampleCNN` uses channels-first 4x50 one-hot input, two Conv1d/ReLU blocks, pooling, and a small fully connected regression head. Each seed initializes a fresh model. The target scaler is mean and population standard deviation fitted only on train labels; dev/calibration never affect it. Constant train targets use scale 1 and record raw_std 0.

## CUDA and parallel execution

Every worker calls `configure_cuda` before model/data allocation. It requires CUDA, an existing device index, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and an explicit per-process memory fraction. There is no CPU training fallback. The default three seeds are 11, 23, 37, each with fraction 0.2; custom runs require at least three unique seeds, one fraction per process, and fractions summing to at most 1.

Parallel execution uses spawn and one ProcessPoolExecutor worker per seed on the same explicit CUDA device. GPU name, capability, requested/actual device, actual dtype, deterministic/TF32 state, and actual `max_memory_allocated` are recorded. No GPU-utilization percentage is inferred or fabricated.

## Convergence and history

There is no maximum epoch, fixed step limit, or accepted `max_epochs/max_steps` config. Each epoch evaluates full train and dev normalized MSE. An improvement is significant only when the corresponding absolute best loss decreases by more than `min_delta`. The plateau counter resets when either train or dev improves significantly; only simultaneous non-significant changes increment it. Training stops only when this counter reaches `patience`.

Every epoch appends train/dev loss, best dev, best epoch, plateau state, significant-improvement flags, epoch seconds, and cumulative seconds to `history.jsonl` with fsync. `latest.pt` is written every epoch; `best.pt` is updated whenever dev reaches a new absolute best. Both use temporary-file + fsync + atomic `os.replace` + directory fsync.

The checkpoint stores model, optimizer, train-only scaler, plateau state, complete history, best model state, seed, and data hash. Resume requires every seed's checkpoint, verifies seed/data/scaler identity, restores all state, preserves prior history, and writes to a new run directory. Final model weights are restored from best. Summaries include initial/best model hash, data/split/allowlist hashes, best epoch, total epochs, total training time, actual GPU/dtype, actual peak memory, and history/checkpoint hashes.

## Commands

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/home/proxy-function-gap-sample-runtime/.deps:src:gate0:runtime/src
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
$PY -m proxy_function_gap.trainer.cli preflight --config configs/trainer.example.json
$PY -m proxy_function_gap.trainer.cli cuda-self-check --config configs/trainer.example.json
# Formal training is not authorized by integration:
$PY -m proxy_function_gap.trainer.cli train --config configs/trainer.example.json --execute-training
```

`make-synthetic-pack` creates only a tiny test pack with `training_authorized=false` and `synthetic=true`. It cannot be used by the formal parallel training engine.

## Integration verification

Three complete rounds of E0/Gate0/split-v2/Sample-runtime/trainer pytest each passed 175 tests. The integrated source/data-boundary audit and E0 independent math check passed in every round. Wheel inspection contains only the intended Python packages and no raw data or official weights.

The real CLI preflight verified a synthetic pack, split manifest, Learner allowlist, all hashes, and role counts, while returning `ready_for_formal_training=false`. Three spawned CUDA self-check processes passed on one NVIDIA GeForce RTX 5090 with device `cuda:0`, dtype `torch.float32`, seeds 11/23/37, and per-process fraction 0.05. Each record includes actual device, allocated memory, and `scientific_result=false`.

A synthetic request with `--execute-training` was rejected by the formal-training permission check before any training worker was launched. No fresh formal proxy training was started during integration.
