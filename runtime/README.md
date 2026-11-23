# Independent Sample 2019 CUDA Runtime

This independent staging recomputes predictions from the official Sample 2019 evolution-model HDF5 weights. It does not copy the official GPL Python code, train a model, modify the main repository, or provide a CPU inference fallback.

## Provenance

Integrated source commit and subtree details are recorded in `PROVENANCE.md`. No official GPL Python code or HDF5 weights are copied into the integrated repository.

- Official repository: `https://github.com/pjsample/human_5utr_modeling`
- Official commit: `d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe`
- Original weights: `/home/proxy-function-gap-sample-audit/official-d53df410/modeling/saved_models/evolution_model.hdf5`
- Retrained weights: `/home/proxy-function-gap-sample-audit/official-d53df410/modeling/saved_models/retrained_evolution_model.hdf5`
- GEO data remain read-only under `/home/proxy-function-gap-sample-audit/data/`.

The PyTorch implementation maps the verified 54-by-4 Keras input to channels-first Conv1d tensors and restores Keras position-major flatten order before the dense layer. Both models run in eval mode on the same CUDA input batch with float32, deterministic algorithms, and TF32 disabled.

## Environment and execution

The staging-private `.deps` directory contains only `h5py==3.14.0`; the confirmed interpreter supplies `torch==2.7.1+cu128` and `numpy==2.2.6`.

```bash
cd /home/proxy-function-gap
PYTHONPATH=/home/proxy-function-gap-sample-runtime/.deps:src:runtime/src:gate0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /root/miniconda3/envs/lucaone_cu128/bin/python -m unittest discover -s runtime/tests -v

PYTHONPATH=/home/proxy-function-gap-sample-runtime/.deps:src:runtime/src:gate0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /root/miniconda3/envs/lucaone_cu128/bin/python -m sample_runtime.run \
  --requested-device cuda:0 --batch-size 128 \
  --output-dir /teams/HoulabData_1788502323/liucunyu/proxy_function_gap/sample_runtime/run1
```

CUDA initialization, model upload, input upload, or execution failure exits nonzero. There is no CPU model path. Moving final numeric outputs back for CSV serialization is not CPU inference.

## Outputs

Each successful run atomically publishes:

- `step_predictions.csv`: 5,710 designed step states with `(library,id,sparse_step,utr)` keys.
- `target_predictions.csv`: 7,106 designed target rows; `sparse_step` is empty because target `info2` is not a step index.
- `provenance.json`: code commit, official commit, weight/data hashes, scaler, requested/actual device, GPU, versions, and dtype.
- `validation.json`: target `info2` check, independent NumPy samples, and connection/row-conservation checks.
- `artifact_hashes.json`: deterministic hashes for prediction tables, model views, manifests, provenance, and validation.
- `run_metadata.json`: variable timestamps and command metadata, excluded from double-run hash equality.

`measured_rl` is experimental. `original_raw_output` and `retrained_raw_output` are standardized model outputs. `original_pred_mrl` and `retrained_pred_mrl` use only the frozen eGFP first-40k scaler (`mean=5.993167213121654`, `scale=1.799413065189219`, `ddof=0`). Target `published_pred_info2` is a published prediction used only for numerical validation, never as a measured label or fitted scaler.

## Gate 0 identity and model-specific permission views

Prediction rows use the same identifiers as Gate 0: `sequence_id=sample2019_<sha256(normalized_utr)>`, `sequence_hash=<sha256(normalized_utr)>`, and `raw_row_id=GSM3130443:row-<one-based-row-number>` with seven digits. Every row is asserted against Gate 0 `sequence_index.tsv.gz`; `(library,id,sparse_step,utr)` remains as the original connection key.

`configs/model_roles_v1.json` controls model-specific roles. The original model is the random-library source proxy. The retrained model has `unknown_retrain_membership` and `blind=false`; its field-safe Learner view is restricted to nonblind diagnostics and cannot support a blind claim. Missing roles and unknown-lineage blind claims are rejected.

Each run writes separate `step` and `target` Learner/Evaluator CSVs under `views/original/` and `views/retrained/`. Learner columns are restricted to Gate 0 IDs, the original connection metadata, UTR, allowed target-design metadata, and the model output for that view. `measured_rl`, `total_reads`, and `published_pred_info2` are Evaluator-only. The target public `info2` remains a numerical-validation input and is never a blind feature.

For split-v2, join model views and role assignments by `raw_row_id`, with `sequence_id` and the original connection key as assertions. `view_manifest.json` provides relative paths, exact columns, row counts, model lineage, blind status, and label-holder roles. Split assignment and label-vault state remain external to this runtime.
