# Independent Sample 2019 CUDA Runtime

This independent staging recomputes predictions from the official Sample 2019 evolution-model HDF5 weights. It does not copy the official GPL Python code, train a model, modify the main repository, or provide a CPU inference fallback.

## Provenance

- Official repository: `https://github.com/pjsample/human_5utr_modeling`
- Official commit: `d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe`
- Original weights: `/home/proxy-function-gap-sample-audit/official-d53df410/modeling/saved_models/evolution_model.hdf5`
- Retrained weights: `/home/proxy-function-gap-sample-audit/official-d53df410/modeling/saved_models/retrained_evolution_model.hdf5`
- GEO data remain read-only under `/home/proxy-function-gap-sample-audit/data/`.

The PyTorch implementation maps the verified 54-by-4 Keras input to channels-first Conv1d tensors and restores Keras position-major flatten order before the dense layer. Both models run in eval mode on the same CUDA input batch with float32, deterministic algorithms, and TF32 disabled.

## Environment and execution

The staging-private `.deps` directory contains only `h5py==3.14.0`; the confirmed interpreter supplies `torch==2.7.1+cu128` and `numpy==2.2.6`.

```bash
cd /home/proxy-function-gap-sample-runtime
PYTHONPATH=.deps:src CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /root/miniconda3/envs/lucaone_cu128/bin/python -m unittest discover -s tests -v

PYTHONPATH=.deps:src CUBLAS_WORKSPACE_CONFIG=:4096:8 \
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
- `artifact_hashes.json`: deterministic hashes for the two prediction tables, provenance, and validation.
- `run_metadata.json`: variable timestamps and command metadata, excluded from double-run hash equality.

`measured_rl` is experimental. `original_raw_output` and `retrained_raw_output` are standardized model outputs. `original_pred_mrl` and `retrained_pred_mrl` use only the frozen eGFP first-40k scaler (`mean=5.993167213121654`, `scale=1.799413065189219`, `ddof=0`). Target `published_pred_info2` is a published prediction used only for numerical validation, never as a measured label or fitted scaler.
