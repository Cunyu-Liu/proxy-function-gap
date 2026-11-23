# Sample 2019 CUDA Runtime Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Recompute original and retrained Sample 2019 evolution-model predictions for all connectable step and target sequences with an independent, reproducible PyTorch CUDA runtime.

**Architecture:** Load the two official HDF5 files read-only and map their tensors into an independently implemented two-convolution PyTorch module. Run both models in eval mode on the same encoded batch on one explicitly selected CUDA device, then join predictions to immutable GEO metadata without changing the source data. The runtime contains no training path and no CPU inference fallback.

**Tech Stack:** Python 3.10.22, PyTorch 2.7.1+cu128, h5py 3.14.0, NumPy 2.2.6, CUDA 12.8, float32 with TF32 disabled.

---

### Batch 1: Runtime and immutable provenance

**Files:**
- Create: `/home/proxy-function-gap-sample-runtime/src/sample_runtime/model.py`
- Create: `/home/proxy-function-gap-sample-runtime/src/sample_runtime/provenance.py`
- Test: `/home/proxy-function-gap-sample-runtime/tests/test_model.py`

**Risk:** High — legacy Keras tensor layouts and CUDA determinism directly affect every prediction.

**Implementation:** Implement only the verified 54-by-4 input, two valid Conv1d layers, flatten, dense, ReLU, and linear output. Load original and retrained HDF5 tensors with explicit path and SHA-256 checks. Require CUDA, call `model.cuda()` and `input.cuda()`, disable TF32, and stop on any CUDA error. Do not import or copy official GPL Python code.

**Minimum verification:** Check tensor shapes, model eval mode, requested/actual device, weight hashes, and a small CUDA forward against an independent NumPy forward.

**Independent review:** Yes — final review of tensor layout and absence of CPU fallback.

### Batch 2: Data connection and artifacts

**Files:**
- Create: `/home/proxy-function-gap-sample-runtime/src/sample_runtime/data.py`
- Create: `/home/proxy-function-gap-sample-runtime/src/sample_runtime/run.py`
- Test: `/home/proxy-function-gap-sample-runtime/tests/test_data.py`

**Risk:** High — library-specific `id/info1/info2` semantics and designed flags can silently create false labels.

**Implementation:** Read the two GEO gzip files read-only. Select exactly 5,710 step and 7,106 target `designed=True` rows. Assign step `sparse_step=info2` and `generation=info1`; retain target `info2` only as `published_pred_info2`. Use the frozen first-40k scaler for both models. Write separate measured and predicted columns, run metadata, target validation, and deterministic CSV artifacts to a new teams directory.

**Minimum verification:** Assert one-to-one keys, exact row counts, unique sequence keys, 50 nt inputs, no null predictions, and no measured values in prediction derivation.

**Independent review:** Yes — final review of connection keys and measured/predicted separation.

### Batch 3: Three-round verification and reproducibility

**Files:**
- Create: `/home/proxy-function-gap-sample-runtime/src/sample_runtime/validate.py`
- Create: `/home/proxy-function-gap-sample-runtime/README.md`
- Test: `/home/proxy-function-gap-sample-runtime/tests/test_model.py`
- Test: `/home/proxy-function-gap-sample-runtime/tests/test_data.py`

**Risk:** High — a successful forward alone does not establish numerical equivalence or deterministic delivery.

**Implementation:** Round 1 records CUDA device, versions, dtype, model eval state, and weight hashes. Round 2 checks all target predictions against public `info2` with the independently recovered affine mapping and checks deterministic samples against NumPy float64. Round 3 asserts one-to-one joins and row conservation, then runs the complete pipeline twice and compares prediction and metadata hashes. Any CUDA or validation failure exits nonzero without a fallback artifact.

**Minimum verification:** Run unit tests, the three validators, and two complete CUDA pipelines on the deliverable itself; inspect final Git status and artifact hashes.

**Independent review:** Yes — final milestone review of evidence and remaining Gate 0 blockers.
