# Gate 0 IDs and Role-Safe Views Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Align prediction rows with Gate 0 identifiers and generate model-specific Learner and Evaluator views without exposing experimental labels or published target predictions to Learner.

**Architecture:** Compute Gate 0 raw row IDs during the immutable GEO scan and independently verify every selected row against Gate 0 `sequence_index.tsv.gz`. A tracked model-role manifest controls per-model view metadata and blind claims. Separate original and retrained views use a strict Learner column allowlist; full labels remain only in Evaluator views.

**Tech Stack:** Existing Python 3.10, PyTorch 2.7.1+cu128, standard-library CSV/JSON, deterministic CUDA inference.

---

### Batch 1: Gate 0 identity and row assertions

**Files:**
- Modify: `/home/proxy-function-gap-sample-runtime/src/sample_runtime/provenance.py`
- Modify: `/home/proxy-function-gap-sample-runtime/src/sample_runtime/data.py`
- Test: `/home/proxy-function-gap-sample-runtime/tests/test_data.py`

**Risk:** High — incorrect source row numbering breaks split and label-vault joins.

**Implementation:** Add `sample2019_` sequence IDs, sequence hashes, GSM row IDs, source row numbers, and raw indices. Assert every row against Gate 0 `sequence_index.tsv.gz` while preserving `(library,id,sparse_step,utr)`.

**Minimum verification:** Full selected-row join and deliberate wrong-ID rejection.

**Independent review:** Yes — final source-to-Gate 0 reconciliation.

### Batch 2: Model roles and permission views

**Files:**
- Create: `/home/proxy-function-gap-sample-runtime/configs/model_roles_v1.json`
- Create: `/home/proxy-function-gap-sample-runtime/src/sample_runtime/views.py`
- Modify: `/home/proxy-function-gap-sample-runtime/src/sample_runtime/run.py`
- Test: `/home/proxy-function-gap-sample-runtime/tests/test_views.py`

**Risk:** High — measured labels or published predictions could become blind features.

**Implementation:** Generate original/retrained, step/target, Learner/Evaluator views. Learner fields are limited to IDs, sequence, design metadata, and one model output. Require explicit model roles; reject blind claims for unknown retrain lineage. Keep measured MRL, read depth, and public info2 exclusively in Evaluator views.

**Minimum verification:** Permission negative tests for measured_rl, total_reads, published_pred_info2, missing roles, and unknown-lineage blind claims.

**Independent review:** Yes — final permission and manifest audit.

### Batch 3: CUDA reruns and reproducible delivery

**Files:**
- Modify: `/home/proxy-function-gap-sample-runtime/src/sample_runtime/validation.py`
- Modify: `/home/proxy-function-gap-sample-runtime/README.md`

**Risk:** High — prior artifacts do not satisfy the new identity and permission contract.

**Implementation:** Run the complete pipeline twice on CUDA after committing a clean code state. Repeat device/weight, target/independent-sample, Gate 0 ID/row, permission-negative-test, and deterministic-hash checks.

**Minimum verification:** All unit tests, two full CUDA runs, independent source and Gate 0 reconciliation, and byte comparison of every deterministic artifact including views.

**Independent review:** Yes — final delivery review.
