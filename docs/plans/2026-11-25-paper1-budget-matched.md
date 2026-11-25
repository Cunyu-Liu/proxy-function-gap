# Paper 1 Budget-Matched Random(N) Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Reconstruct frozen strict-H1 nested proposal prefixes and quantify equal-budget proxy-ranking value and Random candidate-budget effects without model execution or changes to old results.

**Architecture:** A standalone CPU analysis reads remote frozen pools, CNN scores, selections, evaluations, and labels. A separate verifier reimplements selection and statistics rather than importing the producer; a delivery verifier checks hashes and the de-identified local export.

**Tech Stack:** Python 3, NumPy PCG64/SeedSequence, pytest, JSON/TSV, SSH/scp.

---

### Batch 1: Frozen protocol and raw reconstruction

**Files:**
- Create: `/home/proxy-function-gap-paper1-budget/docs/plans/2026-11-25-paper1-budget-matched.md`
- Create: `/home/proxy-function-gap-paper1-budget/checks/paper1_budget_matched_v1/frozen_config.json`
- Create: `/home/proxy-function-gap-paper1-budget/checks/paper1_budget_matched_v1/run_analysis.py`
- Test: `/home/proxy-function-gap-paper1-budget/tests/paper1_budget_matched_v1/test_budget_matched.py`

**Risk:** High — proposal-stream or tie-breaking drift would invalidate pairing.

**Implementation:** Freeze all contexts, caps, seeds, PCG64 spawn keys, estimands, and three all-20 families before seeing new estimates. Reconstruct prefixes and original selections; fail unless all 1,600 original pairs (3,200 selection records) agree on ID/index/regret. Generate Random(N) from the same prefix and calculate regrets.

**Minimum verification:** Synthetic nested-prefix, tie, PCG64, regret-identity, structural-zero, Holm, and Bonferroni tests; retain the pre-implementation failing test log and round-1 raw audit.

**Independent review:** Yes — round 2 separately reconstructs every raw selection and regret.

### Batch 2: Pool-cluster statistics and artifact generation

**Files:**
- Create: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_budget_matched_v1/analysis/paired_records.tsv`
- Create: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_budget_matched_v1/analysis/pool_effects.tsv`
- Create: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_budget_matched_v1/analysis/results.json`

**Risk:** High — cluster resampling and multiplicity determine interpretation.

**Implementation:** Average five seeds within each pool; run 10,000 complete-pool bootstrap draws; retain nominal and all-20 adjusted intervals and Holm/Bonferroni p-values for all three 20-row families. Export full remote ID/regret audits and de-identified local aggregates. Include every negative result.

**Minimum verification:** Round-1 summary identities, counts, means, and intervals; deterministic regeneration check.

**Independent review:** Yes — independent implementation must exactly reproduce delivered numerical values.

### Batch 3: Three-round validation and local delivery

**Files:**
- Create: `/home/proxy-function-gap-paper1-budget/checks/paper1_budget_matched_v1/verify_independent.py`
- Create: `/home/proxy-function-gap-paper1-budget/checks/paper1_budget_matched_v1/verify_delivery.py`
- Create: `deliverables/reports/source_data/paper1_budget_matched_v1/`
- Create: `deliverables/reports/docs/PAPER1_BUDGET_MATCHED_V1.md`

**Risk:** High — a partial or stale sync could misstate evidence.

**Implementation:** Execute raw/independent/delivery rounds, retain all failures, synchronize only aggregate source data, and report T-cell r1/r2 plus all caps in other contexts. Label all computation CPU-only and all evidence retrospective sensitivity.

**Minimum verification:** Three successful round logs in addition to retained failure logs; final tests, independent exact comparison, remote/local SHA256 manifest, and report-value audit.

**Independent review:** Yes — round 3 validates the exact synced deliverables rather than only source imports.
