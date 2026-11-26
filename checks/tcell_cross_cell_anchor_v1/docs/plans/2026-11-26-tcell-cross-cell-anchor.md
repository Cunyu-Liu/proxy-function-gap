# Paired Cross-cell Anchor Implementation Plan

> **Execution policy:** Implement by functional batch under the global lightweight development and verification strategy. Do not apply task-by-task review or full-validation gates unless the batch risk warrants them.

**Goal:** Test whether same-sequence measured HepG2 MRL improves budget-matched T-cell MRL selection over frozen Sample-CNN scores in repeat-linked retrospective pools.

**Architecture:** A label-blind preparation stage reads only T-cell sequence keys, exact-matches measured HepG2 labels, removes Sample train/dev/calibration exact/H1 neighbors, and creates 16 H1-isolated pools of 64 sequences shared by both repeats. A separately gated one-shot evaluator reads T-cell labels and compares frozen CNN, HepG2 anchor, fixed 50/50 within-pool average-rank fusion, and Random(N) on shared nested proposals. Independent verification recomputes the scientific outputs without importing the implementation.

**Tech Stack:** Python, NumPy, PyTorch CUDA `cuda:0`, gzip/csv, hashlib, pytest.

---

### Batch 1: Frozen protocol and label-blind preparation

**Files:**
- Create: `/home/proxy-function-gap-tcell-anchor/configs/anchor_frozen.json`
- Create: `/home/proxy-function-gap-tcell-anchor/anchor.py`
- Create: `/home/proxy-function-gap-tcell-anchor/prepare.py`
- Test: `/home/proxy-function-gap-tcell-anchor/tests/test_anchor.py`

**Risk:** High — eligibility or H1 isolation errors would invalidate the experiment.

**Implementation:** Freeze raw hashes, three CNN checkpoints, deterministic sequence ordering, five proposal seeds, all caps, tie rules, primary success criteria, secondary estimands and bootstrap before reading target labels. Preparation must fail closed on raw/count/eligibility/isolation mismatch and must prove that no T-cell `rl` field is accessed.

**Minimum verification:** Synthetic exact/H1/rank/proposal/regret/bootstrap/Holm tests; full raw hash, eligible count, Sample exclusion and cross-pool H1 audit; CUDA inference manifest.

**Independent review:** No separate agent is authorized; an independent recomputation is required in Batch 3.

### Batch 2: One-shot target evaluation

**Files:**
- Create: `/home/proxy-function-gap-tcell-anchor/evaluate.py`
- Create: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_cross_cell_anchor_v1/target/`

**Risk:** High — target labels can be evaluated only once and cannot drive protocol revisions.

**Implementation:** Require the frozen preflight manifest and matching config hash. Evaluate cap 1/2/4/8/16 for CNN, HepG2 anchor, fixed fusion and Random(N), then compute repeat-specific Holm-adjusted bootstrap tests, paired repeat-linked primary CI, turnover, tail flips, Spearman and oracle opportunity loss. No post-target callback or weight tuning.

**Minimum verification:** Exercise the exact evaluation pipeline on synthetic fixtures before the real call; use a fail-closed one-shot sentinel and preserve all outputs.

**Independent review:** Yes — independent numeric replay in Batch 3, because this is the sole formal target evaluation.

### Batch 3: Three-round verification and sanitized delivery

**Files:**
- Create: `/home/proxy-function-gap-tcell-anchor/independent_verify.py`
- Create: `/home/proxy-function-gap-tcell-anchor/delivery_verify.py`
- Create: `/home/proxy-function-gap-tcell-anchor/export_delivery.py`
- Create: `deliverables/reports/source_data/tcell_cross_cell_anchor_v1/`
- Create: `TCELL_CROSS_CELL_ANCHOR_V1.md`

**Risk:** High — delivery must preserve negative results, exact numeric fidelity and the retrospective/paired evidence boundary.

**Implementation:** Round 1 verifies implementation and pre-target isolation; Round 2 independently recomputes pools, selections, effects, CIs, Holm and secondary summaries; Round 3 reruns tests in a clean process, checks hashes and audits local exports for sequence strings or sequence-level values. Sync only configs, aggregate statistics, code, logs and verification.

**Minimum verification:** Three real rounds must pass; report any retained failed attempt. Do not push code or alter historical artifact directories.

**Independent review:** Yes — Round 2 uses a standalone implementation and Round 3 audits the actual exported deliverable.
