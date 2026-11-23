# Sample 2019 Gate 0 data builder

This staging repository builds file-level Gate 0 artifacts without model training, model inference, outer-result access, or E1 scientific evaluation.

Requirements: Python 3.9 standard library only. Raw inputs are validated against `data_manifest.json` and are never modified.

```bash
cd /home/proxy-function-gap-data
/root/miniconda3/bin/python -m gate0_sample2019.inspect_formats --config configs/gate0_v1.json
/root/miniconda3/bin/python -m gate0_sample2019.build --config configs/gate0_v1.json
/root/miniconda3/bin/python -m unittest discover -s tests -v
/root/miniconda3/bin/python -m gate0_sample2019.verify --config configs/gate0_v1.json
# After verification, commit the staging repository, then record its commit:
/root/miniconda3/bin/python -m gate0_sample2019.finalize --config configs/gate0_v1.json
```

The output directory is created once and existing derived artifacts are never overwritten. Use a new output version for any changed configuration. Normalization evidence is stored in `length_format_validation.json`.

## Artifact contract

- `sequence_index.tsv.gz`: one row per raw CSV record, with raw/normalized sequences, both hashes, raw row ID, source metadata, trajectory/round, replicate and split-group IDs.
- `endpoint_coverage.tsv.gz`: measured/predicted/missing states. `rl` is the measured endpoint when finite; predicted MRL remains empty with `not_available_raw` because raw step predictions were not published.
- `inclusion_exclusion.tsv.gz`: label-eligibility and trajectory-replay eligibility. All 3,193 empty `rl` rows are excluded with `missing_rl_total_zero`; no zero or prediction imputation occurs.
- `unique_sequences.tsv.gz`: exact normalized sequence groups and their raw-format, source-file, replicate and trajectory relationships.
- `split_ready_groups.tsv.gz`: proposed Hamming-1 components, not assigned train/query/outer partitions and not frozen.
- `near_duplicate_edges.tsv.gz` and `overlap_report.json`: verified distance-1 edges and raw/normalized cross-file exact overlap.
- `replicate_summary.json`: replicate-level coverage and the non-independence rule.
- `verification_round*.json`, `negative_leakage_tests.json`, `verification_summary.json`: independent full-data checks and deliberately invalid partition tests.

`raw_row_id` is the GSM accession plus the one-based source row number; the possibly duplicated original CSV index remains in `raw_index`. `raw_row_sha256` is SHA256 of the raw CSV dictionary serialized as UTF-8 JSON with sorted keys and compact separators. `sequence_hash` is SHA256 of the normalized sequence; `sequence_id` is `sample2019_` plus that hash. Raw sequence and all raw metadata remain available for audit.

The Hamming-1 candidate rule is exact: two equal-length sequences at distance one must share either their complete left half or complete right half. Every candidate pair is checked over the full sequence. Variable-length sequences are compared only at equal length; no cross-length similarity is inferred.

The original designed-library CSV index is not unique (46,988 repeated-index rows); it is preserved in `raw_index` and independently counted in round 1. Only the source-row-number-based `raw_row_id` is used as the unique row key.

After `finalize`, `gate0_sample2019.validate_delivery` independently rechecks every derived checksum, every raw input hash, the staging commit and the no-E1/no-training flags, and writes `final_validation.json`.

## Frozen split and label-vault protocol

`configs/split_v1.json` defines `sample2019-split-v1`. Allocation units join near-duplicate components with every component belonging to the same complete trajectory. Role assignment is deterministic within random/designed source-library strata and mixed random/designed overlap strata.

```bash
/root/miniconda3/bin/python -m gate0_sample2019.freeze_split --config configs/split_v1.json
/root/miniconda3/bin/python -m gate0_sample2019.verify_split --config configs/split_v1.json
# After all three rounds pass, commit staging and run:
/root/miniconda3/bin/python -m gate0_sample2019.finalize_split --config configs/split_v1.json
/root/miniconda3/bin/python -m gate0_sample2019.validate_split_delivery --config configs/split_v1.json
```

The official model's first 20,000 high-read rows are `official_seen_test`, not a new blind test. The next 260,000 are official-seen training rows. Rows tied at either cutoff (`total_reads` 2497 or 267) and their complete allocation units are `official_boundary_ambiguous` and can never become outer.

Learner label access is restricted to explicit role/column/path allowlists. Query features are unlabeled and query labels require a selection-hash- and time-checked reveal. Outer features and labels are Evaluator-only, and the outer evaluation method is disabled. Every reveal uses a new run directory; repeating a selection requires a new run ID and preserves earlier run artifacts. No outer regret or outer per-sequence labels are returned by these protocols.

For main-repository integration, use `integration_manifest.json` in the split output directory. It pins the staging commit, required code files, split/role manifests and protocol entrypoints. `split_delivery.json` records the gate status and remaining blockers; `delivery_checksums.json` covers all derived role views and protocol test artifacts.

## Model-specific split v2

`configs/split_v2.json` creates an immutable `sample2019-split-v2`; v1 is never overwritten. Original and retrained evolution models have separate role manifests, row-level assignments, historical outer permissions and E1 decisions. The original fixed-pool E1 context is only `eGFP|unmod|designed_single_assay`; step sequences are complete-trajectory historical measured-state replay only. Retrained evolution remains NO-GO while the 2695 retraining sequence list is unresolved.

```bash
/root/miniconda3/bin/python -m gate0_sample2019.freeze_split_v2 --config configs/split_v2.json
/root/miniconda3/bin/python -m gate0_sample2019.verify_split_v2 --config configs/split_v2.json
# After all three rounds pass, commit staging and run:
/root/miniconda3/bin/python -m gate0_sample2019.finalize_split_v2 --config configs/split_v2.json
/root/miniconda3/bin/python -m gate0_sample2019.validate_split_v2_delivery --config configs/split_v2.json
```

Target/step `designed=False` observations and their full allocation units are control-only. The first 40,000 scaler rows and linked components, official train/test/boundary rows and all near-duplicate components remain isolated. Replicate 2, modified RNA and mCherry are independent contexts and are not pooled with eGFP unmodified replicate 1.

Control assignment takes precedence over official train/test/boundary roles for any allocation unit containing target/step `designed=False`; official exposure remains recorded in `official_model_role` and `exposure_flags`, and the unit still cannot become outer. If all step trajectories are excluded by training/scaler/control lineage, `original_Paper1_historical_replay_authorized` is false rather than inventing a historical outer pool.

## Fresh proxy data packs

`configs/fresh_proxy_v1.json` builds deterministic train/dev/calibration/query packages without opening outer label vaults or the label-bearing total sequence index. Only eGFP unmodified replicate 1 and the designed single assay are included. Rep2, modified RNA and mCherry remain separate and are not packed. Query has no measured-label column and no outer Learner package is produced. Train-only scaling statistics are deferred to the trainer.

```bash
/root/miniconda3/bin/python -m gate0_sample2019.fresh_proxy_pack --config configs/fresh_proxy_v1.json
/root/miniconda3/bin/python -m gate0_sample2019.verify_fresh_proxy --config configs/fresh_proxy_v1.json
# After all three rounds pass, commit staging and run:
/root/miniconda3/bin/python -m gate0_sample2019.finalize_fresh_proxy --config configs/fresh_proxy_v1.json
/root/miniconda3/bin/python -m gate0_sample2019.validate_fresh_proxy_delivery --config configs/fresh_proxy_v1.json
```

## fresh_proxy_cnn_v1 conditional child manifest

`configs/fresh_cnn_v1.json` builds an immutable fresh-proxy child lineage without official weights or distillation. Train/dev/calibration/query come only from the frozen fresh pack. Outer strata are independently reviewed from v1 random outer and the v2 original 3163 designed candidates; unmodified rep1 random, target/step designed=True, official-seen, control and quarantine are not relaxed into fresh outer. Random outer is external-context only and never enters the main E1 pool.

```bash
/root/miniconda3/bin/python -m gate0_sample2019.fresh_cnn_child --config configs/fresh_cnn_v1.json
/root/miniconda3/bin/python -m gate0_sample2019.verify_fresh_cnn --config configs/fresh_cnn_v1.json
# After all three rounds pass, commit staging and run:
/root/miniconda3/bin/python -m gate0_sample2019.finalize_fresh_cnn --config configs/fresh_cnn_v1.json
/root/miniconda3/bin/python -m gate0_sample2019.validate_fresh_cnn_delivery --config configs/fresh_cnn_v1.json
```

A verified child can authorize training start, but `E1_enabled` remains false. Model hash, epoch, convergence, V/Q calibration and configuration freeze are null until real evidence exists. Outer is always historical measured-state, never never-evaluated.

## fresh_proxy_cnn_v2 and production NPZ

`configs/fresh_cnn_v2.json` independently re-reviews v1 random outer without inheriting blanket original/retrained random exclusions. Fresh unmodified rep1 is restored when its allocation unit, near component and trajectory do not overlap fresh train/scaler/dev/calibration/query or designed outer. Original/retrained manifests are unchanged.

`data_manifest.json` and `production/*.npz` use deterministic standard-library NPY/NPZ encoding. NPZ row_id order is exactly the source TSV order and includes UTR50 one-hot, measured_rl only for labeled roles, role/allocation/context/source hashes and `synthetic=false`. Query has no measured-label array. Production outer NPZ remains Evaluator-only. Train-only scaler statistics remain deferred to the trainer.

## Immutable fixed candidate pools

`configs/fixed_pools_v1.json` builds one fixed pool per role: dev=64, calibration=64, outer random=256, outer designed=64. Each complete allocation unit is an independent cluster; sequence/context deduplication, q=1/N unique feasible sequences, cluster assignments, source split hash and paired proposal seed plan are explicit. Outer learner views contain no labels and Evaluator vaults are references only. Small pools are protocol fixtures, not independent replication series; no variance or outer score is computed.

## Disjoint multi-pool v2

`configs/fixed_pools_v2.json` creates immutable `fresh-proxy-fixed-pools-v2` without modifying v1. It builds 32 dev, 16 calibration, 4 outer-random and 16 outer-designed pools. Every pool retains its configured unique-sequence size; sequence, Hamming-1 component, allocation-unit and trajectory conflicts are blocked both across roles and across pools within a role. These are disjoint same-study protocol units, not independent studies or external replications. Outer scoring remains disabled.

```bash
/root/miniconda3/bin/python -m gate0_sample2019.fixed_pools --config configs/fixed_pools_v2.json
/root/miniconda3/bin/python -m gate0_sample2019.verify_fixed_pools --config configs/fixed_pools_v2.json
# Run verification in three logged rounds, commit staging, then finalize and validate:
/root/miniconda3/bin/python -m gate0_sample2019.finalize_fixed_pools --config configs/fixed_pools_v2.json
/root/miniconda3/bin/python -m gate0_sample2019.validate_fixed_pools_delivery --config configs/fixed_pools_v2.json
```
