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
