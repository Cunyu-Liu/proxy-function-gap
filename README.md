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
