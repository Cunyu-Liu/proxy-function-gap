# Fixed-pool manifest strict schema

## Status

The production pool manifest is not complete. This document is the fail-closed interface used by synthetic tests. No real pool regret, risk, or outer result is produced.

## Manifest fields

Required JSON fields:

- `schema_version`: exactly `1.0`.
- `pool_id`: unique pool identifier.
- `pool_role`: `dev` or `calibration`; outer is rejected.
- `split_version`: `sample2019-split-v2`; `split_hash` is required.
- `candidate_file`: `{path, sha256}` for the complete-candidate TSV.GZ.
- `prediction_manifest`: `{path, sha256}` for the three-member inference manifest.
- `label_source`: `{path, sha256}` for the allowed dev/calibration label TSV.GZ.
- `row_order_sha256`: sorted unique row IDs joined by newline with trailing newline, SHA256.
- `candidate_count`: exact unique row count.
- `synthetic`: explicit boolean.
- Optional `proposal_probs`: one non-negative probability per candidate, summing to 1; default uniform.
- Non-synthetic pools additionally require `source_pool_manifest`, `source_pool_file`, `source_pool_final_validation`, `vault_references`, `pool_hash`, `support_definition`, and `support_train_source`. These pin the immutable fixed-pool lineage, final validation, Learner-only label source and train-only support reference by path and SHA-256.

Unknown fields are rejected. A production manifest without the complete frozen-pool provenance is rejected.

## Candidate TSV contract

The exact ordered columns are:

`row_id, sequence_id, utr50, group_id, proxy_mean, proxy_std, proxy_range, support`

No measured label, outer role, Q, or outer metric column is allowed. `group_id` is the leakage-blocking unit used for grouped OOF. `support` is finite and non-negative.

The candidate row order must exactly match the frozen fixed-pool source order. The prediction NPZ may contain the complete dev/calibration role; candidates are aligned as a strict row-ID subset, never by position. The full prediction row-order hash and source hash must match the inference manifest and frozen role source. Candidate proxy mean/std/range must equal the selected ensemble values within 1e-12. Prediction files must contain only `row_id`, `ensemble_mean`, `ensemble_std`, `ensemble_range`, `pred_seed_11`, `pred_seed_23`, and `pred_seed_37`.

The separate label TSV uses `row_id` and `measured_rl`. It may contain the complete allowed role, but every candidate row ID must be present and all IDs and measured values must be unique and finite respectively. Production labels must exactly match both the frozen production role source and the Learner vault reference. Labels are used only by the evaluator or dev/calibration training roles, never by candidate ranking.

Production support is `1 - minimum Hamming distance to any fresh train UTR50 / 50`. It is recomputed from the hash-pinned train source during loading; candidate support values must match within 1e-12.

## Hashes and provenance

Every file is rehashed at load time. Candidate, prediction-manifest, prediction NPZ, and label-source hashes must match. `prediction_manifest.labels_in_output`, `outer_labels_read`, and `outer_metrics_computed` must all be false. No outer NPZ/TSV is read by the loader.

## Selection manifest

Selection records include pool ID, strategy, alpha, proposal cap, paired seed, selected ID, regret, proposals, proxy item-forwards, three-member item-forwards, and per-member item-forwards. `selection_config_sha256` is the canonical JSON hash of all records and must be frozen before Q training. Q is never a candidate-ranking input.
