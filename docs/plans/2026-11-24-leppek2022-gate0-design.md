# Leppek 2022 Gate 0 design

## Scope and decision

Build an independent, read-only data qualification pipeline for Leppek et al. (2022), GSE173083. Do not modify Paper 2/3, train or infer a model, access outer results, impute missing measurements, or relabel `Predicted expression` as an observation.

Three implementation options were considered: extending the Sample 2019 builder, using a notebook-only audit, or a separate versioned module. A separate module is selected because the assays and sequence entities differ, while immutable artifacts and independent verification remain compatible with the existing Gate 0 contract.

## Source and entity model

- Pin GEO SOFT, the GEO file list, all four GEO XLSX supplements, `GSE173083_RAW.tar`, PMC full text/HTML, and all publisher supplementary files by URL, retrieval time, size, and SHA256.
- Article and publisher supplements retain the CC BY 4.0 license stated in PMC. GEO has public access but no explicit data license in the retrieved SOFT; this is recorded rather than inferred.
- The primary entities are 233 pooled full-length constructs and 24 individual CDS designs. Supplementary long-form measurements retain their source workbook, sheet, row, and source identifier. Raw sequencing samples are represented from SOFT metadata; RDAT members remain preserved in their original tar.
- Each source row receives a stable ID and raw-row hash. RNA is normalized to uppercase DNA only for comparison; both raw RNA and normalized DNA are retained.

## Endpoint contract

`Ribosome load`, polysome ratios, in-cell/in-solution degradation coefficients and half-lives are measured or fit-from-measurement endpoints with explicit assay metadata. `Predicted expression` is a derived kinetic prediction, never a measured protein endpoint. CAI, DegScore, Vienna, EternaFold, MFE, SUP, and AUP columns are computational predictions/features. Missing source cells remain missing.

Luciferase values are measured protein-expression evidence only when read from a source-data experiment table, with time, normalization, replicate, cell, chemistry, and construct retained. Exact replicate counts come from Source Data `Fig_4c`; Supplementary Data 5 name-value offsets are reported rather than silently repaired. No pooled MRL label is synthesized. Sample 2019 MRL pooling/training is NO-GO because the sequence entity, reporter context, and assay endpoint differ. Descriptive cross-endpoint/external-mechanism analysis can be GO only for constructs with directly measured endpoints.

## Cleaning, duplicates, and roles

Inclusion is endpoint-specific; a missing endpoint does not erase other valid measurements. Validate sequence alphabet, source length, and full-length assembly from components. Report every exclusion reason without deleting the raw record.

Exact groups use normalized full-length sequence. Near duplicates use a preregistered normalized Levenshtein threshold of at most 1% of the shorter construct (minimum one edit), with full pairwise verification and connected components. Shared UTR/CDS components are separately audited. No outcome is used for clustering.

Roles are frozen at the connected-component level with pooled and CDS families kept as separate strata. Deterministic, seed-pinned balanced allocation targets 60% development, 20% candidate, and 20% permanent evaluation by construct count. A component has exactly one role. Role assignment precedes any model and is independent of endpoint values. The complete audit table is Evaluator-level; future learner views must omit candidate/permanent labels.

## Artifacts and verification

The versioned output contains `data_manifest.json`, `field_dictionary.tsv`, `sequence_index.tsv.gz`, `endpoint_coverage.tsv.gz`, `endpoint_measurements.tsv.gz`, `inclusion_exclusion.tsv.gz`, `distribution_summary.tsv`, `exact_duplicate_groups.tsv.gz`, `near_duplicate_edges.tsv.gz`, `overlap_report.json`, `role_assignments.tsv.gz`, `role_manifest.json`, `gate0_decision.json`, `delivery_checksums.json`, and three verification reports/logs.

Round 1 independently checks source hashes, row counts, raw-row hashes, and cell-level source-to-derived values. Round 2 recomputes sequence assembly, endpoint states, summaries, and all duplicate pairs. Round 3 checks artifact hashes, cluster/role invariants, label semantics, and the gate decision. Any failure stops delivery; no model stage exists in this module.

## Assumptions

- Public GEO access is sufficient for research download, but is not treated as an explicit open license.
- Source-data expression experiments may cover selected constructs rather than the full pool; this limits, rather than disqualifies, a mechanism-only candidate.
- No CUDA work is requested or permitted at Gate 0, so CPU-only execution is intentional and recorded.
