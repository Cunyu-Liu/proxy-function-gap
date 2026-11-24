# Gate 0 source provenance

- Source repository (read-only during integration): `/home/proxy-function-gap-data`
- Source branch: `staging`
- Source commit: `25d6fd4a8133aadde85f6cc4718503c0efe21249`
- Source commit subject: `Build Gate 0 Sample 2019 split-ready artifacts`
- Source tree: `dfff55f48300b5e32f3ccc8e3bb41fa29e4e1024`
- Integrated subtree: `gate0/`
- Initial integration merge: `cc8ec11` (`Add 'gate0/' from commit '25d6fd4a8133aadde85f6cc4718503c0efe21249'`)

Integration used `git subtree add --prefix=gate0 /home/proxy-function-gap-data 25d6fd4a8133aadde85f6cc4718503c0efe21249`. The merge has the previous main commit and the source commit as parents, and the initial integrated `gate0/` tree exactly equals the source tree above. Later packaging and documentation adjustments are committed separately in the main repository.

The source repository and its `staging` branch were not modified. Raw inputs and derived artifacts are not part of this subtree history. `configs/gate0_v1.json` continues to reference `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/`; no raw or derived data is copied into Git.

The upstream author-code evidence has its own independent provenance in `configs/author_code_evidence.json`: repository `pjsample/human_5utr_modeling`, commit `d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe`, with per-notebook hashes.

No split/vault assignment, real model inference, training, or E1 evaluation is introduced by this integration.

## Split-v2 update

- Updated source commit: `05eb96b5e730c9dd6cebc96486220bd6266c30d9`
- Source subject: `Add model-specific Gate 0 split v2`
- Source tree: `aeee1de6b0f219b550e29c728d685468e386332e`
- Integrated merge: `56d3e3b` (`Merge commit '05eb96b5e730c9dd6cebc96486220bd6266c30d9'`)

The split-v2 update was integrated with `git subtree pull --prefix=gate0`. Only `README.md` conflicted: the resolution retained all upstream split-v2 content and the existing integrated-path/provenance documentation. Existing E0 code and the original Gate 0 implementation were not overwritten. The original source commit and the split-v2 source commit are both ancestors of the main repository.

Trainer preflight consumes the frozen split-v2 manifest and explicit Learner allowlist. The trainer does not create or modify splits/vaults.

## Fresh CNN v2 production update

- Source commit: `a2f10ee5511b5334d04fab11306a81c63647120c`
- Source subject: `Complete fresh CNN v2 production delivery`
- Source tree: `f525338f25060d61904b032a59944684d1cfa85e`
- Integrated merge: `77077f8` (`Merge commit 'a2f10ee5511b5334d04fab11306a81c63647120c'`)

The fresh CNN v2 production delivery was integrated with `git subtree pull --prefix=gate0`, preserving the source commit as an ancestor. Existing E0, Gate 0, split-v2, runtime, and trainer code were retained. The authorized selected build is `build_a`; its root child/integration/verification/final-validation chain authorizes fresh training while E1 and outer metrics remain disabled.

## Leppek 2022 GSE173083 independent Gate 0

Primary public sources are GEO `GSE173083` (last update 19 April 2022 in SOFT), SRA RunInfo `SRP315847`, Europe PMC/PMC `PMC8940940.1`, and publisher supplementary files for DOI `10.1038/s41467-022-28776-w`. Exact URLs, byte sizes and SHA256 values are pinned in `configs/leppek2022_v2.json`; raw and derived artifacts remain under `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/leppek2022_gate0` and are not tracked by Git.

The PMC article states CC BY 4.0, which is retained for article and publisher-supplement provenance. The retrieved GEO SOFT does not state an explicit data license; public accessibility is not relabeled as an open license. SRA RunInfo pins 70 runs (50,374 MB reported); raw FASTQ were not downloaded or used to recompute endpoints in this Gate 0 build.

GEO and publisher primary-table copies are independently crosschecked. Source Data `Fig_4c` supplies exact replicate counts. Supplementary Data 5 name-value offsets and the Source Data half-life `n=2` exception are preserved in `supplementary_crosscheck.json`; no outcome-based source selection is performed.

## Schlusser 2024 independent Gate 0

Primary sources are the peer-reviewed article DOI `10.1186/s13059-024-03369-6` (PMC11337900, CC BY 4.0), Zenodo `10463061` (CC BY 4.0 sequence-level tables and plot scripts), Zenodo `13133725` (MIT code snapshot), and GitLab project 2850 commit `3bd1f03579b662b0b0bc9c266ae92f457258727c` (MIT). Zenodo `10463061` is the file-level predecessor of ZIP-only record `10463090`; using it avoids altering or repacking the source tables. Every downloaded table is verified against the published byte size and MD5 and then locally SHA256-recorded.

HepG2 raw sequencing is BioProject PRJNA1045106 / NCBI SRP474241; HEK293 and yeast lineage are PRJNA591214 and PRJNA338918. NCBI RunInfo contains six HepG2 runs (three RNA-seq and three Ribo-seq), while the ENA portal response in this audit contained only the three RNA-seq runs; both records are retained rather than silently treating ENA coverage as complete. Available ENA metadata preserves run accessions and FASTQ MD5 values. FASTQ were not downloaded or independently reprocessed, so the build claims published-table provenance only. GEO reporter lineage is GSE114002 and GSE104252; DART lineage is Niederer 2022 Supplementary Table S2. Public accessibility without an explicit repository data license is not relabeled as CC0.

Raw and derived artifacts remain under `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/schlusser2024_gate0`. No Paper 2/3 source is changed. HepG2 is frozen before label inspection or model execution; other published training resources are not promoted to independent outer evidence.
