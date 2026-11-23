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
