# Gate 0 integration contract

## Scope

The main repository now contains both the E0 fixed-pool synthetic checks and the completed Sample 2019 Gate 0 processing code. Gate 0 comes from commit `25d6fd4a8133aadde85f6cc4718503c0efe21249` and is integrated under `gate0/` with Git ancestry preserved. See `gate0/PROVENANCE.md` for the exact source tree and merge method.

The initial Gate 0 integration did not implement split/vault assignment, run real model inference or training, or authorize E1. A later authorized update adds the completed split-v2 commit `05eb96b5e730c9dd6cebc96486220bd6266c30d9` and Sample runtime commit `3c8d047816a4a216a3954c66112c1e858c7dc2c7` with subtree provenance. Split-v2 roles are now available to the trainer preflight; no new split/vault implementation is written by the trainer.

## Package and test layout

- E0 package: `src/proxy_function_gap/`.
- Gate 0 package: `gate0/gate0_sample2019/`.
- Gate 0 frozen configs: `gate0/configs/`.
- E0 tests: `tests/`; Gate 0 regression tests: `gate0/tests/`.
- setuptools searches both `src` and `gate0`; pytest uses the same import roots and both test directories.
- Both packages require only the Python 3.9 standard library at runtime. pytest remains the only development dependency.

## Data boundary

Raw and derived data stay under `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/` as specified by the Gate 0 config. They are not copied into the main repository. The root `.gitignore` excludes common raw/derived data formats and local data directories, and integration audits check tracked file sizes.

## Commands

From `/home/proxy-function-gap`, the integrated module commands are:

```bash
.venv/bin/python -m gate0_sample2019.inspect_formats --config gate0/configs/gate0_v1.json
.venv/bin/python -m gate0_sample2019.build --config gate0/configs/gate0_v1.json
.venv/bin/python -m gate0_sample2019.verify --config gate0/configs/gate0_v1.json
.venv/bin/python -m gate0_sample2019.finalize --config gate0/configs/gate0_v1.json
.venv/bin/python -m gate0_sample2019.validate_delivery --config gate0/configs/gate0_v1.json
```

These commands are documented for reproducibility but are not executed during integration testing. Tests use only synthetic values and temporary files; no real input manifest is read. Finalization records the integrated main commit. Legacy delivery JSON keys containing `staging` remain for compatibility with already produced artifacts.

## Verification gate

Integration is accepted only after three rounds of the combined E0+Gate0 pytest suite and source/data-boundary audit, the original Gate 0 unittest discovery, the E0 independent math check, and a final committed-state test. Any failure blocks the integration commit and must remain in the run logs.

## Integration review outcome

- Three combined pytest rounds each collected and passed 128 tests; original Gate 0 unittest discovery passed 9 tests.
- E0 independent math checks and integrated source/data-boundary audits passed in every round.
- The first wheel exposed an unintended top-level `tests` package. Package discovery was restricted to `proxy_function_gap*` and `gate0_sample2019*`; a second failure was traced to the stale generated `build/lib/tests` directory. A clean rebuild contains only the two intended packages and their dist-info, with no raw data members.
- The audit covers tracked and candidate files, rejects raw-data suffixes and files over 1 MiB, and verifies the source commit is an ancestor of the integrated repository.
- The initial Gate 0 source HEAD remained at `25d6fd4a…`; concurrent split files were excluded at that stage. The subsequent authorized split-v2 update integrates the completed `05eb96b5…` commit through merge `56d3e3b`, while unrelated working-tree changes remain excluded. No real data processing entry point was executed during the initial integration.
