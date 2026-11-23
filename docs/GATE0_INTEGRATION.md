# Gate 0 integration contract

## Scope

The main repository now contains both the E0 fixed-pool synthetic checks and the completed Sample 2019 Gate 0 processing code. Gate 0 comes from commit `25d6fd4a8133aadde85f6cc4718503c0efe21249` and is integrated under `gate0/` with Git ancestry preserved. See `gate0/PROVENANCE.md` for the exact source tree and merge method.

This integration does not implement split/vault assignment, does not run real model inference or training, and does not authorize E1. Existing split-ready groups remain proposed and unfrozen.

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
- The source repository HEAD remains at the same `staging` commit, and integration did not modify it. During final verification, concurrently created untracked `configs/split_v1.json`, `gate0_sample2019/freeze_split.py`, and `gate0_sample2019/protocol.py` were observed. Their contents were not read, integrated, or modified; they are excluded from the main repository. No real data processing entry point was executed during integration.
