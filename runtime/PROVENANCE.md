# Sample runtime provenance

- Source repository (read-only during integration): `/home/proxy-function-gap-sample-runtime`
- Source commit: `3c8d047816a4a216a3954c66112c1e858c7dc2c7`
- Source subject: `fix: project prediction rows onto declared schemas`
- Integrated subtree: `runtime/`
- Initial subtree merge: `981a3de` (`Add 'runtime/' from commit '3c8d047816a4a216a3954c66112c1e858c7dc2c7'`)

The source commit is an ancestor of the main repository through the subtree merge. The integrated `runtime/src/sample_runtime` package is the source repository's independent PyTorch implementation. No official GPL Python source file or HDF5 weight is tracked or copied into the main repository.

Official repository and weight provenance remains declarative in `src/sample_runtime/provenance.py`: `pjsample/human_5utr_modeling` at `d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe`, GPL-3.0, with absolute external weight paths and SHA-256 values. Those external files are read only by explicit runtime tests/inference and are never packaged.

`STAGING_ROOT` is retained as a compatibility variable name but now points to `/home/proxy-function-gap/runtime`; Git commands therefore record the integrated main commit. Runtime tests require the confirmed external CUDA environment and do not have a CPU inference fallback.
