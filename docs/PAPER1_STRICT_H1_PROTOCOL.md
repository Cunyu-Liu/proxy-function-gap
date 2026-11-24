# Paper 1 strict-H1 reconstruction protocol

## Evidence status

This is a retrospective reconstruction and protocol-sensitivity analysis of Castillo outer pools that have already been inspected. It is not a new blind test and must not be described as independent prospective confirmation. No result-dependent gate, threshold, endpoint, seed, cap or model change is allowed.

## Frozen pool contract

- Contexts: random-end HEK293T N50 r1, defined-end HepG2 r1, defined-end T-cell r1 and T-cell r2.
- Each context has two independently executed deterministic builds, each containing 16 pools of 64 unique valid UTR50 sequences.
- Sample train, dev and calibration sequences are combined for exact and Hamming-1 exclusion before deterministic candidate ranking. Exact and Hamming-1 conflicts are hard failures both within/across selected pools and against Sample.
- Hamming-2 counts are a predeclared sensitivity diagnostic only. They are not used for candidate selection, exclusion, pass/fail, model selection or interpretation-driven threshold changes.
- Three separately written checks cover (1) pool exact/H1 isolation, (2) cross-Sample exact/H1 isolation and (3) double-build learner/label/manifest hashes plus raw, Sample, train-reference and per-pool hashes.
- Existing pool, selection and evaluation directories are not overwritten. New outputs are rooted at `castillo_fixed_pools_v3_strict_h1_retry1` and `paper1_v3_strict_h1_retry1`.

## Frozen analysis

The existing three-member fresh CNN is used for Random/BoN pressure and score matching. The existing three-member 4-mer MLP is used with the CNN for common-error analysis. Both architectures must execute inference on CUDA with deterministic settings; CPU fallback is forbidden. No model is retrained or calibrated on Castillo labels.

The pressure grid is caps `{1,2,4,8,16}` and seeds `{101,202,303,404,505}`. At each cap, each pool contributes the mean paired BoN-minus-Random regret over common seeds; negative values favor BoN. Score matching remains the nearest different candidate by CNN proxy mean at cap 16, with deterministic sequence-ID hash tie-breaking; its effect is optimized-minus-matched absolute error. Common error includes CNN-minus-MLP absolute error and absolute-error correlation.

All uncertainty intervals use pool-cluster bootstrap with 10,000 replicates, seed 20261124 and percentile interval `[2.5, 97.5]`. Per-pool paired effects are retained as aggregate source data. No sequence strings or sequence-level measured labels are exported to the local deliverables.

## Validation and reporting

Pressure regret, score matching and cross-architecture metrics are independently recomputed three times. The aggregate summary is also regenerated and compared three times. Reports must state which previous qualitative conclusions remain unchanged and which change, including null and directionally unfavorable results. Observed outer results remain retrospective/protocol-sensitivity evidence regardless of the number of recomputation rounds.
