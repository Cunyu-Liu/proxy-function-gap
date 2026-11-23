# E1 outer replay protocol

## Authorization

E1 starts only after the measured-MRL ensemble-v2 lineage, multi-pool V and Q artifacts have each passed three verification rounds. The FP-REJ deployment configuration is selected on 32 dev pools and frozen before outer use. Q uses 16 calibration pools only for final probability calibration. No outer result may be used to retune alpha, budget, threshold, features or model type.

## Learner

The Learner reads only the frozen outer unlabeled JSONL views and train-only support references. It executes actual CUDA forwards for the three fresh CNN members and V/Q small models. It writes selection records with `regret=null`, actual member/verifier/V costs, Q decisions and configuration hashes. Full-pool proxy features used by Q add `q_feature_member_item_forwards`; these costs are reported separately from selection costs.

All candidate decisions use the true `utr50` and a fixed 50nt/valid-alphabet mask after T→U conversion. No unregistered GC, uAUG or poly-U threshold is invented. These sequence families remain available for later mechanism analyses.

## Evaluator

The Evaluator is a separate entry point. After Learner artifacts are frozen and validated, it reads the Evaluator-only historical MRL vault, applies the frozen Q decisions and writes `selector_regret` before Q and final `regret` after Q. It does not return per-sequence labels to the Learner. Repeating an evaluation requires new output paths and preserves earlier records.

## Evidence boundary

Outer designed and outer random pools are disjoint complete allocation units from the same Sample 2019 study. They support preregistered historical replay, not independent-study replication or prospective wet-lab validation. Random outer contexts are secondary and are not pooled with the main designed context. Abstention is reported with coverage; regret is never imputed for abstained selections.

The primary comparison is frozen FP-REJ+Q minus BoN at cap 16, using pool-cluster bootstrap. Matching nominal cap alone is not sufficient: actual CNN, V and Q feature costs must accompany every comparison. Development and calibration metrics are not final scientific conclusions.
