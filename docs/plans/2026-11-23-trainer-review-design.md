# Trainer independent-review repair design

## Scope

Repair the trainer without starting formal training. Preserve the existing E0, Gate 0, split-v2, runtime, CUDA-only execution, and no-epoch-cap constraints. All end-to-end training evidence is explicitly synthetic and unauthorized.

## Decisions

1. **Exact role identity instead of count-only checks**: NPZ includes row IDs and the manifest pins per-role fresh TSVs. Compare unique sets and counts exactly, then compare a sorted-ID canonical order hash. This accepts different physical ordering without weakening identity.
2. **Eval-based stopping**: online batch loss is retained only as diagnostics. Full train/dev losses are computed after `eval()` on the same frozen model and used by DualPlateau. Dev absolute best always resets plateau, so min_delta cannot stop a continuing absolute-best sequence.
3. **Identity-rich history**: each epoch records pack, per-role order, effective config, model-state hashes and seed+epoch shuffle metadata.
4. **Strict resume**: expected checkpoint SHA256 is configured per seed. Completed checkpoints are rejected. Running snapshots restore only when code, model definition, effective config, scaler, data, source, split, allowlist and order identities all match. Original initial model hash is propagated.
5. **Explicit synthetic parallel smoke**: the formal engine remains closed to synthetic data. A separately named smoke engine uses ZeroParameterModel for deterministic plateau behavior and exercises three-seed training and resume.

## Alternatives rejected

- Count-only or subset role checks: cannot prove pack identity.
- Online train loss for stopping: mixes changing weights with eval and can stop prematurely.
- Treating sub-min_delta absolute best as plateau: violates the reviewed stopping definition.
- Resuming completed checkpoints or accepting paths without hashes: hides terminal state and provenance mismatches.
- Replacing initial hash on resume: loses the original initialization identity.

## Verification

Three full E0/Gate0/split/runtime/trainer pytest rounds, explicit CUDA self-checks, source/data audits, and CLI three-seed training/resume smoke are required before commit. No scientific performance claim is allowed.
