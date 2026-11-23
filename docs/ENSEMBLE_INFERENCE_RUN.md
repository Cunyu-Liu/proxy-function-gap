# Fresh three-member ensemble inference

## Current verified output

`ensemble_v2` is the only valid production inference output for measured-MRL analysis. It completed train/dev/calibration inference on `cuda:0` with float32 and no CPU fallback. No labels are present in prediction files, no outer data was read, no outer metric was computed, and E1 remains disabled.

- Manifest: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/fresh_proxy_cnn_v2/inference/ensemble_v2/prediction_manifest.json`
- Config: `configs/ensemble_inference_v2.json`
- Rows: train 293,091; dev 21,458; calibration 16,128
- Actual item-forward: 330,677 per member; 992,031 across all three members and roles
- Prediction units: measured MRL after `prediction * train_scale + train_mean`; each scaler has `fit_role=train`

| Seed | Actual model SHA256 |
|---:|---|
| 11 | `13eca7c6a3282f1a1f5c00a3f34cb50d8047d5a16bf146da50fb8f22722b4173` |
| 23 | `2f439c3836426aee7ed21e8b35a22d17673e04d78e78c85f58f74cd53c70b6b6` |
| 37 | `967707e04bbb1fc0d8e210d0855817eb480afd0108e7a7eb47490c8924205a75` |

Each member independently validated checkpoint SHA256, pack/order/source/code/model identity, and executed all rows on CUDA. Files contain only `row_id`, per-member predictions, ensemble mean, sample standard deviation (`ddof=1`), and range.

Three independent checks verified the inverse transform against the normalized ensemble-v1 outputs, recomputed ensemble statistics, and validated all output hashes and no-outer flags. See `ensemble_v2/verification.json`.

## Invalidated outputs

`ensemble_v1` preserved normalized model outputs without inverse target transformation. It and every pool/V artifact derived from it are invalid for MRL science and contain an `INVALIDATED.json` record. They are retained for audit and must not be used. Production pool loading rejects inference manifests without explicit measured-MRL units and inverse-transform evidence.

These predictions are inputs for subsequent dev/calibration interfaces, not scientific evaluation results.
