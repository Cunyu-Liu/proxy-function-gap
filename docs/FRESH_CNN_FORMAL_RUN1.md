# Fresh CNN formal training: run1_retry1

## Status

All three production seeds completed with `converged_dev_plateau=true`. No outer labels were read and no outer metrics were computed. E1 remains disabled.

- Training code commit: `fd5e673785612a34a0f6e334761e190d681449dc`
- Frozen config SHA256: `9c3339987c90acffa8d3de0dc09c70d9e6c5e191b581feee25f54b0988eb78ca`
- Data pack SHA256: `1df96fabb665e3b2819542af02077fe631bed1732ff87c3a664a460d3e62a759`
- Output: `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/fresh_proxy_cnn_v2/training/formal_run1_retry1`
- Device: one NVIDIA GeForce RTX 5090, `cuda:0`, torch `2.7.1+cu128`, CUDA `12.8`, float32
- Seeds: 11, 23, 37; batch size 1024; Adam learning rate 0.001; min_delta 0.0001; patience 5; per-process GPU memory fraction 0.08

The first run failed before the first optimizer step because adaptive max pooling had no deterministic CUDA backward implementation. It was not replaced by CPU training or by disabling determinism. The pooling was changed to deterministic mean reduction, the full suite passed again, and the retry used a new output directory.

## Actual results

| Seed | Epochs | Best epoch | Final online train loss | Final full-train eval loss | Final full-dev eval loss | Best full-train eval loss | Best full-dev eval loss | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 88 | 62 | 0.4431543292 | 0.4527236250 | 0.7753952583 | 0.4674857158 | 0.6672114295 | 230.74 |
| 23 | 61 | 50 | 0.4622794853 | 0.4628752050 | 0.6927347994 | 0.4820291716 | 0.6770937815 | 166.14 |
| 37 | 104 | 77 | 0.4451223353 | 0.4671275034 | 0.6760604314 | 0.4632554173 | 0.6751449035 | 262.37 |

All final plateau counters equal 5. Actual peak allocated memory was 952,163,840 / 952,380,928 / 952,272,384 bytes respectively. Every history contains per-epoch online/full-train/full-dev loss, pack/order/config/model-state hashes, and seed+epoch shuffle identity.

## Three verification rounds

1. Per-seed convergence, complete history, completed/latest and best checkpoints, code/data/order/config/model hashes, and all checkpoint/history hashes passed.
2. Each best checkpoint independently recomputed full-train (293,091 rows) and full-dev (21,458 rows) losses on CUDA. Values exactly matched the best-epoch history.
3. Key metadata checks were rerun, raw history was independently summarized and compared against summaries and hashes, and the complete 201-test suite plus source/data audit passed again.

Artifacts: `verification_round1.json`, `verification_round2_gpu.json`, `verification_round3.json`, `independent_summary.json`, `frozen_config_retry1.json`, `command_retry1.txt`, and the full run directory above.

## Remaining gates

- V and Q calibration is not complete and no calibration-based deployment decision is authorized.
- No outer labels were accessed and no outer metric or regret was computed. E1 remains disabled.
- These results establish fresh training convergence only, not biological improvement over any baseline or official model.
- Model weights, NPZ, checkpoints, and large logs are not committed to Git.
