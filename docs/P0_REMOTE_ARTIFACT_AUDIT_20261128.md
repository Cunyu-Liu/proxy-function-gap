# P0 远端深度完整性审计（2026-11-28）

协调者直接执行（子代理后端 "Invalid token" 故障，无法委派）。远端只读，未修改任何冻结 artifact。

## 1. 本地 source_data 与远端 artifacts SHA256 比对

本地根：`deliverables/reports/source_data/`；远端根：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/`。

| 本地文件 | SHA256（前 16 位） | 远端对应 | 一致 |
| --- | --- | --- | --- |
| `paper1_budget_matched_v1/results.json` | `6226ad942d6abc4e` | `paper1_budget_matched_v1/analysis/results.json` 与 `export/results.json`（两份相同） | ✅ |
| `scenario_gb1_v1/aggregate/test_aggregate.json` | `2b070135fba6fb26` | `scenario_gb1_v1/results/local_aggregate/test_aggregate.json` | ✅ |
| `scenario_gfp_v1/results/primary.json` | `52cabc32c89e09ee` | `scenario_gfp_v1/results/primary.json` | ✅ |
| `scenario_aav_v1/results/main_results.json` | `cd2f42458c4e6c72` | `scenario_aav_v1/results/main_results.json` | ✅ |
| `scenario_tem1_v1/results/direction_results.tsv` | `1cb5bbacf61cd125` | `scenario_tem1_v1/results/test/direction_results.tsv`（dev 切片哈希不同，属正常角色分离） | ✅ |
| `scenario_mrna_non_tcell_v1/artifact/results/decision_summary.tsv` | `35dc93288177dc10` | `scenario_mrna_non_tcell_v1/results/decision_summary.tsv` | ✅ |
| `scenario_mrna_non_tcell_v1/artifact/results/unified_scenario_metrics.json` | `921c7944ca9ebe7b` | `scenario_mrna_non_tcell_v1/results/unified_scenario_metrics.json`（quarantine/concurrent_mixed_output 下同名文件哈希不同，本地对应的是非隔离正式版） | ✅ |

完整哈希见执行日志；7/7 核心文件一致，无远端缺失。

## 2. 冻结协议/清单存在性与哈希（远端）

| 文件 | SHA256 |
| --- | --- |
| `paper1_budget_matched_v1/frozen_config.json` | `2b42cfcc3623e0b410c29cb2e9c7b299596ed65ad838a34375277e1df226c6da` |
| `paper1_budget_matched_v1/PROTOCOL.md` | `bef80787aa4e0243c4d366d492933b0d95d37a764e0a5e4508aa8faf791d53bd` |
| `paper1_budget_matched_v1/artifact_manifest.json` | `95897c83ae98a190a8272917d6c09fd8e26853761ff3f78235692aa9b17b9c70` |
| `scenario_gb1_v1/frozen/frozen_config.json` | `a7c2eff9c503111f110abfbd90e89f666a608b4f7c6c404127fb0eb4a1edb571` |
| `scenario_gb1_v1/artifact_manifest.json` | `42a8266df594c898e136d04a719c5944f8d089e754e9e6212f37b72579ed1bd1` |
| `scenario_aav_v1/frozen_protocol.json` | `012924d3e71ff2353e7b51aa6d2fe305ac5667a97097f930dacf94ca9df5f88c` |
| `scenario_tem1_v1/source_snapshot/frozen_protocol.json` | `3153bd8c5b6bc8590ab64c72b5cf5dadd5316a0413ad53a9e7ce2fa8b9b4e1f7` |
| `scenario_tem1_v1/artifact_manifest.json` | `5601e65575d80d5b936433f7cc90182c0272b21d41d05f9f63de39facbf6c613` |
| `scenario_mrna_non_tcell_v1/protocol/frozen_protocol.json` | `54ef915770bbd798b52d594db5fb2f64faa6853c18950495ebf8f87070eceb61` |
| `scenario_mrna_non_tcell_v1/artifact_manifest.json` | `f0ddf472fb7dd55c3ba15b57d94216cfee1f049e15d1b067b6506577d8427702` |

注意：`scenario_gfp_v1` 目录下无 `frozen*.json`/`*protocol*` 命名文件；其核验证据在 `scenario_gfp_v1/verification/`（`round1_raw_gate.json`、`round2_independent.json`、`posthoc_hamming_correction.json`）。GFP 冻结协议载体待 P1/P4 阶段确认（可能在仓库 configs 或 results/primary.json 内嵌），列为待办，不阻塞当前结论。

`scenario_mrna_non_tcell_v1/protocol/` 保留 `frozen_protocol_attempt1_failed.json`、`frozen_protocol_attempt2_killed.json` 与 `verification/pre_result_protocol_corrections.json`：失败尝试按规范保留，未包装。

## 3. 远端仓库测试套件

- 环境：`/root/miniconda3/envs/lucaone_cu128/bin/python`（README 核验环境），`CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- 默认 `PYTHONPATH`（src/gate0/runtime/src + sample-runtime .deps）下 collect 出现 3 个 ImportError：`tests/pool/test_castillo.py`、`tests/pool/test_strict_h1_summary.py`、`tests/test_q_threshold_domain.py` 依赖 `checks/` 下模块（`validate_castillo_pools`、`summarize_strict_h1_paper1`、`verify_q_head`）。
- 将 `/home/proxy-function-gap/checks` 加入 PYTHONPATH 后全量运行（`-p no:cacheprovider`，未产生缓存文件）：

```
325 passed, 106 subtests passed in 100.45s
```

0 失败。建议：仓库 README/pytest.ini 应补充 `checks` 到 pythonpath（待协调者决定是否提交该小修复）。

## 4. 结论

- 本地交付 source_data 与远端冻结 artifacts 完全一致，可复现链完整。
- 仓库测试套件全绿（需 checks PYTHONPATH）。
- 缺口：GFP 冻结协议载体位置待确认；pytest.ini pythonpath 未含 checks（文档/配置小缺口，非科学缺口）。
- 远端磁盘剩 ~154GB、GPU 剩 ~13.8GB，后续实验须控制体积。