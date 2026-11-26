# Paper 1 T-cell 改进实验代码归档与科学边界

日期：2026-11-26。本提交只归档七个工作包的核心代码、冻结配置、测试和结果报告，不新增训练、推理或科学评价。所有结果仍是历史 outer 已查看后的回顾性证据；不得写成新盲测、独立研究或前瞻确认。

## Paper 1 的可发表科学故事

T-cell 的关键矛盾不是 proxy 完全无效，而是**全局排序有真实等预算价值，高分尾部却发生足以限制功能选择的翻转**。在原 strict-H1 诊断中，CNN BoN 与 Random(N) 使用同一 proposal 前缀；cap16 的 BoN−Random(N) regret 差在 r1 为 **−1.706 MRL（95% CI −2.169 至 −1.211）**，r2 为 **−1.087 MRL（95% CI −1.812 至 −0.336）**。负值表示 BoN 更接近完整池实测最优；对应实测 MRL 在 r1 为 6.140 对 4.434，在 r2 为 6.894 对 5.807。r1 的 cap2/4/8/16 全部通过本轮十项 Holm 校正，r2 仅 cap4/16 通过，cap2/8 的未通过结果保留。

这给出 Paper 1 的正面功能终点：在相同候选与评分预算下，proxy 排序确实提高所选构建体的真实 MRL、降低 regret。与此同时，top16 内 Spearman 只有 r1 0.206、r2 0.144，且两个 repeat 各有 15/16 池的预测第一名被尾部其他候选的实测 MRL 超过。完整池 top16 第一名 regret 可分解为机会损失与尾部内排序损失：r1 为 0.836+2.480=3.316，r2 为 1.953+2.117=4.071。该分解把“扩大 proposal 覆盖”与“修复尾部排序”分开，也解释为何 MAE、全局相关或选择更替不能替代 regret。

因此，发表叙述应是**真实等预算收益与明确失效边界共存**，而不是“所有改进都成功”。后续七个工作包中的失败方法用于判伪具体修复机制，不能包装成性能提升。

## 七个工作包的结论与归档位置

| 工作包 | 冻结结论 | 仓库代码 | 结果报告 |
|---|---|---|---|
| T-cell 误差诊断 | 等预算 BoN 有真实 MRL 收益；高分尾部翻转是主要可修复空间，oracle 分解不是已实现收益 | `checks/tcell_error_diagnosis_v1/` | `docs/TCELL_ERROR_DIAGNOSIS_V1.md` |
| B-strict residual v1 | 两个 auxiliary test 的 cap16 regret 均变差；门禁失败，T-cell 未评价 | `checks/tcell_residual_bstrict_v1/` | `docs/TCELL_RESIDUAL_BSTRICT_V1.md` |
| B-rank v2 | 一方向 CI 跨零，另一方向点估计变差；双辅助门禁失败，T-cell 未评价 | `checks/tcell_rank_bstrict_v2/` | `docs/TCELL_RANK_BSTRICT_V2.md` |
| Cross-cell anchor | HepG2-anchor 在 T-cell cap16 显著变差；合并效应 +0.428897 MRL，95% CI +0.245460 至 +0.616474 | `checks/tcell_cross_cell_anchor_v1/` | `docs/TCELL_CROSS_CELL_ANCHOR_V1.md` |
| Multicontext backbone | 两个 auxiliary test 均变差；门禁失败，T-cell 未评价 | `checks/tcell_multicontext_backbone_v1/` | `docs/TCELL_MULTICONTEXT_BACKBONE_V1.md` |
| Target adaptation v1 | 50k main 在 r1/r2 cap16 分别变差 +0.038929/+0.012551 MRL；MAE 改善没有转化为 regret 改善 | `checks/tcell_target_adaptation_v1/` | `docs/TCELL_TARGET_ADAPTATION_V1.md` |
| ListNet v2 | 联合主成功失败；相对原 CNN/v1 main 的 cap16 改善仅 0.037415/0.007585 MRL，CI 均跨零且低于 0.1 MRL 幅度门槛 | `checks/tcell_listnet_adaptation_v2/` | `docs/TCELL_LISTNET_ADAPTATION_V2.md` |

这些结果共同支持三个方法学结论：

1. **真实功能终点必须是所选 MRL 与 regret。** 全局相关、MAE、pair accuracy 或 turnover 的有利变化不能自动升级为等预算功能收益。
2. **共享 proposal prefix 与 Random(N) 是必要对照。** 它使方法间 proposal headroom 相同，避免把候选机会差异误写成排序改进。
3. **门禁失败和不利目标结果具有科学价值。** residual、rank、backbone 在目标前停止；anchor、target adaptation 和 ListNet 的目标失败完整保留。单次 repeat、个别 cap 或小幅点估计不能替代预注册联合成功标准。

## 来源恢复与公开边界

六个实现工作区在归档时存在，代码直接复制且不改科学实现。诊断工作区已消失；远端冻结 artifact 不含 `source_snapshot` 目录，因此使用本地同步的冻结代码快照恢复，并逐文件对照 `tcell_error_diagnosis_v1/artifact_manifest.json` 的 SHA-256。该差异及全部来源哈希记录于 `docs/key_results/20261126/TCELL_IMPROVEMENT_SOURCE_HASHES_20261126.json`，不虚构远端 snapshot 的存在。

仓库不包含序列字符串、逐序列标签、逐序列预测、模型权重、NPY/NPZ 等大型数组、CSV/TSV 科学表、日志、缓存或失败中间数据。原始科学输入和完整 artifacts 仍留在受控远端；本提交仅提供文本级实现、协议、测试和结论可追溯性。
