# T-cell B-strict 辅助残差修复 v1：辅助门禁未通过，按冻结协议停止

日期：2026-11-26。证据类别为**回顾性方法开发/协议敏感性**；历史 outer 已查看，本工作不构成新盲测或前瞻确认。

## 结论

B-strict 在两个 leave-one-auxiliary-context-out 方向的 dev 上都能轻微降低 cap16 BoN regret，并降低 MSE；但在各自未参与选型的 auxiliary test 上，两个方向的 cap16 regret 均变差。根据 `TCELL_AUXILIARY_SUPERVISION_AUDIT_20261126.md` §8.2，本方案在目标评价前失败并停止。

**没有生成 T-cell corrected 选择，没有运行 T-cell Evaluator，没有读取 T-cell MRL，也没有目标 regret、CI 或主成功判定。** 因此本工作没有证明 T-cell 等预算 MRL 选择得到实质改善。

## 冻结实现

- 新工作区：`/home/proxy-function-gap-tcell-residual`；artifacts：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_residual_bstrict_v1`。
- 冻结三成员 Sample CNN；所有辅助 CNN 推理均在 `cuda:0`，PyTorch 2.7.1+cu128，无 CPU 降级。三成员 checkpoint/model hash 与 Sample train-only scaler 均通过。
- 特征严格为 `p, p², GC, uAUG, longest-poly-U, intercept`。残差为 `MRL-p`，仅用源辅助 train 的残差均值/总体 SD 标准化。
- λ 网格预先固定为 `[0, 1e-6, 1e-4, 1e-2, 1, 100]`，强度网格为 `[0, 0.25, 0.5, 0.75, 1]`。ridge 使用均方损失口径、不惩罚截距、float64 CUDA 闭解；条件数上限 `1e12`。
- 部署公式预注册为 `p + Sample train SD × strength × Xβ`，Sample train SD=`1.6689551812138181`。最终两方向等权组合只在双门禁通过后才会使用，本轮未进入该阶段。
- 只按目标 auxiliary dev 的 cap16 平均 BoN regret 选 λ/强度；平手依次比较 dev MSE、较大 λ、较小强度。test 不参与选型。
- T-cell r1/r2 原始文件在辅助阶段只读取 `utr`，用于无标签全集 exact/H1 排除；另排除 Sample train/dev/calibration exact/H1。

## 原始、排除与分组

| 辅助情境 | 有效唯一 50 nt | T-cell r1 exact | T-cell r2 exact | T-cell 并集 exact | 额外 H1-only | Sample exact/H1 | 严格保留 |
|---|---:|---:|---:|---:|---:|---:|---:|
| N50 | 2,200,533 | 0 | 0 | 0 | 0 | 0 | 2,200,533 |
| HepG2 | 12,563,191 | 316,495 | 319,264 | 324,052 | 244,834 | 488,976 | 11,989,347 |

HepG2 的逐 replicate exact 计数与审计一致。四个原始文件 SHA-256 均与既有 pool manifest 一致；有效行数、唯一性和重复计数通过，重复有效行均为 0。

严格排除后按 H1 connected component 的 `sha256(context|minimum_component_sequence)` 模 10 分为 80% train、10% dev、10% test：

| 情境 | train 行/component | dev 行/component | test 行/component |
|---|---:|---:|---:|
| N50 | 1,763,213 / 694,086 | 220,183 / 86,500 | 217,137 / 86,372 |
| HepG2 | 9,604,728 / 6,466,600 | 1,192,185 / 807,988 | 1,192,434 / 806,703 |

半段索引复核未发现 exact/H1 component 跨角色。

## 辅助选择与前置门禁

“改善量”定义为 frozen-CNN regret 减 corrected-CNN regret，负值表示 corrected 更差。

| 方向 | 所选 λ | 强度 | dev frozen → corrected cap16 regret | dev MSE frozen → corrected | test frozen → corrected cap16 regret | test 改善量 | 门禁 |
|---|---:|---:|---:|---:|---:|---:|---|
| N50→HepG2 | 0 | 0.5 | 4.091181 → 4.089923 | 12.823442 → 12.557254 | 4.082435 → 4.082778 | **−0.000343** | FAIL |
| HepG2→N50 | 1 | 0.75 | 5.800774 → 5.794299 | 12.229010 → 11.883042 | 5.828669 → 5.833308 | **−0.004638** | FAIL |

两方向都说明“dev 上轻微选择改善 + MSE 改善”不足以保证跨辅助 test 的选择收益。完整 30 个候选/方向及所有负结果保存在 `artifacts/auxiliary/*/selection.json`，没有事后扩大网格或更换特征。

### 数值与仿射-only 阴性消融

- N50→HepG2 所选条件数 `1,370,630.35`；HepG2→N50 所选条件数 `41,519.34`。全部拟合有限、闭解收敛、CUDA 执行。
- 仿射-only 使用相同 λ/强度；部署有效正斜率分别为 `0.864922`、`0.820935`。数学上保持 frozen-CNN 排序，dev/test 逐 pool×seed regret 与 frozen-CNN 完全一致。
- 为避免浮点扰动原 CNN 精确同分，阴性消融按等价的 frozen-CNN 排名计算，同时保留仿射变换分数有限性检查。这不影响完整残差模型的候选选择。

## 三轮真实核验与失败保留

1. **原始/主分析轮：PASS。** 25 项原始/数组/分组检查及 126 项辅助候选与 test 指标检查通过；聚焦测试 `8 passed`。
2. **独立轮：PASS。** 不导入主训练模块，重新在 CUDA 求解所选 ridge，逐池重算全部 60 个候选指标/方向及门禁；130 项检查通过，系数最大绝对差 0。目标状态明确为 `NOT_RUN_AUXILIARY_GATE_FAILED`，T-cell 评价次数 0。
3. **交付轮：PASS。** 再次执行独立复算、测试 `8 passed`、导出哈希和禁止字段检查；本地另做 SHA-256 回读与报告一致性核验。

失败没有删除：

- 第一次仿射阴性消融因浮点同分扰动的逐值断言停止，日志和中间目录保留。
- 独立核验随后发现主分析 proposal 位置误用于池候选标签索引；旧辅助门禁结果作废并保留。修正为先索引 proposal 标签再取 argmax，新增回归测试后重跑，本文只报告修正后的最终结果。
- SSH 连接中断和未启动尝试没有伪装成科学运行；科学结果只来自有完整 manifests 与 verification 的运行。

## 产物与边界

- 本地脱敏 source data：`deliverables/reports/source_data/tcell_residual_bstrict_v1/`。包含 config、代码、系数、全部候选、日志和三轮 verification；不含序列字符串、逐序列标签或逐序列预测。
- 远端保留完整数组、序列、失败中间目录和全部 artifacts；代码未 push。
- 因前置门禁失败，本轮不能报告 T-cell cap1/2/4/8/16 或逐池结果，也不能以任何 T-cell 结果改特征、λ、强度或子集。后续若提出新科学假设，必须另立方案，不能把本轮包装为成功的冻结检验。
