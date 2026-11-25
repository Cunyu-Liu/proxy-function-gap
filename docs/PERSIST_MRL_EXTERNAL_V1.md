# PERSIST-seq 独立同终点外部复验（persist_mrl_external_v1）

日期：2026-11-25

## 结论

本次完成的是**独立研究、同终点的回顾性外部复验**，不是盲测或前瞻性验证。Leppek 等（2022）PERSIST-seq / GSE173083 的作者 XLSX 提供实测核糖体负载，其标签此前已被探索性查看，因此证据边界不作升级。

在冻结的 `UTR (5') variants` 主分析中，59 个构建体、49 个唯一 5′UTR 全部合格；对 Sample train/dev/calibration 的 exact 与等长 Hamming-1 审计均未发现冲突。固定全部 59 个构建体为候选池、使用 200 个算法种子和 cap16 后，**BoN 相对 Random 的真实 MRL regret 差为 +0.104723（95% UTR 簇 bootstrap 区间 −0.100964 至 0.212706；双侧符号 p=0.245175）**。区间跨零且点估计方向不利于 BoN，故本次不支持 BoN 改善真实 MRL 选择，也没有达到冻结的显著获益规则。该负结果原样保留，未调整子集、种子、cap、终点或模型。

## 冻结协议与兼容性

冻结协议及哈希见 `source_data/persist_mrl_external_v1/protocol/frozen_protocol.json` 与 `frozen_protocol.sha256`。科学设定在任何预测—标签指标计算前固定：

- 唯一主队列：作者表的 `Group == UTR (5') variants`，预期并核实 59 行、49 个唯一 UTR。
- 纳排：Sequence ID 唯一，5′UTR 为长度至少 4 nt 的 ACGT 序列，`Ribosome load` 为有限数；不按标签或预测筛选。
- 泄漏控制：合并 Sample train/dev/calibration，任一 exact 或等长 Hamming-1 冲突均排除该完整 UTR 的所有构建体。剩余不足 20 个构建体或 10 个 UTR 簇则停止。
- 模型：三个既有 4-mer MLP checkpoint 的 frozen 权重和既有训练 scaler；259 维输入为 256 个归一化 4-mer 频率、GC、uAUG、最长 poly-U/50，三成员算术平均。
- 输入：原生变长完整 5′UTR；**没有训练新模型，没有使用 50 nt CNN，没有截断、补 N 或重校准**。
- 主终点：同一固定池内，Random 取每个 PCG64 随机排列的首候选；BoN 在同一排列的前 16 个候选中取预测最高者。`regret = 池内最高实测 MRL − 所选构建体实测 MRL`，主效应为同种子的 `BoN − Random`，负值才有利于 BoN。
- 算法种子：20261125–20261324，共 200 个。重复算法种子只是随机提案重复，不作为独立生物样本。
- 不确定性：以完整 UTR 为簇，有放回抽取 49 个簇，携带簇内全部构建体；每个重抽样均重建 Random/BoN 提案及 oracle。共 10000 次，簇种子 20261126，报告 2.5%–97.5% 百分位区间。
- 显著性：双侧 bootstrap 符号 p 按冻结公式计算；仅当 95% 区间完全低于零且 p<0.05 时称为显著获益。

首次执行因把实际列名 `Group` 错写为 `Sequence group` 而在纳排前失败；当时没有任何预测—标签指标。仅修正该模式字段后重新冻结，科学规则完全未变。失败记录保存在 `verification/pre_metric_schema_failure.json`。

## 数据、权重与设备核验

- 原始文件：`GSE173083_Table_S1_-_Attributes_for_pooled_233_sequences.xlsx`，193214 bytes；SHA-256 `2e11287568dd3be66cd4cc7656033335a66a76ef71d26528a300a9eae5dda7a8`，与检索报告一致。
- 工作表 `Attributes for pooled 233 seque` 共 233 行；终点字段为 `Ribosome load`，不是 TE、蛋白表达或稳定性。
- 三个 checkpoint、三个 Sample 角色文件均逐一核验 SHA-256；checkpoint 内部模型哈希也与既有 training manifest 一致。
- 推理强制 `cuda:0`、float32、确定性算法、关闭 TF32，并断言模型参数、输入和输出均在 CUDA。设备为 NVIDIA GeForce RTX 5090，PyTorch 2.7.1+cu128；未发生 CPU fallback。仅输出复制回 CPU 用于保存。
- 提案、bootstrap 与相关统计明确为 CPU 计算；预测阶段输入文件不含标签。

## 结果

| 指标 | 点估计 | 95% UTR 簇区间 |
|---|---:|---:|
| **BoN−Random regret（主终点）** | **+0.104723** | **[−0.100964, 0.212706]** |
| Spearman（预测 vs 实测 MRL） | −0.008447 | [−0.326701, 0.315406] |
| Pearson（预测 vs 实测 MRL） | 0.055358 | [−0.266098, 0.315503] |
| MAE（同一未重校准 MRL 尺度） | 5.548939 | [5.295488, 5.785035] |
| Random top1 regret | 0.597666 | [0.385343, 0.659143] |
| BoN top1 regret | 0.702389 | [0.385673, 0.796739] |
| Random oracle-MRL 覆盖率 | 0.015 | [0.005, 0.070] |
| BoN oracle-MRL 覆盖率 | 0.000 | [0.000, 0.175] |

Spearman/Pearson 和 MAE 以构建体为描述单位，但区间仍按 UTR 簇计算；相同 UTR 的重复构建体保留在固定池中，未被当作额外独立生物证据。覆盖率表示所选构建体的实测 MRL 等于当次池内 oracle 的比例。bootstrap 中池与 oracle 会变化，因此 BoN 覆盖率区间上界不为零并不与原始池的 0 冲突。

## 三轮真实核验

1. **原始与主执行核验**：验证 XLSX、Sample 和 checkpoint 哈希；检查 233/59/49 计数、序列和终点合法性、冲突排除与停止阈值；5 项单元测试通过；实际完成 CUDA 推理和冻结统计。
2. **独立实现核验**：使用独立的 openpyxl 读取、特征循环、手写 CUDA 矩阵网络和独立提案/bootstrap 代码，重新计算全部 10000 次重抽样，并逐元素比较三成员预测与 bootstrap 数组。结果见 `verification/round2_independent.json`。
3. **交付核验**：本地同步后重新验证协议、报告数值、逐文件 SHA-256、必需文件与负结果表述。结果见 `verification/round3_delivery.json` 和本地 `delivery_verification.json`。

## 产物位置

- 远端代码：`/home/proxy-function-gap-persist-mrl`（未 push）。
- 远端 artifacts：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/persist_mrl_external_v1`。
- 本地 source data：`deliverables/reports/source_data/persist_mrl_external_v1/`。
- 本报告：`deliverables/reports/docs/PERSIST_MRL_EXTERNAL_V1.md`。

## 限制

这是对单一独立研究、小型 5′UTR 单变构建体集的回顾性外测，不证明普遍失效，也不能外推到其他构建体、细胞或终点。XLSX 是作者发布的处理后序列表，本轮没有进行 FASTQ/reads 级复现。样本量和 UTR 簇数有限；不显著不等于证明无效。历史标签接触使其不能被称为盲测。
