# Paper 1：paired cross-cell HepG2 实验锚点选择审计

日期：2026-11-26。状态：**已完成一次冻结评价；主假设失败，HepG2-anchor 在两个 T-cell repeat 的 cap16 选择均显著变差。**

证据类别：**paired cross-cell retrospective evidence**。HepG2 MRL 是同序列、不同细胞的真实实验测量，不是 T-cell 标签；但相同序列同时提供 HepG2 锚点和 T-cell 评价，因此本实验不是序列独立验证。历史 outer 已查看，本轮也不是新盲测或前瞻确认。不得将 HepG2 MRL 冒称为目标 T-cell MRL。

## 1. 结论先行

- 主比较定义为 **HepG2-anchor BoN regret − Sample-CNN BoN regret**，负值才代表改善。cap16 的效应在 r1 为 **+0.463090 MRL**，r2 为 **+0.394705 MRL**，均为变差。
- repeat-linked pool block bootstrap 10,000 次的合并效应为 **+0.428897 MRL**，95% percentile CI **[+0.245460, +0.616474]**，p=0.00019998。区间完全位于有害方向。
- CNN 合并 cap16 基线 regret 为 2.408230；冻结的最小实质改善门槛为 `max(0.1, 5%×2.408230)=0.120412`。实际绝对改善为 **−0.428897**，即反向超过门槛，主成功为 **FAIL**。
- 预注册的固定 50/50 within-pool rank fusion 也未改善：cap16 合并效应 **+0.154527**，95% CI **[−0.004346, +0.326043]**。未根据结果调整权重。
- 有价值的科学信息是：HepG2 的整体 Spearman 与 CNN 相当或略高，但这种全局相关性没有转化为高预算尾部选择收益。同序列其他细胞实验锚点不能直接替代目标细胞的功能排序。

## 2. 冻结数据与隔离

制备阶段只读取 T-cell 的 `utr` 序列键；T-cell `rl` 只在唯一一次最终 Evaluator 中读取。Learner 输入只有冻结三成员 Sample-CNN 分数与同序列实测 HepG2 MRL，不拟合残差模型，也不使用其他序列残差。

| 项目 | 结果 |
|---|---:|
| T-cell r1/r2 共有有效 50 nt、ACGT 序列键 | 314,563 |
| 与两个 T-cell repeat 均 exact 同序列且具有效 HepG2 MRL | 311,707 |
| 排除 Sample train/dev/calibration exact | 261,042 |
| 排除 Sample train/dev/calibration H1-only | 1,917 |
| 排除后 eligible | **48,748** |
| 最终 repeat-linked 唯一序列 | **1,024 = 16 池×64** |
| 池内/跨池 exact 或 H1 | **0 / 0** |
| 池序列对 Sample exact 或 H1 | **0 / 0** |

池按 `SHA256('tcell_cross_cell_anchor_v1|sequence')` 排序，并贪心排除与已保留序列 exact/H1 的候选；前 1,024 条按连续 64 条分池。r1/r2 使用完全相同的序列和 pool membership。原始 Castillo 文件、三个 Sample 角色、三个 CNN checkpoint 均通过 SHA-256 核验。

冻结配置 SHA-256：`09611fd14cf76588c449e7d5d2fd0a8c3315fb31c5a63690c02b186bce11e2b4`。

## 3. 等预算选择协议

- caps：1/2/4/8/16；proposal seeds：101/202/303/404/505。
- 所有方法共享 nested proposal prefix；N 个 proposal 位置按固定随机种子有放回抽取，重复位置不增加唯一序列数。
- CNN、HepG2-anchor 与固定秩融合在相同 prefix 内取分数最高者，平分时取最先出现的位置。
- Random(N) 取相同 prefix 的第一个位置；因此 cap1 四种方法必然相同。
- regret 为完整 64 序列池的真实 T-cell 最大 MRL 减所选序列的真实 T-cell MRL。
- 主 bootstrap 以 16 个 repeat-linked pool 为簇，两个 repeat 和五个 proposal seeds 始终一起重采样。
- Holm 家族为 2 repeats×5 caps 共 10 个 repeat-specific anchor−CNN 检验。正式目标评价仅一次，评价后无协议回调。

CNN 推理使用 RTX 5090 的 `cuda:0`、PyTorch 2.7.1+cu128、确定性算法、TF32 关闭且无 CPU 回退；三个成员的 checkpoint/model hash 全部通过。

## 4. 全 cap 主结果

下表数值均为真实 T-cell MRL regret；effect=HepG2−CNN，**正值表示 HepG2-anchor 更差**。

| Repeat | cap | CNN regret | HepG2 regret | effect | Holm p（10 项） |
|---|---:|---:|---:|---:|---:|
| r1 | 1 | 4.087472 | 4.087472 | 0.000000 | 1.000000 |
| r1 | 2 | 3.512237 | 3.647425 | +0.135188 | 0.418758 |
| r1 | 4 | 3.070485 | 3.286882 | +0.216397 | 0.111989 |
| r1 | 8 | 2.601777 | 2.934772 | +0.332995 | 0.138586 |
| r1 | 16 | 2.376296 | 2.839386 | **+0.463090** | 0.053995 |
| r2 | 1 | 4.013409 | 4.013409 | 0.000000 | 1.000000 |
| r2 | 2 | 3.446259 | 3.439405 | −0.006854 | 1.000000 |
| r2 | 4 | 3.091283 | 3.093325 | +0.002042 | 1.000000 |
| r2 | 8 | 2.615734 | 2.680850 | +0.065116 | 1.000000 |
| r2 | 16 | 2.440164 | 2.834869 | **+0.394705** | **0.017998** |

r2 cap2 有极小点估计改善，但区间跨零且幅度远低于门槛，不能抵消两个 repeat 在 cap16 的一致反向结果。r1 cap16 的未校正 bootstrap p=0.005999，r2 为 0.001800；Holm 后分别约 0.053995 与 0.017998，方向均为有害。

### cap16 的其他等预算方法

| Repeat | CNN | HepG2-anchor | 50/50 rank fusion | Random(16) |
|---|---:|---:|---:|---:|
| r1 | 2.376296 | 2.839386 | 2.589008 | 4.087472 |
| r2 | 2.440164 | 2.834869 | 2.536506 | 4.013409 |

固定融合的 repeat-specific effect 分别为 +0.212712 和 +0.096342；合并 +0.154527，CI 跨零。它只是预注册次要方法，不构成改善证据。

## 5. 次要机制诊断

### 5.1 全局相关性与池内相关性

| Repeat | 预测器 | 1024 序列总体 Spearman | 池内 Spearman 中位数 | 池内范围 |
|---|---|---:|---:|---:|
| r1 | CNN | 0.722091 | 0.713592 | [0.577610, 0.815449] |
| r1 | HepG2 | **0.730001** | 0.713821 | [0.579334, 0.858549] |
| r2 | CNN | 0.724564 | 0.706479 | [0.635378, 0.797070] |
| r2 | HepG2 | **0.750157** | **0.741121** | [0.644551, 0.824267] |

HepG2 的总体相关略高，但选择目标依赖高分尾部而非全排序平均，因此上述差异不能证明选择改善。

### 5.2 top-tail 翻转

- 两个 repeat 均有 **16/16 池**出现 full-pool CNN top1 与 HepG2 top1 不同；top16 集合平均 Jaccard 仅 **0.470432**。
- CNN top16 内共有 840 个非平分 predictor-order 翻转对。r1 中 T-cell 顺序支持 HepG2 402 对、支持 CNN 436 对、平分 2 对；r2 分别为 401、439、0。
- 因而尾部翻转不是单向修正 CNN 错误，反而略偏向支持 CNN 的目标细胞顺序。

### 5.3 选择更替改善/变差分解

cap16 每个 repeat 有 80 个 pool×seed 比较：

| Repeat | 选择改变 | 改善 | 变差 | 标签不变 | 平均所选 T-cell MRL 变化 |
|---|---:|---:|---:|---:|---:|
| r1 | 61 | 21 | **39** | 1 | **−0.463090** |
| r2 | 61 | 22 | **39** | 0 | **−0.394705** |

两个 repeat 的更替数量相同并非人为对齐，而是相同 proposal 和 pool 下的实际结果；有害更替均多于有益更替。

### 5.4 oracle 机会损失

oracle 机会损失定义为当前 N 个 proposal 中可达的最佳真实 T-cell MRL 减所选 MRL；proposal headroom 为全池最佳减 proposal 可达最佳，二者相加严格等于全池 regret。

| Repeat | 方法 | cap16 oracle 机会损失 | proposal headroom |
|---|---|---:|---:|
| r1 | CNN | 1.075351 | 1.300946 |
| r1 | HepG2-anchor | **1.538440** | 1.300946 |
| r2 | CNN | 1.302036 | 1.138128 |
| r2 | HepG2-anchor | **1.696741** | 1.138128 |

方法间 proposal headroom 完全相同，证实预算和随机机会公平；HepG2-anchor 的损失来自相同可达候选中的错误选择，而不是更差的 proposal 机会。

## 6. 三轮真实核验

1. **实现/隔离轮：PASS。** 10 项 focused tests 最终通过；原始哈希、行数、eligible、Sample exact/H1、池内/跨池 exact/H1、repeat-linked membership、CNN checkpoint 和 `cuda:0` 均核验。制备时 `target_mrl_read=false`、`target_evaluation_count=0`。
2. **独立数值轮：PASS。** 不导入主 `anchor.py/evaluate.py`，重新读取原始目标标签，独立重算所有方法的选择/regret、Random(N)、10 项 Holm、10,000 次 bootstrap、Spearman、top-tail、turnover 和 oracle；**236 项检查通过，最大数值差 0**。独立复现主成功为 false。
3. **交付轮：PASS。** 干净进程再次 `10 passed`；冻结配置、原始/派生数组哈希、一次评价开始/完成哨兵、负结果和证据边界通过。本地导出审计通过：无原始表、序列数组、逐序列标签或预测。

开发失败保留而未隐藏：首次配置 JSON 重序列化导致哈希不一致；一次 CUDA 调用缺少 `CUBLAS_WORKSPACE_CONFIG`；旧 CUDA toolkit 对 sm_120 给出兼容警告。这些均发生在目标标签读取前，最终采用无兼容警告的 CUDA 12.8 环境重跑。早期合成测试还发现并修正了融合数组分配错误及两处错误测试期望。所有记录见 `failed_attempts/summary.json` 与日志。

## 7. 产物与最终边界

- 远端新工作区：`/home/proxy-function-gap-tcell-anchor`。
- 远端完整 artifacts：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_cross_cell_anchor_v1`。
- 本地脱敏 source data：`deliverables/reports/source_data/tcell_cross_cell_anchor_v1/`。
- 代码未 push；历史 residual/rank artifacts 未覆盖。

本轮判伪的是“**直接把同序列 HepG2 MRL 当作 T-cell BoN 排序锚点能够改善 cap16 真实选择**”这一冻结假设，也未见固定 50/50 秩融合的收益。它不证明其他细胞测量没有全局信息价值，但不支持把该价值表述成目标细胞高分尾部的实质选择改进。后续若要改变权重、使用非线性组合或细胞特异尾部模型，必须另立科学假设与协议；不得在本结果上回调并继续称同一冻结检验。真正的序列独立或前瞻确认仍需未触碰数据或新实验。
