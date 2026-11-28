# AAV9 production 场景实验：严格隔离下的等预算决策收益

日期：2026-11-27 至 2026-11-28
实验：`scenario_aav_v1`
状态：三种子 CUDA 训练、外测与三轮核验完成

## 结论

在 Fit4Function 的公开 **AAV9 production** 存档中，严格排除跨库精确与 Hamming-1 近邻后，冻结序列 proxy 的 cap16 BoN 相比同池同候选 RandomN 显著降低实测 regret：平均配对差 **−4.206431** 个原始 Production 富集单位，95% disjoint-pool bootstrap CI 为 **[−4.564111, −3.931098]**。因此，本场景支持“decision-focused proxy 排序可具有等预算决策价值”。

这只是单一血清型、单一终点的**回顾性**固定库重排证据，不支持普遍跨场景阳性，也不是新盲测、de novo 设计或前瞻湿实验。Production 是核酸酶抗性病毒基因组相对 DNA library 的富集，**不是 tropism**、感染、转基因表达或临床疗效。

## 数据准入与隔离

- 固定 Fit4Function commit：`6bfc2ebfe4abcd45cc6fe737e4700242a5090fee`。
- modeling / assessment 原始 SHA-256 分别为 `6aabda2d…87d41` / `0ab5570b…5a07`，与预审计一致。
- 原始行数 74,464 / 74,467，有限 Production 74,435 / 74,440；零值不删除。
- 直接按序列集合确认 **9,993** 个跨库精确 AA 重叠，不依赖 Calibration 标签。
- 另排除 508 个非精确 Hamming-1 重叠，总近邻排除 10,501；外测保留 63,941，其中 6,669 个零值。
- train/dev/test 为 **63,270 / 11,165 / 63,941**。modeling 内以 Hamming≤1 连通簇划分 train/dev；test 对全部 modeling AA 均无精确或 Hamming-1 交叉。
- 上游仅明确代码 BSD-3-Clause，未找到数据统一再分发许可；本地交付不含原始 AA、codon、CSV、checkpoint 或可还原外测序列的数组。

## 真实训练

7-mer 以 20 字母整数 token 和 one-hot 编码，经两层位置保持 1D CNN 与 MLP 输出 Production。训练只读取 train/dev，标签用 train 均值/标准差标准化；dev MSE 唯一控制早停。三个种子都在 RTX 5090 上真实 CUDA 训练并收敛：

| Seed | 最佳 epoch | 最后 epoch | 最佳 dev MSE（标准化） |
|---:|---:|---:|---:|
| 11 | 13 | 33 | 0.872971 |
| 23 | 12 | 32 | 0.864951 |
| 37 | 16 | 36 | 0.875280 |

三个最佳 checkpoint 等权集成，未按外测 regret 选模型。

## 等预算实测 regret

同一池、同一组 N 个候选同时供 BoN 与 RandomN；RandomN 从这 N 个候选均匀抽一个。regret 均为 `oracle 实测 Production − 所选变体实测 Production`，越低越好。cap16 是唯一主终点，cap4/64 为次要，cap1 为 sanity。

| Cap | 池数 | BoN 平均 regret | RandomN 平均 regret | 配对差 BoN−RandomN | 95% bootstrap CI |
|---:|---:|---:|---:|---:|---:|
| 1 | 63,941 | 0.000000 | 0.000000 | 0.000000 | [0, 0] |
| 4 | 15,985 | 0.329732 | 2.099629 | −1.769897 | [−1.863906, −1.692641] |
| **16** | **3,996** | **2.216051** | **6.422482** | **−4.206431** | **[−4.564111, −3.931098]** |
| 64 | 999 | 7.224161 | 13.888537 | −6.664376 | [−7.818690, −5.880822] |

cap16 中 BoN / RandomN / oracle 的平均所选 Production 分别为 **5.164374 / 0.957943 / 7.380425**。BoN 有明显收益，但 regret 仍非零，不能表述为达到 oracle。

cap16 单种子配对差分别为 −4.169844、−4.173694、−4.064315，方向一致。cap4/64 经 Holm 校正后仍为阳性；p 值下溢时保留机器输出，不把其解释为数学上的绝对零概率。

## 排序与 top-tail 诊断

- 外测 Spearman = **0.820908**，Pearson = **0.470005**；重尾分布下二者差异明显。
- Proxy top 1% 的实测均值 **8.172255**，是全池均值的 **8.471699 倍**，但真实 top 1% 均值为 **20.603078**，top-tail 召回仅 **18.75%**。
- Top 5% / 10% 的实测均值为 6.160086 / 5.031465，真实 top-tail 召回为 45.75% / 60.89%。

因此“平均决策改善”与“完整恢复极端最佳变体”不同；顶部漏检和剩余 regret 是需要保留的负向诊断。

## 三轮核验与失败记录

1. **原始轮 PASS**：重算固定源哈希、字段、行数、9,993 精确重叠、额外近邻、零值保留和跨 split 隔离。首次许可证检查因原文 `BSD 3-clause` 大小写而停止，修正字符串断言后通过；科学配置未变，失败日志保留。
2. **独立轮 PASS**：独立从 CSV 重建 train/dev/test、重载三个 CUDA checkpoint、重新生成固定池并复算所有 regret、p 值和区间。1024-batch 重推理与原 2048-batch 最大差 0.004278，未直接放宽阈值；统一 batch 后三种子及集成最大差均为 **0.0**，所有 cap 的 regret 与 CI 差异也为 **0.0**。两次失败日志均保留。
3. **交付轮 PASS**：本地 manifest 逐文件 SHA-256、脱敏边界、逐池 regret 算术、cap 汇总、报告数值与结论边界由 `verify_delivery.py` 复核；详见 `verification/round3_delivery.json`。

## 范围与后续

首选 Fit4Function 的原料、许可内部分析条件和隔离均满足，因此本轮没有启用 Bryant AAV2 packaging 备选，也没有获得跨血清型或跨终点的普遍结论。`Meeting6` 未进入任何训练、标签或评估。所有历史公开数据只用于回顾性 benchmark；后续若要声称 production 之外的功能价值，必须分别获取相应实测终点并重新冻结隔离与统计协议。
