# 跨场景证据综合审计（2026-11-28）

## 1. 总结论

当前证据支持一个**跨场景的评估原则**，不支持一个**跨场景普遍阳性的模型方法**：

> 在生物序列设计中，proxy 的价值必须在真实功能终点、固定候选池和等预算选择下检验；全局相关性和点态误差不能替代功能 regret。proxy 排序在 T-cell MRL、GB1 结合、GFP 亮度和 AAV9 production 中显示可重复的等预算收益，但在 mRNA 稳定性、基因编辑 indel、内源 TE 和 TEM-1 跨药物迁移中没有稳健收益，部分场景甚至显著反向。

这不是“所有 proxy 都有效”，也不是“所有 proxy 都失败”。更准确的贡献是：**decision-focused evaluation 能区分哪些任务中 proxy 真正改善选择，哪些任务中回归指标掩盖了决策失效。**

## 2. 统一主结果表

所有差值均为 `BoN regret − RandomN regret`，负值有利于 proxy。不同终点单位不同，**不得把原始效应量跨场景合并为单一 meta-analysis 效应**。

| 场景 | 真实终点 | 主要 n/簇 | cap16 差值 | 95% CI | 判定 | 关键边界 |
|---|---|---:|---:|---:|---|---|
| T-cell r1 | measured MRL | 16 pools | −1.705760 | [−2.163445, −1.211914] | 阳性；all-20 Holm p=0.004000 | 历史 strict-H1 回顾性 |
| T-cell r2 | measured MRL | 16 pools | −1.086941 | [−1.833685, −0.318589] | 点估计/名义区间阳性；all-20 Holm p=0.064994 | r2 仅 cap4 通过完整校正族 |
| GB1 | IgG-Fc 结合富集 lnW | 55 位置对池 | −0.160894 | [−0.282853, −0.050515] | 阳性 | lnW 不是 Kd；31/55 池改善、24/55 反向 |
| GFP CNN | log 亮度 | 16×1000 池 | −0.260830 | [−0.301866, −0.218031] | 阳性 | 简单加性模型更优；高阶相关衰减 |
| GFP additive | log 亮度 | 16×1000 池 | −0.2800 | [−0.3095, −0.2521] | 阳性 | 不能把收益归因于复杂 CNN |
| AAV9 | Production 富集 | 3,996 disjoint pools | −4.206431 | [−4.564111, −3.931098] | 阳性 | production 不是 tropism/感染/临床疗效 |
| TEM-1 总体 | 实验派生相对抗性 | 52 位置、25方向 | −0.159379 | [−0.207874, −0.113933] | 条件依赖 | Amp→Amp 多数有利；Amp→cefotaxime 15/15 区间跨0 |
| HEK293T N50 | measured MRL | 16 pools | +0.135939 | [−0.989306, +1.135124] | 未支持 | 高 cap 无持续收益 |
| HepG2 defined-end | measured MRL | 16 pools | −1.086886 | [−2.084375, −0.118105] | 名义阳性，校正不支持 | Holm p=0.303970 |
| Castillo indel | normalized indel | 48 UTR、4情境 | −0.2308 至 +0.0462 | 各情境见表 | 0/4 校正阳性 | 小样本；8/48 与 Sample 有 exact/H1 接触 |
| Castillo stability | integrated abundance | 27 experimental UTR | +0.2838 | [+0.0824, +0.4435] | 显著反向 | 8 spike-in 不是候选；6时间点不是重复 |
| Leppek MRL | measured MRL | 59 constructs/49 UTR | +0.104723 | [−0.100964, +0.212706] | 未支持 | 历史 Random1 协议，不与新 RandomN 混排 |
| Schlusser TE | endogenous TE | 6,309 genes | +0.626608 | [−0.828459, +9.998249] | 未支持 | 历史 Random1；TE 极端右偏 |

## 3. 可以跨场景成立的结论

### C1：等预算功能 regret 是必要验收终点

- T-cell：50k 适配显著降低 MAE，却不改善 cap16 regret；ListNet 也未达到实质门槛。
- GFP：总体 Spearman 0.434，9阶为 −0.020，但固定池选择仍显著获益。
- AAV9：Spearman 0.821，但真实 top1% 召回仅 18.75%，剩余 regret 非零。
- GB1：MAE 2.159、Spearman 0.348，但 cap16 regret 显著低于 RandomN。
- TEM-1：Amp→Amp 全局相关中位数 0.312，顶部十分位降至 0.132；跨药物相关接近零且选择收益未解析。

**允许表述：** 点态误差、全局排序和真实选择收益是不同量；模型开发应以功能 regret 为主终点。
**禁止表述：** MAE 无用、相关性无用、所有 ranking loss 都失败。

### C2：proxy 排序价值具有任务和部署条件边界

- 阳性：T-cell MRL、GB1 lnW、GFP 亮度、AAV9 production、TEM-1 Amp→Amp。
- 未支持/反向：HEK293T 高 cap、Castillo indel 校正后、mRNA stability、Leppek MRL、Schlusser TE、TEM-1 Amp→cefotaxime。

**允许表述：** 同一评估框架揭示任务依赖的 proxy 价值。
**禁止表述：** decision-focused 方法普遍有效；proxy 普遍失效；AAV production 收益等于基因治疗疗效。

### C3：简单基线必须保留

- GFP additive 在四个预算均优于 CNN。
- GB1 使用共享替换效应的加性 ridge 已足够产生收益。
- T-cell ListNet 相对原 CNN 仅小幅、未达门槛。

**允许表述：** 复杂模型不是收益的必要条件；必须与简单可审计基线比较。
**禁止表述：** CNN/深度学习天然优于加性模型。

## 4. 统计与协议不可混写之处

1. **RandomN 实现不同：** GB1/TEM-1 使用均匀随机选择的精确期望；GFP 使用64个固定随机重复；AAV 使用同池随机抽取；T-cell 使用共享 nested prefix 的 Random(N)。它们都回答等预算随机选择，但 Monte Carlo 方差处理不同，不能声称完全同一实现。
2. **簇单位不同：** T-cell/GFP 为 pool，GB1 为位置对池，TEM-1 为位置，AAV 为 disjoint pool，indel/stability 为 UTR。区间只描述各自库内抽样不确定性。
3. **终点语义不同：** MRL、lnW、log亮度、Production、indel、integrated abundance、TE、相对抗性不可互换。
4. **证据等级不同：** 全部为历史公开数据回顾性分析；没有新前瞻湿实验。AAV/GB1/GFP 的永久 test 是历史数据内严格隔离，不是独立研究。
5. **多重性不同：** T-cell 使用 all-20 Holm；indel/stability 使用各自20项 Holm；TEM-1 逐方向区间未作多重校正；AAV cap4/64 有次级 Holm。不能把未校正区间写成确认性多重检验。

## 5. 推荐论文结构

### 推荐：一篇主文 + 场景扩展结果

主标题建议：

> **Decision-focused evaluation reveals task-dependent value of sequence proxies across biological design**

结构：

1. **主问题：** 为什么预测准确不等于选得对？
2. **核心机制案例：** T-cell MRL 的等预算收益、top-tail 瓶颈、MAE–regret 脱钩。
3. **跨场景外部有效性：** GB1、GFP、AAV9 production 的阳性；TEM-1、mRNA stability/indel/TE 的边界。
4. **统一结论：** proxy 价值是任务/终点/部署条件的函数；功能 regret 应作为模型开发验收终点。
5. **限制：** 回顾性、历史数据内隔离、终点语义差异、无前瞻湿实验。

不建议把五类场景包装成五篇独立阳性方法论文；它们共享同一评估问题，合并为一篇主文更有逻辑力量。

## 6. 必须修正/避免的叙事问题

- 不把 AAV Production 写成 tropism、infectivity、transgene expression 或 clinical efficacy。
- 不把 GB1 lnW 写成 Kd 或直接结合自由能。
- 不把 TEM-1 派生 fitness 写成直接酶活、MIC 或临床药物反应。
- 不把 GFP 亮度写成纯表达量。
- 不把 stability 的6个时间点当6个生物重复，不把8个 spike-in 当候选。
- 不把 Castillo 48条中仅12条实测 MRL 与45条作者预测 MRL 混用。
- 不把不同单位的原始 regret 差值做跨场景平均。
- 不把历史数据内 grouped holdout 写成新盲测或独立研究复现。

## 7. 三轮独立核验

配套脚本 `verify_cross_scenario_synthesis_20261128.py` 从本地冻结 JSON/TSV 重新读取关键数值，检查：

1. 五类场景主效应、CI、终点语义和样本单位；
2. 阳性/边界分类与禁止跨单位合并；
3. 报告数值、来源哈希和交付完整性。

三轮结果见 `CROSS_SCENARIO_SYNTHESIS_VERIFICATION_20261128_ROUND{1,2,3}.json`。