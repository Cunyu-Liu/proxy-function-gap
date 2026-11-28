# Paper 1 自我反省与发表导向优化（2026-11-28）

## 1. 现在最大的问题

**不是场景不够多，而是场景之间还没有形成足够硬的共同机制主线。**

V7 已经证明 decision-focused 评估在多个任务中有用，但读者仍可能问：

> 这些数据集只是被同一套指标评估了一遍，还是共同回答了一个科学问题？

目前最有力的回答应该是：

> **proxy 的决策价值取决于高分尾部是否保留真实功能排序，而不是全局预测是否准确。**

T-cell 给出机制深描：全局排序有用，但 top-tail 翻转造成剩余 regret；MAE 改善不改善选择。GB1/GFP/AAV 给出跨任务阳性：尾部富集与等预算收益一致。TEM-1/mRNA stability/indel/TE 给出边界：尾部相关性弱、负或终点错配时，收益消失或反向。

## 2. 场景之间有没有关联

有，但必须写成三层关联：

1. **任务层关联：** 都是“从候选序列池中选择少数功能最优候选”的问题。
2. **指标层关联：** 都用真实功能 regret、等预算 RandomN、top-tail 诊断。
3. **机制层关联：** 都检验全局预测质量与高分尾部决策质量是否脱钩。

不能写成：这些分子/终点在生物学上等价。它们不等价。

## 3. 数据、清洗、泄漏、划分、baseline、超参、OOD 检查

### 数据集是否找全

当前已覆盖：
- mRNA：Sample/Castillo T-cell、HEK293T、HepG2、Leppek、Schlusser；
- 蛋白：GB1、GFP；
- 基因治疗：AAV9 production；
- 酶/抗性：TEM-1。

仍缺：AAV tropism/infectivity 的独立实测、更多蛋白稳定性/表达任务、前瞻湿实验。不能声称覆盖所有热门场景。

### 清洗与质量

- 预测字段与实测字段分列；
- 缺失不插补；
- GB1 lnW 下限保留；
- GFP 亮度不是纯表达；
- AAV Production 不是 tropism；
- TEM-1 是实验派生相对抗性；
- stability 的 spike-in 不作候选，时间点不作重复。

### 泄漏与划分

- T-cell：exact/H1、repeat-linked pools；
- GB1：位点阻断；
- GFP：低阶训练、高阶永久 test，位点互斥；
- AAV：跨库 exact/H1 排除；
- TEM-1：位置簇；
- mRNA：UTR/构建体单位，历史 outer 不重称盲测。

### baseline 是否找全

已保留：RandomN、BoN、CNN、加性 ridge、MSE-only、ListNet、HepG2 anchor、辅助残差/pairwise/多情境门禁。仍缺官方 COM/生成式方法在同一固定池面的公平复现；不能声称已比较所有 SOTA。

### 超参

所有模型选择均限定 train/dev；test 一次评价。T-cell 后续修复失败未回调。AAV/GFP/GB1 均按 dev 选择。TEM-1 无模型拟合。

### OOD

已覆盖：跨细胞、跨构建体、跨终点、跨研究、跨突变阶数、跨药物条件。仍缺真正前瞻独立研究。

## 4. 每个实验的意义

- T-cell 等预算：证明 proxy 排序在特定情境有真实价值。
- T-cell tail 分解：定位剩余损失在高分尾部。
- T-cell 适配/ListNet：证明回归改善不等于决策改善。
- HepG2 anchor：证明跨细胞全局相关不能替代目标尾部排序。
- GB1：蛋白结合中简单加性 proxy 可改善选择。
- GFP：低阶监督可支持高阶选择，但复杂 CNN 不必要。
- AAV：production 重排有强收益，但不等于临床疗效。
- TEM-1：同一 proxy 的价值依赖部署药物条件。
- mRNA stability/indel/TE：终点错配或尾部失效时收益消失。

## 5. 还没做完什么

1. **统一机制的因果证明不足。** tail audit 是关联/诊断，不是干预证明。
2. **缺少一个跨场景可部署的改进方法。** 当前最强贡献是评估原则和边界，不是新 SOTA 模型。
3. **缺少前瞻验证。** 所有结果都是历史数据回顾性。
4. **图形和叙事仍需统一。** 已新增 Figure20/21，V8 将围绕 tail reliability 重写。
5. **投稿元数据未完成。** 作者、单位、利益冲突、许可、目标仓库仍需人工确认。

## 6. 下一步发表导向动作

- 用 V8 把主线改为：全局 accuracy 不足以解释选择价值，tail reliability 是跨场景共同诊断。
- 把 Figure20/21 作为主图，Figure14/16/17 作为 T-cell 机制深描，Figure18/19 作为扩展结果。
- 不再把失败修复包装成性能提升；把它们作为“为什么必须用 regret 验收”的证据。
- 若后续要方法论文，必须新预注册一个 tail-aware 选择/训练方法，并在未查看 test 上证明 regret 改善。