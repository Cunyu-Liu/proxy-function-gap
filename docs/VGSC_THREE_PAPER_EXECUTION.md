# VGSC 三个 Paper 嵌入点：执行与门禁记录

日期：2026-11-24

本记录依据冻结产物和现有源码核对三个嵌入点，避免重复执行已完成实验。所有科学结果均来自历史实测 MRL 的独立 Evaluator 评分；训练、开发、校准和 proxy 指标不作为最终功能结论。

## 嵌入点一：Paper 1 压力干预与机制归因

状态：**已执行，按回顾性固定池证据范围形成预印本证据包。**

- 固定池家族、V-only/联合引导、WSC、greedy、BoN/Random、α 与预算网格已在 E0 和生产 pilot 中实现。开发集冻结 FP-REJ 的 α=20、cap=16；未在外层结果上回调 α。
- Sample designed 与 random、Castillo-Hair 四个外部情境、替代 N50 池均完成真实 CUDA 推理和历史标签 regret 评分。Sample designed 的 BoN regret 随 cap 单调下降；四个外部情境未复现持续单调趋势，结论限定为情境依赖。
- 分数匹配四个差值区间均跨 0，不支持额外的优化器特异失真；同架构和 4-mer MLP 跨架构审计显示共同 proxy 风险。归一化输出噪声未产生单调退化，保留为负结果。
- 原始数据没有符合条件的完整逐步实测轨迹，因此不声称历史轨迹重放；论文采用合同允许的固定池降级路线。Castillo 是同研究的外部情境，不冒充独立研究元分析。

证据：`docs/PILOT_V3_RESULT.md`、`docs/PAPER1_EXTERNAL_RESULT.md`、`docs/PAPER1_MECHANISM_RESULT.md`、`docs/PAPER1_CROSS_ARCH_RESULT.md`、`docs/PAPER1_SENSITIVITY_NOISE_RESULT.md` 和 `paper/PAPER1_DRAFT.md`。结果已写入对应 PPT。

## 嵌入点二：Paper 2 风险决策与公平基线

状态：**已执行完整证伪 pilot；主线门禁失败，停止方法推进，不再调参。**

- V 点态过高估计头、Q 池级风险头、FP-REJ/FP-TILT、WSC、greedy、V-only、LCB 和实际调用计数均已实现。生产选择执行真实 CUDA forward，Learner/Evaluator 分离。
- designed 外层 16 池：FP-REJ without Q 的 regret 为 1.0242708233，BoN 为 0.6811140309；差值 +0.3431567924，池簇 bootstrap 95% 区间 [0.2250717297, 0.4670981804]，不利于 FP-REJ。
- FP-REJ+Q 在 designed 上覆盖率为 0，regret 不定义。三个 eligible random 池的小差值区间跨 0，不能支持一般优势。
- 停止条件已触发：匹配预算下不优于 BoN，且 Q 的覆盖失效。不得查看外层后调整阈值、α 或池构造再称同一盲测；不得将 VGSC 写成 Paper 2 有效方法。保留预算、噪声、负结果和基线价值。

## 嵌入点三：Paper 3 获取函数与部署选择器一致

状态：**条件未触发，不启动 Paper 3 主实验，不修改不存在的部署选择器。**

方案要求只有在 Paper 2 最终采纳倾斜部署选择器后，才把 regret-oriented 获取函数中的 \(i_D\) 从后验均值 argmax 改为倾斜分布采样。当前 Paper 2 明确未采纳该选择器，且三论文门禁要求前序命题成立后再推进，因此现在实现倾斜 \(i_D\) 会与实际部署不一致。

仓库当前没有 Bayesian ridge 标签获取主实验或 Paper 3 结果，不能声称已经完成标签效率结论。若未来由新的、未污染的外层证据重新开启 Paper 3，必须先冻结实际部署选择器，再让获取函数中的 \(i_D\) 与之一致；Q/E 隔离、近重复约束、统一 posterior 和隐藏评估标签要求不变。

## 本次核验边界

- 真实 RTX 5090 CUDA 计算通过，未发生 CPU 静默降级。
- 完整测试 220 项通过，源码/大文件边界审计和论文主张审计通过。
- 本地 `HEAD`、`origin/master` 与 GitHub 远端一致；有效代码已上传。
- 不为占用 GPU 而重复已收敛模型或运行无科学目的任务，不重跑已验证产物。
