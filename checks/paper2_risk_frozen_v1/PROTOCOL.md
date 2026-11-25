# Paper 2 ensemble-std 拒选：冻结协议 v1

## 证据边界

本工作包只读取冻结预测、历史选择和实验标签数组，在 CPU 上统计；不训练模型、不重新推理、不进行 CUDA 验证。历史 outer 标签和经验 risk–coverage 曲线已经查看，因此即使通过 calibration 门禁，新的 outer 评分也只能称为“冻结阈值的回顾性确认”，不能称为未接触盲测。

旧实现以 pool×seed 的 BoN top-1 选择决策为单位，把所选候选的 ensemble std 作为风险；它不是 pool-level 风险，也不是池内全候选过滤。旧曲线按当前评分角色的风险次序统计量形成经验覆盖率，outer 曲线因此使用了 outer 分位点。

## 预注册规则

1. 决策单位为每个 pool×paired-seed 的一条 BoN cap16 top-1 决策，种子固定为 101、202、303、404、505。
2. 风险为所选候选的三成员 ensemble 标准差。`risk <= threshold` 时接受原 BoN 选择，否则整项拒选，不用次优候选替代。
3. dev 的风险分位数网格固定为 `{0.50, 0.75, 0.90}`，使用 NumPy linear 分位数。分别计算全量 BoN 平均 regret 减接受决策平均 regret；选降幅最大者，平局依次选 dev 覆盖率更高者、分位数更低者。
4. calibration 只评估 dev 选中的一个阈值，不重新选分位数。通过条件同时包括：至少 8 条接受决策、至少 3 个接受池、点估计降幅大于 0、池簇 bootstrap 95% 区间下界大于 0。
5. 主终点为 `mean regret(all BoN decisions) - mean regret(accepted decisions)`，正值支持拒选。覆盖率是共同报告的运行特征，不是成功定义。
6. 区间以完整池为重采样簇，保留每池全部五个配对种子；10000 次 percentile bootstrap，seed=20261125，报告 2.5% 和 97.5% 分位点。点估计按接受决策加权，bootstrap 同时重算全量与接受均值。
7. calibration 门禁失败即停止，不评分 outer。通过后，在同一次调用中评分 outer_designed（主）和 outer_random（次要），之后不得改变阈值、分位数或主终点。
8. outer_random 只有三个合格池，只作次要证据。任何区间跨 0、反向或低覆盖结果均保留，不以历史经验曲线替换冻结阈值结果。

## 数据与复现

- dev/calibration：从 `pool_validation/production_v3` 原始候选表和允许的 Learner 标签，统一使用 NumPy 固定种子提案重建 BoN，与 outer baseline 的提案算法一致；dev 逐条对照既有 baseline evaluation。旧 calibration selection 使用不同的 Python `random.Random` 提案算法，只作异协议审计，不作逐条身份对照或主分析输入。
- outer：只读取 `paper2_baselines_v1` 已冻结的 selection/evaluation 配对记录，验证文件哈希和记录一致性；不产生新推理。
- 三类核验：原始字段及聚合复算；独立脚本与 pytest；交付物—源数据哈希及报告数值一致性。
- 所有新文件写入 teams 的 `paper2_risk_frozen_v1`，不修改原始数据、Paper 3 或论文主文件。
