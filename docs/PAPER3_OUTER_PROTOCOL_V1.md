# Paper3 永久 outer 一次性回放协议 v1（预注册，未揭盲）

## 状态与授权边界

本协议于 2026-11-25 在任何 Paper3 outer Learner 准备、outer 标签读取或 outer 评分之前冻结。当前只允许 `calibration_dry_run`；Paper2 P0 仍由另一工作包完成，在其门禁明确放行前不得执行正式 outer Learner 或 Evaluator。不得读取永久 outer 标签、逐序列评分或既有 outer 汇总来修改本协议。

永久 outer 已在 Paper1/Paper2 历史研究中出现评分，因此即便本工作包未读取这些结果，正式运行也只能称为 **Paper3 冻结的一次性历史回放**，不能称为全课题从未使用过的新独立盲测，也不能称为独立研究或前瞻湿实验验证。

## 从 V4 继承且不再调整的内容

- V4 配置规范哈希 `d7b0438970b43ff8898924343e1b60f708201a55763f41b1fa4d964440538bd6`，三轮 calibration 科学哈希 `c3eb20bb082c7e8e40fbaeb7c8229b3d84485a5076926fd6c2ba756c731be0e4`；V4 不重跑。
- 表示固定 `kmer4`；Bayesian ridge prior variance=0.1、noise variance=0.16，均来自 V4 development-only 四折阻断 CV，正式 outer 不再 CV。
- 六策略及顺序固定：random、max-variance、diversity、expected-improvement、top-proxy、regret-analytic。random 种子固定 20261124/25/26，其他策略和求积种子沿用 V4。posterior-mean top-1 为唯一部署规则。
- regret-analytic 固定 64 标签节点、32/64/128 价值节点、query block=4、收敛阈值 1e-5；其独立高斯对角协方差近似不是联合后验精确闭式解。
- 所有拟合、推断、获取评分均必须真实使用 CUDA:0、float64；CUDA 不可用或任何 CPU 静默降级立即停止，不生成科学评分。

## Q/E 构造与隔离

每个 64 序列 outer 池在 Learner 无标签视图中按 `sha256(split_seed, pool_id, group_id)` 排序并贪心平衡为 32 个 query 和 32 个 evaluation 序列。分配只使用 pool ID、group ID 和固定种子，不使用 measured label、proxy 分数或既有 outer 指标。同一 group 必须完整进入同一侧；跨 Q/E exact 或 Hamming-1 冲突均使该池 fail closed，不通过移动序列修复。

development 仍固定合并 `dev_000`–`dev_031`。feature/target scaler 只在 development 拟合。Learner 可读取 development 标签、Q/E 无标签特征及经账本揭盲的 Q 标签；不得读取 E 标签。Q 标签只能由独立 vault 按唯一 query ID 返回。部署选择写入含 E row ID 的冻结清单后，Evaluator 才可读取 E 标签并只返回聚合 regret，不返回逐序列标签或中途评分。

所有策略使用相同 development、Q/E、表示、posterior 更新和预算。批大小 4；批内 Hamming≤8 排除、跨批不额外排除、平局按冻结行序。预算固定 0/4/8/16/32；full-Q=32 时所有策略必须使用同一信息集并作同一 E 选择。

## 队列、主假设与终点

- 主队列：`outer_designed_000`–`outer_designed_015` 全 16 池，不按 headroom、V4 表现或 outer 结果筛池。
- 次要队列：`outer_random_000`–`outer_random_002` 三池单独报告，不与 designed 合并，不作主结论。
- 唯一主比较：expected-improvement 减 random 在 16 个唯一 Q 标签时的真实 measured-MRL top-1 regret 差。random 先在每池内对三个冻结种子取均值，再计算逐池配对差和 16 池等权均值。
- 唯一主成功判据：以完整 outer pool 为重采样簇，固定种子 20261125、10000 次 percentile bootstrap，95% 区间的上界小于 0。序列、random 种子和 bootstrap 次数不得冒充独立生物证据。
- 次要终点：0/4/8/16/32 全曲线、0–16 trapezoidal AUC、六策略逐池 regret、zero/full-Q headroom、regret-analytic 收敛和 random outer 分层结果。次要终点不替代主终点，不做事后主终点切换。

## 门禁、停止与失败保留

正式运行前必须同时满足：Paper2 P0 书面放行；本配置、实现和测试哈希冻结；三轮 calibration-only 演练科学字段一致；Q/E、development 隔离、预算/full-Q、CUDA 和独立统计复核全部通过。任一失败均停止且不得揭盲。

一旦正式 outer 开始，所有池、策略、预算和种子必须一次完整执行，禁止依据中途 outer regret 早停、换策略、换表示、改超参、改主预算或补种子。缺标签或 Q/E 隔离失败的池在评分前整池排除并记录；不得插补或部分池评分。若主区间包含 0，则主假设不成立；若 EI 不优于 random/max-variance/top-proxy，保留简单基线；若 zero/full-Q 无 headroom，则不声称标签修复有效。所有负结果、非单调曲线和逐池反例均保留。

正式 outer 只允许一个新输出目录的一次评分。任何冻结后修改都必须建立新协议版本、新独立外测资源，并把当前 outer 降级为已使用开发证据；不得覆盖本轮文件。

## 本轮交付而非科学结论

三轮演练只验证实现、隔离、预算、CUDA 和确定性，calibration 数值不得写成 outer 结论。当前交付不包含 outer query 清单、outer regret、outer 区间或任何 outer 有效性主张。
