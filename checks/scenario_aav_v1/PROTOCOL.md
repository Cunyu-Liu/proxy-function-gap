# AAV9 production 固定池实验协议（结果前冻结）

日期：2026-11-27。实验标识：`scenario_aav_v1`。

## 科学边界

本实验只检验公开 Fit4Function 存档中 AAV9 VP1 588–589 间 7-mer 插入变体的 **production fitness** 排序是否能改善等预算固定池决策。`Production` 是核酸酶抗性病毒基因组相对 DNA library 的 VRPM/DRPM 富集比，不是 tropism、感染、转基因表达或临床疗效。全部数据为回顾性公开数据，不是新盲测或湿实验验证。`Meeting6` 和任何模型预测均不作为实测标签。

## 数据与许可

只使用固定 commit `6bfc2ebfe4abcd45cc6fe737e4700242a5090fee` 的 modeling/assessment production CSV。SHA-256 必须分别等于 `6aabda2d…87d41` 与 `0ab5570b…5a07`，且原始精确 AA 重叠必须为 9,993；否则停止。上游代码许可证为 BSD-3-Clause；未发现数据统一再分发许可，因此只开展内部分析，本地只同步聚合结果、文件哈希和无序列的伪匿名 ID，不再分发原始序列。

## 标签与角色

任务单位为 AA，两个 codon replicate 保留在原行聚合，不拆成独立样本。保留 `Production=0`，在任何划分前排除非有限值并报告数量；不得按标签高低选池。modeling 只用于 train/dev，assessment 只用于外测。外测排除与 modeling 任一 AA（包括非有限标签行）Hamming 距离 ≤1 的所有序列。modeling 内以 Hamming ≤1 连通簇为单位划分，使用冻结种子 20261127 洗牌并依序装入 dev，目标 15%，不得跨簇。划分不使用功能标签。

## Proxy 与收敛

编码为 20 字母整数 token 和可微 one-hot；两层 kernel=3 的位置保持 1D 卷积（64 通道）后展平接 128 单元 MLP，dropout=0.1。训练标签仅用 train 均值和标准差标准化，MSE + AdamW（lr=0.001，weight decay=0.0001），batch=512，最多 300 epoch。以 dev MSE 唯一早停，patience=20，min delta=1e-6；若达到 epoch 上限仍未早停则不宣称收敛。种子 11/23/37 的最佳 dev checkpoint 等权平均为冻结 proxy，不按外测 regret 选种子或调参。

## 等预算终点

唯一主终点为 cap16 的平均配对 `BoN regret − RandomN regret`，单位为原始 Production 富集比。对每个 N，从外测序列随机排列后构造互不重叠的固定池，丢弃不足 N 的尾部；同一池同一 N 个候选同时供 BoN 和 RandomN。BoN 选 proxy 分数最高者，RandomN 独立均匀抽一个；oracle 仅用实测 Production 计算 `max(y_pool)-y_selected`。N=4/64 为次要终点并做 Holm 校正，N=1 只作等预算 sanity。池生成种子基值 20261127，RandomN 种子基值 20261128。

主区间为 2,000 次 disjoint-pool 配对 bootstrap 的双侧 95% percentile 区间，种子基值 20261129。负 regret 差表示 BoN 优于 RandomN；正差和零差均如实保留。另报告逐种子敏感性、Spearman/Pearson、proxy top 1%/5%/10% 的实测均值、真实 top-tail 召回和相对全池均值，均为诊断而非额外主假设。

## 三轮核验

1. 原始轮：固定 URL、commit、许可证、原始哈希、字段、行数、9,993 精确重叠和近邻排除。
2. 独立轮：从原始 CSV 独立重建序列划分，重载最佳 checkpoint，在外测序列重算预测，并从固定候选矩阵重算所有 BoN/RandomN/oracle/regret 与区间。
3. 交付轮：核对本地脱敏 manifest、SHA-256、主结果算术、日志与报告，不允许原始 AA/codon 序列进入本地 artifacts。

只有全部准入、隔离和运行核验通过才报告性能；任何失败均保留，不能改用 Meeting6 或模型预测冒充实测。
