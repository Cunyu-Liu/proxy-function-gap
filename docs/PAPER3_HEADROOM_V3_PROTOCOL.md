# Paper3 development/query acquisition V3 冻结协议

## 目的与证据边界

本轮只检验开发/校准池中的获取函数与数值降噪，不触碰永久 outer，不修改 Paper2 或外部数据文件。V2 的全部 48 个门禁和原 6 个曲线情境是固定输入，不因 V3 结果增删、换池或调门限。V2 配置 SHA256 为 `1004488ba7d19ccbbc3c10d33b232eb698e70ea77a265bde62af18ad0fb4b4de`，三轮科学字段 SHA256 为 `70db97fdc11f91783015b6d5f5869fd00f5465101f238854b507c9df68b49678`。本协议在查看任何 V3 科学结果前提交。

## 固定队列与门禁

- development 固定合并 `dev_000`–`dev_031`；query 固定 `calibration_000`–`calibration_015`；表示固定 `summary4/kmer4/cnn64`，定义、CNN hash 和全开发缩放规则沿用 V2。
- 对全部 3×16=48 情境重算 zero/full-Q 门禁：`zero_regret-full_Q_regret > 0`。无论正负均保留，不根据阳性选择曲线。
- 曲线只运行 V2 已冻结的 6 个情境：`summary4/calibration_001`，以及 `kmer4/calibration_{000,001,002,013,014}`。即使 V3 门禁改变也运行并标记，不加入新的阳性池。
- 预算固定 0/4/8/16/32/64，部署始终为 posterior-mean top-1，regret 用真实 measured 尺度报告。full-Q 时所有策略必须具有同一信息集与选择。

## 内层开发超参数选择

- 每个表示独立进行 4 折开发池阻断交叉验证，折号为 development pool 编号模 4。每折的 feature/target scaler 只拟合该折训练池。
- 网格固定：prior variance `{0.1,1,10}`，noise variance `{0.01,0.04,0.16}`。评分是四折 held-out development measured-MRL MSE 的等权折均值，不使用任何 calibration/query 标签。
- 选择最小均值；并列时依次选择最接近 V2 prior=1、最接近 V2 noise=0.04、较小 prior、较小 noise。每个表示选出的唯一参数用于该表示的全部门禁和全部策略，完整保存所有网格结果。
- 最终全开发 scaler 和 posterior 仍只由合并 development 拟合。没有 outer 调参，也不根据曲线或门禁重新选择先验/噪声。

## 六种策略与公平约束

固定顺序为 `random/max_variance/diversity/expected_improvement/top_proxy/regret_quadrature`。除 random 使用 V2 的三个种子 20261124/25/26 外，其余各一条；所有策略在同一表示、同一逐步 posterior 和同一真实标签预算下比较。

- `random`：固定种子的排列，按下述统一批内规则顺序接受候选。
- `max_variance`：最大化 latent posterior variance，不含观测噪声。
- `diversity`：最大化到合并 development、已揭示 query 和本批已选 query 的最小标准化特征欧氏距离；距离使用同一当前表示。
- `expected_improvement`：相对当前完整 query posterior-mean 最大值的解析 latent Gaussian EI，`sigma*(z*Phi(z)+phi(z))`；不使用标签 Monte Carlo。
- `top_proxy`：按冻结 ensemble proxy mean 降序，作为简单非自适应基线。
- `regret_quadrature`：对单点真实标签揭示后的 posterior expected best-minus-selected regret 取负；评估集合为完整 query，部署 top-1。

每批 4 个标签，批内模型不因 fantasy 更新。所有策略（包括 random 和 top-proxy）逐一接受候选后，排除本批剩余候选中与其 50nt Hamming 距离 ≤8 者；跨批不额外排除，已揭示候选永久排除。平局按冻结池行序。该规则在查看 V3 曲线前固定，并逐批记录排除数。开发池结构检查显示各 query 池最小两两 Hamming 距离为 19，因此预计规则不活跃；若实际排除为零也原样报告。

## 共同随机数、求积与收敛

- regret 的 posterior sampling 使用固定种子和 antithetic 标准正态共同随机数，候选之间及 S/K 层级之间复用最大层随机前缀。S 固定倍增为 32/64/128。
- hypothetical label 不再用朴素 Monte Carlo：使用 8/16/32 节点 Gauss–Hermite 求积，节点和权重确定，K 与 S 同步倍增。条件 posterior outcomes 用联合高斯条件恒等式构造，避免 V2 双层噪声。
- 每个获取步保存三个 `(S,K)` 层的逐候选 expected regret、首选和最高两层最大绝对分差。最终选择固定使用最高层 `(128,32)`，不因结果追加层数或改种子。
- 数值收敛标记：最高两层首选相同，且最高两层共同候选分数最大绝对差 ≤0.01（标准化目标单位）。未达标只标记，不改变选择或门限；完整报告收敛比例。EI 使用解析公式，作为非 MC 对照。
- 三轮独立真实运行要求全部科学字段一致，包括内层网格、48 门禁、6 情境全部曲线、获取顺序、S/K 收敛字段与 CUDA 元数据；运行时间戳和输出路径不进入科学哈希。

## 判定与报告

主要描述为各表示固定 6 池的逐预算 pool-mean regret，以及逐池全部曲线；random 先对三个种子取池内均值。不以跨池均值掩盖反例，不删除非单调或不利结果。若 regret 或其他复杂策略没有稳定优于 random/max-variance/top-proxy，结论保留简单基线并分析数值、表示、模型错设和池组成原因。V3 是回顾性 development/calibration 诊断，不是新盲测，不声称 outer 改进或普遍 acquisition 优越性。
