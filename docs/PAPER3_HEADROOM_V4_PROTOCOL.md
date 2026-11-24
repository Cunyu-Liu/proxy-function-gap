# Paper3 development/query analytic regret acquisition V4 冻结协议

## 目的与证据边界

V4 只替换 V3 `regret_quadrature` 中的双层 Monte Carlo 数值方法，新增 `regret_analytic`，并在完全相同的 development/calibration 边界内重跑六策略。全部 48 个 zero/full-Q 门禁和 V2 冻结的 6 个曲线情境保持不变；不读取、评分或调优永久 outer，不修改 OOD、Leppek、Paper2 或外部数据文件。V3 配置 SHA256 为 `b0ef0257366c11861e0ff07d8223cc895b47531c1cd1940641795f22ef13a5b9`，三轮科学字段 SHA256 为 `163d03fb6b88d0a18a25b7e3a829363fd20075382e39f0e3e0612945361c4a81`。本协议及实现须在查看任何 V4 门禁或曲线科学结果前提交。

## 固定队列、内层选择与公平约束

- development 固定合并 `dev_000`–`dev_031`；query 固定 `calibration_000`–`calibration_015`；表示固定 `summary4/kmer4/cnn64`，CNN checkpoint/hash、开发缩放和池清单沿用 V3。
- 每个表示独立执行 V3 的 4 折 development-pool 阻断 CV，网格固定 prior variance `{0.1,1,10}`、noise variance `{0.01,0.04,0.16}`，只用 held-out development measured-MRL MSE 选择。不得用 query/outer 标签或 V4 曲线改选参数。
- 全部 3×16=48 情境重算 `zero_regret-full_Q_regret > 0` 门禁，正负均保留。曲线固定 `summary4/calibration_001` 和 `kmer4/calibration_{000,001,002,013,014}`，不得按 V4 阳性增删。
- 预算固定 0/4/8/16/32/64，批大小 4，批内 Hamming≤8 排除规则、冻结池行序平局规则和 query 唯一账本沿用 V3。random 固定种子 20261124/25/26，其余策略各一条；部署始终为 posterior-mean top-1，regret 用真实 measured-MRL 尺度。
- 固定顺序为 `random/max_variance/diversity/expected_improvement/top_proxy/regret_analytic`。前五种策略定义与 V3 完全一致，第六种只是同一 one-step fantasy expected-regret 目标在明确独立高斯近似下的确定性实现。

## V4 解析目标与独立高斯近似

对评估候选 latent outcomes `Y∈N(μ,C)` 和待查询点 `q`，观测 `T=qβ+ε`，`s_q²=qΣq+noise_variance`。对标准正态标签节点 `z` 做 fantasy posterior 更新：

`μ'_i(z)=μ_i + Cov(Y_i,T) z/s_q`，`C'=C-Cov(Y,T)Cov(Y,T)^T/s_q²`。

V4 **显式对角化 fantasy 后验协方差**，即假设更新后的候选价值相互独立，方差 `v'_i=C'_{ii}`。部署选择 `a(z)=argmax_i μ'_i(z)`，one-step top-1 expected regret 为

`R(q)=E_z{E[max_i Y'_i | T]-max_i μ'_i(z)}`。

给定 `z`，独立正态最大值的期望用一维求积精确逼近：

`E[max Y']=Σ_i ∫ (μ'_i+sqrt(v'_i)x) φ(x) Π_{j≠i} Φ((μ'_i+sqrt(v'_i)x-μ'_j)/sqrt(v'_j)) dx`。

标签积分与上述价值积分均使用标准正态 Gauss–Hermite 节点/权重，不使用 posterior 或 hypothetical-label Monte Carlo。最终分数为 `-R(q)`。这里的“解析/精确”只指该独立高斯目标消除了两层 MC；它不是原始相关联合后验 multivariate-max 的闭式解，独立近似及其影响必须如实报告。

## 节点、收敛与数值对照（在看到 V4 科学结果前冻结）

- 标签节点在三层均固定为 64，价值节点层固定为 `(32,64,128)`，最终选择固定使用 `(64,128)`。候选按固定块大小 4 在 CUDA 上并行计算；块大小只影响 kernel 调度，不改变数学目标或候选顺序。
- 每个获取步保存三层逐候选分数、首选和最高两层最大绝对分差。收敛标记要求最高两层首选相同，且最大绝对分差 ≤`1e-5`（标准化目标单位）。未达标只标记，不追加节点、不改门槛或改选候选。
- 数值开发修订记录：初版协议提交 `cd6f3cf` 曾设 label/value 均为 `(16,32,64)`。在任何 V4 门禁或曲线运行前，64 候选随机内层数值例显示标签节点同步倍增导致约 `1e-3` 的最高层振荡；因此透明修订为固定 64 标签节点和 `(32,64,128)` 价值节点。五个独立随机数值例（种子 1–5）均通过上述 `1e-5` 门槛。该修订只使用合成数值精度和运行成本，未查看或使用任何 V4 科学结果。
- 固定微小池数值对照：(a) 二候选逐标签 fantasy 重拟合的闭式 normal-max 公式；(b) 三候选、三查询逐查询/逐标签/逐部署选择的穷举标量积分；(c) 固定种子的 antithetic 高采样嵌套 CUDA MC，标签 4096、独立 posterior 32768。保存逐查询分差和 MC 标准误，不用 MC 结果调节点或门槛。
- 所有获取分数、模型训练/推理和数值 MC 均在 CUDA 上执行；穷举标量公式仅作独立 CPU 数学审计，不参与获取或科学推断。

## 三轮、判定与报告

三次独立完整真实运行须使内层网格、48 门禁、6 情境全部曲线、获取顺序、数值对照、求积收敛字段与 CUDA 元数据的科学哈希一致；路径和时间戳不进入科学哈希。独立验证器重算预算、门禁、target scale、full-Q、批内约束和 outer 隔离。

完整报告包括 48 池门禁、六策略 pooled 曲线、6 池逐池曲线、384 个 regret 获取步的收敛统计、微小池/MC 对照和全部负结果。random 先对三个种子取池内均值，不以 pooled 均值隐藏反例。若 `regret_analytic` 或其他复杂策略没有稳定优于 random/max-variance/top-proxy，保留简单基线并分析独立近似、模型错设、top-1 离散跳变和池组成原因；不得追求阳性或声称 outer/普遍 acquisition 优越性。
