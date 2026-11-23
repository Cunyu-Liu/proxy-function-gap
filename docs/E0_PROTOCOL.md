# E0 冻结协议：VGSC-inspired fixed-pool

## 1. 范围与命名

E0 只验证固定完整序列候选池上的软件正确性、合成 regret 和实际计分成本。实现称为 **VGSC-inspired fixed-pool**，不是生成式 VGSC 的官方复现。

- **FP-REJ（主方法）**：完整序列提议上的有界拒绝采样。
- **FP-TILT（解析对照）**：有限候选池上的精确 V 倾斜分布及确定性 top-k。

不使用真实标签，不训练模型，不调用 GPU，不生成前缀或新序列，不进入 E1。

## 2. 候选池、约束与提议

候选池 \(C=\{x_1,\ldots,x_M\}\)，基础分布 \(P_i\geq0\)、\(\sum_iP_i=1\)。UTR 长度、GC、uAUG 和 poly-U 形成硬掩码 \(m_i\)：

\[
\tilde P_i=\frac{m_iP_i}{\sum_jm_jP_j}.
\]

不可行候选不提议、不计分。无可行正概率质量时报错。BoN/WSC/greedy/FP-REJ 的提议有放回；任何最终 top-k 都按序列身份去重、不放回。E0 不实现全长 CDS 阅读框约束。

## 3. 权重与聚合

\(V_i,\hat Q\in[0,1]\)，\(\alpha\geq0\)。V-only 权重为 \(V_i^\alpha\)。联合 FP-REJ 的聚合只计算一次指数：

\[
w_i=\begin{cases}
(V_i\hat Q)^\alpha, & \mathrm{product},\\
\min(V_i,\hat Q)^\alpha, & \mathrm{min},\\
((V_i+\hat Q)/2)^\alpha, & \mathrm{mean}.
\end{cases}
\]

不得写成 \(V_i^\alpha\hat Q^\alpha\) 之外再对整体取指数，也不得写成 \(V_i^\alpha\cdot\hat Q\) 的双重压力形式。\(\alpha=0\) 时权重精确定为 1，即使 V 或 Q 为 0；实现走数学等价路径并省略无意义的 V/Q 计分。

## 4. 六种策略

### Random

从 \(\tilde P\) 不放回抽取 top-k；单轨迹 top-1 只提议一次。proxy/verifier/Q item-forward 全为 0。

### BoN

单轨迹有放回提议 \(C\) 次，逐 item 计算 proxy，按序列身份保留最高 proxy 后排序取不放回 top-k。其他策略不隐式调用该 proxy。

### WSC

单轨迹有放回提议 \(C\) 次。对第 j 个提议计算 \(V_j^\alpha\)，**按序列身份求和加权投票**：

\[
S(x)=\sum_{j:\,x_j=x}V_j^\alpha.
\]

按 \(S(x)\) 降序取不放回 top-k，不再做最终加权随机抽样。alpha=0 时退化为身份频次投票。

### Greedy

单轨迹有放回提议 \(C\) 次，直接做 **verifier reward argmax**，按序列身份保留最高原始 V 并排序。alpha 和 Q 不参与 greedy 排序。

### FP-TILT（解析对照）

仅使用 V：

\[
T_i=\frac{\tilde P_iV_i^\alpha}{\sum_j\tilde P_jV_j^\alpha}.
\]

需要对全部正提议质量的可行身份计分，proposal 数为 0。按 T 排序取不放回 top-k。池级 Q 不是 FP-TILT 输入，不能通过 product/min/mean 或有限拒绝采样的差异声称精确选择增量。

### FP-REJ（主方法）

单轨迹至多提议 \(C\) 次。每个提议计算上述 V-only 或联合权重 w，以概率 w 接受，首次接受即停止。全部拒绝时在缓存提议中按 \(w/(1-w)\) 回退；w=1 必定在接受阶段停止，不进入该除法。单轨迹只输出 top-1 或拒选。

多轨迹接口运行 M 条独立轨迹，共享同一 RNG、总预算和可选 cache；每条非拒选轨迹给其身份一票，最后按票数取不放回 top-k。不会为每条轨迹重置总预算。

## 5. 池级 Q、拒选与全零规则

池级 Q 只允许两种用途：

1. `abstain_below_q`：Q 小于阈值时整轨迹显式拒选，覆盖率为 0；此时 Q 不改变候选排序。
2. FP-REJ 的接受权重，影响接受率、回退率和实际 V/Q 成本。

`zero_weight_policy=abstain` 是默认主协议。FP-TILT 零归一化、WSC 全零投票、greedy 全零 reward、FP-REJ 全零回退权重均显式拒选，返回空选择、`abstain_reason=zero_guidance_weight`，不读取真值或偷偷改用 proxy。

只有调用方显式设置 `zero_weight_policy=uniform` 才允许回退：FP-TILT 在正提议质量身份上均匀不放回，WSC/greedy 在已缓存身份上均匀不放回，FP-REJ 在已缓存提议中均匀选一个。Q=0 在 product/min 且 alpha>0 时产生全零联合权重；mean 为 \((V/2)^\alpha\)，不自动视为全零。

## 6. 主预算与实际成本

**主预算字段是实际 `proxy_item_forwards`，不是名义 C、batch 函数调用次数或 wall-clock。** 同时保留 `verifier_item_forwards` 和 `qhat_item_forwards`，不得用单一总数掩盖不同模型的成本。

- Random 的三种 item-forward 均为 0。
- 无 cache 时，重复提议每次逐 item 计分；BoN/WSC/greedy 每轨迹固定 C 次，FP-REJ 可提前停止。
- ensemble 每个成员、每个 item 都收费，最终分数取成员均值。
- `BatchScorer` 的一次批量调用按输入 item 数收费，另记 scorer_batches；batch 不等于一次 item-forward。
- 默认无 cache。只有显式传入 `ScoreCache` 才按 `(score_kind, sequence_id)` 缓存 ensemble 均值；命中记 cache_hits 且 item-forward 为 0，新身份记 cache_misses。批内重复合并也记一次命中。
- 多轨迹共享总预算和可选 cache；计数不重置。top-k 去重不减少已经发生的提议或计分。
- `acceptance_rate=accepted_proposals/proposals`；多轨迹 `fallback_rate=fallbacks/M`，`coverage=非拒选轨迹数/M`。拒选的 regret 为缺失，不用最差值静默填充。

**相同 C 或 N 不等于相同实际成本。** E0 不做策略优劣结论，后续比较必须冻结并报告实际分模型 item-forward。

## 7. 概率 logit 噪声协议

鲁棒性噪声只施加于 V/Q 概率的 logit，而不是无界 MRL 或任意 proxy reward：

\[
p'=\operatorname{sigmoid}(\operatorname{logit}(\operatorname{clip}(p,\epsilon,1-\epsilon))+\eta),
\qquad \eta\sim N(0,\sigma^2).
\]

sigma=0 精确保留原概率，包括 0 和 1；sigma>0 时端点先裁剪以保证 logit 有限。该定义不适用于无界 MRL。

## 8. 合成真值与 regret

- `adversarial`：decoy 的 proxy 含正结构偏差，真实合成值较低。
- `independent`：proxy_signal 与 V 独立于 synthetic truth 构造。
- `oracle`：proxy_signal=synthetic truth，结构偏差和合成 proxy 噪声全为 0，仅作正确性上界。

合成小扰动只用于构造已知 proxy 误差，不是第 7 节的 V/Q 鲁棒性噪声协议。真值只在选择完成后评分：

\[
R(\pi)=\max_{i:m_i=1}Y_i^{synthetic}-Y_{i_\pi}^{synthetic}.
\]

## 9. E0 门禁

1. 完整 pytest 无失败、无 xfail，覆盖 C=1/2/3、权重退化、身份投票、top-k、多轨迹和成本。
2. 独立数学检查穷举 C=1/2/3 全路径，并核对接受率、回退率和经验选择分布。
3. 源码审计确认真值隔离、无 GPU/训练/前缀路径，协议与实现命名一致。
4. 同一冻结配置两次端到端 JSON 完全一致。

任一未通过则 E0 不通过，不进入 E1。通过也只表示软件正确性门禁通过。
