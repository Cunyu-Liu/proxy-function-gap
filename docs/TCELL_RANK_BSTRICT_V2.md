# T-cell B-rank-v2：strict 辅助 pairwise logistic 排序修正

日期：2026-11-26。状态：**已完成；两个辅助门禁均未通过，按冻结协议停止，未评价 T-cell。** 证据类别为回顾性方法开发/协议敏感性；历史 outer 已查看，本工作不构成新盲测或前瞻确认。

## 1. v1 失败定位：只使用辅助数据

v1 报告与数组显示，MSE 残差拟合的 dev/test 选择表现不一致，不能解释为数值不收敛：

| 方向 | λ / 强度 | dev regret 改善 | test regret 改善 | dev 选择更替 | test 选择更替 | test 更替改善/变差 |
|---|---|---:|---:|---:|---:|---:|
| N50→HepG2 | 0 / 0.5 | +0.001258 | **−0.000343** | 3,443/93,150 | 3,485/93,155 | 1,629 / 1,568 |
| HepG2→N50 | 1 / 0.75 | +0.006476 | **−0.004638** | 342/17,200 | 357/16,960 | 164 / 173 |

正负改善量分别表示 frozen-CNN regret 减 corrected regret。test 选择平均 MRL 变化因此分别为 −0.000343 和 −0.004638。虽然 test 更替中 N50→HepG2 的改善次数更多，但其平均改变量为 −0.009166 MRL，说明“改善次数多”并不等于选择收益为正。

固定 frozen-CNN 完整池 top16、每池全部 120 个无序对并排除实测 MRL 同分后：

| 方向/角色 | frozen 成对准确率 | corrected 成对准确率 | 对→错 | 错→对 | 净错→对比例 |
|---|---:|---:|---:|---:|---:|
| N50→HepG2 dev | 0.509859 | 0.509782 | 27,144 | 26,987 | −0.0000770 |
| N50→HepG2 test | 0.509583 | 0.509510 | 27,262 | 27,112 | −0.0000736 |
| HepG2→N50 dev | 0.504915 | 0.505287 | 3,103 | 3,251 | +0.0003724 |
| HepG2→N50 test | 0.503349 | 0.503354 | 3,104 | 3,106 | +0.0000051 |

N50→HepG2 所选系数 `[p,p²,GC,uAUG,polyU,intercept]` 为 `[-0.05749,-0.00820,-0.17464,0.03646,-0.00375,0.75865]`；HepG2→N50 为 `[-0.00527,-0.01940,-0.00025,-0.00353,0.00304,0.81224]`。大截距不影响池内排序；小幅 p/GC 等修正只更换约 2–4% 的 BoN16 选择，却未稳定改善 tail pair 方向。v1 还用 destination dev 选型，不能作为 source-inner 泛化选型证据。

**新科学假设：** 高分尾部 regret 的直接障碍是成对顺序错误；MSE 会被非尾部与误差幅度主导，故改为只在 source train 固定 CNN-top16 上优化 convex pairwise logistic 修正。该假设可由 source dev 与未参与选型的 target auxiliary test 判伪，不扩大无序模型搜索。

诊断原始结果：`artifacts/diagnostics/v1_failure_diagnosis.json`。诊断没有读取或评分 T-cell MRL。

## 2. 冻结数据、隔离与部署

- 新实现 `/home/proxy-function-gap-tcell-rank`；新 artifacts `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_rank_bstrict_v2`。v1 目录只读，绝不覆盖。
- 沿用 v1 已验证的 strict-H1 辅助数组：在拆分前排除 T-cell r1∪r2 原始无标签全集及 Sample train/dev/calibration 的 exact/H1；拆分为 source train/dev/test，H1 component 不跨角色。
- 冻结三成员 Sample CNN 分数 `p`，特征严格为 `p,p²,GC,uAUG,longest-poly-U,intercept`。五个非截距特征只用 source train 全行均值和总体 SD 标准化；零 SD 为失败。
- 部署唯一公式为 `p + s0 × strength × X_scaledβ`，`s0=1.6689551812138181` 来自 Sample train。pair 差分中截距恒消去，因此保留特征位置但系数固定为 0，避免不可识别截距。
- 不读 T-cell `rl` 进行训练、pair 构造、正则/强度选择、提前停止或门禁。

## 3. 冻结 pair、损失与 CUDA 凸优化

每个 source train 固定 64 候选池仅按 frozen CNN 分数稳定降序取 top16；跑前冻结池和 tail。每池全部 120 个无序对，label 为 `sign(y_i−y_j)`，MRL 同分权重为 0。每个池总权重相同，池内非平分对同权，全局权重和为 1。

对每一对优化：

`weight × softplus(−label × [p_i−p_j + s0(z_i−z_j)β]) + λβᵀβ`。

λ 网格冻结为 `[0,1e-6,1e-4,1e-2,1,100]`，强度网格为 `[0,0.25,0.5,0.75,1]`。只用 source train 拟合，source-inner dev 选择。

使用 CUDA `cuda:0`、float64、带 Armijo 回溯的 damped Newton。没有科学训练步数上限或 Armijo 步数上限。停止条件为梯度无穷范数 `≤1e-8`，或相对损失变化 `≤1e-12` 且梯度无穷范数 `≤1e-5`。每次保存迭代数、梯度、损失、回溯、初末 Hessian 条件数和有限性；条件数上限 `1e12`，任一失败即方案失败，不切换无预定模型。

## 4. source-inner dev 选型与 target auxiliary test 一次门禁

方向固定为 N50→HepG2 和 HepG2→N50。每方向只在 **source train/dev** 构造池、proposal 和 tail；destination dev 完全禁用。source dev cap16 等共享 proposal 前缀的 BoN 平均 regret 选 λ/强度；平手容差 `1e-12`，依次比较 source dev pairwise logistic loss、较大 λ、较小强度。

冻结所选模型后，对 destination auxiliary **test 仅评分一次**。两方向分别要求：

1. cap16 corrected−frozen 平均 regret 严格小于 0；
2. 对五种子先在池内取均值，再按 test pool 做 10,000 次 cluster bootstrap，95% percentile CI 上界严格小于 0。

bootstrap seed=`20261126`。任一方向点估计或 CI 失败即停止，**不读取、不推理、不评价 T-cell**；两方向不能互相平均抵消。

## 5. 双过后的冻结 T-cell 一次评价

仅当双门禁通过，才对两个方向的所选修正（含各自强度）做等权平均，不重拟合、不根据 outer test 改协议。T-cell r1/r2 只评价一次，沿用 v1 caps `1/2/4/8/16`、种子 `101/202/303/404/505`、共享嵌套 proposal 与 tie 规则。

主成功仍须同时满足：两个 repeat 的 cap16 平均 regret 均改善；合并 repeat-linked pool block bootstrap 95% CI 上界小于 0；合并绝对改善至少 `max(0.1 MRL, frozen 平均 regret 的 5%)`；全部隔离/CUDA/数值日志通过。10,000 次、seed=`20261124`。cap2/4/8 为六项 repeat-specific Holm 次级族。目标评分后不得再改科学协议；全部有利与不利结果保留。

## 6. 三轮真实核验与导出边界

1. 实现/泄漏轮：冻结配置和 v1 输入哈希、strict 隔离、tail/pair/权重、尺度、凸损失、Newton 收敛、共享 proposal、门禁 fail-closed、目标未运行状态及 focused tests。
2. 独立辅助轮：不导入主拟合模块，重算 pair/损失/优化/候选选择/一次 test regret/CI；确定性重放并保留所有负结果。
3. 交付轮：干净进程重放验证和测试，核对配置、代码、日志、artifacts 哈希与禁止字段；只有双门禁通过才允许出现 T-cell 结果文件。

本地仅同步 config、聚合、系数、代码、日志和 verification，不同步序列字符串、逐序列标签或预测。机器可读冻结协议：`configs/rank_frozen.json`；其 SHA-256 将在科学运行前写入 prefreeze manifest。

## 7. 最终结果：pairwise 假设仍未通过辅助门禁

两方向都只在 source-inner dev 选型，destination dev 未参与；destination test 仅对所选模型评分一次。改善效应定义为 corrected−frozen regret，负值有利。

| 方向 | 所选 λ / 强度 | source dev frozen→corrected regret | test frozen→corrected regret | test 效应 [95% pool-cluster CI] | 门禁 |
|---|---|---:|---:|---:|---|
| N50→HepG2 | 0 / 0.75 | 5.800774→5.793016 | 4.082435→4.075892 | **−0.006543 [−0.014204, +0.001233]** | FAIL：CI 上界大于 0 |
| HepG2→N50 | 1e-4 / 0.75 | 4.091181→4.075556 | 5.828669→5.849856 | **+0.021187 [−0.007718, +0.050096]** | FAIL：点估计变差且 CI 上界大于 0 |

N50→HepG2 在 test 有小幅平均改善，但 10,000 次、18,631 个 test pool 的 cluster bootstrap 区间跨零，不能满足冻结的证据门槛。HepG2→N50 在 source dev 改善却在 3,392 个 test pool 上反向，说明 source-inner 选择收益仍未迁移。test 选择更替率分别为 9.29% 和 15.50%，更替本身不构成成功。

所选标准化特征系数 `[p,p²,GC,uAUG,polyU,intercept]`：

- N50 source：`[0.48335089,-0.89545977,-0.01619109,0.00699927,-0.00326682,0]`；
- HepG2 source：`[-0.21698754,-0.29993753,-0.01487625,-0.02296983,0.00196364,0]`。

两方向都选择强度 0.75，而不是零强度基线；v2 的负结果不是“没有实施修正”。但没有任何 T-cell corrected 选择、MRL、regret、CI 或主成功评价。`target/` 与 `auxiliary_gate_passed.json` 均不存在，target evaluation count=0。没有根据目标结果改协议，因为目标从未开启。

## 8. 三轮真实执行、CUDA 与失败保留

1. **实现/泄漏轮：PASS。** 四个原始文件、三个 Sample 角色、全部辅助数组哈希一致；strict 排除和角色行数一致。focused tests 最终 `7 passed`。早期 tail/loss 夹具、cuBLAS deterministic 环境、Newton 步长、测试求根区间和符号错误均保留在 `logs/round1_tests*.log`。
2. **独立辅助轮：PASS。** 不导入主模块，逐 λ 重算全部 12 个拟合的损失和解析梯度、60 个 source-dev 候选 regret、两个 test 选择/regret 和 10,000 次 bootstrap。损失最大绝对差 `7.771561172376096e-16`，候选 regret 最大差 0，独立梯度全部通过。验证器字段与 NumPy 标量序列化失败保留，最终见 `verification/round2_independent.json`。
3. **交付轮：科学检查 PASS。** 干净进程再次 `7 passed`；配置 SHA-256 始终为 `16c94ae5caaf9bd166b9a61af9505a69ae776b6689838705e005192747b55afd`，确认 fail-closed、target=0、v1 未修改和负结果保留。见 `verification/round3_delivery.json`。

所有正式 CUDA 拟合均为 float64、RTX 5090、无科学迭代上限。修正解析导数后，N50 的六个 λ 在 1–3 次 Newton 迭代收敛，最终梯度无穷范数范围约 `1.25e-15` 至 `4.43e-10`；HepG2 范围约 `4.20e-16` 至 `1.27e-9`。条件数、损失、Armijo 和有限性均在 `fits.json` 与 CUDA 日志中。

开发失败没有被隐藏：

- 初次 Armijo 重复 Hessian 的性能中断保存在 `failed_attempts/performance_attempt1`；
- 第二次慢运行及逐迭代调试定位到 CUDA 梯度/Hessian 漏乘 `s0/s0²`，保存在 `failed_attempts/performance_attempt2`、`logs/debug_lambda_1e-4.log` 和主日志；
- 修正后首次 test 在产生 regret 前发现全局/局部池索引错误，保存在 `failed_attempts/test_index_attempt1`；修正后从头重跑，本文只使用最终完整运行；
- 验证器与导出器失败均保留。上述调试只使用辅助数据，没有读取 T-cell MRL。

## 9. 产物与边界

- 远端代码：`/home/proxy-function-gap-tcell-rank`。
- 远端完整 artifacts：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_rank_bstrict_v2`；v1 未覆盖。
- 本地脱敏数据：`deliverables/reports/source_data/tcell_rank_bstrict_v2/`；包含配置、诊断、全部候选、系数、日志、代码和三轮 verification，不含序列字符串、逐序列标签或逐序列预测。
- 本结果只判伪“当前固定特征、pair 和强度的 convex pairwise logistic correction 能通过双辅助门禁”，不证明所有排序修正都无效，也不提供任何 T-cell 改善证据。后续若另提新科学假设，必须另立协议，不能把本轮包装为成功的冻结检验。
