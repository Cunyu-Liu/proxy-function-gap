# Paper 1：T-cell 目标情境适配 v1

日期：2026-11-26。状态：**已完成一次冻结 test；主成功为 FAIL。** 50k fine-tuned joint MSE+tail-pairwise 模型在两个 repeat 的 cap16 等预算 regret 均未改善，合并区间跨零，也未达到实质改善幅度。

证据类别：**target-context retrospective grouped holdout**。本轮明确使用 T-cell train/dev 的 r1/r2 标签进行适配和选型，因此不是无标签迁移、独立研究、序列外部验证、新盲测或前瞻确认。历史 outer 已查看；16 个 grouped holdout test 池只提供同一回顾性数据内的序列/H1 隔离评价。

## 1. 结论先行

主效应定义为 `50k adapted-main regret − original Sample-CNN regret`，负值才有利：

| Repeat | 原 CNN cap16 regret | 50k main cap16 regret | main−原 CNN |
|---|---:|---:|---:|
| r1 | 2.389657 | 2.428586 | **+0.038929** |
| r2 | 2.873002 | 2.885553 | **+0.012551** |

- 合并 repeat-linked pool block bootstrap 效应为 **+0.025740 MRL**，95% percentile CI **[−0.072019, +0.140994]**，10,000 次、16 个 repeat-linked pool、seed=20261126。
- 原 CNN 合并 cap16 基线为 2.631329；冻结的最小绝对改善门槛为 `max(0.1,5%×2.631329)=0.131566`。
- 实际绝对改善为 **−0.025740**，即方向相反；两个 repeat 点估计、CI 上界和幅度三项均失败。
- 全部 2 repeats×5 caps 的 main−原 CNN 检验经 Holm 后均为 **p=1**。cap1 各方法因共享第一 proposal 位置而完全相同。
- 因此不能声称“少量目标监督已经修复 T-cell 高分尾部选择”。本轮支持的是：pointwise 误差可显著改善，但当前 50k joint fine-tune 没有转化为更好的 cap16 选择。

## 2. 冻结数据、repeat 角色与隔离

原始 T-cell r1/r2 文件及 Sample train/dev/calibration 的 SHA-256 全部通过。共有 314,563 条 r1/r2 exact 共享、50nt、ACGT 序列；仅按序列键构建 exact/H1 无向连通分量：

- 6,718 条 component 内 H1 边，307,845 个 component，最大 component 为 3 条序列。
- component 整体分配到 **50,000 train、10,000 dev、1,024 test**，test 为 **16×64**，r1/r2 使用完全相同的序列和 pool membership。
- 跨 train/dev/test 的 exact/H1 边为 **0**。unused 序列不进入任何模型或评价。
- 为避免原 Sample CNN 预训练造成 test 近邻泄漏，凡任一 member 与 Sample train/dev/calibration 存在 exact/H1 的整个 component 均不得进入 test。最终 test 对 Sample exact/H1 均为 **0**。
- train/dev 的拟合标签为同序列 `(r1+r2)/2`，两个 repeat 等权；对 MSE，这与两个 repeat 各 1/2 权重等价（只差与参数无关的常数）。pairwise 监督沿用预注册的 mean-replicate 目标。
- test 的 r1/r2 MRL 直到 72 个最终 run、全部选择和配置冻结后才读取一次，并分别评分，不能用均值替代两个 repeat 的 test 结果。

nested train budget 为 1k/5k/20k/50k，均按完整 component 嵌套，严格满足 `1k⊂5k⊂20k⊂50k`。训练/dev MSE 使用最后的不完整池；动态 pairwise 和 dev cap16 只用完整 64 序列池，dev 共 156 个完整池。

机器可读协议：基础配置 SHA-256=`7df31364834289e617f3cb48cf11e1c406ef02dc45a8d51a3515782f8a6a042f`；独立预审修订 SHA-256=`2777178b6d431b6bc945714ad776c15deffa1efcb410353f46585cf5e0ed74ff`。

## 3. 独立预审修订：dev 同时约束两个 repeat

预审指出只按 mean-replicate dev regret 选型可能掩盖单 repeat 失败。该问题在任何 test 推理或 test 标签读取前以独立 amendment 冻结，未改变数据、模型网格、test、成功门槛或已训练权重：

1. 每个候选分别计算 dev r1 与 dev r2 的 cap16 regret。
2. 主选型指标为 **`max(r1 regret,r2 regret)`**，不再只看均值。
3. 平手容差 `1e-12`，依次比较 mean-replicate regret、dev combined loss、较小 LR、较小 λ。
4. 两个 dev repeat 可精确重建原均值标签；三 seed 始终等权，from-scratch 与 fine-tune 使用相同架构、数据、网格、三 seed、plateau 和 worst-repeat 选型。

早先 mean-only 选择的是 LR=0.001、λ=1；修订后 main 选择 **LR=1e-5、λ=0.01**，dev r1/r2 regret 分别为 **2.535442/2.678299**。因此早先 9 个 nested run 被完整保留为 `failed_attempts/superseded_mean_only_nested`，不计入最终集合；按新超参重新完成 9 个 nested run。没有根据 test 做这次修订或替换。

## 4. 模型、网格与真实 CUDA 自然收敛

基线为冻结三成员 Sample CNN 的算术平均。main、MSE-only 和 nested 的 seed 11/23/37 分别从对应原成员初始化；from-scratch 使用相同 `FreshSampleCNN` 的 seeded random 初始化。网络容量不变。

- 主网格：LR=`[1e-5,1e-4,1e-3]`，λ=`[0.01,0.1,1]`，共 9 个组合/seed。
- MSE-only：λ=0，使用同一 LR 网格并独立按 worst-repeat dev 选 LR。
- from-scratch：使用同一 3×3 网格并独立按 worst-repeat dev 选 λ/LR。
- nested：只使用修订后 50k main 的 LR/λ，不重新调参。
- 动态尾部为当前预测 top16 的全部 120 个无序对，`softplus(-sign(y_i-y_j)(p_i-p_j))`，实测同分对权重为 0；top16 membership detach，分数梯度正常反传。
- 目标缩放只用对应 budget 的 train 均值和总体 SD。Adam、float32、RTX 5090、CUDA 12.8、确定性算法、TF32 关闭、无 CPU 回退。
- `DualPlateau(min_delta=1e-4,patience=5)`，任何 dev 绝对最佳重置 plateau；**没有最大 epoch**。

最终共 **72 个 run、22,082 epoch**，单 run 43–1,814 epoch，最佳 epoch 4–1,809；全部自然收敛。三个 seed 进程真实并行，保存完整 history、最佳 checkpoint、有限损失/梯度、初始化/数据/模型哈希。最终选择为：

| 模型 | LR | λ | 三 seed 最佳 epoch |
|---|---:|---:|---|
| main 50k | 1e-5 | 0.01 | 80 / 84 / 70 |
| MSE-only 50k | 1e-5 | 0 | 80 / 84 / 70 |
| from-scratch 50k | 1e-5 | 1 | 1037 / 939 / 1658 |

MSE-only 的 worst-repeat dev 为 2.676207，略低于 joint main 的 2.678299；该开发期不利信号没有被隐藏或用来事后更换主模型。

## 5. 唯一一次 test：全 cap 主结果与 Random(N)

所有方法共享 nested proposal prefix，seeds=101/202/303/404/505；Random(N) 取相同 prefix 的第一个位置。regret 始终相对于完整 64 序列池的 repeat-specific 实测 oracle。

| Repeat | cap | 原 CNN | main 50k | main−原 CNN | MSE-only | from-scratch | Random(N) | main Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r1 | 1 | 3.694147 | 3.694147 | 0 | 3.694147 | 3.694147 | 3.694147 | 1 |
| r1 | 2 | 3.257228 | 3.311612 | +0.054385 | 3.311612 | 3.218465 | 3.694147 | 1 |
| r1 | 4 | 2.572932 | 2.603418 | +0.030485 | 2.603418 | 2.478433 | 3.694147 | 1 |
| r1 | 8 | 2.624468 | 2.584807 | −0.039661 | 2.584807 | 2.420123 | 3.694147 | 1 |
| r1 | 16 | 2.389657 | 2.428586 | +0.038929 | 2.428586 | 2.254828 | 3.694147 | 1 |
| r2 | 1 | 4.830002 | 4.830002 | 0 | 4.830002 | 4.830002 | 4.830002 | 1 |
| r2 | 2 | 4.091160 | 4.110078 | +0.018918 | 4.110078 | 4.131200 | 4.830002 | 1 |
| r2 | 4 | 3.337055 | 3.315566 | −0.021489 | 3.315566 | 3.299283 | 4.830002 | 1 |
| r2 | 8 | 3.085590 | 3.163563 | +0.077973 | 3.163563 | 3.185627 | 4.830002 | 1 |
| r2 | 16 | 2.873002 | 2.885553 | +0.012551 | 2.885553 | 3.066365 | 4.830002 | 1 |

main 与 MSE-only 在所有 cap 的选择/regret 完全相同，说明本轮所选小 λ 的 pairwise 项没有产生额外的最终选择收益。from-scratch 在 r1 cap16 有点估计改善 **−0.134828**，但 95% CI **[−0.340526,+0.058553]** 跨零；r2 反而变差 **+0.193363**，95% CI **[−0.196777,+0.571721]**。它不能跨 repeat 替代主模型或构成修复证据。

## 6. 嵌套学习曲线：20k 的次级改善不能替代 50k 主结论

| Train budget | dev cap16 mean-replicate regret | r1 test regret（效应） | r2 test regret（效应） |
|---:|---:|---:|---:|
| 1k | 1.827241 | 2.435756（+0.046100） | 2.946595（+0.073593） |
| 5k | 1.794924 | 2.416233（+0.026576） | 2.941328（+0.068326） |
| 20k | 1.800521 | **2.327279（−0.062378）** | **2.850507（−0.022495）** |
| 50k | 1.787570 | 2.428586（+0.038929） | 2.885553（+0.012551） |

20k 在两个 repeat 均有小幅点估计改善，但幅度低于主门槛、不是预注册主预算，且 50k 反向，学习曲线非单调。不能看到 test 后改选 20k 或把它包装成主成功。该现象只说明目标监督量与尾部选择之间存在不稳定关系，后续若研究必须另立协议。

交付审查曾发现 50k nested 行误聚合了全 cap；已仅用已保存的 cap16 数组修正为表中数值，原始错误 JSON 保存在 `failed_attempts/aggregate_learning_curve_attempt1`。没有新 test 推理、没有重读 test 标签，主结果及其他行不变。

## 7. 高分尾部翻转与选择更替

main 与原 CNN 的 full-pool top16 Jaccard 均为 0.871732，两个 repeat 都有 8/16 池发生 top1 翻转：

| Repeat | 原 CNN tail-pair accuracy | main tail-pair accuracy | 变化 | 翻转对支持 main / 原 CNN |
|---|---:|---:|---:|---:|
| r1 | 0.533090 | 0.527879 | −0.005211 | 130 / 140 |
| r2 | 0.586458 | 0.602083 | +0.015625 | 150 / 120 |

尾部方向在 r1 略变差、r2 略改善，并不一致。cap16 每个 repeat 的 80 个 pool×seed 比较中均有 16 个选择更替，改善/变差各 8 个；但平均所选 MRL 变化分别为 **−0.038929/−0.012551**，与 regret 变差一致。更替次数或翻转数量本身不能证明修复。

## 8. 误差与 regret 不同步

| Repeat | 模型 | MAE | RMSE | bias | Spearman |
|---|---|---:|---:|---:|---:|
| r1 | 原 CNN | 1.352159 | 1.645257 | +1.069305 | 0.697937 |
| r1 | main | **0.914506** | **1.274272** | +0.370288 | 0.703776 |
| r1 | from-scratch | 1.014330 | 1.380693 | +0.433579 | 0.641142 |
| r2 | 原 CNN | 1.062273 | 1.482733 | +0.399929 | 0.719154 |
| r2 | main | **1.018716** | **1.451990** | −0.299088 | 0.728935 |
| r2 | from-scratch | 1.137297 | 1.552134 | −0.235797 | 0.671161 |

main 的 MAE 在 r1/r2 分别降低 **0.437653/0.043556**，RMSE 和 Spearman 也改善，但 cap16 regret 同时变差，两个 repeat 的 MAE 与 regret 变化均不同向。池级 MAE–regret Spearman 分别为 0.638/0.121，只是描述性关联。

这直接排除了“整体拟合误差降低就等同高分尾部选择修复”的解释。目标适配主要改善了 pointwise 拟合/校准，而主终点所需的尾部成对顺序和实际 BoN 选择没有同步改善。

## 9. 三轮真实核验与开发失败保留

1. **原始/实现/隔离轮：PASS。** 原始哈希、314,563 共有键、component 整体隔离、精确预算、Sample-clean test、nested subset、三成员初始化、真实 CUDA 和 test=0 通过。最终干净测试为 **11 passed**；包含 worst-repeat 选型、H1 component、动态尾部梯度、MSE-only、plateau、共享 Random prefix、Holm、模型容量和真实 CUDA。
2. **独立数值轮：PASS。** 不导入实现模块，重新读取原始 test 标签，独立重算跨角色 H1、Sample test 隔离、nested subset、72 个训练摘要、worst-repeat dev 选型、全部 test 选择/regret、bootstrap/Holm、嵌套曲线与主要误差指标。共 **204 项比较，最大绝对差 0**。
3. **交付轮：PASS。** 干净进程再次 **11 passed**；基础配置、预审 amendment、72 run、训练先于 test、evaluation count=1、无 post-test 科学协议变更和聚合修复链均通过。本地回读 **208 个 manifest 文件**，禁止导出文件为 0。

所有失败均保留而未隐藏：`select.py` 标准库遮蔽导致改名；两次 prepare 在标签读取前因 unused 角色断言停止；原模型哈希的 torch/NumPy dtype 表示及 manifest 键错误在首个优化步前修正；早期 mean-only nested 被预审修订 supersede；50k nested 聚合错误只修正报告不重评 test。SSH 断连和监控超时不被当作科学结果。对应日志、失败快照和哈希链均保留。

## 10. 产物与最终边界

- 远端代码：`/home/proxy-function-gap-tcell-adaptation`；代码未 push。
- 远端完整 artifacts：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_target_adaptation_v1`；全部权重保留 teams。
- 本地脱敏聚合：`deliverables/reports/source_data/tcell_target_adaptation_v1/`；包含配置、72 run 的 summary/history、选型、聚合 test、代码、日志、失败记录和三轮 verification，不含序列字符串、逐序列标签/预测、`.npy/.npz/.pt` 或原始表。
- 本报告：`TCELL_TARGET_ADAPTATION_V1.md`。

本轮只判伪了“当前 50k grouped target supervision + 所选 joint MSE/dynamic-tail pairwise fine-tune 能在两个 repeat 实质改善 cap16 选择”。它不支持改选 20k、用 from-scratch 的单 repeat 有利点估计替代主结果，也不证明所有目标监督或所有尾部损失无效。真正的独立确认仍需未触碰数据或新实验。
