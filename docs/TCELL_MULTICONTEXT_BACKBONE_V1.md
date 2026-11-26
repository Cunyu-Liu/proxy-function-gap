# Paper 1：多情境尾部排序共享骨干 v1

日期：2026-11-26。状态：**已完成；两个辅助门禁均未通过，按冻结协议停止，未评价 T-cell。** 证据类别为回顾性方法开发/协议敏感性；历史 outer 已查看，本工作不构成新盲测或前瞻确认。

## 1. 结论先行

- 冻结三成员原 Sample CNN 为基线，新主模型保持相同两层 CNN 容量，在扩展多情境 Sample 上联合优化 context-train-only 标准化 MSE 与当前预测 top16 的 pairwise logistic loss。
- 主模型 λ 网格 `[0.01,0.1,1]` 仅由 Sample dev 与 N50/HepG2 source-inner dev 的标准化 cap16 regret 选择，最终 **λ=0.1**。destination dev 未参与选型。
- 两个 destination auxiliary test 都只对所选主模型评分一次，效应定义为 `主模型 regret − 原 CNN regret`，负值才有利：

| 方向 | 原 CNN cap16 regret | 主模型 cap16 regret | 效应 [95% pool-cluster CI] | 门禁 |
|---|---:|---:|---:|---|
| N50→HepG2 | 4.092569 | 4.137198 | **+0.044629 [+0.021840,+0.068346]** | FAIL：显著变差 |
| HepG2→N50 | 5.828669 | 5.856719 | **+0.028050 [−0.049121,+0.104614]** | FAIL：点估计变差且 CI 跨零 |

两方向各做 10,000 次 cluster bootstrap，seed=`20261126`；分别使用 18,626 和 3,392 个完整 destination test 池。任一方向失败都足以停止，因此没有创建 `target/`，没有读取 T-cell MRL，没有目标选择、regret、CI 或主成功评价，**target evaluation count=0**。

## 2. 冻结角色、隔离与多情境接口

- 新实现：`/home/proxy-function-gap-tcell-backbone`；完整 artifacts：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_multicontext_backbone_v1`。
- 机器可读冻结协议：`configs/backbone_frozen.json`，SHA-256=`02cbfca7ae4feff294fa99deedf0ee5169088d8eb83429e6df40f6ea55c952b5`。
- 新 Sample 接口使用已验证的 `sample2019-fixed-utr-v1` 规范化序列，要求 50nt、ACGT、有限 MRL；包含 eGFP unmod r1/r2、pseudo r1/r2、m1pseudo r1/r2、mCherry r1/r2 和合法 designed 序列，排除 varying-length、official/control/scaler/quarantine 等不合法角色。
- T-cell 制备阶段只读取 r1/r2 的序列键，分别核验 349,633/352,724 条；`rl` 列未访问。任何与目标 exact/H1 的 Sample 行及其整个 allocation component 均在拆分前排除。
- exact/H1/trajectory allocation component 按冻结 SHA256 规则整体进入 train 或 dev，不跨角色。扩展数据最终为 **272,673 train + 27,449 dev**：

| 情境 | train | dev |
|---|---:|---:|
| eGFP unmod r1 | 2,347 | 281 |
| eGFP unmod r2 | 5,628 | 609 |
| eGFP pseudo r1 | 3,464 | 381 |
| eGFP pseudo r2 | 4,511 | 466 |
| eGFP m1pseudo r1 | 3,298 | 370 |
| eGFP m1pseudo r2 | 5,554 | 640 |
| mCherry unmod r1 | 120,711 | 13,447 |
| mCherry unmod r2 | 95,355 | 10,654 |
| eGFP unmod designed | 2,856 | 301 |

目标 exact/H1 排除对 eGFP 随机库影响很大，保留数据以 mCherry 为主；这是冻结严格隔离的结果，不能事后为获得更有利分布而放宽。部分 eGFP dev 只有 4–10 个完整池，选择估计因此有明显不确定性。

- 原数据 MSE+tail-rank ablation 使用未改动的原 fresh-CNN train/dev：**293,091/21,458**，不与新拆分混用。扩展 MSE-only ablation 使用上述扩展数据。
- 扩展 Sample 会覆盖旧 strict 辅助集，因此不能原样复用。新过滤排除了 HepG2 中 1,699 条 exact 和 1,790 条 H1-only；N50 无新增排除。保留 HepG2 train/dev/test 为 9,601,975/1,191,814/1,192,069，N50 保持 1,763,213/220,183/217,137。

## 3. 模型、动态尾部损失与 λ 选择

主模型和两个 tail-rank 候选均保持原 `FreshSampleCNN`：Conv1d(4,32,k5)、MaxPool、Conv1d(32,64,k5)、全局均值、Linear(64,32)、Linear(32,1)，没有扩容量。

- 每个情境仅用自身 train 标签拟合均值和总体 SD，零 SD 失败；零样本部署直接比较标准化网络输出的顺序，不使用 T-cell 统计量。
- 固定 64 序列池，batch 为 16 池/1,024 序列。每一步按当前预测分离 top16 成员，构造全部 120 个无序对；标签平分对权重为 0，梯度仍通过所选分数反向传播。
- 总目标为等情境 MSE 加 λ 倍等情境动态尾部 pairwise logistic loss。每个情境内池等权，避免大行数情境直接支配总目标。
- λ 选择为 9 个扩展 Sample dev 情境与两个 auxiliary source dev 情境的标准化 cap16 平均 regret。平手容差 `1e-12`，依次比较 dev 联合损失和较小 λ。
- 主模型候选 selection metric：λ=0.01 为 1.927557，λ=0.1 为 **1.866910**，λ=1 为 1.884409。
- 原数据 tail-rank ablation 独立选择 λ=1；其 source-dev 描述性表现不能替代 destination 门禁，也未被用于放宽主模型门禁。

## 4. 三种子真实 CUDA 收敛

三个 seed=`11/23/37` 并行运行，每个均完成 7 个冻结候选：主模型 3 个 λ、扩展 MSE-only 1 个、原数据 tail-rank 3 个 λ，共 **21 个正式 run**。

- RTX 5090、PyTorch 2.7.1+cu128、float32、确定性算法、TF32 关闭、无 CPU 回退。
- 三进程真实前向/反向探针均通过，每进程约 0.95GB；正式并行训练时 GPU 利用率约 93–99%，总显存约 15GB。
- 使用原 DualPlateau：`min_delta=1e-4`、`patience=5`，任何 dev 绝对最佳重置 plateau；科学最大 epoch 为 `null`。
- 全部 run 自然收敛，合计 **1,659 epoch**；总 epoch 范围 47–127，最佳 epoch 范围 7–98。原数据 tail-rank seed23 的 λ=0.01 到 epoch 89 才停止、最佳 epoch80，说明未人为提前截断。
- 主模型所选 λ=0.1 的三个最佳 epoch 为 8/15/9；权重 model hash 分别为 `10651633…a37`、`1b0af673…55a`、`2d7edff4…5d0`。
- 每个 run 均保存完整 `history.jsonl`、最佳 checkpoint、最佳 epoch、有限损失/梯度、真实 CUDA、训练数据哈希、权重哈希、最佳 dev 预测和总结。本地只保留曲线和哈希，最佳权重留在 teams。

## 5. 辅助门禁与目标 fail-closed

主模型冻结后才执行 destination auxiliary test。每方向均使用共享嵌套 proposal、五个种子 `101/202/303/404/505`、cap16 和完整池 oracle。两方向点估计均为正，即主模型更差；N50→HepG2 的 CI 完全位于有害方向。

由于 `both_directions_pass=false`：

- 不执行 T-cell 新模型推理；
- 不读取 T-cell `rl`；
- 不评价 r1/r2 的任何 cap；
- 不评价两个 ablation 的 T-cell 表现；
- 不修改 λ、损失、隔离、门禁或成功阈值；
- 不把 source-dev 收益、训练损失下降或选择更替解释成目标改善。

因此本工作没有获得“具体 T-cell MRL 等预算选择实质提升”的证据，只判伪了当前冻结共享骨干能通过双辅助门禁。两个 ablation 已完整训练并冻结，但没有越过主模型门禁去查看目标，符合预定归因边界。

## 6. 三轮真实核验

1. **原始/实现/隔离轮：PASS。** 原始 Sample/Castillo 哈希、目标序列键计数、exact/H1 component 拆分、扩展 Sample 对辅助集的重新过滤、原 pack 哈希均通过。最终本地与远端 focused tests 均为 `7 passed`；包含 H1、动态尾部梯度、不完整池、MSE-only、plateau、共享 proposal、bootstrap、模型容量和真实 CUDA。
2. **独立轮：PASS。** 不导入训练模块，审计 21 个 run 的有限性、最佳 epoch 和收敛；独立重算全部 λ 的 Sample/source-dev 选择、destination regret 和 10,000 次 bootstrap。共 **57 项比较，最大绝对差 0**。确认双门禁失败且 target 目录不存在。
3. **交付轮：PASS。** 干净进程再次 `7 passed`；配置 SHA-256、Sample/auxiliary 文件哈希、21 个 run、fail-closed target=0 和 no-push 均通过。见 `verification/round3_delivery.json`。本地再次核对 21 份 summary/history、选择算术和禁止文件；最终无 `.npy/.npz/.pt`。

## 7. 保留的开发失败与导出边界

未隐藏任何失败：

- 初次合成测试发现 H1 半段扫描方向、one-hot 维度和两个夹具期望错误；修复后回归通过。
- Sample 制备第一次错误地直接 zip 交错 lineage；第二次漏建 `data/`；两次均保留在 `failed_attempts/prepare_sample_attempt1/2`。
- 正式训练前四次启动分别暴露嵌套数据键、pool permutation 设备引用、CUDA tensor 的 NumPy 索引和不完整池 reshape；均发生在首个优化步前，相关目录与日志保存在 `failed_attempts/training_start_attempt1–4`。
- 第一次辅助选择在旧 Unicode 序列数组兼容处停止，destination test 尚未启动；修复后仅执行一次正式 destination 评分。
- 独立验证首次 shell `tee` 因 artifacts 日志目录不存在返回失败，验证器本身每次均通过；随后创建目录并保存最终日志，未重新推理 destination。
- 第一次本地导出路径错误且包含逐序列 source-dev 预测；部分包未作为交付，最终导出已删除所有逐序列数组并复核禁止文件为 0。
- 多次 SSH `Connection closed` 与等待监控失败不代表科学任务失败；正式远端进程独立运行，未以历史结果替代本轮执行。

本地 `deliverables/reports/source_data/tcell_multicontext_backbone_v1/` 仅含配置、脱敏聚合、完整训练曲线、代码、日志和 verification；不含序列字符串、逐序列标签或预测。代码未 push，历史 residual/rank/anchor artifacts 未覆盖。

## 8. 最终边界

本轮结论是：**当前多情境 Sample + 同容量动态尾部 pairwise 共享骨干在两个辅助迁移方向均未改善 cap16 等预算 regret，不能开启 T-cell 评价。** 它不证明所有共享骨干或所有排序损失无效，也不提供任何 T-cell 正负结果。后续若改变数据隔离、尾部定义、损失或模型，必须另立新协议，不能把本轮包装为成功的冻结检验或新的盲测。
