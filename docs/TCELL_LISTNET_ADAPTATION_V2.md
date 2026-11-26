# Paper 1：T-cell 真标签 ListNet 目标适配 v2

日期：2026-11-26。状态：**81 个 run 全部自然收敛，唯一一次 test-v2 已完成；联合主成功为 FAIL。** 证据类别：**target-context retrospective grouped holdout**。v1 train/dev 标签继续用于目标适配和选型；test-v2 是从历史原始终端重新构建的 component-disjoint grouped holdout，不是独立研究、外部验证、前瞻盲测或湿实验确认。历史 outer 已查看的边界保留。

## 1. 科学假设与唯一核心修改

v1 所选 joint 模型与 MSE-only 的最终选择相同，且 MAE 改善没有转化为 regret 改善。v2 的唯一核心修改是：在 v1 固定的完整 64 序列 train 组内，用全部候选的真实 mean-replicate MRL softmax 与预测 softmax 的交叉熵替代“当前预测 top16 内 pairwise logistic”。

对组内标准化预测 `p` 和真实 MRL `y`：

`ListNet = -sum_i softmax(y/tau)_i * log_softmax(p/tau)_i`

总目标为 `ListNet + mse_weight × MSE`。MSE 保留最后的不完整池，ListNet 只使用完整池。不再根据当前预测选择尾部、构造 pair 或给 pair 赋权；64 个位置全部接受真实标签驱动的 listwise 梯度。

## 2. 冻结数据与 test-v2 隔离

- 原始终端 r1/r2 的 SHA-256 与 v1 完全一致；共有 314,563 条合法 50nt ACGT 序列，6,718 条 H1 边、307,845 个 exact/H1 连通分量，最大分量 3 条序列。
- 50,000 train、10,000 dev、mean-replicate 标签、固定序列池和 dev proposals 全部只读复用 v1。train 有 781 个完整池，dev 有 156 个完整池。
- 新 test-v2 先排除任一 member 属于 v1 train/dev/test 的整个 component，再排除与 Sample train/dev/calibration 有 exact/H1 接触的 component。剩余 39,222 个 eligible component。
- 按 v2 冻结哈希顺序分配完整的 1,004 个 component、1,024 条序列，并以完整 component 打包为 16×64 池。test-v2 对 v1 三角色的 exact/H1 冲突、对 Sample 的 exact/H1 泄漏、跨 test 池 component 均为 0。
- r1/r2 使用完全相同的序列、pool membership 和 proposal prefix。准备阶段只读取无标签序列键和 v1 已冻结 train/dev 标签数组；test-v2 MRL 在全部模型与全部 test-v2 预测冻结前不可读。

机器可读冻结配置 SHA-256：`f7d7c2fc38b5d32d4e7f648b47f6894f3072c7870a869eddca3d8e46a76587e2`。

## 3. 模型、网格与自然收敛协议

网络保持 `FreshSampleCNN` 不变。每个 ListNet run 的 seed 11/23/37 分别从对应的冻结 Sample CNN 成员初始化，最终三 seed 等权平均。原 CNN、v1 main 和 v1 MSE-only 只读复用冻结权重；from-scratch 和 MSE-only 不重跑。

- LR：`[1e-5,1e-4,1e-3]`；温度：`[0.5,1,2]`；MSE 权重：`[0.1,1,10]`，共 27 组合×3 seeds=81 个正式 run。
- Adam、float32、RTX 5090、CUDA 12.8、确定性算法、TF32 关闭、无 CPU 回退。
- `DualPlateau(min_delta=1e-4,patience=5)`，任何 dev 绝对最佳重置 plateau，没有最大 epoch。最佳 checkpoint 只由 dev combined objective 决定。
- 候选选择只用 dev 的 mean-replicate cap16 regret；1e-12 平手后依次比较 dev combined loss、较小 LR、较低温度和较小 MSE 权重。test-v2 不参与温度、权重、LR、seed、预算或 checkpoint 选择。
- ListNet 的 dev 搜索为 27 组合×3 seeds，v1 main 历史搜索为 9 组合×3 seeds；相同的是架构、50k 目标训练标签、dev 指标、proposal 和三 seed 部署方式，不能宣称总训练计算或 dev 搜索预算相同。
- 四臂均对全部 1,024 条 test-v2 序列做三成员全池离线评分，即每方法 3,072 member-item forwards、每池 3×64；随后只在共同 N-item prefix 内离线 argmax。cap16 是 proposal/选择预算，不是“每池仅 16 次 forward”的模型评分成本。非 prefix 分数不参与选择，四臂计算口径相同。

## 4. 预注册终点与成功门槛

主终点是 per-sequence `(r1+r2)/2` MRL 所定义的等预算 regret，r1/r2 分别结果为次要并完整保留。所有方法共享 5 个 nested proposal seeds=`101/202/303/404/505` 和 caps=`1/2/4/8/16`，regret 相对于同池 64 序列实测 oracle。

主成功必须同时满足两个 cap16 比较：

1. ListNet−原 CNN 的 repeat-linked pool block bootstrap 95% CI 上界小于 0，且绝对改善至少 `max(0.1,5%×原CNN基线regret)`；
2. ListNet−v1 main 的同一 CI 与幅度条件也通过；
3. 所有隔离、真实 CUDA、自然收敛、预测冻结和一次评价检查通过。

bootstrap 为 16 个 component-disjoint pool 的 10,000 次有放回整簇抽样，seed=`20261126`，percentile 95% CI。

主推断层级在揭盲前明确为**联合 co-primary intersection–union test（IUT）**：原 CNN 与 v1 main 两项各自使用未校正的 95% percentile CI，只有两项方向和量级门禁全部通过，才宣称“ListNet 同时优于两者”。这两个 CI 不是 Holm 同时区间，也不能分别声称为 Holm 校正后的主结论。另算的 2 comparators×5 caps=10 个 mean-replicate bootstrap 符号型 p 值使用 Holm，属于全 cap 次级多重性层级；只有联合主门禁通过后才解释该族，主失败时其余 cap 只作描述。r1/r2 分开的 CI/p 不另设确认性多重族，均为次要描述。失败后不得换终点、模型、温度、权重、cap 或预算。

## 5. 训练与选型结果

81/81 个正式 run 全部在真实 RTX 5090 CUDA 上自然收敛，共 **21,868 epoch**；单 run 21–1,048 epoch，最佳 epoch 3–320。所有 loss/gradient 有限，三 seed 进程没有 CPU 回退，没有科学最大 epoch，也没有以最后 checkpoint 冒充收敛。累计各 run 训练时间为 19,195.6 秒；完整 history、best-dev checkpoint、初始化和哈希均保留。

唯一 dev 选择为：

| 模型 | LR | 温度 | MSE 权重 | 三 seed 最佳 epoch | dev cap16 mean-replicate regret |
|---|---:|---:|---:|---:|---:|
| ListNet+MSE 50k | 1e-4 | 0.5 | 10 | 13 / 48 / 27 | **1.759305** |

所选 run 名为 `listnet_50000_lr_0.0001_tau_0.5_mse_10`。作为 dev 参考，v1 main 的 mean-replicate cap16 regret 为 1.787570，但 dev 改善不能替代 test-v2 结果。27 个候选的全部 dev 指标和三 seed 摘要均保存在 `selection/listnet.json` 与 training histories。

四臂预测在标签前一次性冻结：original CNN、v1 main、v1 MSE-only、ListNet 各三个 checkpoint 和全部 1,024 序列分数均有 SHA-256。预测冻结 manifest 记录 `test_v2_mrl_read=false`，之后评价目录以 exclusive-create 开启，评价计数为 1。

## 6. test-v2 主结果

效应定义为 **ListNet regret−comparator regret**，负值有利。主估计量先逐序列平均 r1/r2，再计算同一 64 序列池 oracle 的 regret。

| Comparator | Comparator cap16 regret | ListNet cap16 regret | 效应 [未校正95% CI] | 绝对改善 | 所需幅度 | Holm-10 p | 主门禁 |
|---|---:|---:|---:|---:|---:|---:|---|
| 原 CNN | 1.766447 | 1.729031 | **−0.037415 [−0.183204,+0.102494]** | 0.037415 | 0.1 | 1 | FAIL |
| v1 main | 1.736616 | 1.729031 | **−0.007585 [−0.028698,+0.012474]** | 0.007585 | 0.1 | 1 | FAIL |

两个未校正 CI 上界均大于 0，两个改善幅度也都低于 0.1 MRL。因此联合 co-primary IUT 失败，不能宣称 ListNet 同时优于原 CNN 和 v1 main。这里的未校正 CI 不是 Holm 同时区间；cap16 的 Holm-10 p=1 也均不支持次级全 cap 推断。没有因结果更换温度、MSE 权重、LR、cap、预算或终点。

### 6.1 mean-replicate 全 cap 次要结果

主门禁失败，下表及 Holm-10 仅作描述，不把个别 cap 的点估计包装成成功。

| cap | 原 CNN | v1 main | v1 MSE-only | ListNet | Random(N) | ListNet−原 CNN [95% CI] | ListNet−v1 main [95% CI] | 两项 Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.044182 | 3.044182 | 3.044182 | 3.044182 | 3.044182 | 0 [0,0] | 0 [0,0] | 1 / 1 |
| 2 | 2.524848 | 2.510994 | 2.510994 | **2.509004** | 3.044182 | −0.015844 [−0.120888,+0.063402] | −0.001990 [−0.005970,0] | 1 / 1 |
| 4 | **2.044026** | 2.074275 | 2.074275 | 2.075813 | 3.044182 | +0.031787 [−0.011470,+0.080222] | +0.001538 [−0.027558,+0.032172] | 1 / 1 |
| 8 | **1.749059** | 1.793429 | 1.793429 | 1.789545 | 3.044182 | +0.040485 [−0.064086,+0.151149] | −0.003884 [−0.029234,+0.017581] | 1 / 1 |
| 16 | 1.766447 | 1.736616 | 1.736616 | **1.729031** | 3.044182 | −0.037415 [−0.183204,+0.102494] | −0.007585 [−0.028698,+0.012474] | 1 / 1 |

cap2/4/8 的方向不一致，cap16 的小幅点估计改善不能外推为全预算稳定改善。v1 main 与 MSE-only 在所有 mean-replicate cap 的 regret 仍完全相同，尽管逐序列分数可有微小数值差异。

## 7. 次要 repeat、top-tail 与 MAE–regret 诊断

### 7.1 r1/r2 分别全 cap regret

两个 repeat 的结果是次要描述，不另设确认性多重族。所有 repeat-specific CI、p、选择和 turnover 完整保存在 `target_v2/secondary.json`。

| Repeat | cap | 原 CNN | v1 main | v1 MSE-only | ListNet | Random(N) | ListNet−原 CNN | ListNet−v1 main |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r1 | 1 | 3.908336 | 3.908336 | 3.908336 | 3.908336 | 3.908336 | 0 | 0 |
| r1 | 2 | 3.462082 | 3.456151 | 3.456151 | 3.459386 | 3.908336 | −0.002696 | +0.003236 |
| r1 | 4 | 2.982340 | 2.993158 | 2.993158 | 3.007198 | 3.908336 | +0.024858 | +0.014040 |
| r1 | 8 | 2.848152 | 2.839579 | 2.839579 | 2.835761 | 3.908336 | −0.012391 | −0.003818 |
| r1 | 16 | 2.845312 | 2.755641 | 2.755641 | 2.776587 | 3.908336 | −0.068725 | **+0.020946** |
| r2 | 1 | 3.937315 | 3.937315 | 3.937315 | 3.937315 | 3.937315 | 0 | 0 |
| r2 | 2 | 3.344900 | 3.323123 | 3.323123 | 3.315908 | 3.937315 | −0.028992 | −0.007216 |
| r2 | 4 | 2.862998 | 2.912679 | 2.912679 | 2.901715 | 3.937315 | +0.038717 | −0.010964 |
| r2 | 8 | 2.407253 | 2.504566 | 2.504566 | 2.500615 | 3.937315 | +0.093362 | −0.003951 |
| r2 | 16 | 2.444869 | 2.474878 | 2.474878 | 2.438763 | 3.937315 | −0.006106 | **−0.036115** |

ListNet 相对 v1 main 在 r1 cap16 变差、r2 改善，平均后仅 −0.007585。r2 的这一次要 95% CI 为 [−0.075229,−0.004838]，但它不抵消 r1 反向，也不改变 mean-replicate 联合主门禁失败。相对原 CNN，两个 repeat cap16 点估计虽均略有利，幅度和区间同样不达标。

### 7.2 真实 top-tail 翻转与选择更替

以下 pair 均固定在 comparator 的完整池预测 top16 内，真实方向由对应 MRL 决定；“翻转有利/不利”只计算 ListNet 与 comparator 预测顺序发生变化的对。

| Context | Comparator | top16 Jaccard | top1 翻转池 | 翻转有利/不利 ListNet | Comparator pair accuracy | ListNet pair accuracy | 变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| mean | 原 CNN | 0.857843 | 7/16 | 151 / 115 | 0.575521 | 0.594271 | **+0.018750** |
| mean | v1 main | 0.941176 | 3/16 | 66 / 44 | 0.576042 | 0.587500 | **+0.011458** |
| r1 | 原 CNN | 0.857843 | 7/16 | 140 / 126 | 0.529167 | 0.536458 | +0.007292 |
| r1 | v1 main | 0.941176 | 3/16 | 62 / 48 | 0.558333 | 0.565625 | +0.007292 |
| r2 | 原 CNN | 0.857843 | 7/16 | 147 / 119 | 0.575521 | 0.590104 | +0.014583 |
| r2 | v1 main | 0.941176 | 3/16 | 61 / 49 | 0.563021 | 0.569271 | +0.006250 |

mean-replicate cap16 的 80 个 pool×seed 比较中，ListNet 相对原 CNN 更替 23 个，改善/变差为 15/8；相对 v1 main 更替 6 个，改善/变差为 5/1。真实 pair 准确率和更替计数略有有利信号，但平均 MRL 改变量只有 0.037415/0.007585，说明这些局部翻转的幅度不足以形成预注册的功能选择收益。

### 7.3 MAE、RMSE 与 regret 仍不同步

| Context | 模型 | MAE | RMSE | bias | Spearman |
|---|---|---:|---:|---:|---:|
| mean | 原 CNN | 1.099992 | 1.361727 | +0.791727 | 0.728399 |
| mean | v1 main | 0.790973 | 1.093642 | +0.111647 | 0.730150 |
| mean | v1 MSE-only | 0.790997 | 1.093652 | +0.111644 | 0.730162 |
| mean | ListNet | 0.793077 | 1.094663 | +0.125848 | **0.732135** |
| r1 | 原 CNN | 1.351561 | 1.617066 | +1.076405 | 0.700767 |
| r1 | v1 main | **0.906767** | **1.245150** | +0.396325 | 0.700547 |
| r1 | v1 MSE-only | **0.906756** | **1.245134** | +0.396322 | 0.700598 |
| r1 | ListNet | 0.910905 | 1.250094 | +0.410526 | **0.700898** |
| r2 | 原 CNN | 1.114450 | 1.538451 | +0.507049 | 0.669614 |
| r2 | v1 main | **1.053745** | 1.454248 | −0.173031 | 0.672934 |
| r2 | v1 MSE-only | 1.053774 | 1.454278 | −0.173034 | 0.672897 |
| r2 | ListNet | 1.055413 | **1.451541** | −0.158830 | **0.676288** |

ListNet 相对原 CNN 的 mean MAE 降低 0.306915，cap16 regret 同向降低 0.037415；但相对 v1 main，MAE **增加 0.002104**，regret 却略降 0.007585，二者反向。r2 相对 v1 main 也出现 MAE 增加 0.001667 而 regret 降低 0.036115。故“真实标签 listwise 目标进一步降低 pointwise 误差”不成立，局部 pair 改善也没有转化为达到幅度和统计门槛的 regret 修复。

## 8. 三轮核验与失败保留

1. **原始/实现/隔离轮：PASS。** 配置和 10 个执行/验证文件在 prefreeze 哈希冻结；13 项 focused tests 在真实 CUDA 上通过，覆盖 ListNet 数值恒等式、64/64 位置梯度、partial-pool 权重、双基线门槛、共同 prefix、Holm 和真实 CUDA。独立 sequence-only 实现从头重建 307,845 个 component、39,222 个 eligible component 和全部 test-v2 池，逐索引一致，v1/Sample/跨池冲突均为 0。独立预审另以人工张量验证真实目标的解析梯度/有限差分和合成标签恢复；这些只证明数学与实现，不当作 T-cell 效果证据。
2. **独立数值轮：PASS。** 不导入主实现，重新构建 test-v2，核对 81 个训练摘要和 27 个 dev 候选，独立 CUDA 重算四臂预测，重读唯一一次 test 标签并重算所有 context/method/cap 选择、regret、bootstrap、Holm、主幅度、top-tail 和 MAE。共 **446 项检查，最大绝对差 0**；dev 选型差和四臂预测差也为 0。见 `verification/round2_independent.json`。
3. **交付轮：PASS。** 干净进程再次 **13 passed**；配置/代码 prefreeze 哈希未变，81 个 checkpoint/history/summary、四臂预测冻结、evaluation count=1、post-evaluation changes=0 和回顾性边界均通过。远端导出 manifest 含 209 个文件；本地逐文件 SHA-256 回读全部通过，禁止的二进制、逐序列标签/预测和原始表导出为 0。另同步统计澄清和本地回读日志。

首次 prepare 因错误地把完整多序列 component 视为池内重复而在标签读取前停止。该失败完整保留；只把断言修正为“同一 component 的全部成员必须位于同池”，没有修改配置、序列或池。一次本地监控任务超时中断时远端训练仍继续，随后只接续监控，没有重复提交训练。上述事件均未触发 test-v2 重评。

## 9. 产物边界

- 远端代码：`/home/proxy-function-gap-tcell-listnet`；代码不 push。
- 远端完整 artifacts：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_listnet_adaptation_v2`；全部权重保留 teams。
- 本地脱敏聚合：`deliverables/reports/source_data/tcell_listnet_adaptation_v2/`；只包含配置、聚合、训练 history/summary、代码、日志和 verification，不含序列字符串、逐序列标签/预测、`.npy/.npz/.pt` 或原始表。

本轮判伪的是“当前冻结的 full-64 ListNet+MSE 目标及 27 组合 dev 网格，能在新 component-disjoint test-v2 上同时显著且实质优于原 CNN 与 v1 main”。轻微 pair/点估计改善和单个 repeat 的次要区间不能改写该结论，也不证明所有 listwise 目标无效。按预注册失败规则，不再事后更换终点或追加模型搜索。
