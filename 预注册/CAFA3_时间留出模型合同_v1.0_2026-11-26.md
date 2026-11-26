# CAFA3 时间留出模型合同 v1.0

日期：2026-11-26 ｜ 状态：**预注册；在模型性能结果产生前冻结** ｜ 数据：`cafa3_timeholdout_v1`

## 1. 数据边界与任务

- 唯一输入是 `E6_蛋白迁移/results/cafa3_timeholdout_v1/frozen_manifest.json` 对应的冻结产物；先核验 `frozen_manifest.json` 的 `summary_sha256` 与所需输入 SHA，外部数据不改写。
- t0 用于训练、早停、tau 与阈值选择；三个实名 fold 按冻结白名单读取。每折 held-out 同时输出 validation 选参轨迹和选择后的 `t0_proxy_test` 指标，但后者与该折 validation 是同一批蛋白，**不是独立 test，不用于无偏泛化主张**。
- t1 是最终时间 OOD，只在某 fold×track×ontology 的模型、epoch、tau 和阈值均由 t0 validation 冻结后评价一次。NK/LK、频率、完全重复及同源桶仅为冻结后的报告切片，绝不用于选参。
- t0 三折仅保证完全序列 MD5 组不跨折，不能称为全同源隔离。30/50/90 压力轨只按冻结白名单剔除 t0 train/validation；t1 成员与真值不变。BLASTP 未检出不等于无同源。

## 2. 冻结表示与模型

- 官方 `esm==3.2.1` ESMC-300M，参数全部冻结，显式 CUDA，无 CPU 降级。序列最多保留前 510 个氨基酸，最终层真实氨基酸位置 mean pooling，排除 BOS/EOS/pad，输出 float16 缓存，维度必须为 960；ID 顺序、数组形状与有限值均验收。
- BCE：960→GO 标签的线性头，AdamW，lr=1e-3、weight_decay=1e-4、batch=512，seeds=0/1/2。按 ontology 分开拟合，标签为历史 OBO 的 `is_a`/`part_of` 传播多标签。
- deep-kNN：原始 960 维嵌入 L2 归一化，cosine top-k=20，权重 `max(similarity,0)+1e-6`，确定性结果，不伪造三 seed。
- BCE logit adjustment 为 `logit - tau*log((t0_direct_frequency+0.5)/(n_t0+1))`，候选 tau=0/0.5/1，按三 seed 平均 validation 低频标签（direct frequency<100）macro AUPRC 选择；平分取较小 tau。kNN 固定 k=20，不使用该 logit adjustment。
- 某 fold/track 中传播训练频率为 0 的标签整列 masked，不参与 BCE，不能把未知/NaN 当负例；其输出固定为 0。训练频率大于 0 时，密集矩阵的 0 仅表示 CAFA 开放世界计算中的未标注状态，不解释为实验阴性。任何 NaN/Inf 输入、logit 或概率均使运行失败。

## 3. 选参与停止

- epoch 上限 1000 是故障保险，不是收敛证据；patience=20，min_delta=1e-4。每 epoch 用 validation masked BCE 选择最佳 checkpoint，严格改善超过 1e-4 才清零停滞计数。
- tau 仅用 validation；Fmax 阈值与 Smin 阈值分别在 0.01–1.00（步长 0.01）validation 网格上选择。Fmax 平分取较低阈值，Smin 平分取较低阈值；每 seed/model 独立冻结。test/t1 不重新扫描最优阈值。
- 三 seed 的 epoch、tau、阈值及 validation 历史落盘。正式运行不减少 seed、fold、ontology 或 track；小规模 smoke 单独标记，不能并入正式结果。

## 4. 指标与分母

- 每个 ontology×NK/LK 报告 protein-centric Fmax、Smin 及所选阈值下 coverage；另有 overall 与 direct frequency `0/1-4/5-19/20-99/100-499/>=500` macro AUPRC。macro AUPRC 只平均该切片至少一个正例且有有限预测的标签，同时报告有效标签数；undefined 保留 NaN，绝不填 0。
- Smin 的 information content 由冻结 t0 propagated protein frequency 计算：`-log((frequency+1)/(66841+1))`；`ru` 为未预测真值的平均 IC，`mi` 为错误预测的平均 IC。t1 只报告 validation 所选阈值处的值，不以 t1 的 `max(F)` 或 `min(S)` 冒充冻结评价。
- t0 validation 评价使用相同历史 OBO 从 direct clean 注释传播的真值。t1 使用冻结 propagated 真值，只评价 `has_official_leaf_truth=1` 的官方角色成员；BPO NK 覆盖必须为 848/849，无真值成员不构造成全零负例，UNLISTED 不进入 NK/LK。
- 输出逐标签 AUPRC、frequency bucket、NK/LK 正例数，及 protein-level 指标分母。完全重复和同源桶只作附加切片，不重新选择模型。

## 5. 执行与验收

- 产物：`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/cafa3_timeholdout_esmc300m_p20_v1/`；代码、状态与日志在 `/home/ProxyFunctionGap/`。
- 独立 `nohup`、独立 `flock`，每阶段前检查 GPU 空闲显存与 MemAvailable，不足每 300 秒重查；不得 kill、暂停、抢占或修改 RiNALMo、LoRA/full 及其他队列。
- 正式组合为 3 folds×4 tracks（main/id30/id50/id90）×3 ontologies；每个组合含 BCE 三 seed 与一个 deterministic deep-kNN。逐组合原子完成标记与断点跳过，失败必须传播，禁止静默缺组合。
- CUDA smoke 必须验证真实样本 ID 对齐、960 维、全部有限及 BCE 反向传播。最终审计检查全部 36 组合、108 个 BCE seed 和 36 个 kNN、白名单、阈值来源、有限值、t1 只读，以及从冻结阈值网格和逐标签结果重算选择值与 macro AUPRC。
