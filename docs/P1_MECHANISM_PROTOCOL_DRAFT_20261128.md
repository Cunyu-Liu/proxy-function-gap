# P1 机制实验预注册协议草案：不确定性惩罚尾部选择（UPTS v1）

日期：2026-11-28。状态：**已执行完毕；辅助门禁失败，按冻结协议停止，未评价 T-cell。** 结果与三轮核验见 `TCELL_UPTS_V1_20261128.md`。
证据类别：回顾性方法开发/协议敏感性。历史 T-cell outer 已查看，本实验即使成功也**不构成新盲测、独立研究或前瞻确认**。

## 1. 科学问题与假设

**问题**：T-cell 剩余 regret 主要在高分尾部内部排序（r1：2.480/3.316；r2：2.117/4.071，见 `tcell_error_diagnosis_v1`）。在固定候选池、固定 proxy 分数、固定 proposal 前缀的前提下，仅改变尾部选择规则能否降低真实功能 regret？

**假设 H-UPTS**：三成员 CNN 分歧（`proxy_std`）标记了部分被高估的尾部候选；在前缀内按 `s_κ(x) = proxy_mean(x) − κ·proxy_std(x)` 取 argmax（κ>0）相对原 BoN（κ=0）可降低 cap16 真实 MRL regret，且该改善纯归因于 within-tail 排序（opportunity loss 构造上不变），与全局 MAE/Spearman 无关（分数冻结，回归指标构造上不变）。

**可证伪性**：若分歧不携带尾部高估信号，辅助门禁将失败，实验按冻结规则停止。已知风险：CNN/MLP 误差 Pearson 0.936/0.963（共同误差主导），`proxy_std` 量级仅 ~0.08 MRL，信号可能不足。失败原样保留。

**与已试方法的区别**：七个工作包（residual v1、rank v2、multicontext backbone、target adaptation、ListNet v2、cross-cell anchor、误差诊断）均为训练侧修正、anchor 或诊断；LCB/分歧惩罚选择仅在 E0 合成语境存在（README、`V_Q_SELECTION_PROTOCOL.md`），从未在 Castillo 固定池上评价。本实验是选择侧干预，不训练、不推理、不改分数。

## 2. 勘察结论（评价面可行性）

- Castillo GSE232927 原始库仅 6 个 context 文件：tcell r1/r2、hepg2 r1、hek293t N50 r1、N25 r1/r2。四个 50nt context 全部已查看并冻结（`paper1_budget_matched_v1/PROTOCOL.md`）。
- N25 被 `docs/PAPER1_EXTERNAL_PROTOCOL.md` 明确排除（25nt 与冻结 50nt 模型不兼容，禁止 padding/重训）。
- PERSIST-seq（Leppek GSE173083）已探索性查看，null（+0.104723，CI 跨零）。
- **结论：现有数据资产内无未查看同终点评价面。** 采用执行提示词 P1 选项 C 框架 + 选项 B 干预：辅助数据选型/门禁（§8.2），通过后目标 test 一次冻结评价（§8.3），全程诚实标注回顾性。选项 A 分解作为归因补充（§6）。

## 3. 冻结输入（全部只读，冻结时逐一登记 SHA256）

| 角色 | 路径（远端根 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap`） |
| --- | --- |
| T-cell 固定池 | `data/derived/castillo_fixed_pools_v3_strict_h1_retry1/defined_tcell_r{1,2}` |
| 冻结 CNN 分数（含 `proxy_mean`/`proxy_std`/`proxy_range`） | `paper1_v3_strict_h1_retry1/defined_tcell_r{1,2}/cnn_scores/*.json` |
| 冻结选择/评价 | `paper1_v3_strict_h1_retry1/defined_tcell_r{1,2}/{selection,evaluation}` |
| 等预算重建规则 | `paper1_budget_matched_v1/PROTOCOL.md`（`_draw_index`、PCG64 位置流、嵌套前缀） |
| 辅助严格数组（N50/HepG2，含 `member_predictions.npy`、`labels.npy`、`split.npy`、哈希清单；已对 T-cell r1∪r2 与 Sample exact/H1 排除） | `tcell_residual_bstrict_v1/auxiliary/{n50_hek293t_r1,defined_hepg2_r1}/` |
| 辅助池/选择机器（复用，只读） | `/home/proxy-function-gap/checks/tcell_residual_bstrict_v1/`（冻结时确认其辅助 cap16 BoN regret 评价面定义并原样复用） |
| 简单 baseline 分数 | 冻结 4-mer MLP 分数（`paper1_v3_strict_h1_retry1/*/cross_architecture`，冻结时登记） |

## 4. 方法定义（预注册，禁止事后扩展）

- 选择规则：对每个 context×pool×seed×cap 的冻结嵌套前缀（长度 N，含重复提案位置），UPTS(κ) 选前缀内 `s_κ = proxy_mean − κ·proxy_std` 最大者的**首次出现位置**（与冻结 BoN 的 first-occurrence 平手规则一致）。
- κ 网格（唯一超参，预先固定）：`[0, 0.5, 1, 2, 4, 8, 16]`。依据：`proxy_std` 典型 ~0.08，κ=16 对应惩罚 ~1.3 MRL，覆盖 0 到约一个尾部 score 展幅；网格在看到任何辅助结果前冻结。
- **恒等核验**：κ=0 必须逐值复现全部 1,600 个冻结 BoN 选择（index/ID/regret），否则实现失败，停止。
- 不做其他选择规则变体（不加 top-k median、rank-shrinkage 等第二臂），避免搜索扩张。

## 5. 辅助选型与前置门禁（不看 T-cell 标签）

1. 两方向 leave-one-context-out：N50→HepG2 与 HepG2→N50，沿用 residual v1 已冻结的辅助 train/dev/test 拆分与池评价面。
2. 选型：仅用 destination **dev** 的 cap16 平均 BoN regret 选 κ；平手取较小 κ。test 不参与选型。
3. 预注册机制诊断（仅辅助，无论结果均报告）：辅助 top16 内 `proxy_std` 与符号误差 `(p−y)`、绝对误差 `|p−y|` 的 Spearman；若分歧与高估无正关联，如实记录为假设不利证据。
4. **门禁（沿用 §8.2，任一失败即停止，不评价 T-cell）**：
   - 原始哈希/拆分/隔离核验失败；
   - 所选 κ 在**两个**方向的 auxiliary test 上不能同时降低 cap16 平均选择 regret（相对 κ=0）；
   - 只改善 MSE/校准而改善不了选择；
   - 需要读取 T-cell MRL 才能决定 κ 或任何配置。
5. 门禁失败 → 保留全部辅助数值与诊断，写失败报告，提出下一假设；不得换 κ 网格/评价面重claim。

## 6. 目标一次冻结评价（仅在双门禁通过后）

- 评价面：strict-H1 T-cell r1/r2 全部 16 池×64 候选，caps {1,2,4,8,16}，seeds {101,202,303,404,505}，同一冻结嵌套前缀。cap1 构造上恒等（前缀长度 1）。
- 主终点：cap16 的 `UPTS(κ*) − frozen-BoN` 逐池平均真实 MRL regret 差（负值有利）；regret = 完整池实测 oracle − 所选实测 MRL。
- 对照：RandomN（冻结重建）、frozen BoN（κ=0）、冻结 MLP BoN（简单 baseline；T-cell 无加性 ridge，如实披露，不新训）。
- 统计：完整池为簇，10,000 次 cluster bootstrap，PCG64 种子 `20261128 + 100*context_index + cap_index`（冻结时登记）；95% 百分位区间；次级 cap2/4/8 与 cap1 零行纳入每 context 10 行 Holm 族（与误差诊断口径一致，不合并、不事后换族）。
- **主成功判据（沿用 §8.3，全部满足）**：
  1. r1 与 r2 cap16 点估计均 < 0；
  2. 两情境合并池簇 95% CI 上界 < 0；
  3. 每 context 绝对降低 ≥ `max(0.1 MRL, 5%×冻结基线平均 regret)`（r1 阈值 0.178，r2 阈值 0.252，按冻结基线 3.555/5.043 计算，冻结时以 artifact 精确值重算并登记）；
  4. 隔离与全部日志核验通过。
- 任一不满足 → 判"未证明实质改进"，保留全部数值；不得换 cap/子集/κ 重claim。
- **机制归因（选项 A/B 结合）**：前缀固定 → opportunity loss 逐池恒等；Δregret 全部计入 within-tail 排序损失变化。逐池报告：被降级/升级候选的 `proxy_std`、within-tail loss 变化、选择更替数。全局 MAE/Spearman 构造上不变 → 若成功，直接展示"回归指标零变化下的决策改善"脱钩证据。

## 7. 执行与三轮核验

- 计算性质：**纯 CPU 统计与记录重建**（分数已冻结，无训练/推理；按硬约束 4 明确标注 CPU-only）。无 GPU 需求，不占显存。
- 新代码根：`/home/proxy-function-gap-tcell-upts-v1`；新 artifacts 根：`/teams/.../tcell_upts_v1`（体积小，JSON/TSV 聚合；逐候选序列/标签仅留远端）。既有目录只读。
- 轮 1：原始哈希、前缀重建、κ=0 恒等（1,600 对 index/ID/regret）、辅助拆分隔离核验。
- 轮 2：独立实现（另写一份选择+bootstrap 统计）逐值比对。
- 轮 3：仓库测试 + 远端/本地 manifest 与 SHA256 一致性；失败日志保留。
- 本地交付：`deliverables/reports/source_data/tcell_upts_v1/`（聚合表、协议、哈希、日志）+ 报告文档；不含序列字符串与逐序列标签。

## 8. 停止规则与失败处理

- 门禁失败、恒等核验失败、数值病态 → 立即停止，保留证据，写失败原因与下一假设。
- 目标评价后无论结果如何，本冻结检验结束；后续方向必须另立新协议。

## 9. 待协调者冻结前拍板的开放问题

1. κ 网格上限 16 是否足够（不扩网格；若辅助 dev 最优落在边界，按冻结网格取值并如实记录边界命中）。
2. residual v1 辅助池评价面的精确定义（池大小/种子/簇）需在冻结时从其 manifest 逐字确认并哈希登记。
3. bootstrap 种子偏移与 Holm 族定义按 §6 冻结，不再改动。
4. 投稿叙事定位：若成功，作为"预注册辅助门禁下的选择侧机制干预（回顾性）"进 Results；若失败，作为"共同误差主导、分歧信号不足"的判伪证据进 Discussion/Limitations。两种结局均服务主线。