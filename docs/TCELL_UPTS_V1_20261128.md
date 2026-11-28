# T-cell UPTS v1：不确定性惩罚尾部选择——辅助门禁失败报告（2026-11-28）

状态：**已完成；双方向辅助门禁未通过，按冻结协议停止，未评价 T-cell。**
证据类别：回顾性方法开发/协议敏感性。历史 outer 已查看，本工作不构成新盲测或前瞻确认。
计算性质：纯 CPU 统计与记录重建；无训练、无推理、无 GPU（按执行提示词硬约束 4 标注）。

## 1. 科学问题与假设

T-cell 剩余 regret 主要在高分尾部内部排序（r1：2.480/3.316；r2：2.117/4.071）。假设 H-UPTS：三成员 CNN 分歧 `proxy_std` 标记被高估的尾部候选；在**同一冻结嵌套 proposal 前缀**内按 `s_κ = proxy_mean − κ·proxy_std` 取 first-occurrence argmax（κ>0）可降低 cap16 真实 MRL regret。前缀固定 → opportunity loss 构造上不变；分数冻结 → 全局 MAE/Spearman 构造上不变。任何效应纯归因 within-tail 排序。

与已试七个工作包（训练侧修正/anchor/诊断）不同，本实验是选择侧干预，`proxy_std` 信号从未在 Castillo 固定池上被利用。

## 2. 冻结协议与输入

- 配置：`configs/upts_frozen.json`，SHA256 `790d0727660738712c1e8a88cda5ab2025627af50a8e0963c981148abef1c07b`（计算任何指标前冻结）。
- 64 个输入文件哈希登记于 `input_hashes.json`（辅助数组、冻结 dev/test 池与提案、residual v1 selection/gate、budget results/PROTOCOL/frozen_config、T-cell cnn_scores 32 份、池清单）。
- κ 网格预注册 `[0, 0.5, 1, 2, 4, 8, 16]`，唯一超参；选型仅用 destination dev cap16 平均 BoN regret，平手（1e-12）取较小 κ；test 不参与选型。
- 门禁（§8.2）：所选 κ 在**两个** leave-one-context-out 方向的 destination test 上同时严格降低 cap16 regret；任一失败即停止，不读 T-cell MRL。
- 评价面逐字复用 residual v1 冻结池/提案（SHA256 钉住）；`proxy_std` = 三成员 `std(ddof=1)`（与 `inference.py:152` 一致）；成员均值与冻结 `cnn_predictions.npy` 最大绝对差 **0.0**（两方向）。
- §8.3 目标幅度阈值已预先冻结（`frozen_thresholds.json`）：r1 ≥ 0.177759 MRL（基线 3.555181），r2 ≥ 0.252169 MRL（基线 5.043377）。本轮未使用。

## 3. 结果（全部保留）

### κ=0 恒等核验

两方向 dev/test cap16 平均 regret 与 residual v1 `selection.json` baseline **逐值相等**（精确 ==，非容差）。

### 辅助 dev 选型网格（cap16 平均 BoN regret，MRL）

| κ | N50→HepG2 dev | HepG2→N50 dev |
| ---: | ---: | ---: |
| 0 | 4.091181 | 5.800774 |
| 0.5 | **4.086141**（选中） | 5.797403 |
| 1 | 4.087868 | 5.802234 |
| 2 | 4.091257 | 5.795600 |
| 4 | 4.115204 | **5.793045**（选中） |
| 8 | 4.161173 | 5.800778 |
| 16 | 4.192927 | 5.825576 |

### 辅助 test 门禁（未参与选型）

| 方向 | 选中 κ | κ=0 test regret | UPTS test regret | 改善（正=有利） | 门禁 |
| --- | ---: | ---: | ---: | ---: | --- |
| N50→HepG2 | 0.5 | 4.082435 | 4.070261 | **+0.012174** | PASS |
| HepG2→N50 | 4 | 5.828669 | 5.858227 | **−0.029558** | **FAIL** |

`all_directions_pass = false` → `target_evaluation_permitted = false`，`target_mrl_read = false`。**未生成 T-cell 选择，未运行 T-cell Evaluator，无目标 regret/CI。**

### 预注册机制诊断（destination dev 池 top16，仅报告不选型）

| 方向 | n | Spearman(std, 符号误差 p−y) | Spearman(std, |误差|) | top16 平均 std |
| --- | ---: | ---: | ---: | ---: |
| N50→HepG2 | 298,032 | 0.003599 | 0.004685 | 0.1103 |
| HepG2→N50 | 55,040 | 0.009172 | 0.017150 | 0.1243 |

## 4. 解释与论文含义

1. **假设被辅助数据判伪**：尾部内分歧与高估误差的相关接近零（0.004–0.017），`proxy_std` 量级（~0.11 MRL）远小于尾部误差（~1.5 MRL）。三成员共享同一训练数据与架构归纳偏置，误差高度相关（CNN/MLP Pearson 0.936/0.963），分歧主要反映数值噪声而非功能性高估。
2. 单方向（N50→HepG2）的小幅 dev/test 改善（~0.012 MRL）不足以跨越双方向门禁，且幅度远低于 §8.3 实质门槛；**不得**挑选该方向单独声称成功。
3. 对主线的贡献：这是第八个被预注册门禁判伪的修复方向，且是第一个**选择侧**干预。它与训练侧失败（residual/rank/backbone/ListNet/adaptation）共同表明：T-cell 尾部翻转不能由冻结 proxy 的内部不确定性或辅助共享修正修复，剩余 within-tail 损失需要 proxy 之外的信息。论文应将其作为"为什么需要新信息/新监督而非重排已有分数"的机制边界证据，而非方法贡献。
4. **发表定位建议**：累计判伪证据支持 Paper 1 以评估原则+机制诊断+边界为主贡献（执行提示词第二节第 2 条的判断成立）；方法论文路线在当前数据资产内缺乏可部署改进。

## 5. 下一假设（未执行，另立协议才可检验）

- H-next-A（机会侧）：固定评价预算下扩大提案覆盖（N>16 前缀）能否以 opportunity loss 降低为主改善 regret——与 within-tail 干预正交，可用同一冻结分数 CPU 检验；但会改变预算语义，须预注册新 estimand。
- H-next-B（信息侧）：尾部翻转是否由特定序列特征（如 uAUG 上下文、poly-U 位置）系统性驱动——需在辅助数据上做特征条件误差分析，仅诊断不修复。
- 若两者均无信号，接受边界，把资源转向 P3 独立数据与 P5 论文升级。

## 6. 三轮核验

| 轮次 | 内容 | 结果 |
| --- | --- | --- |
| 1 | 64 输入哈希核验、κ=0 恒等（精确 ==）、成员均值一致性、7 项单元测试（含与 residual v1 `cap16_regret` 逐值比对、提案公式重建） | PASS |
| 2 | 独立纯 Python 循环重算 dev/test regret（容差 1e-9 内一致）、提案从冻结公式独立重建逐值相等、scipy spearman 独立复算诊断 | PASS（`verification/round2_independent.json`） |
| 3 | 本地↔远端 8 个 artifact + 6 个代码文件 SHA256 全一致；测试复跑 7 passed | PASS（`verification/round3_delivery.json`） |

## 7. 文件与路径

- 远端代码：`/home/proxy-function-gap-tcell-upts-v1/`（configs/src/scripts/tests）。
- 远端 artifacts：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/tcell_upts_v1/`（input_hashes、frozen_thresholds、auxiliary_gate、两方向 selection/gate、verification 轮 2/3）。
- 本地交付：`deliverables/reports/source_data/tcell_upts_v1/{code,artifact}/`；本报告 `deliverables/reports/docs/TCELL_UPTS_V1_20261128.md`。
- 协议草案：`deliverables/reports/docs/P1_MECHANISM_PROTOCOL_DRAFT_20261128.md`（已执行，门禁失败）。
- 逐候选序列与标签仅留远端；本地无序列字符串、无逐序列标签。
- GitHub：尚未提交（由协调者在 P6 统一提交代码/协议/报告，不含数据）。