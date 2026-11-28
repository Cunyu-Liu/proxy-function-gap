# 数据 / Baseline / OOD 综合审计（V10，2026-11-28）

状态：审计完成。本表是对"数据集找全？清洗？分布？泄漏？质量？划分？benchmark？baseline？超参？OOD？"检查清单的权威回应，逐行溯源到冻结 artifacts。所有结果为历史公开数据回顾性分析；无前瞻湿实验。

## 1. 数据集覆盖与缺口

| 家族 | 场景 | 来源 / 许可 | 终点语义（不可互换） | n（评价群） | 数据角色 |
| --- | --- | --- | --- | ---: | --- |
| mRNA | T-cell MRL r1/r2 | Sample 2019 GSE114002 训练 + Castillo-Hair 2024 GSE232927 defined-end | measured MRL | 1024/repeat（16×64 池） | 机制深描 |
| mRNA | HEK293T N50 / HepG2 defined | Castillo-Hair GSE232927 | measured MRL | 1024/context | 跨细胞 OOD |
| mRNA | Castillo indel | Castillo-Hair 编辑汇总 | normalized indel（非 MRL） | 48 UTR / 4 情境 | 跨终点边界 |
| mRNA | Castillo stability | Castillo-Hair 稳定性计数 | integrated abundance（非表达） | 27 实验 UTR（8 spike-in 排除） | 反向边界 |
| mRNA | Leppek MRL / Schlusser TE | GSE173083 / Schlusser Zenodo 10463061 | measured MRL / endogenous TE | 59 构建体 / 6309 基因 | 历史 null 对照 |
| protein | GB1 binding | ProteinGym Olson 2014（CC-BY） | IgG-Fc 结合富集 **lnW（非 Kd）** | 20063 test 突变 / 55 位置对池 | 阳性扩展 |
| protein | GFP brightness | ProteinGym Sarkisyan 2016 | log 亮度（**非纯表达**） | 16000（16×1000 池，≥5 阶） | 阳性扩展 |
| gene therapy | AAV9 production | Bryant 2021 Fit4Function | **Production 富集（非 tropism/感染/临床）** | 63941 / 3996 disjoint 池 | 阳性扩展 |
| enzyme | TEM-1 resistance | Stiffler 2015 MaveDB 00000086（CC0） | 实验派生相对抗性（**非酶活/MIC/临床**） | 52 test 位置 / 25 方向 | 条件迁移边界 |

**缺口（如实）**：AAV tropism/infectivity 独立实测、蛋白稳定性/表达新终点、真正前瞻独立研究均未纳入。P3 已锁定候选 Cao 2021（GSE176581，5′UTR 蛋白产量）/ Ogden 2019（GSE139657，AAV2 肝 tropism）/ Tsuboyama 2023（稳定性）；本轮评价 Cao（见 `SCENARIO_CAO2021_V1`），其余列后续工作。不能声称覆盖所有热门场景。

## 2. 清洗与质量

- 预测字段与实测字段分列；缺失不插补（`endpoint_coverage` 三态：实测/预测/缺失）。
- GB1 lnW 下限保留（censored 不删除）；GFP 亮度≠纯表达；AAV Production≠tropism；TEM-1 派生 fitness≠直接酶活。
- stability：8 spike-in 不作候选，6 时间点不作 6 个生物重复。
- Castillo 48 条中仅 12 条实测 MRL，45 条作者预测 MRL 不混用。
- 序列精确/近重复（exact/H1）去重与隔离；GC/同聚物/uAUG 特征敏感性已做（`paper1_sequence_ablation_v1`）。

## 3. 数据分布

各场景标签分布见新增 Extended Data 图（`figures/v8/ED_label_distributions`）。关键特征：MRL 右偏、AAV Production 极端右偏（top1% 召回仅 18.75%）、Schlusser TE 极端右偏、stability 全局 Spearman −0.766（proxy 与终点反向）。分布异质是"不得跨单位合并原始效应"的根据。

## 4. 泄漏控制与划分

| 场景 | 隔离规则 | 簇单位 | 划分 |
| --- | --- | --- | --- |
| T-cell | exact/H1 + repeat-linked pools | pool（16/repeat） | Sample train/dev/cal → Castillo 永久 test |
| GB1 | 位点阻断（position-blocked） | 位置对池（55） | dev 选 λ → 永久 test |
| GFP | 低阶 train（1–2）/ 高阶永久 test（≥5），位点互斥 | pool（16） | 位点 hash 70/30 |
| AAV | 跨库 exact/H1 排除 | disjoint pool（3996） | train/dev → 永久 test，test_labels_read=false |
| TEM-1 | 位置簇（同位置全 variant 同 split） | 位置（52） | 位置 hash 60/20/20，seed 20261127 |
| mRNA | UTR/构建体单位；历史 outer 不重称盲测 | UTR/构建体 | 历史冻结 |

反泄漏测试：标签打乱后方法不应保持优势；近重复跨组移动应被检测阻止；冻结后再揭盲生成新运行记录。所有 grouped holdout **不是**前瞻盲测/独立研究。

## 5. Benchmark / Baseline 全表

### 5.1 本项目已评价 baseline

| Baseline | 实现来源 | 超参选择角色 | 官方? |
| --- | --- | --- | --- |
| RandomN / BoN（等预算） | `paper1_budget_matched_v1` 等 | 无超参；冻结 | 本项目协议 |
| 三成员 CNN ensemble | `fresh_proxy_cnn_v2`、retry1 cnn_scores | train-only scaler，dev 早停 | 本项目 |
| 4-mer MLP（简单 k-mer） | retry1 cross_architecture | 同上 | 本项目 |
| 加性 ridge（≈one-hot 加性线性） | `scenario_gb1_v1`、`scenario_gfp_v1` | dev 选 λ/α | 本项目 |
| COM-style / LCB / distance / diversity-LCB | `paper2_baselines_v1` | dev-only 网格 | COM 独立适配（非官方权重） |
| MSE-only / ListNet / residual / rank / backbone / anchor / adaptation | `tcell_*` | 辅助 train/dev + §8.2 门禁 | 本项目 |
| UPTS（不确定性惩罚选择） | `tcell_upts_v1` | 辅助 dev 选 κ，双方向门禁 | 本项目（门禁失败） |
| **Oracle upper bound（补充描述行）** | `tcell_error_diagnosis_v1`（within-prefix headroom 1.835/2.955 MRL）；AAV `oracle_mean_production` 7.380 | 诊断上界，非部署 | 本项目 |

### 5.2 外部方法可比性（P4 审计，六者均不可官方等价复现）

| 方法 | 判断 | 原因 |
| --- | --- | --- |
| 官方 COM | 需适配且不等价 | 已有诚实标注 CNN 适配（非 UTR-ResNet、实测 MRL 非 oracle-relabel、固定池重排非 de novo） |
| Design-Bench | 不可比 | learned oracle 评分 + UTR 任务即 Sample（泄漏）+ 生成池外序列 |
| Reddy promoter | 不可比 | 不同分子/终点（启动子湿实验） |
| GABO/aSCR | 需适配且不等价 | 约束生成轨迹，固定池无轨迹 |
| T3PO-mRNA / mRNAutilus | 不可比 | 生成式池外序列 |

### 5.3 P4 小缺口处理（本轮）

- **one-hot linear**：GB1/GFP 的加性 ridge（共享替换效应）即 one-hot 加性线性模型，蛋白场景已覆盖；T-cell 纯 one-hot ridge 未单列，4-mer MLP 作为简单 k-mer 基线已评价。**如实披露**：T-cell 纯 one-hot ridge 未单独部署。
- **简单 ensemble（CNN+MLP 均值）**：cross-arch 分析存在两路分数，但未作为部署选择 baseline 单独评价；**披露为未部署补充项**（不新训、不在已查看 test 上重 claim）。
- **oracle upper bound**：已补为描述行（§5.1），不需新实验。

## 6. 超参表（全部 dev-only 选择，test 一次评价）

| 场景 | 模型 | 关键超参 | 选择角色 | 收敛证据 |
| --- | --- | --- | --- | --- |
| T-cell | CNN ensemble | 3 成员 seeds 11/23/37；train-only scaler | dev 早停 | strict-H1 协议 |
| GB1 | 加性 ridge | λ=10.0（dev 选）；5 seeds 11/23/37/53/71 | dev | `training_convergence.json` |
| GFP | additive + CNN | CNN c32/k9/lr1e-3/wd1e-3/do0.1；5 seeds；α dev 选 | dev | `dev_selection.json` |
| AAV | CNN | 3 seeds；dev MSE 早停（best epoch 12–13） | dev | `all_converged=true`，`converged_by_frozen_rule=true` |
| TEM-1 | 无拟合 | 源条件 fitness 直接作 proxy | — | 无模型 |
| UPTS | 选择规则 | κ∈{0,0.5,1,2,4,8,16}，辅助 dev 选 | 辅助 dev | 门禁失败停止 |

禁止：看到 test 后换阈值/cap/子集/模型/终点再称同一冻结检验成功。

## 7. OOD 覆盖（独立单位 ≠ 算法种子）

| 维度 | 证据 | 独立单位 | 状态 |
| --- | --- | --- | --- |
| 跨细胞 | T-cell vs HepG2 vs HEK293T | context/池 | 已覆盖 |
| 跨构建体 | defined-end vs random-end | 构建体/UTR | 已覆盖 |
| 跨终点 | MRL/stability/indel/TE/production/lnW/brightness/resistance | 各场景簇 | 已覆盖（禁止合并原始效应） |
| 跨研究 | Sample→Castillo/Leppek/Schlusser | 研究/库 | 已覆盖（回顾性） |
| 跨突变阶数 | GFP 低阶→高阶（order9 ρ 衰减） | 位点互斥池 | 已覆盖 |
| 跨药物/条件 | TEM-1 Amp→cefotaxime（15/15 跨零） | 位置簇 | 已覆盖 |
| 前瞻/真正独立研究 | Cao 2021（本轮，跨终点回顾性独立研究） | 研究 | 部分补强；仍非前瞻 |

算法种子（101–505 等）只是提案重复，**不作独立生物重复**。

## 8. 三轮核验引用

- 跨场景综合：`CROSS_SCENARIO_SYNTHESIS_VERIFICATION_20261128_ROUND{1,2,3}.json`（PASS）。
- 统一尾部关联：`cross_scenario_tail_audit_v2/association_verification_round{1,2,3}.json`（PASS）。
- 各场景冻结 verification（round1_raw / round2_independent / round3_delivery）见各 `scenario_*/verification/`。

## 9. 结论

数据/baseline/OOD 覆盖足以支撑"评估原则 + 机制诊断 + 边界"主贡献；外部 SOTA 不可等价复现已逐条披露；简单基线（加性 ridge）在 GFP/GB1 优于 CNN，证明基线公平性被认真对待。剩余缺口（T-cell 纯 one-hot ridge、CNN+MLP 部署 ensemble、前瞻独立研究）如实列入 Limitations，不伪造、不硬凑。