# P4 baseline 与 OOD 可比性审计（2026-11-28）

状态：审计完成；未训练、未评价、未修改冻结 artifact。协调者直接执行（子代理后端持续故障）。
来源：本地 `PREPRINT_REFERENCE_AUDIT_20261126.md`、`FULL_AUDIT_20261124.md`、远端 `docs/PAPER2_BASELINES_PROTOCOL.md`/`PAPER2_BASELINES_RESULT.md`，以及本轮官方页面核验（Design-Bench GitHub、COM arXiv、GABO NeurIPS 原页）。

## 1. 外部方法在同一固定池评价面的可比性

本项目评价面定义：**给定候选池 + 等预算嵌套 proposal 前缀 + BoN/RandomN + 真实功能标签 regret（pool-cluster bootstrap）**。

| 方法 | 来源（已核验） | 官方评价面 | 终点 | 代码/权重 | 判断 | 原因 |
| --- | --- | --- | --- | --- | --- | --- |
| **官方 COM**（Trabucco et al. ICML 2021） | arXiv:2107.06882；作者在 design-baselines 仓库（本项目协议引用 commit `785dbcfa`） | 离线 MBO：对 learned surrogate 做 de novo 优化，design-bench oracle 评分 | UTR/GFP 等任务用 **oracle-relabelled** 或 exact oracle | 官方实现可得；**官方 UTR-ResNet 权重未发布为可零样本评分 Castillo 池的资产** | **需适配且不等价**（已有 COM-style adaptation） | 本项目已做独立 PyTorch COM 适配（`paper2_baselines_v1`）：CNN 而非 UTR-ResNet、实测 MRL 而非 oracle-relabel、固定池重排而非 de novo 生成、本仓库 split/收敛协议。四处差异已逐条披露；不得称官方复现 |
| **Design-Bench 协议** | github.com/brandontrabucco/design-bench（本轮核验）；arXiv:2202.08450 | 生成 128 个新候选，用 **approximate oracle**（held-out 训练的 ResNet/Transformer 等）评分，normalized 100th percentile | UTR-ResNet-v0 = **Sample 2019 280k**（ρ 0.8617）；GFP-Transformer-v0 = Sarkisyan 56k | pip 包 + 服务器下载 | **不可比（评价面不同构 + 泄漏）** | (a) 用 learned oracle 而非实测标签，存在 train-test discrepancy（其 README 明示）；(b) UTR 任务即 Sample——本项目 train/dev/cal，作 test 即泄漏；(c) 生成池外新序列 ≠ 固定池选择。可在 related work 引用其 normalized-score 协议差异，不可直接比数值 |
| **Reddy et al. NeurIPS 2024 promoter** | DOI 10.52202/079017-2954；作者 Reddy/Geng/Herschl/Kolli/Kumar/Hsu/Levine/Ioannidis（本地审计已核验） | COM + 集成 LCB + 多样性选择设计**细胞类型特异启动子**，湿实验验证 | 启动子活性（湿实验） | 论文+代码（NeurIPS 页） | **不可比（不同分子/终点/评价面）** | 启动子 ≠ 5′UTR/衣壳/酶；其 LCB/多样性**思想**已被本项目 paper2_baselines 在 Sample 面适配评价（lcb/diversity_lcb），但分子与终点不同，不能跨搬数值 |
| **GABO/aSCR**（Yao et al. NeurIPS 2024） | DOI 10.52202/079017-1397；本轮核验 NeurIPS 原页；code github.com/michael-s-yao/gabo | 生成对抗 MBO：source critic 约束优化轨迹接近参考分布，offline **generative** design tasks | design-bench 式 oracle | 官方代码可得 | **需适配且不等价** | aSCR 约束的是生成轨迹；固定池重排无轨迹可约束。若把 source critic 当池内惩罚项属新适配，非官方等价，须另立协议 |
| **T3PO-mRNA** | OpenReview KTCInSlPpL，题名 *Full-length mRNA Design with Reward-Guided Masked Diffusion Model Fine-Tuning*（本地审计：论坛 403，作者/年份未核实） | 树搜索找高 reward 轨迹→微调扩散模型，**生成全长 mRNA** | reward（多性质） | 未核实 | **不可比（生成式，池外序列）** | 与固定池选择不同构；本地 FULL_AUDIT_20261124 已判定可不比但须说明。投稿前人工补核作者/年份 |
| **mRNAutilus** | arXiv:2605.31296（2026-05-29，本地审计已核验存在） | MDM + 蒙特卡洛树引导，**多目标生成 mRNA** | 多治疗性质 | 预印本，代码未核实 | **不可比（生成式）** | 同上 |

**结论**：六个外部方法中，无一能在本项目固定池实测 regret 面上做**官方等价**复现。COM 已有诚实标注的适配；其余或评价面不同构（生成式/oracle 评分）、或分子终点不同、或数据泄漏。论文 Limitations 必须披露，不得用不等价实现冒充官方比较。

## 2. 已有 baseline 清单（实现来源 / 超参角色 / 是否官方）

| Baseline | 实现/artifact 路径 | 超参选择角色 | 官方? |
| --- | --- | --- | --- |
| RandomN / BoN（等预算） | `source_data/paper1_budget_matched_v1/`；远端 `paper1_budget_matched_v1` | 无超参；冻结重建 | 本项目协议 |
| 三成员 CNN ensemble（proxy_mean） | 远端 `fresh_proxy_cnn_v2`、`paper1_v3_strict_h1_retry1/*/cnn_scores` | train-only scaler，dev 早停 | 本项目 |
| 4-mer MLP（简单 k-mer baseline） | 远端 `paper1_v3_strict_h1_retry1/*/cross_architecture` | 同上 | 本项目 |
| 加性 ridge | `source_data/scenario_gb1_v1/`、`scenario_gfp_v1/` | dev 选择 | 本项目 |
| COM-style / LCB / distance / diversity-LCB | 远端 `paper2_baselines_v1`；`docs/PAPER2_BASELINES_PROTOCOL.md` | **dev-only** 网格（λ/γ），calibration 不参与选型 | COM 为独立适配（非官方权重）；LCB/多样性为 Reddy 风格思想适配 |
| MSE-only / ListNet / residual / rank / backbone / anchor / target-adaptation | `source_data/tcell_*` 各目录 | 辅助 train/dev + §8.2 门禁 | 本项目 |
| UPTS（不确定性惩罚选择） | `source_data/tcell_upts_v1/`（本轮新增） | 辅助 dev 选 κ，双方向门禁 | 本项目（门禁失败） |
| Oracle upper bound（分解诊断） | 远端 `tcell_error_diagnosis_v1`（within-prefix oracle 1.835/2.955 headroom） | 诊断，非部署 baseline | 本项目 |

**缺口（如实披露）**：
- **one-hot linear**：无独立实现；4-mer MLP 是 k-mer 频率简单模型等价物，但 one-hot 线性未单列。
- **简单 ensemble（CNN+MLP 均值）**：cross-arch 分析存在，但未作为部署选择 baseline 评价。
- **oracle upper bound 作为 baseline 行**：仅有分解诊断，未进 baseline 表（可补描述行，不需新实验）。
- 官方 COM/生成式：见 §1，不可等价复现。

## 3. OOD 覆盖表（独立单位 ≠ 算法种子）

| OOD 维度 | 证据/artifact | 独立单位 | 状态 |
| --- | --- | --- | --- |
| 跨细胞 | T-cell vs HepG2 vs HEK293T；`tcell_cross_cell_anchor_v1`、`scenario_mrna_non_tcell_v1` | context/完整池（16 池/context） | 已覆盖 |
| 跨构建体 | defined-end vs random-end；mRNA UTR/构建体单位 | 构建体/UTR 簇 | 已覆盖 |
| 跨终点 | MRL vs stability vs indel vs TE vs production vs lnW vs brightness vs resistance | 各场景池/簇 | 已覆盖（终点异质，禁止合并原始效应） |
| 跨研究 | Sample→Castillo（external）、Sample→Leppek PERSIST、Schlusser | 研究/库 | 已覆盖（回顾性） |
| 跨突变阶数 | GFP 低阶 train→高阶永久 test（order9 Spearman 衰减） | 位点互斥池 | 已覆盖 |
| 跨药物/条件 | TEM-1 Amp→cefotaxime（15/15 跨零） | 位置簇 | 已覆盖 |
| 前瞻/真正独立研究 | 无 | — | **缺口**（P3 已记 limitation；Cao/Ogden 为候选未评价） |

所有区间均为 pool/UTR/位置簇 cluster bootstrap；算法种子（101–505、200 种子等）只是提案重复，**不作独立生物重复**。

## 4. 可直接进 Limitations 的披露文案

> 我们未与官方 COM 或生成式 MBO 方法（Design-Bench 协议、GABO/aSCR、T3PO-mRNA、mRNAutilus）在同一固定池实测 regret 面上做等价比较。COM 只有诚实标注的独立适配（CNN 替代 UTR-ResNet、实测 MRL 替代 oracle-relabel、固定池重排替代 de novo 生成）；Design-Bench 用 learned oracle 评分且其 UTR 任务即本文训练来源 Sample，存在评价面不同构与泄漏；生成式方法产出池外新序列，与固定池选择问题不同构；Reddy 启动子工作的分子与终点不同。因此本文主张限于固定池等预算选择面，不声称对 de novo 设计方法的优劣。

## 5. 来源链接

- Design-Bench: https://github.com/brandontrabucco/design-bench ；arXiv:2202.08450
- COM: https://arxiv.org/abs/2107.06882 （ICML 2021）
- GABO/aSCR: https://papers.nips.cc/paper_files/paper/2024/hash/4dd0a016d7d253d02473e4778414ab0b-Abstract-Conference.html ；https://github.com/michael-s-yao/gabo
- Reddy: https://doi.org/10.52202/079017-2954
- T3PO-mRNA: https://openreview.net/forum?id=KTCInSlPpL （403，待人工补核）
- mRNAutilus: https://arxiv.org/abs/2605.31296
- 本地：`deliverables/reports/docs/PREPRINT_REFERENCE_AUDIT_20261126.md`、`FULL_AUDIT_20261124.md`