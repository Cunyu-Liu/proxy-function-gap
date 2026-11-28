# P0 状态报告（2026-11-28）

执行依据：`EXECUTION_PROMPT_PAPER1_OPTIMIZATION_20261128.md`。本报告为 P0 阶段验收产物。

## 1. 复现与一致性检查结果

| 检查 | 结果 |
| --- | --- |
| `verify_paper1_tail_reliability_v8.py --round 1/2/3` | 三轮 PASS |
| `verify_cross_scenario_synthesis_20261128.py --round 1/2/3` | 三轮 PASS（各 7 项检查；输出 `CROSS_SCENARIO_SYNTHESIS_VERIFICATION_20261128_ROUND{1,2,3}.json`） |
| `figures/v5/verify_cross_scenario_figures.py` | PASS，Figure18/19 确定性重建字节一致 |
| 远端分支/commit | `paper23-gates-v1` @ `7259351`（与执行提示词一致），`git status` 干净 |
| 远端 artifacts | `paper1_budget_matched_v1`、`scenario_gb1_v1`、`scenario_gfp_v1`、`scenario_aav_v1`、`scenario_tem1_v1`、`scenario_mrna_non_tcell_v1` 均存在于 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap` |
| GPU | RTX 5090 32GB；其他项目（lucaone、rna-sc-p2）占 ~18.7GB，剩余 ~13.8GB 可用 |
| 团队盘剩余 | ~154GB（99% 已用，新实验须控制 artifact 体积） |

本地工作区不是 git 仓库；代码提交均在远端 `/home/proxy-function-gap`。

## 2. 当前最强阳性证据（可发表主张）

1. **T-cell MRL r1 等预算 BoN−RandomN**：cap2/4/8/16 全部通过 all-20 Holm；cap16 Δ=−1.705760，CI [−2.163445, −1.211914]。
2. **GB1 IgG-Fc lnW**（不是 Kd）：加性 ridge cap16 Δ=−0.160894，CI [−0.282853, −0.050515]；31/55 池改善、24/55 反向，异质性保留。
3. **GFP brightness**：低阶训练/高阶永久 test，加性 cap16 Δ≈−0.2800，CI [−0.3095, −0.2521]；加性优于 CNN。
4. **AAV9 production**（不是 tropism/临床疗效）：cap16 Δ=−4.206431，CI [−4.564111, −3.931098]；但 true top1% recall 仅 18.75%。
5. **机制脱钩证据**：50k target adaptation 降 MAE 但 cap16 regret 不改善；HepG2 anchor 全局相关不低但 T-cell cap16 选择显著变差。
6. **边界/反向证据**：mRNA stability cap16 显著反向 +0.2838 CI [+0.0824, +0.4435]；TEM-1 Amp→cefotaxime 15/15 区间跨 0；indel 0/4 校正阳性。

## 3. 当前最弱环节

1. tail reliability 是跨场景**诊断关联**，无干预证明（tail_audit.json 明示 `retrospective: true`，各场景 tail metric 不统一）。
2. 无跨场景可部署的改进方法；最强贡献是评估原则+机制诊断+边界。
3. 全部结果为历史公开数据回顾性分析；grouped holdout 不是前瞻盲测。
4. 官方 COM/生成式 baseline 未在同一固定池面公平复现。
5. 投稿元数据（作者/单位/COI/许可/目标期刊）未确认。

## 4. 不能升级的结果（冻结，禁止事后回调）

- T-cell r2：仅 cap4 通过 all-20 Holm；cap2/8/16 只能写名义阳性，不能升级。
- ListNet 新 test：点估计小幅改善，未达统计+幅度门槛，不能写成成功。
- 50k target adaptation：MAE 改善不能转写为决策改善。
- HepG2 cap16：名义阳性，Holm 不支持。
- HEK293T 高 cap：无持续收益（null）。
- TEM-1 Amp→cefotaxime：null，不能声称跨药物有效。
- mRNA stability：harmful，不能包装。
- AAV：不能声称恢复极端最优（top1% recall 18.75%）、不能写 tropism/临床疗效。
- 所有 grouped holdout：不能写"前瞻盲测/独立验证"。

## 5. 不能重复跑并重claim的已完成实验（test 已查看，一次冻结）

- `paper1_budget_matched_v1`：T-cell r1/r2 等预算 BoN/RandomN 全部 cap。
- `tcell_target_adaptation_v1`（50k adaptation）、`tcell_listnet_adaptation_v2`（ListNet 新 test）。
- `tcell_cross_cell_anchor_v1`（HepG2 anchor）。
- `scenario_gb1_v1`、`scenario_gfp_v1`、`scenario_aav_v1`、`scenario_tem1_v1`、`scenario_mrna_non_tcell_v1` 的冻结 test 评价。
- `tcell_residual_bstrict_v1`、`tcell_rank_bstrict_v2`、`tcell_multicontext_backbone_v1` 等辅助门禁实验。

以上实验的 artifact 只可复核/重算统计，不可换阈值/cap/子集/模型后重claim同一冻结检验。

## 6. 下一步最值得做的一个实验（P1 假设）

**假设 H-P1**：在固定候选池、固定 proxy 分数的前提下，仅改变高分尾部的重排/选择规则（tail-aware re-ranking，用辅助/dev 数据校准，不接触目标 test），可以降低真实功能 regret；且该改善幅度与尾部排序可靠性（within-tail ranking）的改善一致，而与全局 MAE/Spearman 改善不一致。

- 设计方向：执行提示词 P1 选项 B（固定池、只改尾部重排）为主，选项 C（辅助门禁后再进目标 test）为门禁框架；选项 A（opportunity vs within-tail 分解）作为机制归因补充。
- 主终点：真实功能 regret（endpoint-specific）；对照：RandomN、原 proxy BoN、加性 ridge。
- 成功门槛：点估计方向 + cluster CI 不跨 0 + 预注册最小实质幅度；失败即停并保留失败。
- 关键前提：必须先确认**未查看的评价面**（新 repeat/新隔离切片或严格辅助门禁路径），由 P1 设计审计确定；不得在已查看的 r1/r2 test 上重claim。

## 7. 并行推进安排

- P1 机制实验设计审计（远端 T-cell artifacts 勘察 + 冻结协议草案）。
- P3 独立数据检索（独立 5′UTR MRL、AAV tropism/infectivity、蛋白稳定性/表达）。
- P4 baseline/OOD 可比性审计（COM、Design-Bench、Reddy、GABO/aSCR、T3PO、mRNAutilus）。
- 远端深度完整性审计（本地 source_data 与远端 artifact 哈希比对 + 仓库测试套件）。

状态：`DRAFT_NOT_SUBMITTED`。