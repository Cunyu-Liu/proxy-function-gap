# 统一预印本主张—证据台账（V5，2026-11-26）

> V5 纳入 `PAPER1_BUDGET_MATCHED_V1.md`、`PAPER1_SEQUENCE_ABLATION_V1.md` 和 `PERSIST_TIE_AUDIT_V1_1.md`。科学结论以 V5 为准；V4 及更早台账、稿件和验证输出保留作历史记录。所有新分析都是历史 outer 上的回顾性敏感性，不是新盲测或因果证明；衰减、符号反转、跨零区间和门禁失败必须原样保留。

## 1. 交付结构与证据分级

- 主文：`deliverables/paper/UNIFIED_PREPRINT_MAIN_V5_20261126.md`。主文：Paper 1 现象/机制；以等预算阳性开篇，紧接组成衰减，再进入跨架构、终点和 PERSIST 完整 MLP 契约。
- S2：`deliverables/paper/UNIFIED_PREPRINT_SUPPLEMENT_S2_V5_20261126.md`。S2：限制与证伪；选择器、Q 和 ensemble-std 冻结 calibration 均未建立决策收益。
- S3：`deliverables/paper/UNIFIED_PREPRINT_SUPPLEMENT_S3_V5_20261126.md`。S3：诊断；48 个门禁和固定队列曲线只限 development/calibration。
- 历史：V4 三份 Markdown/PDF、builder、manifest 以及 docs 下 V4 story/ledger/verifier/三轮输出不改写；V3.1 既有 archive 继续保留。

状态分为：**协议内回顾性结果**、**回顾性/探索性敏感性**、**冻结门禁失败或未评价**。“协议内”不意味着新盲测、前瞻湿实验或独立生物重复。

## 2. 核心主张台账

| ID | 主张与允许措辞 | 状态 | 核心证据 | 结论边界/禁止措辞 |
|---|---|---|---|---|
| C01 | 等候选预算 BoN(N)−Random(N)：T-cell r1 cap2/4/8/16 效应 −0.799949/−1.448756/−1.579340/−1.705760，全部通过 all-20 Holm，p均0.004000 | V5 首要阳性；历史 outer 回顾性敏感性 | `paper1_budget_matched_v1/results.json`；`pool_effects.tsv`；Figure 14 | 不得写成新盲测、前瞻重复或普遍收益；all-20 含 cap1 结构零；Random(N) 是同前缀同候选数对照 |
| C02 | T-cell r2 cap2–16 点估计均负，但只有 cap4 −1.014876 [−1.544490,−0.465832] 通过 Holm，p=0.006399 | 受限阳性 | 同上；cap2/8/16 Holm p=0.177582/0.222178/0.064994 | 不得概括为 r2 全 cap 校正存活；其余仅名义结果 |
| C03 | HepG2 cap2 等预算通过 Holm；cap4 名义但 Holm p=0.055994；N50 无名义支持效应 | 情境边界 | HepG2 cap2 −0.715625，Holm p=0.006399；N50 cap8/16 点估计为正 | 不隐藏 N50 不利方向或 HepG2 非稳健项；不与 T-cell 合并成统一方向 |
| C04 | 无 Random(N)−Random(1) 对比通过独立 all-20 Holm；非零估计为随机化实现/MC 变异 | 对照边界 | `results.json` 的第三个完整家族；HepG2 cap2 名义为正但校正不显著 | 不声称增加随机候选数有系统收益或危害；cap1 p=1 保留 |
| C05 | 旧 BoN(N)−Random(1) 压力结果不变：T-cell 两重复 cap2–16 的八个 all-20 Bonferroni 区间均负 | 历史回顾性对照 | `paper1_strict_h1_v3`；T-cell r1 cap16 −1.243 [−1.710,−0.769]，r2 cap16 −1.355 [−2.083,−0.653] | 不是等预算曲线；不与新对比混合；HepG2 cap8 校正后仅探索，N50 无持续改善 |
| C06 | GC/uAUG/polyU/support 调整后 T-cell 收益衰减，无调整效应通过 all-20 Holm | 回顾性组成敏感性；不利结果保留 | `paper1_sequence_ablation_v1/analysis/results/primary_paired.tsv`；r1 仅 cap4/8 名义区间负，Holm p=0.662/0.642；r2 全跨零且 cap4 +0.043 | 不得写调整后仍独立稳健；不得把观察性调整写成因果、中介或随机干预 |
| C07 | uAUG 排除使 T-cell 效应大幅缩小且无 Holm 存活；polyU 排除仅 r1 cap4 与 r2 cap8 存活 | 家族异质性 | `sequence_family_sensitivity.tsv`；polyU 截止 `longest_poly_u>=3` 由候选特征盲态冻结；两项 p=0.016 | 不宣称 uAUG/polyU 因果；保留池数减少、cap16 非校正显著、r1 cap8 边界 p=0.061 |
| C08 | 分数匹配未识别一致的优化器特异误差增量 | 协议内零结果，回顾性诊断 | 四情境差 +0.174/+0.326/−0.303/−0.097，区间全跨零 | 不证明高分外推为因果机制，也不证明优化无害 |
| C09 | CNN 与 4-mer MLP 绝对误差高度相关；更低 CNN MAE 不等于独立功能验证 | 协议内回顾性结果 | 相关 0.878–0.962；四个 CNN−MLP MAE 配对区间均排除零 | 只能称误差非独立/共同风险；不得因果断言共享训练数据主导 |
| C10 | 归一化输出高斯噪声未产生单调退化 | 回顾性敏感性 | N50 cap16：6.519/6.018/6.143/6.205/6.526 | 只否定该协议下必然单调放大；不推荐噪声修复 |
| C11 | Schlusser 一次性跨终点评测未观察到 MRL-BoN 改善内源 TE 的证据 | 一次性跨终点协议 | BoN−Random +0.626608；基因簇 [−0.828459,9.99825]；转录本簇 [−0.892444,9.96536]；rho 0.072073 | 不写 BoN 显著更差或 Random 显著更好；不是同终点 MRL、直接蛋白产量或跨研究元分析 |
| C12 | Leppek 终点排序不等价：half-life vs 6h 表达 171/276 反转、top6 重叠 1/6 | 探索性描述 | 24 CDS 原始字段复算；pooled MRL–half-life r=−0.346 | 不给显著性/因果/泛化；不拼接 CDS 与 pooled；Predicted expression 非实测 |
| C13 | PERSIST 完整变长 4-mer MLP 契约无可检测迁移 | 已完成小型独立研究同终点回顾性外测；不是盲测 | 59 构建体/49 UTR；exact/等长 Hamming-1=0；冻结变长 MLP CUDA；BoN−Random +0.104723 [−0.100964,0.212706]，p=0.245175；Spearman −0.008447；`PERSIST_MRL_EXTERNAL_V1.md` | 不支持 BoN 获益，不证明无效；不是 T-cell 直接反驳、普遍失效证明或前瞻验证；完整训练谱系/架构/构建体不可省略 |
| C14 | PERSIST 原 tie 规则为 prediction descending/Sequence ID/row order；SHA256 替代规则主效应/CI/p不变 | 源码确认的回顾性敏感性 | `PERSIST_TIE_AUDIT_V1_1.md`；9 个并列种子、5 个顺序敏感种子；原选择效应零变化；bootstrap 主效应/CI/p精确不变 | v1 仍主分析，v1.1 不按方向选版；局部 bootstrap 变化不得隐藏；不把 tie 稳健性写成外部有效性 |
| C15 | 所测 verifier-guided 与 COM-style/diversity-LCB 选择器未建立普遍 regret 优势 | 回顾性证伪 | FP-REJ−BoN +0.343 [0.225,0.467]；designed COM/diversity 区间跨零；random COM +1.811 [0.123,2.660] | 只限所测实现；COM-style 非官方；random 仅3池；FP-TILT 非等成本 |
| C16 | Q 概率校准改善，但不判别真实 pool regret，按 Q 拒选不降 accepted regret | 回顾性诊断；calibration 留出 | log loss 0.327→0.069；Q vs BoN rho −0.200 [−0.718,0.401]；AUC 0.297 [0.094,0.727] | 不把 log loss 改善写成风险判别或决策收益；dev32 内样本、calib16 留出 |
| C17 | ensemble-std 回顾性信号未通过冻结calibration门禁，outer 硬锁未评分 | **冻结门禁失败** | dev q=0.50、阈值 0.104026897；calib coverage 0.425，accepted 2.866883 vs full 2.861463，降幅 −0.005420 [−0.509271,0.628623]；`outer_scored=false` | 旧 outer 经验分位点不构成盲态证据；不得回调阈值、评分本版本 outer 或虚构 Paper2 阳性 |
| C18 | 全查询标签修复 headroom 存在且表示依赖 | 回顾性 development/calibration 诊断 | 48 门禁：summary4 0/16、kmer4 5/16、cnn64 1/16；kmer4 最大 3.686 | 不是普遍修复率、可部署上界或 outer 有效性；full-Q 仅诊断信息上限 |
| C19 | 固定 kmer4 五池 EI@16 描述性优于简单/复杂基线；复杂获取不支配 | 回顾性诊断 | EI 1.220084、random 3.497016、max-var 2.644595、regret-analytic 2.131689；32 标签 max-var 2.068558 | 不写标签高效修复已证实；固定队列非16池平均；保留逐池反例和 summary4 完全停滞 |
| C20 | Paper3 V4 384/384 收敛只证明近似目标数值稳定，不证明 acquisition 有效 | 数值诊断 | `posterior_dependence=diagonal_independent_fantasy_posterior_approximation`；三轮科学哈希一致 | 不省略对角独立近似；不把求积收敛写成 outer 泛化或生物效果 |
| C21 | 风险控制与标签修复必须独立过门禁，不能继承 Paper 1 阳性或 PERSIST 阴性 | 研究治理结论 | S2 冻结失败和 S3 未外测的直接证据状态；V5 主链与两份补充分别核验 | 禁止“Paper1 有效所以风险控制有效”“PERSIST 无效所以必须修复”“S2 失败仍放行序贯系统” |

## 3. 推理链审计

| 环节 | 实验现象 | 允许的下一问题 | 不允许的跳跃 |
|---|---|---|---|
| 等预算 | r1 全 cap 存活，r2 cap4 存活 | 排序收益是否依赖所选序列组成？ | 候选数增加与排序价值混同；普遍收益 |
| 组成调整 | 全部 T-cell 调整效应衰减且无 Holm 存活 | 哪些家族/契约限定收益？ | GC/uAUG/polyU/support 因果或中介已证明 |
| 家族敏感性 | uAUG 广泛衰减，polyU 选择性保留 | 如何在独立数据检验可移植性？ | 隐藏符号反转、低池数或未存活项 |
| 分数匹配/跨架构 | 匹配零结果；误差相关 0.878–0.962 | 终点和完整训练契约是否迁移？ | 高分外推或共享数据主导被因果证明 |
| 跨终点/PERSIST | TE 无改善证据；PERSIST 完整 MLP 无可检测迁移 | 新方法能否独立控制决策风险？ | T-cell 被直接反驳、普遍失效、BoN 获益或盲测完成 |
| 风险控制 | 选择器/Q 阴性；候选冻结门禁失败 | 新风险定义能否在新版本独立评估？ | 旧 outer 回调阈值；本版本 outer 评分；继承 Paper1 阳性 |
| 修复诊断 | 表示依赖 headroom，EI 部分预算较好 | 是否有独立理由冻结 outer acquisition？ | headroom=标签效率；S2 失败仍放行序贯系统 |

## 4. 源数据与复算入口

- V5 独立脚本：`deliverables/reports/docs/verify_unified_preprint_v5_20261126.py`，实际执行 `--round 1|2|3`。
- V5 数值/主张输出：`UNIFIED_PREPRINT_SOURCE_VERIFICATION_V5_20261126_ROUND{1,2,3}.json` 与对应 `.log`。
- V5 PDF/发布：`build_unified_preprint_v5.py`、`build_unified_preprint_v5.log`、`UNIFIED_PREPRINT_V5_RELEASE_MANIFEST.json`。
- 等预算：`PAPER1_BUDGET_MATCHED_V1.md`、`paper1_budget_matched_v1/results.json`、`pool_effects.tsv`、`frozen_config.json` 与既有三轮核验。
- 序列：`PAPER1_SEQUENCE_ABLATION_V1.md`、`paper1_sequence_ablation_v1/analysis/results/{summary.json,primary_paired.tsv,sequence_family_sensitivity.tsv}`、冻结协议及既有三轮核验。
- 风险控制：`PAPER2_RISK_FROZEN_V1.md`、`paper2_risk_frozen_v1/frozen_protocol.json` 与 `outer_refused.log`。
- PERSIST：原 `persist_mrl_external_v1` 冻结结果和根目录 `PERSIST_TIE_AUDIT_V1_1.md`、`persist_mrl_external_v1/tie_audit_v1_1/results/{summary_v1_1.json,comparison_v1_vs_v1_1.json}`。
- 所有 V5 新计算为本地只读 CPU 复算；不访问远端、不训练、不推理、不揭盲、不评分 outer，不修改 figures/PPT/source_data。

## 5. 当前结论

**最强主结果是“特定情境中等预算 proxy 排序有价值”，不是“proxy 普遍可靠”。** T-cell r1 cap2–16 与 r2 cap4 的 Holm 存活必须优先报告；四特征调整后的衰减、uAUG/polyU 家族差异、N50/HepG2 边界、跨架构非独立、终点不可互换和 PERSIST 完整 MLP 契约无可检测迁移必须同等可见。PERSIST tie 审计只加强数值稳健性，不改变回顾性、非盲测和跨零不确定性。S2 已止于失败的 calibration 门禁，S3 永久 outer 未动；风险控制与标签修复各自需要新独立门禁。
