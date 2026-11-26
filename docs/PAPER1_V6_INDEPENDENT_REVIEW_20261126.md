# Paper 1 V6 独立终审

日期：2026-11-26
审查对象：`deliverables/paper/PAPER1_TCELL_DECISION_V6_20261126.md` 及其 claim–evidence ledger
审查方式：严格审稿人口径的只读科学审查；独立重读本地聚合证据与 px2 远端冻结记录，不重开实验
修正版本：`deliverables/paper/PAPER1_TCELL_DECISION_V6_1_20261126.md`

## 1. 审稿结论

**V6 需要文档级修正后再作为交付版本；未发现核心实验数值错误、不同 test 的不当合并、校正族替换、辅助门禁越权或 target=0 违例。** 必改问题是摘要和讨论把 50k 在两个 repeat 的 MAE 改善笼统写成 “markedly”，掩盖了 r1 大幅下降与 r2 仅小幅下降的差别。另有两项 minor：Random(N) 单次随机化实现的条件性未明说，以及 ListNet “point improvements” 存在 pointwise/point-estimate 歧义。

上述问题均已在新建 V6.1 中修正，V6 原件保留。V6.1 的三轮数值、文本与 PDF 核验全部通过。**本结论是内部独立科学一致性终审通过，不等于外部同行评审接收、前瞻盲测、独立生物复制或因果验证。**

## 2. Major issues

### M1. 50k 的 MAE 改善幅度被摘要和讨论过度概括

- V6 位置：`deliverables/paper/PAPER1_TCELL_DECISION_V6_20261126.md:13` 与 `:147`。
- 原文分别使用 “markedly reduced MAE” 和 “produced markedly lower MAE”，语境覆盖两个 repeat。
- 冻结结果为：r1 MAE 1.352159→0.914506，下降 0.437653（约 32.4%）；r2 MAE 1.062273→1.018716，下降 0.043556（约 4.1%）。r1 可称大幅下降，r2 只能称小幅下降。
- 该问题不改变 MAE–regret 脱钩结论：两个 repeat 的 cap16 regret 分别增加 0.038929 和 0.012551，合并效应 +0.025740 [−0.072019,+0.140994]，主门禁失败。
- **处理：已在 V6.1 摘要和讨论改为“substantially in r1 and modestly in r2”，并在 ledger C11 固定该边界。**

未发现其他需要改数值、改实验结论或重开实验的 major issue。

## 3. Minor issues

### m1. Random(N) 的实现正确，但单次抽取的条件性应披露

- 实现逐条使用 `SeedSequence(20261125)` 与 `(context_index,pool_index,seed_index,cap_index)` spawn key，由 PCG64 在相同 nested prefix 中抽取一个位置；cap1 与 BoN/Random(1) 结构恒等。实现见 `deliverables/reports/source_data/paper1_budget_matched_v1/source_snapshot/run_analysis.py:56`。
- 这不是把 Random(N) 错写成 Random(1)，也不是候选预算不等；但每条 pool–seed–cap 记录只有一次 selector 位置抽取。pool bootstrap 以这些已实现的随机抽取为条件，不积分 selector-randomization Monte Carlo 不确定性。
- 因此等预算结论应限定于当前冻结历史池与随机化实现，不应外推为所有随机实现下的确认性平均收益。V6 已有回顾性限制，但未明确这一层。
- **处理：V6.1 Limitations 与 ledger C04 已补充该条件性；未改任何随机流或结果。**

### m2. ListNet “point improvements” 有歧义

- V6 位置：`deliverables/paper/PAPER1_TCELL_DECISION_V6_20261126.md:13`。
- “point improvements” 可能被读成 pointwise 改善，但相对 v1 main 的 mean-replicate MAE 实际增加 0.002104；有利的是 cap16 regret 点估计 −0.007585，而非所有误差指标。
- 相对原 CNN，ListNet cap16 regret 点估计为 −0.037415 [−0.183204,+0.102494]；相对 v1 main 为 −0.007585 [−0.028698,+0.012474]。两个未校正主 CI 均跨零，改善均低于 0.1 MRL，Holm-10 p 均为 1，联合 co-primary IUT 失败。
- **处理：V6.1 改为 “small favorable point estimates that failed both co-primary gates”，消除 pointwise 暗示。**

## 4. Optional issues / 后续建议

### o1. 标题可进一步显式标注回顾性证据，但当前不构成事实错误

标题直接陈述等预算 CNN ranking 降低 regret，摘要随即给出 repeat 和 Holm 边界，draft status 也明确回顾性属性，因此可保留。若目标期刊要求标题即含研究设计，可考虑加入 retrospective 字样；本次未做非必要标题改写。

### o2. 16 个池与 repeat 依赖仍是主要外推限制

每个主评价仅 16 池，r1/r2 共享序列库，pool/repeat-linked bootstrap 不产生新的生物复制。V6 已充分披露，无需文本修复。真正增强外部效度需要新的隔离评价或前瞻实验，不能靠追加 bootstrap 或重新随机化解决。

### o3. 投稿前应补齐最新版代码与证据包的可访问归档

V6 已如实说明公开 GitHub 分支不包含完整 V6 证据包、原始标签、权重和最新版 T-cell 分析代码。此披露准确，但可复现性仍有限。建议投稿前形成冻结归档和 DOI；本次不新增发布或 DOI 声明。

## 5. 核心主张逐项核对

| 审查项 | 独立结论 | 必须保留的边界 |
|---|---|---|
| 等预算 Random(N) 与校正族 | **通过。** r1 cap2/4/8/16 效应 −0.799949/−1.448756/−1.579340/−1.705760，all-20 Holm 均约 0.004000；r2 四点估计及名义 CI 均有利，但仅 cap4 −1.014876 通过，Holm p=0.006399。三个对比分别使用完整 4 contexts×5 caps 家族并保留 cap1 结构零。 | 不得用 T-cell-only 10 项族替换 all-20 主口径，也不得声称 r2 全 cap 校正显著。 |
| Random(N)−Random(1) | **通过。** 独立 all-20 族无存活项，符合同一均匀提案分布下的随机化实现差异。 | 不解释为增加 N 会系统改变随机候选质量；注意 m1 的 Monte Carlo 条件性。 |
| pointwise/global/tail | **通过。** MAE 1.567454/1.543969 与 global Spearman 0.591052/0.529222 并存；full-pool top16 Spearman 0.206058/0.143984，15/16 池预测 winner 被尾部其他候选超过。 | 高 MAE 不等于无选择价值，正全局相关也不等于尾部可靠。 |
| tail 分解 | **通过。** r1 3.315826=2.479998+0.835829；r2 4.070571=2.117221+1.953350。实际 BoN16 prefix 分解 r1 3.555181=1.719882+1.835298，r2 5.043377=2.088227+2.955150。 | full-pool 使用全部 64 个既有分数，只是诊断；prefix opportunity 在同前缀比较中抵消。均为会计恒等式，不是因果中介或已实现 oracle 收益。 |
| 50k MAE/regret 脱钩 | **数值结论通过，V6 措辞需 M1 修正。** MAE 两 repeat 均下降，cap16 regret 均不降；主模型与 MSE-only 全 cap 实现选择相同，20k/from-scratch 不救援主结论。 | 只判伪当前 50k joint 方案，不推广为所有目标监督无效；不得事后选择 20k 或单 repeat。 |
| ListNet v2 | **通过。** test-v2 为新的 1,024 序列、16 池 component-disjoint test，排除 v1 三角角色及 Sample exact/H1；主终点为 mean-replicate，repeat-specific 为次要。 | 不与 v1 或 strict-H1 横比绝对 regret；主 IUT 的未校正 CI 与次级 Holm-10 不混用，r2 有利次要结果不能推翻主失败。 |
| HepG2 anchor | **通过。** r1/r2 cap16 anchor−CNN 为 +0.463090/+0.394705；合并 +0.428897 [+0.245460,+0.616474]，p=0.00019998。Holm-10 仅 r2 cap16 通过（0.017998），r1 为 0.053995。50/50 fusion 合并 +0.154527，CI 跨零。 | 合并结果显著有害不等于两个 repeat 各自均校正显著；同序列跨细胞证据不是序列独立外测或因果机制。 |
| HepG2 headroom | **通过。** 两方法 proposal headroom 完全相同；r1 oracle opportunity loss 1.075351→1.538440，r2 1.302036→1.696741，满足 regret=opportunity+headroom。 | 变差来自同可达候选中的选择，不是不公平 proposal 机会。 |
| 辅助门禁与 target=0 | **通过。** residual、pairwise、multi-context 均为 destination auxiliary test 失败并 fail-closed；px2 远端三个 `target/` 目录均不存在。 | 只能称辅助开发门禁失败，不能写成 T-cell 成功或 T-cell 失败；residual 不虚构 CI。 |
| test 间隔离与叙事合并 | **通过。** strict-H1、v1、test-v2、HepG2 anchor 的数据角色、估计量和校正层级分别陈述，正文明确禁止跨池绝对 regret 比较。 | grouped holdout 仍是历史数据内回顾性证据，不升级为独立研究或前瞻盲测。 |

## 6. V6.1 文档修正范围

- `deliverables/paper/PAPER1_TCELL_DECISION_V6_1_20261126.md`：修正 M1/m2，补充 m1 限制，版本号更新为 6.1。
- `deliverables/reports/docs/PAPER1_TCELL_DECISION_V6_1_CLAIM_EVIDENCE_LEDGER_20261126.md`：同步 C04/C11 和验证产物名称，全部实验数值不变。
- `deliverables/paper/build_paper1_tcell_decision_v6_1.py`：仅更新版本化输入输出和页眉。
- `deliverables/reports/docs/verify_paper1_tcell_decision_v6_1.py`：新增修正文本的必要 token、旧措辞禁用项和随机化条件性检查。
- 未修改 V6 原件、`source_data/`、figures、PPT、模型、权重或远端实验结果；未训练、未推理、未重开 target evaluation。

## 7. 三轮真实独立核验

### Round 1：数值与证据一致性

执行：`python3 deliverables/reports/docs/verify_paper1_tcell_decision_v6_1.py --round 1`。
结果：**PASS，302/302**。覆盖池均值、效应、CI、Holm、tail/prefix 恒等式、50k/ListNet/HepG2 门槛、辅助门禁、输入哈希及 ledger 映射。输出为 `deliverables/reports/docs/PAPER1_TCELL_DECISION_V6_1_ROUND1_NUMERIC_20261126.{json,log}`。

另在 px2 通过完整 SSH 只读调用独立实现，未导入生产分析模块、未写远端结果：**3,200 个原始选择、1,600 个 Random(N) 选择、320 个 cap1 恒等、320 个池效应和三个 all-20 家族全部 PASS**。五份关键远端 results/pool 文件与本地 SHA-256 完全一致。

### Round 2：文本、科学边界与版本修正

执行：`python3 deliverables/reports/docs/verify_paper1_tcell_decision_v6_1.py --round 2`。
结果：**PASS，94/94**。检查回顾性/非盲测/非因果措辞、test 分离、校正族、门禁、target=0、参考文献和图引用，并确认新 50k/ListNet/Random(N) 文本存在、旧问题措辞不存在。输出为 `deliverables/reports/docs/PAPER1_TCELL_DECISION_V6_1_ROUND2_NARRATIVE_20261126.{json,log}`。

### Round 3：PDF 构建与交付核验

先执行 V6.1 builder，再执行 `--round 3`。结果：**PASS，45/45**。PDF 共 13 页，必需文本与三幅图存在，字体全部嵌入且含 Unicode 映射，无空白页、替代字符、越界文字或越界对象，无严重 LaTeX 警告；在空临时目录重建的 PDF 与交付 PDF 字节一致。输出为 `deliverables/reports/docs/PAPER1_TCELL_DECISION_V6_1_ROUND3_PDF_20261126.{json,log}`。

交付 PDF：`deliverables/paper/PAPER1_TCELL_DECISION_V6_1_20261126.pdf`
SHA-256：`c7ca7e7f4fe56df6f1bd750ca10b673fdef0255f134648bbb71113b771476c2a`
发布清单：`deliverables/paper/PAPER1_TCELL_DECISION_V6_1_RELEASE_MANIFEST_20261126.json`。

## 8. 最终建议

以 **V6.1 取代 V6 作为后续内部投稿准备底稿**。当前最稳健的科学故事是：冻结 CNN 在历史 T-cell 池中具有等预算真实 MRL 选择价值；高 MAE 与有用全局排序可以并存，尾部翻转解释剩余决策损失；特定 50k、ListNet 和直接 HepG2 anchor 未提供满足预设门槛的额外尾部收益。所有后续方法宣传仍须以新隔离评价中的 measured functional regret 为准，不能用 MAE、pair accuracy、oracle headroom 或辅助 dev 收益替代。
