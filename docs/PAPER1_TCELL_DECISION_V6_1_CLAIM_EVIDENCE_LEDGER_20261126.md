# Paper 1 T-cell decision-focused V6.1：独立主张—证据台账

日期：2026-11-26
主文：`deliverables/paper/PAPER1_TCELL_DECISION_V6_1_20261126.md`
范围：只读核验既有脱敏聚合结果；V6.1 不训练、不推理、不重开任何 target 评价，不修改 `source_data/`、`figures/` 或 PPT。V6.1 仅修正 V6 独立终审发现的文字过度概括与歧义，并补充 Random(N) 单次随机化实现的条件性限制；全部实验数值不变。

## 1. 总体证据分级

- **首要阳性结果**：strict-H1 历史 outer 上等候选预算 BoN(N)−Random(N) 的真实 MRL regret 对比。它是回顾性敏感性分析，不是新盲测、前瞻复制或因果证明。
- **机制定位**：pointwise/global-rank/top-tail 分解是回顾性诊断；oracle 平移与 oracle 重排只解释误差和可修复空间，不是已实现方法收益。
- **目标适配**：v1 与 v2 分别使用从历史原始终端重建的 component-disjoint grouped holdout，并在各自协议内一次评价；仍不是独立研究、外部验证或前瞻盲测。两个 test 集的绝对 regret 不横向比较。
- **跨细胞锚点**：同序列 HepG2→T-cell 是 paired cross-cell retrospective evidence；HepG2 MRL 不是 T-cell 标签，但同序列设计也不是序列独立验证。
- **辅助开发门禁**：residual/pairwise/multi-context 仅评价辅助 destination test，target count=0；不得写成 T-cell 成功或 T-cell 失败。
- “effect”始终指配对数值差，不指因果效应。Holm/bootstrap 是各源协议内的回顾性不确定性/多重性量，不把历史数据升级为前瞻确认。

## 2. 逐条主张—证据对应

| ID | 主文主张与精确数值 | 独立证据入口 | 核验方式 | 必须保留的边界 |
|---|---|---|---|---|
| C01 | 标题/摘要首先报告冻结 Sample CNN 在两个 T-cell repeat 的等预算真实 MRL regret 收益，而非宽泛 proxy contract 或全负结果 | `paper1_budget_matched_v1/pool_effects.tsv`；Figure 14 | 从320行池表按 context×cap 重均值；核对标题与摘要语义 | 只限历史 strict-H1 T-cell 池；不称普遍、前瞻或因果收益 |
| C02 | r1 cap2/4/8/16 效应为 −0.799949/−1.448756/−1.579340/−1.705760，all-20 Holm p均0.004000 | `paper1_budget_matched_v1/results.json`；`PAPER1_BUDGET_MATCHED_V1.md` | 池表重算效应；JSON核对CI/p；Figure 14核对图注 | 完整20项含cap1结构零；不是新盲测 |
| C03 | r2 四个点估计和名义CI均有利，但仅cap4 −1.014876 [−1.544490,−0.465832] 通过all-20 Holm，p=0.006399 | 同上 | 重算cap2/4/8/16均值；核对cap2/8/16 Holm p=0.177582/0.222178/0.064994 | 不得写r2全cap校正存活；T-cell-only另一校正族不得替代all-20主口径 |
| C04 | 等预算BoN与Random(N)共享nested prefix、N与proposal机会；无Random(N)−Random(1)项通过独立all-20 Holm | `frozen_config.json`；`results.json`；`pool_effects.tsv` | 核对cap1恒等和第三对比族校正结果 | 随机化实现/MC变异不得解释为增加N的系统功能收益；每条pool–seed–cap记录只有一次冻结位置抽取，pool bootstrap以该实现为条件，不积分选择器MC不确定性 |
| C05 | CNN pointwise MAE r1/r2=1.567454/1.543969，bias=+1.064393/+0.407480，但global Spearman=0.591052/0.529222，池内均值=0.583869/0.515055 | `tcell_error_diagnosis_v1/analysis/pointwise_metrics.tsv`；`summary.json` | TSV与JSON逐字段交叉核对；MSE恒等式复算 | 高MAE与有用全局排序并存；不得称整体排序完全失效 |
| C06 | oracle全局中位数平移后MAE=1.108358/1.475718，但常数平移不改变池内排序、BoN或regret | `pointwise_metrics.tsv`；`TCELL_ERROR_DIAGNOSIS_V1.md` | 核对平移后MAE与排序不变数学性质 | 目标标签oracle诊断，不是合法部署校准或regret修复 |
| C07 | full-pool top16 Spearman=0.206058/0.143984，pair inversion=43.1%/45.0%，第一名在15/16池被尾部候选超过 | `analysis/top_tail_summary.tsv` | 读取cap16 CNN行并核对winner_flipped=0.9375×16 | full-pool使用64个既有分数，只是诊断；不是BoN16等预算方法 |
| C08 | full-pool top16 regret r1=3.315826=2.479998 within-tail+0.835829 opportunity；r2=4.070571=2.117221+1.953350；within-tail占74.8%/52.0% | `top_tail_summary.tsv` | 独立执行加法恒等和比例计算 | 是会计分解，不是因果中介；oracle headroom不是已实现收益 |
| C09 | 实际BoN16 prefix regret r1=3.555181=1.719882+1.835298；r2=5.043377=2.088227+2.955150（展示四舍五入2.088+2.955） | `analysis/selection_summary.tsv` | 读取CNN cap16 BoN行，复算regret=opportunity+within-prefix | 同前缀对比机会损失精确抵消；full-pool与prefix分解不混用 |
| C10 | 异常候选占6.05%/9.96%，贡献MSE 41.5%/66.5%，BoN16异常事件5/80与11/80 | `analysis/summary.json` 的`outliers` | 核对CNN两条记录、比例和事件数 | 只有r2显示选择富集信号；不按实测异常身份删候选或推广统一规则 |
| C11 | v1 50k main使MAE r1 1.352159→0.914506（Δ−0.437653，大幅下降），r2 1.062273→1.018716（Δ−0.043556，小幅下降），但cap16 regret分别+0.038929/+0.012551 | `tcell_target_adaptation_v1/target/results.json` | 独立计算MAE差、cap16 regret差并与`error_regret_synchronization`核对 | target train/dev使用标签；retrospective grouped holdout，不是零样本/独立研究；不得把两个repeat笼统写成均为markedly降低 |
| C12 | v1合并main−original=+0.025740 [−0.072019,+0.140994]，最低所需降低0.131566，10项Holm p均1，主门禁FAIL | `target/results.json` 的`primary`与`by_repeat_and_cap` | 两repeat效应均值复算；基线2.631329×5%与0.1取大；检查success=false | 点估计方向、CI和幅度均失败；不得包装50k适配成功 |
| C13 | v1所选main与MSE-only全cap选择/regret相同；tail pair变化r1 −0.005211、r2 +0.015625；20k和from-scratch不救援 | `target/results.json`；`TCELL_TARGET_ADAPTATION_V1.md` | 全cap方法regret逐项比对；核对nested/ablation字段 | 不事后改选20k、单repeat或from-scratch；只判伪当前50k方案 |
| C14 | v2 test为新的1,024序列/16池component-disjoint grouped holdout，排除v1三角角色和Sample exact/H1；预测冻结后一次评价 | `tcell_listnet_adaptation_v2/target_v2/results.json`；`verification/round1_independent_split.json`；一次评价哨兵 | 检查evaluation_count=1、配置/隔离/冻结哈希与验证结果 | 不是独立研究、外部验证或前瞻盲测；不与v1绝对regret横比 |
| C15 | ListNet mean cap16 regret=1.729031；相对原CNN −0.037415 [−0.183204,+0.102494]，相对v1 main −0.007585 [−0.028698,+0.012474]；两项降幅<0.1、Holm-10 p=1、联合IUT FAIL | `target_v2/results.json` 的`primary`及mean_replicate cap16 | 独立用方法regret求差；核对CI、阈值、p和success=false | 只能写小幅且不足的点估计，不得写显著或实质性能提升；主CI非Holm同时区间 |
| C16 | ListNet相对v1 main cap16在r1 +0.020946、r2 −0.036115；mean MAE增加0.002104但regret点估计降低0.007585；pair accuracy仅小幅+0.018750/+0.011458 | `target_v2/results.json` 的repeat、pointwise、top_tail字段 | 重算repeat/mean MAE和regret差，核对pair字段 | r2次要有利CI不抵消r1反向和mean主门禁失败；局部pair改善不等于决策修复 |
| C17 | HepG2-anchor cap16 r1/r2 effect=+0.463090/+0.394705；合并+0.428897 [+0.245460,+0.616474]，p=0.00019998，方向显著变差 | `tcell_cross_cell_anchor_v1/target/results.json` | 两repeat求均值；核对合并CI/p、success=false | 正号表示anchor−CNN有害；同序列跨细胞回顾性证据，不是序列独立验证 |
| C18 | HepG2 cap16 Holm-10仅r2通过p=0.017998，r1 p=0.053995；50/50融合合并+0.154527 [−0.004346,+0.326043] | 同上`by_repeat_and_cap`与`secondary` | 核对两个校正p和融合CI | 不得把“两个repeat各自均校正显著”写入主文；融合不改善 |
| C19 | HepG2全局Spearman=0.730001/0.750157，不低于CNN 0.722091/0.724564；top16 Jaccard=0.470432、16/16池top1不同；有害更替39>21和39>22 | 同上`secondary`及cap16`turnover` | 核对全局相关、tail和turnover字段 | 全局相关不等于top-tail选择；不推断所有跨细胞模型无效 |
| C20 | HepG2与CNN proposal headroom相同；oracle opportunity loss r1 1.075351→1.538440、r2 1.302036→1.696741 | cap16`oracle`字段 | 检查headroom逐项相等并复算与总regret关系 | 反向来自同可达候选中的选择，不是不公平proposal机会 |
| C21 | residual v1两方向corrected−frozen=+0.000343/+0.004638，辅助门禁失败，target=0 | `tcell_residual_bstrict_v1/artifacts/auxiliary/*/gate.json`；`auxiliary_gate.json` | 源报告“frozen−corrected改善量”转换符号并与gate核对；检查无target | 只属开发证据，不是T-cell失败；v1门禁不虚构CI |
| C22 | pairwise v2两方向=−0.006543 [−0.014204,+0.001233] 与+0.021187 [−0.007718,+0.050096]，均未过门禁，target=0 | `tcell_rank_bstrict_v2/artifacts/auxiliary/*/gate.json`；`auxiliary_gate.json` | 核对所选模型、test effect、CI和fail-closed | 一项点估计有利但CI跨零；无任何T-cell corrected结果 |
| C23 | multi-context两方向=+0.044629 [+0.021840,+0.068346] 与+0.028050 [−0.049121,+0.104614]，门禁失败，target=0 | `tcell_multicontext_backbone_v1/selection/main.json`；`auxiliary_gate.json` | 核对两方向destination test、CI及target目录不存在 | 不写成T-cell失败；不推广为所有多情境或排序损失无效 |
| C24 | 结论应是T-cell模型开发以真实功能regret为首要终点，辅以selected MRL、同前缀Random(N)、全cap/repeat和预设实质门槛；MAE/RMSE/Spearman/pair accuracy是诊断 | C01–C23的综合逻辑，不新增实验数值 | 第二轮叙事审计检查结论是否由直接决策证据支持 | 不得写MAE无用、所有排序目标无效或CNN普遍最优；未来改进需新隔离评价 |
| C25 | Figure 14/16/17均被主文和完整图注引用；方法、限制、参考文献、数据代码可用性齐全 | V6.1 Markdown与独立图文件 | 第三轮PDF文本/字体/溢出/渲染/图像对象及干净重建检查 | 图只读引用，不在V6.1修改；若另一代理图文件未到位，构建必须失败而非伪造 |

## 3. 禁止的叙事跳跃

1. 不得由 r2 四个名义负区间跳到“r2全cap通过all-20 Holm”。
2. 不得由全局Spearman为正跳到“top-tail可靠”，也不得由MAE高跳到“CNN无选择价值”。
3. 不得把oracle平移、oracle重排、proposal headroom或pair accuracy当作已实现regret改善。
4. 不得由50k MAE降低跳到cap16改善；不得改选20k、from-scratch或单个repeat。
5. 不得把ListNet −0.037415/−0.007585包装成显著、实质或通过预设门槛的提升。
6. 不得跨v1/v2/strict-H1/anchor不同池比较绝对regret来证明方法优劣。
7. 不得将HepG2同序列锚点称作T-cell标签、序列独立外测或因果细胞机制；不得写两个repeat都Holm显著。
8. 不得将residual/pairwise/multi-context写成T-cell失败，因为它们的target evaluation count=0。
9. 不得使用“blind/prospective/causal/independent replication”描述任何V6.1实验；协议内一次评价仍须称retrospective/grouped holdout。
10. 不得把结论写成全负结果或宽泛proxy contract。首要故事是已验证的等预算T-cell收益及regret指标对后续开发的价值。

## 4. 三轮验证的真实范围与输出

- **Round 1—主张数值**：`verify_paper1_tcell_decision_v6_1.py --round 1`。只读重载聚合TSV/JSON，重算池均值、差值、比例、恒等式、门槛与门禁状态；输出 `PAPER1_TCELL_DECISION_V6_1_ROUND1_NUMERIC_20261126.{json,log}`。
- **Round 2—叙事边界**：`verify_paper1_tcell_decision_v6_1.py --round 2`。检查标题/摘要阳性重心、每条证据状态、校正族、test集隔离、target=0语义、图引用、章节与参考文献；输出 `PAPER1_TCELL_DECISION_V6_1_ROUND2_NARRATIVE_20261126.{json,log}`。
- **Round 3—PDF clean rebuild**：从交付Markdown在空临时目录重新构建，检查必需文本/图像、逐页文本与光栅、字体嵌入与Unicode、页面对象越界、空白页、PDF头尾，并要求与交付PDF字节一致；输出 `PAPER1_TCELL_DECISION_V6_1_ROUND3_PDF_20261126.{json,log}` 和 `deliverables/paper/PAPER1_TCELL_DECISION_V6_1_RELEASE_MANIFEST_20261126.json`。

三轮范围互不替代：叙事PASS不证明数值，数值PASS不证明PDF，PDF PASS不把回顾性证据升级为盲测或因果证据。
