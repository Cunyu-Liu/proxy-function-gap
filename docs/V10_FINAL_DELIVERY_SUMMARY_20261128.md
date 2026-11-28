# Paper 1 V10 最终交付总结（2026-11-28）

执行依据：`EXECUTION_PROMPT_PAPER1_OPTIMIZATION_20261128.md` + 用户本轮五点新要求（图升级、叙事链、数据/baseline/OOD 检查、现象级证明、发表导向实验）。状态：**DRAFT_NOT_SUBMITTED**。
GitHub：`Cunyu-Liu/proxy-function-gap`，分支 `paper23-gates-v1`，V10 commit `c8d9852`（主线 `2c247ed → c8d9852`）。

## 1. 本轮相对 V9 的增量

| 维度 | V9 | V10 |
| --- | --- | --- |
| 叙事 | Results 顺序断裂（审计先于场景、UPTS 脱离机制线） | 七问链：等预算价值→尾部瓶颈→回归修复?→选择修复?→跨任务价值→统一诊断→独立研究? |
| 机制主线 | 异质 tail 指标目测散点（Figure20） | 统一 top-decile recall 重算 + 预注册关联检验 ρ=0.879/p=0.00076/LOO 0.83–0.95 |
| 图 | 8 张报告/幻灯片风格（大标题烘焙） | 6 主图 + 1 ED，期刊式紧凑多面板、a/b/c 标签、语义色盲安全配色、矢量+300dpi |
| 数据审计 | 分散 | consolidated 数据/baseline/OOD 审计表 + oracle 补充行 + 缺口披露 |
| 独立数据 | P3 检索（Cao 描述为蛋白产量） | 资格审计修正 Cao 终点为内源 TE；Cao/Ogden 访问受限如实记录未执行 |

## 2. 阶段完成情况

| 阶段 | 状态 | 验收 |
| --- | --- | --- |
| A 状态复核 | ✅ | V9 三验证器三轮 PASS；远端干净 |
| B1 统一 tail 指标 | ✅ | 11 行同定义重算；AAV top1% 召回 0.1875 精确复现；80 输入哈希 |
| B2 关联检验 | ✅ | 协议先冻结；ρ=0.879/p=0.00076；三轮独立核验 PASS；本地↔远端哈希一致 |
| B3 数据/baseline/OOD | ✅ | consolidated 审计表；one-hot/ensemble/oracle 缺口如实披露 |
| C 独立研究 | ✅（门禁） | Cao 终点修正+访问受限；Ogden NCBI 被拦；未伪造评价 |
| D 图组 | ✅ | 6 主图+ED 三格式；三轮验证 PASS（含确定性重建字节一致） |
| E 论文/PPT/release | ✅ | V10 MD/PDF 三轮审计 PASS（主张/图链/PDF/clean rebuild）；PPT 7 页；release index 58 artifacts 三轮哈希 PASS；GitHub push 成功；仓库测试 325+106 全绿 |

## 3. 最终交付物（本地路径）

- 论文：`deliverables/paper/PAPER1_TAIL_RELIABILITY_V10_20261128.md` / `.pdf`
- 图：`deliverables/figures/v8/Figure{1..6}_v8_*.{png,pdf,svg}` + `ED_Figure1_v8_label_distributions.*`
- PPT：`deliverables/ppt/proxy_function_gap_paper1_v10_20261128.pptx`（7 页）
- 报告：`V10_P0_RECHECK`、`CROSS_SCENARIO_TAIL_ASSOCIATION`、`DATA_BASELINE_OOD_AUDIT_V10`、`SCENARIO_CAO2021_V1`（均 `_20261128.md`）
- 验证 JSON：`PAPER1_TAIL_RELIABILITY_V10_VERIFICATION_ROUND{1,2,3}.json`、`figures/v8/v8_verification_round{1,2,3}.json`、`source_data/cross_scenario_tail_audit_v2/association_verification_round{1,2,3}.json`
- Release index：`deliverables/paper/release_index_v10.json`（58 artifacts，commit c8d9852，三轮哈希 PASS）
- 新分析代码：`source_data/cross_scenario_tail_audit_v2/{tail_audit_v2.py(远端),run_association.py,verify_association.py}`、`figures/v8/*.py`

## 4. 不能声称的结论清单（硬边界，继承 V9 + 本轮）

1. 不能声称 tail reliability **因果决定**决策价值（跨场景关联 n=10，回顾性，非因果；差值 CI 触 0，"显著优于全局相关"未确认）。
2. 不能声称**前瞻盲测/独立湿实验/临床疗效**；独立候选 Cao/Ogden 未执行。
3. AAV production ≠ tropism/感染/临床；不能声称恢复极端最优（top1% recall 18.75%）。
4. GB1 lnW ≠ Kd；GFP 亮度 ≠ 纯表达；TEM-1 派生 fitness ≠ 酶活/MIC/临床。
5. T-cell r2 cap2/8/16 不能升级 Holm 阳性；HepG2 cap16 仅名义。
6. ListNet/50k adaptation/MAE 改善 ≠ 决策改善；stability harmful/indel 0/4/HEK null/TEM 跨药 null 不能包装阳性。
7. UPTS 门禁失败、T-cell 未评价；不能挑单方向声称成功。
8. 不能声称与官方 COM/Design-Bench/GABO/T3PO/mRNAutilus/Reddy 等价固定池比较。
9. 算法种子 ≠ 独立生物重复；不能跨终点单位合并原始效应。
10. 不能声称跨场景可部署 SOTA 方法；主贡献=评估原则+定量尾部诊断+证据地图+边界。
11. **Cao 仓库数据是内源 TE，不是蛋白产量**；不得 mislabel。

## 5. 未完成事项（人工/后续）

- 投稿元数据：作者/单位/COI/许可/目标期刊/数据代码 DOI（需人工确认，不编造）。
- 参考文献 [18] T3PO-mRNA 作者/年份补核（OpenReview 403）。
- 独立数据评价：Cao 内源 TE（需可行数据通道+fasta 拼接+泄漏审计+冻结协议）；Ogden AAV2 tropism（需 NCBI 替代通道+蛋白 proxy）；Tsuboyama 稳定性。
- 仓库小缺口：pytest.ini pythonpath 建议补 `checks`（本轮测试以 `PYTHONPATH=checks CUBLAS_WORKSPACE_CONFIG=:4096:8` 运行全绿；配置项非科学缺口）；T-cell 纯 one-hot ridge / CNN+MLP 部署 ensemble 未单列。

## 6. 主线结论（可发表）

> 在生物序列设计中，proxy 的价值不由全局预测 accuracy 单独决定，而由真实功能终点、等预算选择、高分尾部可靠性与部署条件共同决定。一些任务中 proxy 排序确实降低真实 regret（T-cell r1、GB1、GFP、AAV9 production），另一些任务中回归指标改善并不改善选择（adaptation/ListNet/anchor），且尾部翻转不能由冻结 proxy 内部不确定性重排修复（UPTS 判伪）。统一 top-decile 尾部富集指标与等预算决策收益存在经置换检验支持、leave-one-out 稳健的跨场景正关联（ρ=0.879，p=0.0008）。因此模型开发必须用 functional regret 作为验收终点，而非只看 MAE/Spearman。

## 7. 测试运行说明

仓库全套测试在远端以 `/root/miniconda3/envs/lucaone_cu128/bin/python -m pytest` 运行，需 `CUBLAS_WORKSPACE_CONFIG=:4096:8`（确定性 CUDA 硬约束）与 `PYTHONPATH=checks`（verify_q_head 模块）：**325 passed, 106 subtests passed**。缺这两个环境变量时出现的 CUDA/收集错误为环境配置问题，非 V10 代码引入（V10 仅新增 paper/figures/docs/ppt，未改 src/trainer/runtime/tests）。