# Paper 1 发表导向优化——最终交付总结（2026-11-28）

执行依据：`EXECUTION_PROMPT_PAPER1_OPTIMIZATION_20261128.md`。状态：**DRAFT_NOT_SUBMITTED**。
GitHub：`Cunyu-Liu/proxy-function-gap`，分支 `paper23-gates-v1`，最终 commit `2dcf4af6f2a079d41abea2a07b32f8cf647db824`（主线 `7259351 → d8062a3 → fd22195 → 2dcf4af`）。

## 1. 阶段完成情况

| 阶段 | 状态 | 关键产物 / 验收 |
| --- | --- | --- |
| P0 复现与基线 | ✅ | 三验证器全 PASS；7/7 核心文件本地↔远端 SHA256 一致；远端测试 325+106 全绿；`P0_STATUS_REPORT`、`P0_REMOTE_ARTIFACT_AUDIT` |
| P1 机制实验 | ✅（判伪） | UPTS v1 预注册选择侧干预，辅助门禁失败、未读 T-cell 标签；三轮核验 PASS；`TCELL_UPTS_V1`、`P1_MECHANISM_PROTOCOL_DRAFT`、`source_data/tcell_upts_v1/` |
| P2 跨场景方法验证 | 不触发 | P1 未产生可部署改进方法（提示词 P2 前提不成立）；跨场景证据维持 V8/V9 现状 |
| P3 独立数据检索 | ✅ | 8 条检索记录 + 资格审计；推荐 Cao 2021（GSE176581）/Ogden 2019（GSE139657）；`P3_INDEPENDENT_DATA_SEARCH` |
| P4 baseline/OOD 审计 | ✅ | 六外部方法均不可官方等价复现；baseline/OOD 清单 + Limitations 文案；`P4_BASELINE_OOD_AUDIT` |
| P5 论文与图升级 | ✅ | V9 主文 + Figure22（三格式）+ 确定性 PDF；V9 验证器三轮 PASS（主张/图链接/PDF/clean rebuild 字节一致） |
| P6 发布归档 | ✅ | GitHub 推送成功；仓库测试全绿；release index 59 artifacts 三轮哈希 PASS；V9 PPT（7 页）；人工事项清单 |

## 2. 最终交付物（本地路径）

- 论文：`deliverables/paper/PAPER1_TAIL_RELIABILITY_V9_20261128.md` / `.pdf`
- 图：`deliverables/figures/v7/Figure22_v7_upts_auxiliary_gate_falsification.{png,pdf,svg}`（新增）；v3–v6 既有主图保留
- PPT：`deliverables/ppt/proxy_function_gap_paper1_v9_20261128.pptx`（7 页，构建器 `build_paper1_v9_deck.py`）
- 实验报告：`deliverables/reports/docs/TCELL_UPTS_V1_20261128.md`、`P0_STATUS_REPORT_20261128.md`、`P0_REMOTE_ARTIFACT_AUDIT_20261128.md`、`P3_INDEPENDENT_DATA_SEARCH_20261128.md`、`P4_BASELINE_OOD_AUDIT_20261128.md`、`P1_MECHANISM_PROTOCOL_DRAFT_20261128.md`
- 验证 JSON：`PAPER1_TAIL_RELIABILITY_V9_VERIFICATION_ROUND{1,2,3}.json`、`CROSS_SCENARIO_SYNTHESIS_VERIFICATION_20261128_ROUND{1,2,3}.json`、`source_data/tcell_upts_v1/artifact/verification/round{2,3}_*.json`
- Release index：`deliverables/paper/release_index_v9.json`（59 artifacts，SHA256+bytes，commit，三轮核验）
- UPTS 代码与 artifacts：`deliverables/reports/source_data/tcell_upts_v1/{code,artifact}/`；远端 `/teams/.../tcell_upts_v1`

## 3. 不能声称的结论清单（硬边界）

1. 不能声称 tail reliability **因果决定**所有任务的决策价值（跨场景审计是诊断关联；UPTS 是单点选择侧干预且失败）。
2. 不能声称任何**前瞻盲测/独立湿实验/临床疗效**；全部回顾性，grouped holdout ≠ 独立研究。
3. 不能把 **AAV production** 写成 tropism/infectivity/转导/临床疗效；不能声称恢复极端最优（top1% recall 18.75%）。
4. 不能把 **GB1 lnW** 写成 Kd；**GFP 亮度**写成纯表达；**TEM-1 派生 fitness** 写成直接酶活/MIC/临床药物反应。
5. 不能把 **T-cell r2** 的 cap2/8/16 升级为 Holm 阳性（仅 cap4 通过 all-20）；HepG2 cap16 仅名义阳性。
6. 不能把 **ListNet/50k adaptation/MAE 改善** 写成决策改善；不能把 stability（harmful）/indel（0/4）/HEK293T（null）/TEM-1 跨药（null）包装成阳性。
7. 不能声称 **UPTS 成功**或 T-cell regret 改善（门禁失败，目标未评价）；不能挑选 N50→HepG2 单方向有利结果声称成功。
8. 不能声称与**官方 COM/Design-Bench/GABO/T3PO/mRNAutilus/Reddy** 做了等价固定池比较（仅 COM-style 适配，已披露差异）。
9. 不能把**算法种子**当独立生物重复；不能跨终点单位合并原始效应做 meta-analysis。
10. 不能声称提出**跨场景可部署的 SOTA 方法**；主贡献是评估原则 + 机制诊断 + 边界。

## 4. 未完成事项清单（人工/后续）

- 投稿元数据：作者署名/顺序、单位/通讯、利益冲突、许可、目标期刊/仓库、数据/代码 DOI（需人工确认，不得编造）。
- 参考文献 [17] T3PO-mRNA 作者/年份补核（审计时 OpenReview 403）。
- 独立数据评价（未来工作，需另立冻结协议）：Cao 2021 GSE176581（5′UTR 蛋白产量，MLP 零样本可行）、Ogden 2019 GSE139657（AAV2 肝脏 tropism，需新训蛋白 proxy + 同实验室重叠审计）；Tsuboyama 2023 稳定性扩展。
- 仓库小缺口：GFP 冻结协议载体位置待确认；pytest.ini pythonpath 建议补 `checks`（文档/配置项，非科学缺口）。
- 若要走方法论文路线：需未查看同终点数据或新实验上的 regret 改善（当前资产内无未查看面）。

## 5. 主线结论（可发表）

> 在生物序列设计中，proxy 的价值不由全局预测 accuracy 单独决定，而由真实功能终点、等预算选择、高分尾部可靠性与部署条件共同决定。一些任务中 proxy 排序确实降低真实 regret（T-cell r1、GB1、GFP、AAV9 production），另一些任务中回归指标改善并不改善选择（adaptation/ListNet/anchor），且尾部翻转不能由冻结 proxy 的内部不确定性重排修复（UPTS 判伪）。因此模型开发必须用 functional regret 作为验收终点，而非只看 MAE/Spearman。