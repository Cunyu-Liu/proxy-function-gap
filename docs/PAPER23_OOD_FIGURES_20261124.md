# Paper2 / Paper3 / OOD 图现代化（v2，2026-11-24/25）

依据 `plot-is-all-you-need` + `scientific-figure-making` 规范，与 Paper1 v2 同一视觉系统（现代暖冷配色、逐池/逐情境证据、显著性编码、自包含图注、PNG/PDF/SVG）。所有数值来自本地冻结 `reports/source_data/`，`verify_v2_paper23_ood.py` 三轮校验全 PASS（`v2_paper23_ood_validation_report.json`）。

| 文件 | 面板 | 科学含义 |
|---|---|---|
| `Figure7_v2_paper2_baselines` | a 五方法 cap16 regret 柱（±池 SE）；b 配对差值森林 vs BoN | Paper2 选择器路线证伪：所有方法 48 次 CNN forward 预算匹配，配对区间全部跨 0；COM-style/diversity-LCB 无优势。置换负对照保留（BoN 0.672 vs permuted 1.951）。 |
| `Figure8_v2_paper3_headroom` | a 48 门禁 headroom 热图（3 表示×16 池）；b kmer4 六策略 pooled regret—预算曲线 | Paper3 阳性：headroom 存在且表示依赖（kmer4 5/16、cnn64 1/16、summary4 0/16）；EI 在 16 标签兑现 headroom（1.220 vs random 3.497），regret-analytic 不支配。 |
| `Figure9_v2_schlusser_te` | a TE 极端右偏分布（log 轴 min/Q1/median/Q3/max）；b cap16 跨终点配对森林 | 跨终点 OOD 确认性：MRL proxy 不改善 HepG2 内源 TE，配对区间全跨 0，MRL–TE Spearman 0.072。配对差消去共享 oracle。 |
| `Figure10_v2_leppek_endpoints` | a 6 终点配对相关（Pearson●/Spearman◇，CDS vs pooled）；b CDS-24 排序反转+top-6 重叠 | 终点错配机制：溶液半衰期 vs 6h 表达反转 62%（171/276）、top-6 重叠 1/6；pooled MRL vs 溶液半衰期 r=−0.346。proxy 契约必须写明终点。 |

## 校验要点（轮1）

- Paper2：五方法 cap16 均 48 CNN forward（预算匹配）；lcb/distance/com/diversity_lcb 配对区间均跨 0。
- Paper3：正门禁计数 summary4=0/kmer4=5/cnn64=1；kmer4 曲线 budget0 与 budget64 六策略各自相等；EI@16 < Random@16。
- Schlusser：TE max>4000 且 median<1（极端右偏）；cap16 四个配对基因簇区间均跨 0。
- Leppek：半衰期 vs 6h 反转比例=171/276，top-6 重叠=1。

## 证据边界（图注已含）

Paper2/Paper3 为 retrospective development/calibration 诊断（永久 outer 未动）；Schlusser 为确认性一次性揭盲（单研究，非跨细胞泛化）；Leppek 为探索性描述（无 p 值/CI、无模型选择）。COM 为 COM-style 适配非官方权重。所有不利/跨 0 结果如实保留。