# Paper 1 主图现代化重制（v2，2026-11-24）

依据 `plot-is-all-you-need` 与 `scientific-figure-making` 两个 skill 的规范重制。配色采用现代暖冷协调族（teal/violet/coral/sand + 深蓝锚点），语义一致：负值（有利于 BoN）用冷色、正值用暖色；显著性用标记形状编码（● Bonferroni 稳健 / ◇ 仅名义显著 / ○ 区间跨 0）。所有数值与冻结 `source_data/` 及 `strict_h1_multiplicity_sensitivity.json` 逐位一致。

## 校验

`verify_v2_figures.py` 三轮独立通过（`v2_validation_report.json`，overall=PASS）：
- 轮1 数据完整性：20 个压力单元格的逐池均值=配对汇总=多重校正 JSON；8 个 Bonferroni 存活单元格全部为 T-cell cap2–16；4 情境分数匹配区间均跨 0；4 情境跨架构相关 0.87–0.97 且 CNN MAE 配对区间排除 0。
- 轮2 渲染确定性：5 张图重渲染 SVG 逐字节一致（固定 hashsalt）。
- 轮3 格式与标签：PNG/PDF/SVG 均有效，PDF 单页，SVG 合法 XML。

## 新图清单与科学故事

| 文件 | 面板 | 讲什么 |
|---|---|---|
| `Figure2_v2_selection_pressure` | a–d 四情境压力曲线（含 16 个逐池点云+均值 CI+显著性标记）；e cap16/cap8 森林图；f 多重校正稳健性计数 | 核心阳性现象：T-cell 两重复 cap2–16 收益经 all-20 Bonferroni 存活；N50 无持续收益、HepG2 cap8 探索性。**点云让读者看到证据分布而非只看均值线**。 |
| `Figure3_v2_score_matching` | a 差值森林图；b 逐池配对差值条带 | 机制零结果：四情境 optimized−matched 区间均跨 0 → 无优化器特异误差增量。 |
| `Figure4_v2_common_proxy_error` | a 误差相关棒棒糖图；b CNN vs MLP MAE 哑铃图 | 机制阳性：换架构误差仍高度相关（0.878–0.962），CNN MAE 更低但非独立验证。 |
| `Figure5_v2_pressure_heatmap`（新） | 4×5 配对效应热图，RdBu 居中 0，★ 标 Bonferroni 存活 | **一眼读懂全貌**：压力收益的情境依赖格局，T-cell 整片冷色稳健、N50/HepG2 不稳健。适合作图形摘要核心面板。 |
| `Figure6_v2_error_concordance`（新） | 逐池 CNN MAE vs MLP MAE 散点+身份线+逐情境相关 | 机制阳性的直观版：点贴着身份线 → 不同架构犯同样的错，分歧≠独立功能验证。 |

## 与旧图的关系

旧图（`Figure2_strict_h1_*` 等）完整保留，未覆盖。v2 为新增发表候选；论文若采用 v2，需把正文图引用与图注同步更新（图注已自包含证据边界：retrospective strict-H1 protocol sensitivity，非新盲测）。

## 设计要点（供后续 Paper2/3/OOD 图复用）

- 字体 Arial/Helvetica，基准 9.5pt，面板标签 13pt 粗体 a/b/c。
- 去顶/右边框，浅灰网格 `#E6E8EE`，零线虚线 `#9AA0B0`。
- 逐池证据用半透明点云呈现，均值用粗线/菱形强调，避免"只画均值"的老旧风格。
- 显著性用形状+边框编码，色盲友好；热图用发散色标居中 0。
- 导出 PNG(300dpi)/PDF/SVG 三格式，SVG 文本可编辑（fonttype=none），PDF 字体 Type42。