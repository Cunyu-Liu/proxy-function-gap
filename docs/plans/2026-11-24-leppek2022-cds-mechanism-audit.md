# Leppek 2022：24 条 CDS 跨终点描述性机制审计预注册

冻结时间：2026-11-24T11:19:39Z。配置：`gate0/configs/leppek2022_cds_mechanism_audit_v1.json`。本计划在计算任何相关、排序反转或汇总结果之前提交；分析只读 Gate 0 v2 和已固定哈希的原始 Source Data，不改写既有派生数据。

## 研究边界

- 主要对象是 Gate 0 v2 的全部 24 条 individual HBB-Nluc CDS，不按 development/candidate/permanent-evaluation 角色、效应方向或结果强弱筛选。角色标签不参与统计。
- 233 条 pooled 构建体只提供独立的 MRL/核糖体负载—RNA 稳定性机制背景。两套构建体不拼接、不互相补值，也不声称 pooled 有实测蛋白表达或 CDS 有 MRL。
- `Predicted expression`、CAI、结构预测等永不进入实测终点。CDS 的实测表达仅指 HEK293T、PSU、HBB-Fluc 对照下的 Nluc/Fluc 6h 和 24h 均值。
- 不训练或运行模型推理，不访问 outer 结果，不调整 outer 参数，不做泛化显著性声称。全部为 CPU 描述性分析。

## 输入与可追溯性

1. Gate 0 v2 `endpoint_measurements.tsv.gz`、`endpoint_coverage.tsv.gz`、`sequence_index.tsv.gz`、`exact_duplicate_groups.tsv.gz` 和 `near_duplicate_edges.tsv.gz`。先核验 `delivery_checksums.json` 自身 SHA256，再逐文件核验其登记的哈希。
2. 原始 `41467_2022_28776_MOESM6_ESM.xlsx` 的 `Fig_4c`，固定 SHA256，用于逐构建体精确 replicate n 与名称映射。主终点数值仍取 Gate 0 的 Table S5 原始实测字段；Source Data 的 START 归一化均值不覆盖主值。
3. 所有输出写入新版本目录，构建器拒绝覆盖；保留全部结果，包括零相关、负相关、反转和缺失。

## 冻结终点与配对

- CDS 主终点：`cds_in_solution_half_life`、`nluc_fluc_6h`、`nluc_fluc_24h`。三组两两配对均报告，不根据结果选择。
- pooled 背景终点：ribosome load、三种 polysome 比率、in-cell half-life、in-solution half-life。相关仅限三对：ribosome load–in-cell half-life、ribosome load–in-solution half-life、in-cell–in-solution half-life。
- 每对使用两个终点均非缺失的构建体，报告 n、Pearson r 和采用平均秩处理 ties 的 Spearman rho；不计算 p 值、置信区间或显著性标记。Pearson 仅描述线性，Spearman 仅描述单调排序。
- START 归一化终点、降解系数及 SD 保留在 source data/覆盖审计中，但不另起结果驱动配对。共同常数归一化不会改变 Pearson/Spearman，因此不重复包装成独立敏感性证据。

## 排序反转

- 对 CDS 的三组主配对，枚举全部无序构建体对（完整 24 条时每对 276 对）。两终点顺序严格相反记为 discordant，严格相同记为 concordant；任一终点并列记为 tied 并单列。
- 排序反转比例 = discordant / (concordant + discordant)，ties 不进入分母。另固定报告每对 top-6（25%）集合重叠、独有成员和全部平均秩；top-6 以数值降序、构建体 ID 升序打破并列。
- 排序反转是描述性异质性，不解释为因果冲突或模型失效。

## 覆盖、缺失和重复测量

- 分 source family/endpoint 报告总数、实测 n、缺失 n、覆盖率和误差字段 n；明确 pooled 实测表达为 0，CDS MRL 为 0，缺失不插补。
- 检查构建体 ID、序列及 endpoint 记录的重复；引用 Gate 0 exact/near-duplicate 结果，并在 CDS 内单列近重复边，不能将重复测量当独立构建体。
- 逐 CDS 报告 Fig_4c 的 half-life、6h、24h 精确 n 和来源行。特别保留 half-life n=2 的构建体；表达均值/SD 是已发表重复汇总，不能扩展成 96 个独立表达构建体。

## 图形与交付

一张 2×3 多面板图：A 覆盖；B/C pooled MRL 与两种稳定性；D/E CDS 溶液半衰期与 6h/24h Nluc/Fluc；F 6h–24h 表达排序及反转。所有面板标 n 和 exploratory，CDS 展示已发表 SD 误差棒；不拟合趋势线。

交付包括自包含中文报告、PNG/PDF/SVG、逐构建体聚合 source data、覆盖/相关/排序/重复审计表、机器可读摘要和输入/输出 SHA256。三轮验证分别独立重算数值与来源映射、检查图形内容/尺寸/可解析性、核验全部交付哈希与可复现性；三轮都必须 PASS 才能交付。
