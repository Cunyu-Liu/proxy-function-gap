# Leppek 2022 GSE173083 数据落地与 Gate 0

## 最终结论

**正式版本：`leppek2022-gse173083-gate0-v2`。仅允许跨终点描述性研究与 24 条 CDS 的小样本外部机制审计；不允许 Sample 2019 MRL 同终点训练、校准、复验或 regret 评价。**

- 新独立研究候选：`Leppek2022_cross_endpoint_and_external_mechanism_v1`。
- 233 条 pooled 构建体有实测 ribosome load 和 polysome 比率；203 条有细胞内降解拟合，191 条有溶液降解拟合。
- 233 条 pooled 构建体没有任何实测蛋白表达列。203 条 `Predicted expression` 全部保留为预测，未进入 measured 集合。
- 24 条 CDS 有实测 Nluc/Fluc 6h、24h 均值和 SD，以及实测时间曲线拟合的溶液稳定性；不能外推为通用蛋白产量或跨细胞/跨构建体结论。
- 未运行模型训练或推理，未访问 outer 结果，未按阳性结果选择数据。Gate 0 为 CPU-only；完整项目测试中的既有 CUDA 测试实际在 CUDA 环境执行。

早期 `leppek2022_gate0_v1` 因 MFE 哨兵值进入分布而标记为 `SUPERSEDED_NOT_FINAL_DELIVERY`。原始文件、v1 和三轮技术验证日志均保留，但不得引用 v1 作为最终分布或交付。

## 数据来源、版本和许可

完整 URL、字节数、SHA256 和许可逐文件固定在 `gate0/configs/leppek2022_v2.json` 与派生 `data_manifest.json`；v2 只改变哨兵清洗和输出版本，原始输入哈希不变。

| 来源 | 版本与用途 | 许可记录 |
|---|---|---|
| [GEO GSE173083](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE173083) / [SOFT](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE173nnn/GSE173083/soft/GSE173083_family.soft.gz) | SOFT last update 2022-04-19；样本、assay、SRA 关系 | GEO 公开下载；SOFT 未声明显式数据许可，未擅自写成 CC0 |
| [GEO filelist](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE173nnn/GSE173083/suppl/filelist.txt) 与 Table S1/S5 | 233 条 pooled 属性、24 条 CDS 属性；GEO supplementary 2021-03-26 | 同上；原论文为 CC BY 4.0 |
| [GEO RAW tar](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE173nnn/GSE173083/suppl/GSE173083_RAW.tar?download=1) | 41,420,800 bytes；6 个 RDAT；SHA256 `7357eadc1f81f0eac621863b8cf3ad4e488eaac6879e7d597b98ce2d7b7b54c1` | 同上 |
| [SRA RunInfo SRP315847](https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc=SRP315847) | 70 个 Illumina runs，RunInfo 报告 50,374 MB；BioProject PRJNA723762 | 公开 SRA 元数据；RunInfo 未声明显式许可 |
| [Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8940940/fullTextXML) / [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8940940/) | PMC8940940.1；PMID 35318324；DOI `10.1038/s41467-022-28776-w` | 正文明确 CC BY 4.0 |
| Publisher MOESM1–6 | Supplementary Information、Data 1–9、Reporting Summary、Source Data；固定 ZIP/member 哈希 | 随原文章 CC BY 4.0 记录 |

SRA 原始 reads 未全部下载，未从 FASTQ 独立重算已发表终点。当前落地包括发表汇总表、RDAT、SRA RunInfo、PMC 正文和完整补充文件；不得声称完成 FASTQ 级复现。

远端原始区：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/leppek2022_gate0/raw`。正式派生区：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/leppek2022_gate0/derived/leppek2022_gate0_v2`。

## Assay、细胞、时间和构建体

| 数据实体 | 实测终点 | 情境 |
|---|---|---|
| 233 pooled full-length mRNA | 16 组分 polysome sequencing 的 ribosome load 与比率 | HEK293T；m7G/polyA pooled RNA；S1 未单列 polysome 收获时间，标为未报告而非猜测 |
| 同一 pooled 文库 | 细胞内 RNA 降解系数、半衰期及拟合误差 | HEK293T；1、7、12、24h；3 repeats；m7G/polyA |
| 同一 pooled 文库 | 溶液 RNA 降解系数、半衰期及拟合误差 | 非细胞；CHES pH10、10mM MgCl2；0、0.5、1、2、3、4、5、6、16、24h；非 m7G/polyA |
| 233 pooled | `Predicted expression` | 由 ribosome load 与细胞内降解构建的动力学预测，**不是实测蛋白** |
| 24 individual HBB-Nluc CDS | Nluc/Fluc 6h、24h 均值、SD、START 归一化值 | HEK293T；HBB-Fluc control；PSU；Source Data 中表达 n=4 |
| 同一 24 CDS | 溶液降解系数、半衰期、误差和 START 归一化值 | 个体降解时间曲线拟合；精确 n 来自 Source Data Fig_4c |
| Source Data Fig_4a | 18 个额外 ID 的 192 条 Nluc/Fluc 重复记录 | 原始文件和 sheet 清单保留；不同化学/时间情境不混入本轮 233+24 主角色表 |

CAI、GC、DegScore、Vienna、EternaFold、SUP/AUP、MFE/ensemble 和 MLD 均标为计算特征/预测，不作为实验测量。

## 原始—派生与来源差异审计

- 233 条 pooled 的 GEO 与 Supplementary Data 1 共有 9,786 个单元格：0 差异。
- 24 条 CDS 的 GEO 与 Supplementary Data 6 共有 2,016 个单元格：0 数值精度内容差。
- Source Data Fig_4c 与 GEO 的 96 个 6h/24h 均值或 SD 对照全部小于 1% 相对差，最大约 0.428%；归一化汇总口径不互相覆盖。
- Supplementary Data 5 与 Source Data 存在 134 个名称—数值/n 差异，完整列于 `supplementary_crosscheck.json`。精确 n 取自名称和表达映射与 GEO 一致的 Source Data Fig_4c，不按效应方向选版本。
- `1220198B1 / Astros-Brourd-Eli-mod3` 的 half-life n=2，与图注笼统 n≥3 不完全一致；保留精确 n，不写“所有终点 n≥3”。
- `dG(MFE) EternaFold` 的 5 个源单元为 `2e20`：`130003D1、420059B1、620129B1、930019D1、930024D1`。v2 保留原始字符串、将数值状态标为缺失并从分布排除。修正后该预测特征 n=186、均值 −161.525、范围 −374.591 至 21.8808。

## 清洗、纳排和分布

所有 257 条全长序列通过字符、长度与构建关系检查。209 条 pooled 采用 `GGG+5UTR+CDS+C1+barcode+C2+3UTR+C3`，24 条 humanHBB pooled 采用 `GGG+5UTR+CDS+3UTR+C1+barcode+C2+C3`，24 条 CDS 直接采用 HBB 全长并验证 CDS 一致性。

- endpoint 网格：257 × 36 = 9,252 行；另有不重复的非缺失 measurement 长表。
- 纳入：1,883 个实测/实测派生值、2,485 个预测特征值。
- 未测/未报告：4,879 个网格单元；不做零值或均值插补。
- 哨兵排除：5 个单元；原始值仍在 coverage 和 raw source 中。
- 全部 38 个 family/endpoint/status 分布见 `distribution_summary.tsv`，分别标记 measured 或 predicted。

## Exact、近重复与冻结角色

全长 RNA 统一转为大写 DNA 仅用于比较，原 RNA 保留。257 条主序列均唯一，exact duplicate group 为 0。近重复定义为全长 Levenshtein distance ≤ `max(1, floor(0.01 × shorter_length))`，使用连通分量成簇；共有 34 条非 exact 近重复边、236 个簇。

按簇冻结 development/candidate/permanent-evaluation，分配只使用序列、来源家族和固定种子，不使用终点值：

| 来源 | development | candidate | permanent evaluation |
|---|---:|---:|---:|
| pooled 233 | 139 | 47 | 47 |
| CDS 24 | 14 | 5 | 5 |

跨角色 exact/near 边为 0。v1/v2 角色表解压后逐字节一致。共享 UTR/CDS 组件边有 25,514 条，其中 14,432 条跨角色；因此即使全长近重复不跨角色，也不能声称组件级完全独立或跨研究泛化。完整覆盖表属于 Evaluator 级审计产物，未来 Learner 视图必须过滤 candidate/permanent 标签。

## Gate 0 能承担与不能承担的结论

**可承担：**

1. 233 条 pooled 构建体中，实测 ribosome load、RNA 稳定性和 polysome 比率的构建体特异描述性跨终点关系。
2. 24 条 CDS 中，实测 RNA 稳定性与 6h/24h Nluc/Fluc 的小样本外部机制审计，保留精确重复数和来源差异。

**不可承担：**

1. 用 `Predicted expression` 声称 pooled 233 的实测蛋白表达。
2. Sample 2019 50-nt UTR MRL 同终点训练、校准、验证、outer regret 或跨研究可靠性。
3. 未测细胞、递送、化学修饰、时间或构建体上的普适功能结论。
4. 未经 FASTQ 重算的原始读段级复现，或共享组件上的完全独立泛化。

## 交付文件与复现

正式目录包含 `data_manifest.json`、`field_dictionary.tsv`、`sequence_index.tsv.gz`、`endpoint_coverage.tsv.gz`、`endpoint_measurements.tsv.gz`、`inclusion_exclusion.tsv.gz`、`distribution_summary.tsv`、`exact_duplicate_groups.tsv.gz`、`near_duplicate_edges.tsv.gz`、`component_overlap_edges.tsv.gz`、`overlap_report.json`、`role_assignments.tsv.gz`、`role_manifest.json`、`sra_run_index.tsv`、`source_data_inventory.json`、`supplementary_crosscheck.json`、`gate0_decision.json`、`build_summary.json`、`delivery_checksums.json` 和三轮验证报告。

```bash
cd /home/proxy-function-gap
export PYTHONPATH=gate0
.venv/bin/python -m gate0_leppek2022.acquire --config gate0/configs/leppek2022_v2.json
.venv/bin/python -m gate0_leppek2022.build --config gate0/configs/leppek2022_v2.json
.venv/bin/python -m gate0_leppek2022.verify --config gate0/configs/leppek2022_v2.json --round 1
.venv/bin/python -m gate0_leppek2022.verify --config gate0/configs/leppek2022_v2.json --round 2
.venv/bin/python -m gate0_leppek2022.verify --config gate0/configs/leppek2022_v2.json --round 3
```

构建器拒绝覆盖已有输出目录；变更必须创建新版本。配置、构建日志和三轮日志位于 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/leppek2022_gate0/logs`。

三轮结果：Round 1 核验 19 源文件、257 原始行和 9,252 源单元格；Round 2 重算 257 序列组装、38 个分布、32,896 个序列对距离和 34 条近重复边；Round 3 核验 18 个交付文件哈希、236 簇、257 角色和标签/门禁语义，全部 PASS。

完整项目标准验证使用项目 CUDA 环境与 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，结果为 **279 passed + 106 subtests passed**。本次 Gate 0 自身没有 GPU 训练或推理。
