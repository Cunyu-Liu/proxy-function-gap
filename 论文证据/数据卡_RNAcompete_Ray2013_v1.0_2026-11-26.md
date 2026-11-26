# RNAcompete Ray2013 数据卡 v1.0

日期：2026-11-26 ｜ 数据版本：`ray2013_nature12311` ｜ 状态：**原始数据与 v1.0 双轨 split 已冻结验收；未训练、未跑模型基线**

## 1. 来源与固定版本

- 论文：[Ray et al., 2013, Nature 499:172–177](https://doi.org/10.1038/nature12311)。论文指定 GEO `GSE41235` 和 Hughes Lab 官方补充站。
- 官方补充站：[RNAcompete Eukarya Supplementary Data](https://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/)。本版本使用实验主表、探针设计、归一化探针强度、7-mer E-score 和论文补充方法；版本由 DOI、文件字节数和 SHA256 共同固定，而不把服务器后续迁移时间当作生物学新版本。
- GEO 交叉核验：[`GSE41235`](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE41nnn/GSE41235/) 的 `GSE41235_RAW.tar` 为 1,660,405,760 B；本次模型任务使用官方已归一化矩阵，不声称已下载或重新处理 GEO 全部原始扫描数据。
- 远端根目录：`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/ray2013_nature12311/`。校验见 `SHA256SUMS`，规模审计见 `master_audit.json`、`design_audit.json`、`e_scores_audit.json` 和 `norm_data_audit.json`。

## 2. 任务标签与实际规模

这是**体外短 RNA–RBP 结合实验**。归一化探针值是连续强度：补充方法描述列仿射归一化、按行的 robust z-transform 校正初始 RNA 丰度，再做列 robust z-transform；它不是 eCLIP 的体内交联二分类，也不能直接改称 log-odds 标签。E-score 文件给出每个 7-mer 的 SetA、SetB 和 SetAB 富集分数，适合 motif/排序分析。

| 文件/层级 | 实测规模与含义 |
|---|---|
| 实验主表 | 244 个 RNCMPT 实验/244 个 hybridization ID；205 个“物种×基因”唯一组合，24 物种；222 个质粒 ID |
| 探针设计 | 243,504 行；其中非对照 241,399 行，阳性/阴性控制 1,503/602 行 |
| 非对照序列 | 220,011 条唯一 RNA，21,388 行为精确序列重复；长度 30–41 nt |
| normalized 矩阵 | 241,357 行（SetA+SetB）×247 列（3 个元数据列+244 实验列）；241,357 个探针 ID 全部对齐设计；58,103,945 个有限值、787,163 个 NaN、0 个空字段；42 个设计 ID 不在矩阵（独立逐 ID 复核修正旧审计的 102） |
| SetA/SetB | 120,326/121,031 行；唯一 RNA 109,642/110,348，A/B 精确序列交集为 0；另有 42 个非对照行（21 条唯一 RNA）不属于这两个标签，不能随机塞入 A/B |
| 7-mer E-score | 16,382 行×733 列=7-mer 列加 244 实验×3 轨；零非有限值；244 个实验 ID 与主表完全对齐 |
| 缺失 7-mer | 理论 16,384 个中缺 `GAAGAGC`、`GCUCUUC`；保留缺失，不自动补零 |

旧执行文档的“207 RBP、231 实验、241,000 条 38–41 nt”不能描述此固定版本。241,399 是非对照探针行数而非唯一 RNA 数；实际长度还包含 30、32–36 nt。207/231 不得与 244/205 混用。

## 3. 清洗合同

1. 以 `Motif_ID` 连接实验主表和矩阵列，以探针 `ID` 连接设计和强度行；保留 SetA/SetB、杂交条件和重复探针来源，不按行号隐式连接。
2. 使用设计文件的 RNA 列而不是 60 nt DNA 模板列；统一 `T→U` 仅限模型输入表示，保留原文件不变。
3. 对精确 RNA 序列建立全局组，重复探针不当作独立生物重复。拟合时是否聚合及其聚合规则须在训练前固定，评价不得重复计数。
4. 非有限值按上游 NaN 语义处理；两个缺失 7-mer 保留缺失。42 个非 SetA/B 行单独标记，不能随机分配到 SetA/B。
5. 结构/MFE 是设计协变量，不是结合标签。连续强度主任务不自行发明阳性阈值；若增加分类轨，必须独立预注册阈值和负例语义。
6. 归一化已跨同批实验使用行/列信息，故不能把每列当作完全独立的实验生物重复，也不能把 SetA/SetB 当作独立生物重复。

## 4. 同源与外部泛化划分（v1.0 已冻结）

冻结版本 `rnacompete-ray2013-v1.0-id90-strict3train` 已生成序列簇独占五折和整基因留出五折，严格三折 train 的泄漏复核全部通过。详见 [冻结审计](RNAcompete_Ray2013_冻结审计_v1.0_2026-11-26.md) 和该版本 `split_manifest.json`（SHA256=`868524abd1f9c76501c8c8212b6bc531290796c267bc898b45047c7bede39f43`）。SetA↔SetB 仍是后续独立版本，不属于当前已冻结轨。

- **闭集序列回归轨。** 全部实验共享一个全局序列分组：先精确序列去重，再以全局比对/CD-HIT-EST 90% identity 聚类；同簇只入一个 fold。每轮 test 一折、validation 相邻一折、train 显式枚举其余三折；`fold=-1` 和未分配序列不得通过补集进入 train。
- **SetA↔SetB 轨。** 复现论文的跨探针集泛化思想，分别 A 训练/B 测试和 B 训练/A 测试；精确重复和越阈值近同源序列必须从 train 清除，42 个其他行单独报告或不入该轨。validation 只能来自训练侧，test 不用于选参。
- **整基因留出轨。** 若研究未见 RBP，按“物种×Gene_systematic_ID”整组留出，而不是随机拆同一基因的不同 RNCMPT 构建/条件。独立单实验模型不能直接声称具备未见 RBP 预测能力。
- **指标。** 每实验连续预测报告 Spearman、Pearson 与 MSE，再按实验宏平均；SetA/B 分开报告。7-mer 任务可另做排序/富集复现，但不能把 7-mer E-score 与探针强度混成同一目标。
- **与当前 eCLIP 的互补性。** RNAcompete 提供体外、短序列、连续结合强度和 motif 泛化，隔离细胞环境与体内交联负例问题；它不验证体内位点、细胞条件 OOD 或 eCLIP neg2 的真阴性语义。

## 5. 许可证与边界

- 官方论文页和补充站支持公开下载，GEO 也公开原始数据；本次未在补充站、主表或压缩包说明中找到针对该数据包的显式开源许可证。公开可下载不等于 CC0 或任意再分发授权，外部再发布前须向权利方核实。
- 本版本只证明数据来源、文件完整性、schema 和划分设计已经审计，**不代表已有 RNAcompete 模型 run、外部性能或 SOTA**。
- 若需从原始扫描强度重新归一化，`raw_data.txt.gz`、GEO 原始包及 `z_scores.txt.gz` 仍需单独下载和复现；当前任务以官方归一化矩阵为准，不声称完成原始再处理。
