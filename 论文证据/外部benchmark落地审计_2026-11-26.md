# 外部 benchmark 落地审计：RNAcompete 与 CAFA3

日期：2026-11-26 ｜ 范围：公开来源、版本、任务、规模、许可/下载、清洗与时间/同源设计 ｜ 路线：泄漏受控基准与强基线

> 本任务没有修改 GPU/训练代码，没有启动训练或模型验证，没有运行 M1b–f，也没有产生 checkpoint。CUDA 数值探针、patience20/min_delta1e-4/1000 故障保险属于后续模型 run 的强制门控，不因本次只有数据审计而伪造留痕。当前不得写 SOTA。

## 0. 三态结论

- **已下载：**RNAcompete Ray2013 的实验主表、完整探针设计、normalized probe 矩阵、7-mer E-score、Nature clustered E-score 和补充方法；CAFA3 的 t0 training、targets、固定 commit 的 NK/LK 清单、三本体 leaf-only ground truth 与 GO OBO。大文件均在规定 `/teams` 目录， gzip/tar/分段字节数与 SHA256 已核验。
- **已实测：**只实测了解压、schema、行列数、ID 对齐、有限值/NaN、精确重复和 benchmark 规模；**没有任何 RNAcompete/CAFA 模型指标或同协议 baseline run**。
- **待做：**生成冻结清洗表与 split manifest、实际同源相似度计算、naïve/BLAST/k-mer/FM 等基线、validation-only 选参及最终 test。原始 RNA 扫描再处理、CAFA5、再分发许可确认不属于本次已完成项。

版本化数据卡：[RNAcompete Ray2013 v1.0](数据卡_RNAcompete_Ray2013_v1.0_2026-11-26.md)；[CAFA3 2016-09 v1.0](数据卡_CAFA3_2016-09_v1.0_2026-11-26.md)。

## 1. 来源、任务与互补性判决

| benchmark | 固定公开来源 | 真实任务标签 | 与当前问题的互补性及边界 | 下载判决 |
|---|---|---|---|---|
| RNAcompete Ray2013 | [Nature 论文](https://doi.org/10.1038/nature12311)、[Hughes Lab 官方补充站](https://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/)、GEO `GSE41235` | 体外短 RNA 的连续 normalized 结合强度；另有 7-mer SetA/SetB/SetAB E-score | 补充 eCLIP 未覆盖的体外 motif、短序列和连续强度泛化，减少细胞环境/体内负例混杂；不能替代体内位点、细胞条件或 neg2 评价 | 核心加工数据已落地；不声称 GEO 原始包已落地 |
| CAFA3 | [CAFA 官网](https://biofunctionprediction.org/cafa/)、[Zhou et al. 2019](https://doi.org/10.1186/s13059-019-1835-8)、官方评估仓库 commit `910af41cd1e8b7751a68061de2c3fe77cd7c6148` | 2016-09 t0 实验 GO 注释训练，后期新增实验注释作 t1；MFO/BPO/CCO 多标签，NK/LK 分开 | 提供 GRIMM 静态 EC 划分没有的真实时间漂移；不能由 GRIMM EC_v2 或 pseudo-OOD 替代，也不是当前 GOA 回填 | t0、targets、t1 真值和 OBO 均已落地 |

## 2. 实测规模与旧数字纠错

### RNAcompete `ray2013_nature12311`

- 主表为 **244 实验、205 个物种×基因组合、24 物种、222 质粒 ID**，不是旧执行文档的“207 RBP、231 实验”。E-score 的 244 个实验 ID 与主表完全对齐。
- 完整探针文件为 243,504 行，非对照 **241,399 行、220,011 条唯一 RNA**；长度 30–41 nt。旧“241,000 条 38–41 nt”混淆了探针行与唯一序列，也漏掉其他长度。
- SetA/SetB 为 120,326/121,031 行，唯一 RNA 109,642/110,348，精确序列交集为 0；42 个其他非对照行对应 21 条唯一 RNA。精确交集为 0 不代表没有近同源。
- normalized 为 **241,357 行×247 列**（3 元数据+244 实验），所有探针 ID 均能对齐设计；58,103,945 个有限值、787,163 个 NaN、0 空字段，102 个设计 ID 不在矩阵。NaN 保留上游图像伪影语义，不补零。
- E-score 为 16,382 行×733 列，零非有限值；缺 `GAAGAGC` 和 `GCUCUUC`，不自动补零。

### CAFA3 `cafa3_2016_2017`

- t0 说明日期为 **2016-09-24**：66,841 条 Swiss-Prot 序列/蛋白，386,197 条蛋白–GO 注释，24,266 个不同 GO。
- targets 为 **130,827 个唯一蛋白 ID、126,138 个不同序列 MD5**；4,689 个 ID 超出唯一序列数，训练去重不能改变官方评价分母。
- MFO/BPO/CCO 分别有 1,101/2,145/1,097 个 benchmark 蛋白、631/1,673/327 个 GO。全并集为 **3,328 蛋白，NK=1,177、LK=2,151**，2,631 个不同 GO；所有 benchmark ID 均找到 target FASTA。
- 旧“28,679 targets、3,228/1,035、677 术语”不是此固定 CAFA3 包的实测规模，撤回用于本版本的表述。
- 上游不一致：BPO NK 的 `T2089630000462` 不在 `leafonly_BPO.txt`。原样记录，不静默修补或删除后冒称官方协议。

## 3. 落地路径与关键校验

远端共同根目录：`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/`。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `rnacompete/ray2013_nature12311/SupplementaryData1_RNAcompete_master_file.xlsx` | 2,677,507 | `3da82ac06deae33264ff2d49f83cd5302dc9d12ccb559032b84d93fdaf42f261` |
| `rnacompete/ray2013_nature12311/RBD_v3_design_unified.txt` | 47,537,452 | `47abcc7272ad43ec8b0bcb210d3c1c4529e986ba19ea45a6638617eebd5e536a` |
| `rnacompete/ray2013_nature12311/norm_data.txt.gz` | 249,998,462 | `efa7fade5fe6c3a189ece3f381550f73f3bb3d935598d211d69d29d9e3ae47ff` |
| `rnacompete/ray2013_nature12311/e_scores.txt.gz` | 40,372,602 | `066ad68fb697d4a0c7b4402659ec9d21badc86ef93664e43ccccf59142e60818` |
| `cafa/cafa3_2016_2017/CAFA3_targets.tgz` | 36,853,861 | `4b5fb9a287e2761ea3066cc1735311baa6ada52cc5347ed3c97b2d4befad2301` |
| `cafa/cafa3_2016_2017/CAFA3_training_data.tgz` | 22,672,252 | `0815e7fd2e8e19176a15656c96b4386743e8ce5aa08e1062dc57e1712c9aab7c` |
| `cafa/cafa3_2016_2017/910af41.../precrec/go_cafa3.obo` | 32,192,871 | `aafba59201c3bee0105d689dc5f1bf98a6ddbf386c341c3a7cf00ec2b6fdde3d` |

完整哈希见各目录 `SHA256SUMS` 和 CAFA commit 目录 `source_manifest.json`。RNA 审计 JSON 为 `master_audit.json`、`design_audit.json`、`norm_data_audit.json`、`e_scores_audit.json`；CAFA 为 `audit_summary.json`。下载与解析辅助脚本留在 `/home/ProxyFunctionGap/external_benchmark_*.py`，大文件没有写入仓库。

首轮超时遗留的远端进程曾继续写探针文件；该暂态截断文件及计数已弃用，旧进程已终止，完整文件经 Range 分段、原子替换和最终哈希重新验收。报告只引用最终文件。

## 4. 清洗与划分合同摘要

### RNAcompete

1. `Motif_ID` 连实验、`Probe_ID` 连矩阵；使用设计中的 RNA 列，不用 60 nt DNA 模板列。
2. 连续强度是主目标，不自行发明阳性阈值；7-mer E-score 是另一任务。NaN、缺失 7-mer、42 个非 SetA/B 行和重复探针均显式保留审计。
3. 闭集按全局精确 RNA 序列与 90% identity 簇隔离；test 一折、validation 相邻一折、train 显式枚举其余三折，`fold=-1` 不得进入 train。
4. SetA↔SetB 另做跨探针集轨，清除 train 侧精确重复和近同源；整基因留出按“物种×Gene_systematic_ID”，不随机拆构建/条件。Spearman/Pearson/MSE 按实验宏平均。

### CAFA3

1. 主轨固定 2016-09 t0 训练/选参，固定 commit 的 NK/LK t1 只作最终评价；不得用当前 GOA 回填或 t1 调参。
2. t0 内 validation 优先更早历史快照；没有则做精确序列组/同源簇独占验证，并明确不是第二次时间 OOD。
3. 原始 CAFA 轨保留同源并按 t1→t0 最大 identity 分桶；压力轨只按预注册 90%/50%/30% 阈值剔除训练近邻，不重排 t1。
4. 评价使用同 commit `go_cafa3.obo`，主报 Fmax/Smin/coverage，NK/LK 与三本体分开；BLAST/naïve 是后续必要基线。无 fold 的 ID 不得通过补集表达式进入 train。

## 5. 许可证与精确外部阻塞

- **RNAcompete：**官方论文、补充站和 GEO 公开可下载，但本次未找到该数据包的显式开源 LICENSE；不得写成 CC0 或可任意再分发。内部研究数据已落地，外部再发布需另核权利方条款。
- **CAFA3：**论文页标明正文 CC BY 4.0、未有其他声明的数据适用 CC0；官方评估仓库 GPL-3.0 是代码许可，不能统一覆盖 UniProt/GO 上游数据。两个 CAFA 压缩包没有独立 LICENSE，再分发仍需核对上游条款。
- **CAFA5：**Kaggle `cafa-5-protein-function-prediction` 匿名数据 API 实测 HTTP **401 Unauthorized**；本次没有凭据或竞赛规则接受记录，故未下载、未使用，不能写成 CAFA5 已落地。
- **CAFA4：**官网 `CAFA4-export.tgz` 与 `cafa4ontologies.zip` 实测均为 **404 Not Found**；未采用第三方不明镜像。上述阻塞不影响已经完整取得的 CAFA3。
- **原始 RNA 再处理：**GEO `GSE41235_RAW.tar`（1,660,405,760 B）、`raw_data.txt.gz`、`z_scores.txt.gz` 未纳入本次核心加工数据下载；若要做原始强度重现仍是待办，不能声称已完成。

## 6. 三遍验收与下一步门控

- 第一遍（来源/完整性）：官方 URL 交叉核对；Range 字节数、gzip/tar、SHA256、OBO 固定 commit 均通过。
- 第二遍（schema/语义）：行列、ID、NK/LK、连续标签、NaN、重复、异常 BPO ID 均直接解析；旧规模数字已纠正。
- 第三遍（交付边界）：两份数据卡与本报告只声明数据/schema 实测；无模型、无 SOTA、无 CAFA 替代、无 M1b–f、无 GPU 代码改动。

下一步必须先冻结具体清洗表和 split manifest，再启动同协议 baseline。训练时必须另附显式 CUDA 数值探针、run/history/checkpoint/日志、validation-only 选参与 test 终评证据；数据下载完成不解除这些门控。
