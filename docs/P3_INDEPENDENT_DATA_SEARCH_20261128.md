# P3 独立数据检索与资格审计（2026-11-28）

状态：检索完成；**未下载大数据、未训练、未评价**。所有候选仍属历史公开数据（回顾性）；未找到前瞻湿实验数据。
执行方式：协调者直接执行（子代理后端 "Invalid token" 持续故障）。检索工具：内置浏览器 Bing（Google 搜索 ERR_QUIC_PROTOCOL_ERROR、DuckDuckGo fetch 失败、NCBI FetchURL 遇 reCAPTCHA，改用浏览器渲染）。

## 1. 检索记录

| # | 日期 | 引擎/方式 | 检索式 | 关键命中 |
| --- | --- | --- | --- | --- |
| 1 | 2026-11-28 | Bing | Ogden 2019 Science AAV2 comprehensive mutagenesis transduction dataset | Science 2019;366:1139–1143, PMID 31780559 |
| 2 | 2026-11-28 | Bing | Ogden 2019 AAV2 capsid fitness landscape GEO GSE data availability supplementary | **GSE139657**（OmicsDI 命中） |
| 3 | 2026-11-28 | 浏览器 GEO | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139657 | 169 samples；补充 CSV：LiverTropism/packaging/thermostability/antibody/maap/mouse |
| 4 | 2026-11-28 | Bing | Tsuboyama 2023 Nature mega-scale protein stability dataset download Zenodo GitHub | Nature 2023 + Zenodo 记录 |
| 5 | 2026-11-28 | Bing + Zenodo 渲染 | zenodo.org records "Mega-scale experimental analysis of protein folding stability" | **https://zenodo.org/records/7992926**（v2_230420） |
| 6 | 2026-11-28 | Bing | "5'UTR" MPRA "ribosome load" OR "mean ribosome" dataset GEO 2022 2023 2024 library human | Karollus 2021（Sample 衍生）、UTR-Insight 2025、Castillo-Hair 2024、Lim 2021、Plassmeyer 2023 |
| 7 | 2026-11-28 | Bing + FetchURL | Nature June 2024 "5'UTR" translation "cell types" mRNA therapies massively parallel | **Castillo-Hair et al. Nat Commun 15:5284 (2024), DOI 10.1038/s41467-024-49508-2**；数据可用性声明确认 polysome 数据存于 **GSE232927** |
| 8 | 2026-11-28 | Bing + FetchURL | Cao "High-throughput 5' UTR engineering" "protein production" Nature Communications 2021 data availability GEO | **Cao et al. Nat Commun 12:4138 (2021), DOI 10.1038/s41467-021-24436-7**；GSE176581 + GitHub zzz2010/5UTR_Optimizer |

## 2. 候选数据集资格审计

### 2.1 推荐纳入（独立研究）

| 候选 | 来源/许可 | 终点语义 | 样本量/体积 | 与已用数据泄漏风险 | 冻结模型兼容性 | 资格判断 |
| --- | --- | --- | --- | --- | --- | --- |
| **Ogden 2019 AAV2 肝脏 tropism**（首选 AAV 方向） | GSE139657 补充 CSV（GEO 公开）；论文 Science 2019, PMID 31780559；Church lab | **in vivo 小鼠肝脏 tropism**（另有 packaging/thermostability/antibody/MAAP）；不是 production、不是临床疗效 | 169 samples；LiverTropism CSV 752KB，packaging 19.2MB，thermostability 5.7MB，antibody 2MB，mouse 13.2MB；RAW 207MB | 与 Bryant 2021 Fit4Function **同实验室（Church）不同研究/库/血清型（AAV2 vs 项目所用 AAV9 production）**；须做序列级 exact/H1 重叠审计后方可称独立评价面 | RNA CNN/MLP 不适用（蛋白衣壳）；需新建蛋白 proxy（可复用 GB1/GFP 加性 ridge/CNN 管线），**需 GPU 训练** | **合格候选**；工作量中；先审计重叠再冻结纳排 |
| **Cao 2021 5′UTR 蛋白产量**（首选 5′UTR 方向） | GSE176581（raw FASTQ）+ GitHub `zzz2010/5UTR_Optimizer`（处理特征）+ 论文 Source data；CC-BY 4.0；MIT Lu/Kellis lab | **蛋白产量**（报告基因表达），不是 MRL、不是核糖体负载；跨终点须明示 | 合成+天然 5′UTR 库（高通量筛选）；体积待下载审计 | 独立研究；与 Sample/Castillo 无已知同库关系；仍须 exact/H1 审计 | **4-mer MLP 可零样本处理变长 5′UTR**（PERSIST 先例）；50nt CNN 不兼容变长序列 | **合格候选**；工作量低（零样本 MLP + 固定池 BoN/RandomN）；终点为蛋白产量，主张须限定 |
| **Tsuboyama 2023 mega-scale 稳定性** | Zenodo **7992926**（v2_230420）：`Tsuboyama2023_Dataset1_20230416.csv`、Dataset2/3、DMS lists、`Processed_K50_dG_datasets.zip`；Nature 2023；**要求注册使用**（自定义条款，须核对许可） | **热力学折叠稳定性 ΔG**（cDNA display proteolysis），不是表达、不是结合 | ~80 万 ΔG（论文宣称 up to 900,000 domains/variants）；体积大（zip 待审计） | 独立研究；与 GB1/GFP 无同库关系 | 蛋白 proxy 需新训（加性 ridge/CNN 管线可复用）；**需 GPU 训练** | 合格候选；工作量大；作为稳定性新终点扩展，非本轮必需 |
| Rocklin 2017 蛋白稳定性/表达 | Science 2017, DOI 10.1126/science.aan0693；数据在补充材料（待核实下载入口） | 酵母展示稳定性/表达（设计蛋白） | ~15k 设计 | 独立 | 需新训蛋白 proxy | 备选；本轮未深入核实，标注待验证 |
| Lim 2021 前列腺癌 5′UTR MPRA | Nat Commun 12:4217 (2021)（检索命中，未深入核实） | 5′UTR 突变翻译/蛋白影响（疾病变异） | 待核实 | 独立（待审计） | 变长 → MLP 零样本可能 | 备选；待验证 |
| Plassmeyer 2023/2024 神经发育障碍 5′UTR | medRxiv 10.1101/2023.11.02.23297961；Cell Reports Methods 版本（待核实） | 翻译+蛋白产量（疾病变异） | 待核实 | 独立（待审计） | 变长 → MLP 零样本可能 | 备选；待验证 |

### 2.2 标红：非独立 / 泄漏 / 不可用

| 候选 | 原因 |
| --- | --- |
| **Castillo-Hair 2024 Nat Commun 15:5284** | 数据可用性声明明确：polysome 数据存于 **GSE232927**——即本项目已用 accession。**非独立研究**。其 megaTAL 基因编辑数据（Supp Data 1/2、Zenodo 10.5281/zenodo.11398662）属**同研究新终点/未查看切片**，可作回顾性扩展但不得称独立研究 |
| Sample 2019 GSE114002 | 已用作 train/dev/cal |
| Optimus 5-Prime / Karollus 2021 PLOS Comput Biol / UTR-Insight 2025 | Sample 衍生或聚合已用数据，泄漏 |
| Leppek 2022 GSE173083 / Schlusser 2024 | 已查看（PERSIST null） |
| Castillo N25 r1/r2 | 25nt 与冻结 50nt CNN 不兼容；冻结外部协议已排除，禁止 padding/重训制造兼容 |

## 3. 推荐排序与理由

1. **Cao 2021（GSE176581 + Source data）**：唯一可**零样本**（冻结 4-mer MLP）评价的独立 5′UTR 功能面，工作量最低，直接回应"所有结果回顾性/同研究"软肋；代价是终点为蛋白产量（跨终点），主张必须限定为"独立研究、跨终点、回顾性"。
2. **Ogden 2019（GSE139657 LiverTropism）**：填补 AAV tropism/infectivity 实测空白，in vivo 终点价值高；需新训蛋白 proxy 与重叠审计，工作量中。
3. Tsuboyama 2023：稳定性新终点，体积/许可/训练成本高，列为后续扩展。

## 4. 任何新评价前的强制审计（未执行）

- 原始来源 URL/DOI、许可、字段、样本量、终点语义冻结登记；
- 对 Sample train/dev/cal 与 Castillo 全库 exact/H1 泄漏审计；Ogden 另需与 Bryant Fit4Function 库重叠审计（同实验室）；
- 纳排/隔离/终点/主假设一次冻结；test 只评一次；
- 不得把回顾性 grouped holdout 写成前瞻盲测；不得把蛋白产量写成 MRL、把 tropism 写成 production/临床疗效。

## 5. 不可得清单

- 前瞻湿实验/真正未查看同终点 5′UTR MRL 独立库：本轮检索未发现合格者（多数 5′UTR MRL 公开库为 Sample/Castillo/Leppek 衍生或已用）；如实记录为 limitation。
- NCBI FetchURL 被 reCAPTCHA 拦截、Google 搜索 QUIC 错误：已用浏览器渲染替代，不影响结论。

## 6. 来源链接

- GSE139657: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139657
- Ogden 2019: https://www.science.org/doi/10.1126/science.aaw6463（PMID 31780559）
- Tsuboyama Zenodo: https://zenodo.org/records/7992926
- Castillo-Hair 2024: https://www.nature.com/articles/s41467-024-49508-2
- Cao 2021: https://www.nature.com/articles/s41467-021-24436-7 ；https://github.com/zzz2010/5UTR_Optimizer ；GSE176581
- Lim 2021: Nat Commun 12:4217（待核实 URL）
- Rocklin 2017: https://doi.org/10.1126/science.aan0693（待核实数据入口）