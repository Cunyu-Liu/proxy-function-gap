# Schlusser 2024 多研究数据、benchmark 与确认性 OOD

## 结论与证据边界

本 Gate 0 使用 Schlusser et al., Genome Biology 2024（DOI `10.1186/s13059-024-03369-6`，PMC11337900）的公开序列级表格，不重新训练或运行任何模型。正式数据集为 `schlusser2024-gate0-v1`，项目冻结前 HEAD 为 `fe7c0aad23cf074918427297872d8c8c5d0ff9d1`。

**Schlusser 新产生的 HepG2 endogenous mRNA TE 研究被整研究冻结为后续确认性 OOD。当前仅完成数据与协议资格核验，不授权模型执行，未计算 outer 指标，也未向 Learner 返回逐序列 TE。** 该研究在冻结前未用于本课题训练、调参或结果选择；冻结发生在逐序列标签检查和模型试跑之前。未来评估必须在最终 proxy、候选生成和 Evaluator 协议冻结后另行授权，并创建新运行目录。

这轮工作不修改 Paper 2/3，不把已有研究的训练资源改名为独立 outer，也不按阳性结果挑选研究。

## 实际构建与三轮验证结果

13 个 Zenodo 文件全部通过发布方 MD5，24 个数据/元数据/上游代码文件进入 acquisition manifest。实际覆盖如下；所有主研究表的实测终点字段均无空缺失，未用预测值填补。

| 研究 | 原始行 | 唯一完整 5′UTR | 固定 50 nt 兼容行 | 唯一末端 50 nt |
|---|---:|---:|---:|---:|
| Optimus50 | 279,996 | 279,996 | 279,996 | 279,996 |
| Optimus100 | 105,000 | 105,000 | 70,630 | 68,015 |
| Yeast50 | 489,348 | 489,348 | 489,348 | 489,348 |
| DART | 6,962 | 6,962 | 3,084 | 2,201 |
| Yeast TE | 4,312 | 4,312 | 3,489 | 3,486 |
| HEK293 TE | 11,226 | 11,210 | 9,262 | 9,207 |
| HepG2 TE | 15,720 | 15,696 | 13,482 | 13,350 |

4 个 Optimus 预测表分别含 5,326（Optimus5p）或 5,040（MTtrans、Framepool、TranslateLSTM）行，`rl` 与 `predicted_rl` 均逐行区分，没有 prediction→measured 提升。

HepG2 的 15,720 行中，2,238 行因固定 50 nt/ACGT 输入不兼容排除，5,889 行因 Sample exact overlap 排除，1,253 行因 Sample Hamming-1 overlap 排除，最终隔离 Learner 为 **6,340 行**。按唯一末端 50 nt 计，exact overlap 为 5,832，非 exact 的 Hamming-1 overlap 为 1,237。大量重叠说明原表绝不能直接宣称为独立测试；序列隔离是必要协议，而不是结果后筛选。Evaluator 保留全部 15,720 行及排除原因，Learner 不含 TE 或预测字段。

- Round 1：13 个 Zenodo 文件字节数/MD5 与研究角色复核通过。
- Round 2：7 个研究的行数、序列、唯一性、兼容性、缺失和预测语义重算通过；未使用标签分布或模型指标。
- Round 3：6,340 行隔离 Learner、15,720 行 Evaluator inclusion、Sample exact/Hamming-1 零泄漏、交付哈希和 no-outer/no-GPU 门禁通过。
- `gate0_decision.json`：`confirmatory_ood_data_ready=true`，但 `authorized_model_execution=false`、`outer_metrics_computed=false`、`scientific_result_claimed=false`。
- 项目完整测试在 RTX 5090、PyTorch 2.7.1+cu128/CUDA 12.8 环境三轮均为 **296 passed + 106 subtests passed**。首次收集路径错误和后续包断言修正日志均保留；三次通过前的 shell 汇总错误发生在 pytest 已全部结束之后，不是测试失败。

## 公开来源、版本、哈希与许可

| 来源 | 固定版本 | 用途与许可 |
|---|---|---|
| Genome Biology / PMC | DOI `10.1186/s13059-024-03369-6`；PMC11337900；PMID 39164757 | 实验与字段语义；正文 CC BY 4.0 |
| Zenodo `10463061` / DOI `10.5281/zenodo.10463061` | 文件级版本；后续 `10463090` 为 ZIP-only 版本 | 序列级表格和 plot scripts；CC BY 4.0；逐文件 MD5/字节数固定于配置 |
| Zenodo `13133725` | TranslateLSTM 发表快照 | 代码/模型谱系；MIT；不因下载权重而运行推理 |
| Basel GitLab project 2850 | commit `3bd1f03579b662b0b0bc9c266ae92f457258727c` | 预处理实现与 MIT LICENSE；commit 固定 URL |
| NCBI SRP474241 / ENA PRJNA1045106 | SRR26949418–SRR26949423 | NCBI RunInfo 为 3 Ribo-seq + 3 RNA-seq；ENA portal 本次仅返回 3 RNA-seq，差异与 RNA FASTQ MD5 均保留 |
| ENA PRJNA591214 | Alexaki HEK293 runs | HEK293 endogenous TE 谱系；run/FASTQ MD5 保留 |
| ENA PRJNA338918 | GSM2278862–64、GSM2278844–46 对应 runs | yeast endogenous TE 谱系；run/FASTQ MD5 保留 |
| GEO GSE114002 / GSE104252 | Sample 2019 / Cuperus 2017 | reporter 原始谱系；公开下载，不把公开可访问擅自写成 CC0 |
| Niederer 2022 Supplementary Table S2 | DART | 天然酵母 5′UTR reporter RRS；不是 endogenous mRNA TE |

实际下载文件同时记录 SHA256。ENA FASTQ 未全部下载，未从 reads 独立重算 TE；因此这里只声称“发表序列级表格的获取和审计”，不声称 FASTQ 级复现。原始、派生和日志位于 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/schlusser2024_gate0/`，不进入 Git。

## 研究、序列与实验终点

| 研究表 | 体系与序列 | 实测或实测派生终点 | 本课题角色 |
|---|---|---|---|
| Optimus50 | HEK293，50 nt 随机 5′UTR-eGFP IVT reporter | MRL | Sample 2019 已用于本 proxy 训练/开发；仅 source benchmark |
| Optimus100 | HEK293，25–100 nt 随机、天然片段、SNP 和 3′端片段 reporter | MRL | 同一 Sample 研究/模型谱系；不是独立 outer |
| Yeast50 | 酵母 HIS3 质粒，约 50 万条随机 50 nt 5′UTR | 生长富集/生长速率 | Cuperus 训练资源；不同终点，不晋升为确认性 outer |
| DART | 天然酵母 5′UTR，部分 uAUG→AGU，24 nt reporter CDS，酵母提取物 | ribosome recruitment score (RRS) | **天然序列 reporter**，不是内源 mRNA；Niederer 训练资源 |
| Yeast TE | 内源酵母转录本，CDS RPF 与 RNA-seq | TE = CDS ribosome footprints / RNA-seq | Weinberg/Mittal 内源训练资源；仅覆盖审计 |
| HEK293 TE | GRCh38.105 主导转录本，CDS RPF 与 RNA-seq | TE | Alexaki 内源训练资源；不冒充未使用 outer |
| HepG2 TE | Schlusser 新增 HepG2 RNA-seq/Ribo-seq，主导内源转录本 | TE | **整研究冻结的确认性 OOD** |

MRL、RRS、生长速率和 TE 不合并为同一个“表达量”。TE 是由实验 reads 派生的测量终点，不等于直接蛋白产量。天然 5′UTR 被放入 DART reporter 后仍属于 reporter 体系。上游 `combine.py` 在发表表生成时保留 `TE > 0`，并对部分缺失元数据填 0；这是原始数据既有纳排/处理，不是本课题按结果新增筛选。原始脚本和发表表共同保留，不能将发布表解释为未经终点过滤的全体转录本。

Optimus 的 `test_data_*_rl.tsv` 单独审计：实测 MRL 与不同模型预测列并存时，prediction 只作为模型输出，绝不回填实测标签。ClinVar variant TE LFC 也是 TranslateLSTM 预测，不能承担实验功能验证。

## 划分与反泄漏协议

1. 外层分配单位是**完整研究**，不把同一研究的随机序列拆分后冒充研究级复验。
2. HepG2 在标签检查前冻结。Learner 只得到 `row_id`、来源 ID、5′UTR 和兼容的末端 50 nt；Evaluator-only 保险库保存 TE。
3. 当前 Sample proxy 固定输入要求 50 nt 且字母为 ACGT。统一大写、U→T，并取靠近 start codon 的末端 50 nt；短于 50 nt 或含其他字符的记录保留在原始与覆盖表，但不进入固定输入 Learner。此纳排只由技术兼容性决定。
4. 在不知道 TE 的情况下，对 HepG2 末端 50 nt 与 Sample train/dev/calibration 全角色执行 exact 和 Hamming-1 排除。原始行、排除原因和保险库不删除，隔离 Learner 仅保留通过者。
5. 相同末端 50 nt 的不同转录本不视为独立生物学重复；整研究不再随机拆分。未来评估必须保留转录本/基因簇关系，不能只按行数扩大样本量。
6. 所有其他表保持 source-benchmark 角色。尤其 Optimus/HEK293/yeast/DART 是 Schlusser 模型训练或比较资源，不直接改名为本课题独立 outer。
7. 本轮 `outer_metrics_computed=false`、`labels_returned_to_learner=false`、`model_training_or_inference_run=false`。后续任何揭盲都必须生成新配置、日志和结论，不能覆盖本冻结包。

## 可审计交付

正式派生目录包含：

- `data_manifest.json`：DOI、commit、许可、原始文件 SHA256、配置哈希和覆盖；
- `coverage/<study>.json`：实际列名、行数、唯一序列、长度、缺失、实测/预测字段；
- `prediction_field_coverage.json`：Optimus 模型预测字段的独立审计；
- `sequence_index/<study>.tsv.gz`：逐来源行序列索引，不含 TE；
- `learner/hepg2_te_sequences_isolated.tsv.gz`：技术兼容且通过 Sample 序列隔离的无标签视图；
- `evaluator/hepg2_te_labels.tsv.gz` 与 `evaluator/hepg2_te_inclusion.tsv.gz`：Evaluator-only TE 与逐行纳排原因；
- `role_manifest.json`、`overlap_report.json`、`gate0_decision.json`；
- `delivery_checksums.json` 和三份 `verification_round_*.json`。

实际行数、兼容数、排除数及三轮结果以派生 JSON 为准；本文数值与上述已验证 JSON 一致，未按标签或模型结果挑选。

## 复现命令

```bash
cd /home/proxy-function-gap
export PYTHONPATH=gate0
.venv/bin/python -m gate0_schlusser2024.acquire --config gate0/configs/schlusser2024_v1.json
.venv/bin/python -m gate0_schlusser2024.build --config gate0/configs/schlusser2024_v1.json
.venv/bin/python -m gate0_schlusser2024.verify --config gate0/configs/schlusser2024_v1.json --round 1
.venv/bin/python -m gate0_schlusser2024.verify --config gate0/configs/schlusser2024_v1.json --round 2
.venv/bin/python -m gate0_schlusser2024.verify --config gate0/configs/schlusser2024_v1.json --round 3
```

构建器拒绝覆盖既有派生目录；任何配置变更必须使用新版本。下载配置、构建配置、提交、标准输出和三轮日志保存在 `schlusser2024_gate0/logs/`。本 Gate 0 为 CPU-only，没有 GPU 训练或推理；完整项目既有 CUDA 测试必须使用项目 CUDA 环境执行，不能用 CPU 测试冒充 GPU 证据。
