# V5参考文献与投稿文本审计

- 审计日期：2026-11-26
- 原始对象：`deliverables/paper/UNIFIED_PREPRINT_MAIN_V5_20261126.md` 及其PDF
- 修订对象：`deliverables/paper/UNIFIED_PREPRINT_MAIN_V5_1_20261126.md` 及其PDF；后续可用性修正版为`deliverables/paper/UNIFIED_PREPRINT_MAIN_V5_2_20261126.md`及其PDF
- 三轮原始记录：`deliverables/reports/docs/PREPRINT_REFERENCE_AUDIT_20261126_THREE_ROUNDS.log`
- 范围：主文全部参考文献、DOI/arXiv/OpenReview链接、正文编号与主张、标题/摘要/限制/数据代码可用性、PDF
- 明确排除：figures、PPT、`source_data`、补充材料、outer、模型训练/推理、远端px2

## 结论

**发现并已在新V5.1中修正6条既有文献的作者错误（[4]、[7]、[8]、[10]、[13]、[15]），补充2条在限制部分被点名但原先未引用的文献（T3PO-mRNA [16]、mRNAutilus [17]），并为17条文献加入稳定来源链接。** 没有改变任何数值、科学结论、证据等级、标题或outer状态。

V5.1最终三轮正文映射和PDF检查均为PASS。所有17条均被正文引用，编号连续，无未定义或未引用条目；PDF为25页，含17个唯一外部URI、24个链接注释，字体全部嵌入且具Unicode映射，无空白页、替换字符、越界文字或越界对象，干净重建字节一致。

**唯一尚未闭合的元数据事项：T3PO-mRNA [16]。** OpenReview论坛和API在当前环境均被人机验证/403阻断，不能诚实声称已从原始页面核实其作者和年份。V5.1仅列题名、submission ID和原始URL，不编造作者或年份；投稿前应在可正常访问OpenReview的环境补齐。这不影响“本文未与该类生成方法进行固定池重排比较”的限制表述。

## V5保留与文件边界

V5原文件未改，最终SHA256与审计开始时一致：

| 文件 | SHA256 |
|---|---|
| `UNIFIED_PREPRINT_MAIN_V5_20261126.md` | `2dd2f28a8d0adf7210d805fc7ad5bbda26d1f829f80b50e2b57c7d51adfc8ac0` |
| `UNIFIED_PREPRINT_MAIN_V5_20261126.pdf` | `98d967d1f4338340750719899b58df495836ca819b1cd097fa00651e4a5d965f` |

新增/写入仅限：

- `deliverables/paper/UNIFIED_PREPRINT_MAIN_V5_1_20261126.md`
- `deliverables/paper/UNIFIED_PREPRINT_MAIN_V5_1_20261126.pdf`
- `deliverables/paper/UNIFIED_PREPRINT_MAIN_V5_2_20261126.md`
- `deliverables/paper/UNIFIED_PREPRINT_MAIN_V5_2_20261126.pdf`
- `deliverables/reports/docs/PREPRINT_REFERENCE_AUDIT_20261126.md`
- `deliverables/reports/docs/PREPRINT_REFERENCE_AUDIT_20261126_THREE_ROUNDS.log`

没有使用远端SSH，没有重开或评分outer，没有改动figures/PPT/`source_data`。

## 第一轮：原始出版元数据与链接核验

核验依据优先级为出版方页面、DOI落地页、PMLR/NeurIPS/arXiv/OpenReview原始记录；Crossref/OpenAlex仅用于定位或交叉核对，不以记忆替代原始记录。内置浏览器在当前子代理未启用，因此使用网页读取和HTTP工具；AutoGLM搜索的本地token服务不可用，未将其失败结果作为证据。

| 编号 | 原始来源与核验结果 | V5→V5.1处理 |
|---|---|---|
| [1] Skalse et al. | [DOI 10.52202/068431-0687](https://doi.org/10.52202/068431-0687)：现DOI页题名为 *Defining and Characterizing Reward Hacking*，作者Joar Skalse、Nikolaus Howe、Dmitrii Krasheninnikov、David Krueger，NeurIPS 35，2022。NeurIPS旧目录显示 *Reward Gaming*，属于版本题名差异；现DOI支持V5现题。 | 保留题名作者年份；补DOI和卷次35。 |
| [2] Trabucco et al., COM | [PMLR v139](https://proceedings.mlr.press/v139/trabucco21a.html)：题名、Brandon Trabucco/Aviral Kumar/Xinyang Geng/Sergey Levine、ICML 2021均匹配。 | 元数据不变；补PMLR URL。 |
| [3] Design-Bench | [PMLR v162](https://proceedings.mlr.press/v162/trabucco22a.html)：题名、Trabucco/Geng/Kumar/Levine、ICML 2022均匹配。 | 元数据不变；补PMLR URL。 |
| [4] GABO/aSCR | [DOI 10.52202/079017-1397](https://doi.org/10.52202/079017-1397)及[NeurIPS原页](https://papers.nips.cc/paper_files/paper/2024/hash/4dd0a016d7d253d02473e4778414ab0b-Abstract-Conference.html)：实际为Michael S. Yao、Yimeng Zeng、Hamsa Bastani、Jacob Gardner、James C. Gee、Osbert Bastani，2024。 | **纠错：**`M. Yao`→`M. S. Yao`；删除错误的`C. Zhang`；补列其余作者为`et al.`；补DOI和卷次37。 |
| [5] Deep ensembles | [NeurIPS 2017原页](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)：题名、Lakshminarayanan/Pritzel/Blundell、2017匹配。 | 元数据不变；补原页和卷次30。 |
| [6] Weinstein et al. | [DOI 10.52202/068431-0396](https://doi.org/10.52202/068431-0396)：题名、Eli Weinstein/Alan Amin/Jonathan Frazer/Debora Marks、NeurIPS 35、2022匹配。 | 元数据不变；补DOI和卷次35。 |
| [7] Reddy et al. | [DOI 10.52202/079017-2954](https://doi.org/10.52202/079017-2954)及[NeurIPS原页](https://papers.nips.cc/paper_files/paper/2024/hash/a9619dd0f0d54a5cf7734add1dc38cd1-Abstract-Conference.html)：实际作者为Aniketh Janardhan Reddy、Xinyang Geng、Michael H. Herschl、Sathvik Kolli、Aviral Kumar、Patrick D. Hsu、Sergey Levine、Nilah M. Ioannidis，2024。 | **纠错：**`Hersch`→`Herschl`；原第4作者直接写S. Levine造成遗漏，改为前三位加`et al.`；补DOI和卷次37。 |
| [8] GFlowNets | [PMLR v162](https://proceedings.mlr.press/v162/jain22a.html)：实际13位作者，第4位为Jarrid Rector-Brooks，2022。 | **纠错：**原列表不完整且`Rector`缺`Brooks`；改为前三位加`et al.`；补PMLR URL。 |
| [9] Sample 2019 | [DOI 10.1038/s41587-019-0164-5](https://doi.org/10.1038/s41587-019-0164-5)：题名、7位列作者、*Nature Biotechnology* 37:803–809、2019匹配。 | 元数据不变；DOI改为显式链接。 |
| [10] Castillo-Hair 2024 | [DOI 10.1038/s41467-024-49508-2](https://doi.org/10.1038/s41467-024-49508-2)：第3作者是Ban Wang，*Nature Communications* 15:5284、2024。 | **纠错：**`P. S. Wang`→`B. Wang`；DOI改为显式链接。 |
| [11] Leppek 2022 | [DOI 10.1038/s41467-022-28776-w](https://doi.org/10.1038/s41467-022-28776-w)：题名、Leppek/Byeon/Kladwang等、13:1536、2022匹配；原始页明确GSE173083及PERSIST-seq。 | 元数据不变；DOI改为显式链接。 |
| [12] Schlusser 2024 | [DOI 10.1186/s13059-024-03369-6](https://doi.org/10.1186/s13059-024-03369-6)：题名、Niels Schlusser/Asier González/Muskan Pandey/Mihaela Zavolan、25:227、2024匹配。 | 元数据不变；DOI改为显式链接。 |
| [13] Inference-time reward hacking | [DOI 10.52202/085713-2064](https://doi.org/10.52202/085713-2064)及[arXiv检索原页](https://arxiv.org/search/?query=%22Inference-time+reward+hacking+in+large+language+models%22&searchtype=title)：实际作者Hadi Khalaf、Claudio Mayrink Verdun、Alex Oesterling、Himabindu Lakkaraju、Flavio Calmon，NeurIPS 38，2025。 | **纠错：**原作者`M. Khalaf, R. Anil, A. Beirami, et al.`不属于该论文；替换为5位实际作者，补DOI和卷次38。 |
| [14] VGSC | [arXiv:2611.06606](https://arxiv.org/abs/2611.06606)及arXiv Atom API：题名、Ekaterina Fadeeva/Xueyan Li/Mrinmaya Sachan/Maxim Panov、2026匹配；v1为2026-11-04。 | 元数据不变；补arXiv URL。 |
| [15] ASPEST | [arXiv:2304.03870](https://arxiv.org/abs/2304.03870)及arXiv Atom API：实际6位作者Jiefeng Chen、Jinsung Yoon、Sayna Ebrahimi、Sercan Arik、Somesh Jha、Tomas Pfister；v1为2023，v3为2024。 | **纠错：**删除不属于该论文的`C. Zhang`，补齐S. Arik、S. Jha、T. Pfister；保留首年2023并补URL。 |
| [16] T3PO-mRNA | [OpenReview KTCInSlPpL](https://openreview.net/forum?id=KTCInSlPpL)：项目已有记录指向题名 *Full-length mRNA Design with Reward-Guided Masked Diffusion Model Fine-Tuning*。本次原始论坛跳转challenge，API为403；ICLR 2026正式论文目录未检出该题名。 | 新增引用，但**不填未核实的作者/年份**；投稿前人工补核。 |
| [17] mRNAutilus | [arXiv:2605.31296](https://arxiv.org/abs/2605.31296)及Atom API：题名、Sawan Patel/Sophia Tang/Yesol Kim等11位作者、2026匹配；v1为2026-05-29。 | 新增前三位加`et al.`、年份和arXiv URL。 |

### 四个生物数据集/研究的针对性核验

- **Sample / GSE114002：**[9]的出版元数据匹配；[10]原始出版页的Data availability也将既有fixed-end N50库指向GSE114002。正文将其用于Sample训练和历史fixed-pool上下文，与来源角色一致。
- **Castillo-Hair / GSE232927：**[10]原始页明确该研究生成数据存入GSE232927，覆盖HEK293T、HepG2、T细胞和gene-editing语境；正文没有把它写成新的前瞻独立复验。
- **Leppek / GSE173083：**[11]原始页明确PERSIST-seq和In-line-seq原始数据在GSE173083。正文将CDS/pooled constructs与`UTR (5') variants`分开，未把预测表达当实测表达，匹配原研究测量范围。
- **Schlusser 2024 HepG2：**[12]原始页讨论reporter与内源系统间的预测限制，并列出RNA-seq/Ribo-seq来源；正文将其限定为MRL代理到内源TE的一次冻结跨终点评价，而非MRL同终点复验或直接蛋白输出，表述匹配。

NCBI GEO直接网页本次触发reCAPTCHA，因此GEO归属由上述出版方Data availability原文核验，而不是把reCAPTCHA页面误作数据记录。

## 第二轮：正文—文献映射与主张匹配

自动解析结果：V5原有15条均被引用；V5.1有17条、31次编号引用（范围展开后），`cited_unique == reference_numbers == [1,…,17]`，无重复、缺失或越界引用。

| 正文位置/主张 | 文献 | 审计判断 |
|---|---|---|
| Introduction/Related work：reward hacking | [1] | 匹配；原文定义优化不完美代理导致真实目标变差。 |
| offline MBO、COM与Design-Bench | [2,3] | 匹配；未把Design-Bench误写成保守方法本身。 |
| source critic regularization | [2,4] | 匹配；[4]确为GABO/aSCR。 |
| deep ensembles与不确定性 | [5] | 匹配。 |
| molecular fitness的misspecification/non-identifiability | [6] | 匹配。 |
| 生物设计中的conservative optimization、uncertainty、diversity | [7,8] | 匹配且表述较宽泛，没有声称本文复现其方法。 |
| 四个5′ UTR/mRNA研究及测量终点 | [9–12] | 匹配；PERSIST-seq正确归到Leppek [11]。 |
| inference-time reward hacking | [13] | 主张匹配；V5作者错误已修。 |
| VGSC是 imperfect generators/verifiers 下的stochastic decoding | [14] | 精确匹配摘要；正文明确本文只是VGSC-inspired固定池falsification pilot，没有声称复现推理任务结果。 |
| ASPEST连接active learning/selective prediction | [15] | 精确匹配摘要；用于区分abstention与label acquisition合理。 |
| COM comparator范围 | [2] | 正文明确为COM-style adaptation，不是官方COM UTR-ResNet或官方权重；没有不当归属。 |
| T3PO-mRNA和mRNAutilus属不同evaluation surface，未与fixed-pool reranking比较 | 新增[16,17] | [17]原始摘要明确支持de novo/full-transcript生成；[16]仅作范围指引，不承担性能主张。限制表述保守、匹配。 |

未发现“引用编号指向另一篇论文”或“文献支持不了正文强主张”的剩余问题。摘要没有依赖编号文献的外部事实性扩张；其全部数值和限制与V5原文一致。

## 投稿文本一致性

- **标题：**保留不变。“helps at equal budget”对应T-cell r1/r2及HepG2 cap 2的限定阳性结果；“but only within…”对应跨上下文差异、组成调整衰减和PERSIST未检出迁移，没有普遍性过度承诺。
- **摘要：**保留全部数值与证据边界不变。回顾性、非前瞻、非因果、PERSIST非盲、abstention校准失败、outer未评分、label repair仅诊断均明确。
- **限制：**保留科学内容，仅给T3PO-mRNA/mRNAutilus补[16,17]。COM非官方复现、FP-TILT非等成本、跨终点非MRL复验、outer锁定等关键边界仍在。
- **Anonymous authors：**在V5.1 draft status明确写成“投稿前占位、需填写作者和单位”，不是技术错误；未把T3PO未知作者伪装成Anonymous。
- **Data and code availability：**V5.1曾明确写无public repository；后续主代理提供并核验了公开分支，证明这一表述已过时。V5.2已修正为公开代码分支`paper23-gates-v1`和构建commit `7cd8521133ecadd8cbd9588375404c2ff66206c0`，同时保留“无release DOI”和完整证据包不公开的边界；详见下方V5.2补充记录。
- **证据路径：**逐一检查了Data availability列出的文档。`paper3_headroom_v4_20261124.md`实际位于`deliverables/reports/`，其余列出文档位于`deliverables/reports/docs/`；V5.1已明确这两个目录，消除路径歧义。

## 第三轮：链接、映射、PDF最终复核

### 三轮执行结果

最终三轮均实际执行同一组17个URL的HTTP GET、编号/主张token映射和PyMuPDF PDF读取；详细逐URL状态见伴随日志。

- 三轮 `MAPPING=True`，均为31次展开引用、17条连续文献。
- 三轮 `PDF=True`：25页、24个链接注释、17个唯一URI、无缺失URI、无缺失必要文本。
- PMLR、NeurIPS 2017、4个生物DOI、3个arXiv链接均返回HTTP 200并到达预期域名。
- 5个NeurIPS DOI均正确302/解析至相应`proceedings.com`条目，但最终 automated GET 返回403；这些条目已通过网页读取工具获得题名、作者、年份和页码，因此记为“目标站点拒绝自动客户端”，不是错误DOI。
- OpenReview [16]返回HTTP 200但实际跳转`/challenge?...`，不能记为论文内容访问成功；其作者/年份保持未核验。
- 早期试跑曾出现个别30秒超时；最终保留的三轮日志中，所有NeurIPS DOI均至少完成到正确落地URL。超时不作为链接失效证据。

### PDF构建与版式

使用既有V5构建器的同一Pandoc→Tectonic路径，仅在内存中把主文source/output切换为V5.1并将页眉改为v5.1；未覆盖V5或重建补充材料。初次构建发现裸URL没有形成PDF链接，已改为显式`<https://…>`后重新构建和重跑检查。

最终V5.1 PDF：

| 检查 | 结果 |
|---|---|
| 文件 | `deliverables/paper/UNIFIED_PREPRINT_MAIN_V5_1_20261126.pdf` |
| SHA256 | `7837277b711948ecb1bb94f137f54857e5833fb894e879b0e31ccc98755a95bc` |
| 页数 | 25 |
| PDF结构/文本 | PASS；无缺失关键文本、无替换字符 |
| 字体 | 全部嵌入、全部Unicode |
| 空白/无文本页 | 0 |
| 页面外文字/对象 | 0/0 |
| 外部链接 | 17个唯一URI、24个链接注释；17条文献URL全部存在 |
| 干净重建 | 字节一致 |
| 科学文本不变量 | 通过；V5→V5.1差异仅draft说明、两条新引用、可用性澄清和参考文献区 |

构建仅有既存字体绝对路径、underfull hbox和输入PDF版本提示；没有overfull、undefined或missing-character严重警告。

## V5.2补充：公开代码可用性修正与三轮复核

### 修正内容

主代理随后确认公开代码分支已经存在。本次独立读取以下两个原始GitHub页面，均访问成功：

- 公开分支：<https://github.com/Cunyu-Liu/proxy-function-gap/tree/paper23-gates-v1>
- 最新publication-build代码commit：<https://github.com/Cunyu-Liu/proxy-function-gap/commit/7cd8521133ecadd8cbd9588375404c2ff66206c0>，页面标题为“Archive V5 publication build and validation code”。

V5.1中“No upload, GitHub push, public repository”因此是已过时且错误的可用性声明。V5.2将其替换为：

- 公开分支`paper23-gates-v1`及上述完整commit哈希；
- 明确`No release DOI has been created`；
- 明确公开分支是code-only archive，不包含完整冻结证据包、原始数据、序列标签、模型权重、PDF、PPTX、缓存或日志；这些仍保留在上述本地/冻结证据存储中。

V5.2 draft status仅增加本次可用性修正说明；参考文献、标题、摘要、所有科学数值、限制、outer状态和其他正文均未改变。V5与V5.1文件均保留。

### 三轮PDF、链接与clean rebuild结果

三轮均重新执行19个唯一URL的HTTP GET、原PDF完整检查和临时空目录干净重建；每轮的重建PDF也重新通过检查。

| 检查 | 第1轮 | 第2轮 | 第3轮 |
|---|---|---|---|
| GitHub公开分支/commit | 200/200 | 200/200 | 200/200 |
| 原PDF检查 | PASS | PASS | PASS |
| 重建PDF检查 | PASS | PASS | PASS |
| clean rebuild字节一致 | 是 | 是 | 是 |
| 引用映射 | 17条、31次展开引用，PASS | 同左 | 同左 |

三轮共同结果：25页；19个唯一URI、28个链接注释；无空白页、替换字符、越界文字或越界对象；字体全部嵌入且支持Unicode；必要公开分支、commit、无release DOI和code-only限制文本均存在；旧“无公开仓库”表述均不存在。5个NeurIPS DOI仍解析到正确条目后由自动客户端收到403，OpenReview仍跳转challenge，均沿用前述访问限制判断，不是V5.2新增问题。

| 文件 | SHA256 |
|---|---|
| `UNIFIED_PREPRINT_MAIN_V5_2_20261126.md` | `b717f77ccab5f149cd0cbbade85ebc1136d94b3810dc5649058cc5cbc2e3f82a` |
| `UNIFIED_PREPRINT_MAIN_V5_2_20261126.pdf` | `142fb7744a32fa01b61c342ff4798055e74673cef2ec587b685c43805cfde47d` |

本次仅新增V5.2主文MD/PDF并更新本报告；没有修改V5/V5.1、既有三轮日志、figures、PPT、`source_data`或其他文件。

## 投稿前剩余动作

1. 在可正常通过OpenReview人机验证的环境打开[16]，补齐T3PO-mRNA的作者、年份和最终发表状态；在此之前不要把该条写成已正式发表。
2. 将主文YAML和页眉中的`Anonymous authors`替换为真实作者与单位；这是正常投稿步骤，不是本次科学或文献技术错误。
3. 公开代码分支已经存在并在V5.2中列明；若目标期刊要求公开数据或正式归档，仍需另行提供相应资源或创建release DOI。当前V5.2仅明确没有release DOI，不再声称无公开仓库。
