# RNAcompete Ray2013 预注册数据合同 v1.0

日期：2026-11-26 ｜ 数据版本：`ray2013_nature12311` ｜ benchmark 版本：`rnacompete-ray2013-v1.0-id90-strict3train`

> 本合同在任何 RNAcompete 序列聚类、实验纳入筛选、split 或模型训练之前冻结。当前交付仅为数据审计、任务定义与 split 冻结，不是模型结果。合同变更必须升级版本并保留旧 manifest，不得根据 test 表现回改。

## 1. 固定输入与证据边界

远端只读输入根目录：

`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/ray2013_nature12311/`

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `SupplementaryData1_RNAcompete_master_file.xlsx` | 2,677,507 | `3da82ac06deae33264ff2d49f83cd5302dc9d12ccb559032b84d93fdaf42f261` |
| `RBD_v3_design_unified.txt` | 47,537,452 | `47abcc7272ad43ec8b0bcb210d3c1c4529e986ba19ea45a6638617eebd5e536a` |
| `norm_data.txt.gz` | 249,998,462 | `efa7fade5fe6c3a189ece3f381550f73f3bb3d935598d211d69d29d9e3ae47ff` |

任务使用 Ray et al. 2013（DOI `10.1038/nature12311`）官方归一化探针强度，标签是体外短 RNA–RBP 连续结合强度。不得称为 eCLIP 二分类、体内结合位点或 log-odds。7-mer E-score 是另一目标层级，本版不与探针强度混合。公开下载不代表具有任意再分发许可；冻结产物留在 `/teams`，仓库只提交代码和小型汇总。

## 2. 实验、探针和标签对齐

1. 主表第二工作表的一行是一个实验，以 `Motif_ID` 连接 normalized 矩阵实验列；不以列位置或行号猜测连接。244 个矩阵实验必须与 244 个主表 ID 双向一一对应。
2. 设计文件首行是 10 列控制行 schema；`ControlType=false` 的 14 列非对照行追加 `DNA_Seq/RNA_Seq/Structure/MFE`。只纳入 `ControlType=false`，阳性/阴性控制不进入模型任务。
3. 以 `Probe_ID`（矩阵）= `ID`（设计）连接，RNA 输入取设计的第 12 列 `RNA_Seq`。矩阵和设计中的 Probe_Set、RNA_Seq 必须逐行一致；不一致即阻断，不自动修复。
4. 主任务只含 `SetA/SetB` 且存在于 normalized 矩阵的探针。不属于这两个集合的 42 个非对照探针（21 条唯一 RNA）与 60 个缺失于 normalized 的 SetA/B 探针分别以 `not_SetA_B/absent_from_normalized` 原因进入 `excluded_probes.tsv`，不得随机分配集合、填补标签或删除后隐匿不计。
5. RNA 统一为去首尾空白的大写 `ACGU`；只允许为聚类工具临时将 `U` 写成 `T`，输出仍保存 RNA。出现其他非空字符、空序列、重复 Probe_ID 或重复实验 ID 即阻断，不推测修正。
6. 实验初始候选为全部 244 个对齐实验。划分后，某实验若在任一 fold 的 train/validation/test 角色中有限、去重 RNA 标签数 `<100`，或全数据有限标签不足 2 个不同值，则从**闭集序列轨**排除并记录原因；不得按模型分数、 motif 是否已知或标签方向筛选。RBP-OOD 另要求 `Species` 和 `Gene_systematic_ID` 非空，缺失即阻断而非新建猜测基因。

## 3. NaN 与重复探针

- normalized 中 `NaN` 只表示该实验×探针未获得有限强度；原始资料不足以把每个缺失进一步解释为技术失败或未结合。NaN 不填零、不填列均值、不当作负例，也不参与中位数、频率或指标分母。`+/-Inf`、空字段或解析失败与 NaN 分开计数并阻断冻结。
- 同一实验内相同 RNA 的重复探针不是独立生物重复。模型输入和评价单位均为唯一 RNA：该 RNA 的全部有限探针强度取**中位数**作为连续标签；同时保留 `n_probe_rows/n_finite/n_nan` 和重复探针极差审计。该规则是等权序列评价的对齐规则，不是异常值清洗；不过滤离群探针、不截断标签、不自行发明阳性阈值。
- 若同一序列在同一实验没有有限值，该实验×序列保持未观测，不生成标签行。设计中的 Structure/MFE 仅可作协变量，不作为结合标签。

## 4. 同源聚类与闭集序列五折

1. 主任务 SetA/B 的全部唯一 RNA 先做精确去重，再以 CD-HIT-EST 聚类：`-c 0.9 -n 8 -G 1 -g 1 -r 1 -aL 0.9 -aS 0.9 -d 0`，线程/内存仅影响运行，记录实际软件版本和完整命令。双向比较及双侧覆盖阈值是保守泄漏控制，不声称等价于生物学同源家族真值。
2. 一个簇只能属于一个 fold。按簇内唯一 RNA 数降序、固定种子 `20261126` 的随机值破同分，依次分给当前序列数最少的 fold；fold 数固定为 5。未聚类、未映射或 `fold=-1` 均阻断，不进入任何 split。
3. 第 `f` 轮：`test=f`，`validation=(f+1) mod 5`，`train` 显式枚举剩余三个 fold。训练和验证都不得使用 test 标签。所有实验共享相同序列 fold，避免同序列或越阈值近同源跨 split。
4. 每折自动核验：唯一 RNA 在五个 fold 中恰好出现一次；train/val/test RNA、精确序列和 cluster 两两不相交；train fold 恰好三个。另输出原始重复探针和跨实验共享序列审计，不把技术重复数量当作有效样本量。

## 5. 整基因留出 RBP-OOD 五折

- 留出单位是 `Species + Gene_systematic_ID`，同一基因的不同构建、质粒、杂交和条件整组留出。固定种子 `20261126`，以组内实验数作负载、组大小和随机值破同分，贪心分配到 5 个负载尽量均衡的 fold。
- 第 `f` 轮同样显式使用三个 train fold、一个相邻 validation fold 和一个 test fold。核验三个角色物种×基因组两两不相交。
- 此轨要求可跨 RBP 学习/迁移的模型；独立单实验模型不得用于声称未见 RBP 预测能力。不同 RBP 共用探针序列是本任务的设定，不将 RBP-OOD 描述为序列 OOD；闭集序列轨与 RBP-OOD 不得混为同一证据。

## 6. 频率、长尾切片与预定指标

- 实验频率定义为全数据中至少有一个有限探针强度的唯一 RNA 数 `n_finite_unique`，不是原始探针行数，也不是 NaN 填补后的数量。
- 在闭集序列轨合格实验内按 `n_finite_unique` 降序、`Motif_ID` 升序确定秩；前 `ceil(N/3)` 为 head，随后 `ceil(N/3)` 为 mid，其余为 tail。该切片只由频率决定，在模型运行前冻结。
- 每个实验×fold×集合（All/SetA/SetB）在唯一 RNA 上报告 Spearman、Pearson 和 MSE，再对 fold、实验宏平均；同时报告有限评价单位数。Spearman/Pearson 在常数真实值或常数预测时记 NaN 并报告有效实验数，不改成 0 或 1。
- 总结按 Overall/head/mid/tail 分层；SetA/B 分开报告，不以样本量加权掩盖尾部实验。RBP-OOD 先逐实验计算，再分别给实验宏平均和物种×基因宏平均。可附簇内最近邻 identity 分层，但必须另报各层支持数，不能与总体直接横比。
- 本版不创建 AUROC/AUPRC 分类结论，不将连续强度任意阈值化。SetA→SetB/SetB→SetA 可作为后续独立版本，但本版只把 SetA/B 作为分层，不用 test 选参。

## 7. 阻断条件与冻结产物

以下任一情况必须使脚本非零退出并保留审计日志：原始 SHA 不符；实验/探针 ID 或 RNA 不一致；非法序列；`Inf` 或空标签字段；非 SetA/B 行数不符；聚类映射不完整；fold 重复/遗漏；跨 split 精确序列或 cluster 泄漏；RBP 组跨角色；输出目录已含不同版本的正式 manifest。不得跳过检查后写成 accepted。

冻结输出根目录：

`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/rnacompete-ray2013-v1.0-id90-strict3train/`

小产物包括：`experiments.tsv`、`probes.tsv.gz`、`excluded_probes.tsv`、`sequence_folds.tsv.gz`、`rbp_folds.tsv`、`experiment_frequency.tsv`、`duplicate_audit.tsv`、`label_availability.tsv.gz`、`leakage_audit.tsv`、`split_manifest.json`、`SHA256SUMS` 和运行日志。大型 cluster 临时文件也在上述 `/teams` 目录，不写入仓库。`split_manifest.json` 显式枚举五轮 train/val/test fold、软件版本、输入/输出哈希、排除原因和验收状态；SHA256SUMS 覆盖全部正式文件。

## 8. 后续训练约束（本任务不执行）

任何后续模型训练/验证必须显式选择并证明 `cuda`，禁止 CPU 静默降级；沿用五折 strict3train、validation-only 早停、`patience=20`、`min_delta=1e-4` 并恢复最佳 checkpoint。主回归早停指标须在训练脚本运行前另定，不能查看 test 后更换。数据冻结、审计表和 split 验收不构成模型性能、外部泛化、SOTA 或机制因果证据。
