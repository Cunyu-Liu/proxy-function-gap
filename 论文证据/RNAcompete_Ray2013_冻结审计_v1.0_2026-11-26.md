# RNAcompete Ray2013 冻结审计 v1.0

日期：2026-11-26 ｜ 状态：**数据审计与 split 冻结 accepted；未训练、未跑模型、无性能结果**

## 1. 冻结身份与产物

- 版本：`rnacompete-ray2013-v1.0-id90-strict3train`。
- 预注册：[RNAcompete Ray2013 数据合同](../预注册/RNAcompete_Ray2013_预注册数据合同_v1.0_2026-11-26.md)。
- 冻结生成提交：`7b91436`；`split_manifest.json` SHA256=`868524abd1f9c76501c8c8212b6bc531290796c267bc898b45047c7bede39f43`。
- 正式目录：`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/external_benchmarks/rnacompete/rnacompete-ray2013-v1.0-id90-strict3train/`。
- 代码：`code_rareseq/rnacompete_ray2013_freeze.py`、`rnacompete_ray2013_contract_test.py` 和 `run_rnacompete_ray2013_freeze.sh`。大型序列、cluster、标签审计和 manifest 均在 `/teams`，未写入仓库。

## 2. 实验与标签对齐

- 三份原始文件的 bytes/SHA256 全部匹配预注册；主表 244 个 `Motif_ID` 与 normalized 244 个实验列双向完全对应，覆盖 205 个物种×基因、24 物种。
- 241,357 个矩阵 Probe_ID 全部是设计文件子集；设计 RNA、Probe_Set 与矩阵逐行一致。主任务含 241,357 个 SetA/B 探针、219,990 条唯一 RNA。
- 旧 `norm_data_audit.json` 的 `design_ids_not_matrix=102` 有算术和连接错误。独立重算为 **42**，恰好全部是缺失于矩阵的 `RBD_Control_*` 非 SetA/B 探针（21 条唯一 RNA）；SetA/B 缺失数为 0。旧原始审计文件未覆盖，新 manifest 的 `alignment` 字段和 `excluded_probes.tsv` 为修正证据。
- normalized 共 58,103,945 个有限强度、787,163 个 NaN、0 空字段、0 Inf。原始字段存在前导空格 NaN，loader 与完整冻结均用 NumPy 浮点转换保持其 NaN 语义；任何不可转换内容都会报错。
- NaN 不填零、不作负例、不进入频率或指标分母。每个实验×唯一 RNA 取全部有限探针强度中位数，全 NaN 序列保持未观测。

## 3. 重复探针与跨实验共享

每个实验均有 21,367 条重复唯一 RNA。全部 244 个实验均存在有限重复探针强度不完全一致的情况；逐实验重复极差中位数的中位数为 0.51048，全局最大极差为 6702.827024。该离散程度被完整保留在 `duplicate_audit.tsv`，只执行预注册的等权序列中位数对齐，**未做离群删除、截断、阈值化或标签清洗**。

全部 219,990 条 RNA 在至少两个实验有有限观测，65,532 条在全部 244 个实验有有限观测。这是共享探针设计的事实：闭集序列轨用全局 fold 防止跨 split 泄漏；RBP-OOD 中跨 RBP 共享序列是任务设定，不能据此声称序列 OOD。

## 4. 冻结 split 与泄漏验收

CD-HIT-EST 4.8.1，参数 `-c 0.9 -n 8 -G 1 -g 1 -r 1 -aL 0.9 -aS 0.9 -d 0`。219,990 条唯一 RNA 形成 219,985 个簇；簇独占分配到五折。

| 协议 | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---:|---:|---:|---:|---:|
| 序列轨唯一 RNA | 43,998 | 43,998 | 43,998 | 43,998 | 43,998 |
| RBP-OOD 实验数 | 49 | 49 | 49 | 49 | 48 |

每轮 test=`f`、validation=`(f+1) mod 5`、train 显式枚举其余三折。RBP-OOD 以物种×`Gene_systematic_ID` 整组分配，同一基因不同构建/条件不拆组。

`leakage_audit.tsv` 共 31 行：序列五折三种角色两两比较、RBP 五折三种角色两两比较及 SetA/SetB 检查，精确 RNA、cluster、物种×基因组的跨角色重叠最大值均为 **0**。增强复读另核对 probe→sequence→fold 对齐、SetA+SetB=All、NaN+有限=总数、逐实验五折频率总和和 RBP 组互斥，全部通过。

## 5. 频率分布与评价入口

244/244 实验均满足每折至少 100 个有限唯一 RNA 且全数据至少两个不同中位数标签，无实验排除。频率只按有限唯一 RNA 数排序，head/mid/tail 为 82/82/80。

| 层级 | 实验数 | 最小 | 中位数 | 最大 |
|---|---:|---:|---:|---:|
| head | 82 | 219,940 | 219,972 | 219,990 |
| mid | 82 | 219,023 | 219,795 | 219,940 |
| tail | 80 | 124,518 | 218,980 | 219,021 |

后续每实验×fold×All/SetA/SetB 在唯一 RNA 上报告 Spearman、Pearson、MSE，再宏平均；Overall/head/mid/tail 分层，RBP-OOD 同时给实验和基因宏平均。本版不创建二分类阈值或 AUROC/AUPRC 结果，不把 7-mer E-score 混入探针强度。

## 6. 验证记录

- 本地与服务器均通过 `RNACOMPETE_CONTRACT_TEST_OK`：重复中位数、前导空格 NaN、全 NaN 不填补、严格三折、频率分层、cluster 解析和 loader。
- 全量冻结输出 `RNACOMPETE_FREEZE_ACCEPTED`，244 实验，耗时 253.0 秒。
- 独立复读输出 `RNACOMPETE_FROZEN_VERIFICATION_OK`：14 个受 SHA 管理的文件（另含 SHA256SUMS）、244 实验、219,990 RNA、219,985 簇全部通过。
- 真实 loader 抽查 `RNCMPT00100`：219,856 个有限唯一 RNA，与 `experiments.tsv` 一致；重复 RNA `c8130670…4583d5` 的探针 `RBD_v3_000001/RBD_v3_114782` 原始有限值为 1.122218/1.317084，loader 中位数 1.219651 精确匹配。

任何后续训练/验证必须另按合同显式 CUDA、strict3train、validation-only `patience=20` 和最佳 checkpoint 恢复。本次 accepted 只表示数据与划分可复现，不是模型性能、外部泛化、SOTA 或机制因果证据。
