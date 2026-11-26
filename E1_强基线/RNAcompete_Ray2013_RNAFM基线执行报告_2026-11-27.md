# RNAcompete Ray2013 冻结 RNA-FM 模型基线执行报告

日期：2026-11-27 ｜ 数据：`rnacompete-ray2013-v1.0-id90-strict3train`

> 当前交付是预注册、独立可执行实现、工程测试与运行门控。完整fold0三seed和五折矩阵未验收前，不作模型性能、外部泛化或SOTA结论。

## 1. 合同与代码

- 预注册：`预注册/RNAcompete_Ray2013_模型基线预注册合同_v1.0_2026-11-26.md`。
- 共享嵌入：`code_rareseq/rnacompete_ray2013_rnafm_embed.py`。
- 标签缓存、双轨linear/MLP/kNN、validation选参、独立test评价、指标重放与汇总：`code_rareseq/rnacompete_ray2013_baselines.py`。
- 合成合同测试：`code_rareseq/rnacompete_ray2013_baselines_test.py`。
- 独立nohup资源门控驱动：`code_rareseq/run_rnacompete_ray2013_baselines.sh`。

冻结编码器为`multimolecule/rnafm` revision `7d6e73ad3b48e042b378f9a788a56ccb4d573a27`，640维mean-masked表示。连续强度采用masked MSE；NaN不填值、不作负例。序列ID90轨使用244任务输出，整RBP-OOD轨使用可零样本迁移的共享强度头，二者分开报告。

## 2. 选参与矩阵边界

fold0/seed0仅以validation宏Spearman选择weight decay或k，早停固定`patience=20`、`min_delta=1e-4`、1000 epoch故障保险。选择文件冻结后才允许独立evaluate读取test。神经模型规划五折×seed0/1/2；默认驱动只执行fold0三seed并验收，`all`模式在fold0门控通过后才扩fold1–4。kNN确定性执行，不伪造三seed重复。

## 3. 已完成工程验证

本地及远端`/root/miniconda3/envs/lucaone_cu128/bin/python`均通过：

```text
python3 -m py_compile rnacompete_ray2013_rnafm_embed.py rnacompete_ray2013_baselines.py
bash -n run_rnacompete_ray2013_baselines.sh
RNACOMPETE_MODEL_CONTRACT_TEST_OK: NaN mask, metrics, split boundaries, IDs, replay, ties
```

合成测试覆盖：有限标签masked MSE与全NaN batch阻断；Spearman/Pearson常数NaN；严格三个train fold；RBP组不相交角色；共享头预测列对齐；冻结sequence/experiment ID；非有限嵌入阻断；test逐实验指标独立重放；正则/k并列规则。测试还发现并修复了ID object数组与`allow_pickle=False`不兼容，正式产物改用显式Unicode ID。

## 4. 远端执行与证据边界

- 代码：`/home/ProxyFunctionGap/`；大型模型快照、嵌入、标签、checkpoint、预测和日志：`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/rnacompete_ray2013_model_baselines_v1/`。
- 驱动使用独立锁、每阶段GPU/RAM门控，资源不足等待；不抢占RiNALMo、LoRA/full或其他任务。仅停止过本任务冗余的HF下载器，未停止任何其他模型任务。
- HF直连不可用后，模型从镜像固定revision保存为独立`/teams`快照；嵌入manifest记录来源、revision与权重SHA256。此处理不改变模型或数据协议。
- 真实CUDA smoke、正式共享嵌入、fold0三seed与五折结果状态见下节；数据准备、缓存或工程测试均不当作性能证据。

## 5. 当前运行状态

- 唯一RNA标签缓存已真实生成并验收：形状`219990×244`，52,995,625个有限值、681,935个去重后未观测值；逐实验有限计数与冻结审计一致。SHA256为`ff4ed845881c885899bb75c9f1772ab35b45d1fb8878aeb526b73c55a31a786e`。这是数据产物，不是性能。
- 合成CUDA训练头门控已在RTX 5090通过：masked-MSE反向、21 epoch早停（best=1）、244任务validation指标和checkpoint/预测落盘。首次运行发现NaN占位顺序错误并修复，未接受失败产物；该合成run只证明执行链。
- 独立nohup驱动已于00:23 +08:00启动，PID `4008443`；状态与日志为`/home/ProxyFunctionGap/logs/rnacompete_ray2013/`。驱动当前等待固定快照完整，之后依次执行真实模型CUDA smoke、6GB空闲显存门控的正式共享嵌入、validation选参与fold0三seed；默认不自动扩五折。
- 固定RNA-FM权重已组装并核验：398,175,831字节，SHA256 `a0ff096b19510a0d2ef020c171611ce592a527a829c1a4ba29ffffe4422c35d4`。
- **真实模型CUDA smoke通过并已独立重放**：RTX 5090，`cuda:0`数值/反向/优化器探针通过；16条冻结RNA的ID逐字匹配，`16×640`、fp16、全有限值，最大token长度40。ID SHA256为`d38c7e9db32c761bec0a356444f493b0df601bdd5211fe8510cf5cd0a657847a`，嵌入SHA256为`15af8c867815d9aa401c21e3d5e759fe8c1538e86cd662f65e781aebee412f14`。证据：`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/rnacompete_ray2013_model_baselines_v1/smoke/smoke_cuda.json`。
- smoke于00:39 +08:00完成后，正式共享嵌入进入6GB空闲显存门控；首次等待记录为4,525MiB空闲、约54.8GB RAM可用，未降低门限或抢占其他任务。驱动继续存活，后续将自动执行正式嵌入、validation选参与fold0三seed。
- 当前尚无可报告test指标；smoke、标签缓存、合成训练门控均未被当作性能结果。
