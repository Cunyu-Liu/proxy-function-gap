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
- 独立nohup状态与日志为`/home/ProxyFunctionGap/logs/rnacompete_ray2013/`；当前fold0驱动PID `4025704`。驱动先完成共享嵌入和validation选择，再执行fold0三seed；默认不自动扩五折。
- 固定RNA-FM权重已组装并核验：398,175,831字节，SHA256 `a0ff096b19510a0d2ef020c171611ce592a527a829c1a4ba29ffffe4422c35d4`。
- **真实模型CUDA smoke通过并已独立重放**：RTX 5090，`cuda:0`数值/反向/优化器探针通过；16条冻结RNA的ID逐字匹配，`16×640`、fp16、全有限值，最大token长度40。ID SHA256为`d38c7e9db32c761bec0a356444f493b0df601bdd5211fe8510cf5cd0a657847a`，嵌入SHA256为`15af8c867815d9aa401c21e3d5e759fe8c1538e86cd662f65e781aebee412f14`。证据：`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/rnacompete_ray2013_model_baselines_v1/smoke/smoke_cuda.json`。
- smoke于00:39 +08:00完成。此后将资源门控视为工程校准而非科学协议：保持batch、模型、损失和指标不变，将正式嵌入及小型头/检索门限调整为2.5GB，并新增CUDA peak记录；未抢占或停止其他模型任务。
- **正式共享嵌入已完成并独立验收**：00:58 +08:00生成219,990×640 fp16缓存，所有ID与冻结`sequence_folds`逐字一致、所有行有限。ID SHA256 `977ba15917446585fc790fa470ad7b4d9c8d1e6987d783330ce869771e711019`，嵌入SHA256 `4e697e2d0031ed137fb42b68ff7d2447b93dea38af5e8b9b8f4a4743deea8b32`。实测CUDA peak allocated 618,245,632B、reserved 809,500,672B，正式前向耗时178.77秒。
- fold0/seed0序列linear正则搜索已完成并独立审计：weight decay `1e-4/1e-2/1.0`的best/stopped epoch分别为`14/34、14/34、23/43`，validation宏Spearman分别为`0.4136545/0.4137150/0.4174495`，宏MSE分别为`36.7700205/36.7689095/36.6969891`。最终冻结weight decay `1.0`，selection SHA256为`c702d3e5bddf3e7fd3261ff4ddb6575291bb919327422e3f051cafc5d723ac2a`。三候选均满足patience20、配置`test_read_during_training=false`且无test预测产物。以上只证明validation选择协议，不是test或五折性能。
- 全部6个选择已冻结：sequence linear/MLP weight decay均1.0、sequence kNN k20；RBP-OOD linear weight decay1e-4、MLP weight decay1.0、RBP kNN k1。统一`audit_selection_boundaries`确认6/6均`test_used=false`、selection fold0、selection树test产物0。序列MLP第三候选best/stop为80/100，validation宏Spearman0.4590587；RBP linear选择validation宏Spearman0.1085603，均只用于选择。
- kNN工程修复只处理缺失标签：在最多256个train近邻中逐任务取最多k个有限标签邻居，检索batch64避免广播OOM；预测有限性只在有真值的位置强制，NaN真值不进入指标。失败尝试未生成kNN selection或test产物。fold0/最终门禁另强化为selection SHA核验、history/patience核验、独立CUDA checkpoint重算test并与保存预测比对。
- fold0正式训练于03:39启动。首个`sequence_linear_f0_s0`已于03:57完成train/evaluate并独立验收：732行逐实验指标重算一致，43,998×244 test ID正确，checkpoint CUDA重算与保存预测一致，best/stop `23/43`，selection SHA256 `c702d3e5bddf3e7fd3261ff4ddb6575291bb919327422e3f051cafc5d723ac2a`匹配。单run tail（80实验）Spearman/Pearson/MSE为`0.4273671/0.3601801/12.2443230`，仅证明该run产物可重放，**不是三seed或五折总结**。驱动随后进入sequence linear seed1；fold0完整门禁尚未形成，未启动五折。
- `sequence_linear_f0_s1`已完成并通过同样独立重放：732行指标一致、checkpoint CUDA预测一致、best/stop `9/29`，selection SHA匹配。单run tail Spearman/Pearson/MSE为`0.4271312/0.3602961/12.2981157`，不与seed0混写为三seed结论。seed2 train已完成（best/stop `16/36`，validation宏Spearman `0.4165756`），但evaluate因1,553MiB空闲低于2,000MiB门限保持等待，尚无seed2 test产物。

## 6. fold0门禁与五折扩展（非五折结果）

fold0门基于05:42通过，12个神经run和2个kNN评价全部完成；统一selection零test审计、history/patience、selection SHA、独立CUDA checkpoint预测及732行/逐RBP指标重放均通过。`all`驱动PID `4041649`于05:45启动并再次通过fold0门禁，05:46开始fold1 sequence linear seed0，未重复fold0训练。

完整fold0独立汇总保存于`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/rnacompete_ray2013_model_baselines_v1/fold0_independent_summary.tsv`，SHA256 `a8d34192f662a0fa1d079d10b0cfdcafcc3ebbde4cba0bafda0362b3a6cdd616`。神经结果为三seed mean±SD；kNN为单次确定性run。

| 轨/切片 | 模型 | Spearman | Pearson | MSE |
|---|---|---:|---:|---:|
| 序列 Overall | linear | 0.417303±0.000395 | 0.340733±0.000330 | 64.468618±0.033255 |
| 序列 Overall | MLP | 0.459187±0.000468 | 0.405052±0.000861 | 62.573564±0.091634 |
| 序列 Overall | kNN | 0.342253 | 0.250568 | 66.544919 |
| 序列 tail | linear | 0.427857±0.001060 | 0.360603±0.000635 | 12.266298±0.028215 |
| 序列 tail | MLP | 0.475213±0.001142 | 0.426579±0.000657 | 11.734165±0.007426 |
| 序列 tail | kNN | 0.352484 | 0.263434 | 13.290196 |
| RBP-OOD实验宏 | linear | 0.069643±0.004631 | 0.067406±0.004816 | 22.468351±0.058328 |
| RBP-OOD实验宏 | MLP | 0.085236±0.007883 | 0.085477±0.007518 | 22.341350±0.032320 |
| RBP-OOD实验宏 | kNN | 0.168446 | 0.129412 | 22.625864 |

RBP-OOD物种×基因宏Spearman为linear `0.064375±0.004697`、MLP `0.078851±0.007741`、kNN `0.156648`（40组）。这些仅是完整fold0门禁结果；fold1–4正在运行，**不得解释为五折性能、稳定外部泛化或SOTA结论**。

扩展首个`sequence_linear_f1_s0`已完成并独立重放：732行指标一致、43,998×244 ID正确、checkpoint CUDA预测一致，best/stop `45/65`，沿用fold0冻结selection SHA。该单run不与fold0或其他fold混写；驱动已进入fold1 seed1，最终门禁和五折汇总尚未形成。
- `sequence_linear_f1_s1`随后完成并独立重放通过：732行指标、ID、checkpoint CUDA预测及selection SHA一致，best/stop `16/36`。05:53进度为14/60神经eval、2/10 kNN，fold1 seed2训练中；最终门禁不存在，五折汇总未生成。
