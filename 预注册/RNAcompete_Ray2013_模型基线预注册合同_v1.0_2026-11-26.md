# RNAcompete Ray2013 冻结 RNA-FM 模型基线预注册合同 v1.0

日期：2026-11-26 ｜ 数据：`rnacompete-ray2013-v1.0-id90-strict3train` ｜ 状态：**模型运行前冻结**

> 本合同继承 `RNAcompete_Ray2013_预注册数据合同_v1.0_2026-11-26.md`，不修改其输入、标签、聚类、实验纳入、频率层或五折角色。数据审计、嵌入 smoke 与工程门控都不是模型性能。任何test评价后不得回改本版配置；必要变更必须升级版本。

## 1. 目标、表示与标签

- 编码器固定为 `multimolecule/rnafm`，冻结全部参数，隐层维度必须为640。tokenizer使用官方特殊符号和padding；序列完整输入，不截断。对 `attention_mask=1` 的token隐状态求均值（mean-masked，包含特殊token、排除padding），autocast bf16，缓存fp16。记录模型解析revision、代码/输入SHA256、CUDA设备及数值探针。
- 共享缓存严格按冻结 `sequence_folds.tsv.gz` 的 `sequence_id` 顺序保存；逐批及全文件检查有限值。smoke与正式缓存物理分离，正式缓存缺行、ID错位、非640维、非有限值均阻断。
- Ray2013 normalized探针强度是连续回归标签。同一实验×唯一RNA的有限探针取中位数；NaN仅为未观测，不填零/均值、不当负例，也不进入损失或指标分母。主损失为观测元素上的masked MSE。
- 闭集序列轨逐任务仅用train标签计算标签均值/标准差，训练在标准化空间进行，预测后逆变换；常数或不足两个训练标签阻断。整RBP-OOD轨只用train RBP的全部有限标签计算一个全局均值/标准差，绝不使用validation/test RBP标签做归一化。

## 2. 两条证据轨

### 2.1 序列ID90严格三折训练轨

每折沿用冻结角色：三个train fold、一个相邻validation fold、一个test fold。模型以640维向量为输入，一次输出244个实验强度；每个batch只对其有限标签计算masked MSE。频率head/mid/tail只在汇总时切片，主选参使用全部244实验的validation宏Spearman，不按test或tail表现选参。

模型：
1. `linear`：640→244线性多任务头。
2. `mlp`：640→256→GELU→dropout0.1→244浅层多任务头。
3. `knn`：训练序列嵌入L2归一化后的余弦检索；在最多256个最近序列中，逐实验取最多k个最近的有限标签邻居，再对其标签取相似度加权平均。权重为 `max(cosine,0)^2`；若全为零退化为等权。NaN邻居不占据k个有限邻居名额、不进入权重分母；256个候选内没有有限邻居标签时输出NaN，若该位置本身有有限评价标签则阻断正式评价，未观测标签位置不参与指标。该masked近邻规则只处理缺失标签，不使用validation/test标签选择序列。

### 2.2 整RBP-OOD轨

按冻结 `Species + Gene_systematic_ID` 五折整组留出。train/validation/test分别使用对应RBP实验，序列可共享，这不是序列OOD。为使未见RBP可预测，本轨两种神经模型均使用**所有train RBP共享的单一强度输出头**，不是训练后随机初始化的244个任务专属输出：

- `linear`：640→1；`mlp`：640→256→GELU→dropout0.1→1。
- 一个序列batch对全部train RBP标签广播同一预测，按有限标签masked MSE训练。validation/test逐未见RBP评价，不使用其标签训练或标准化。
- `knn`先从train序列的每个train RBP取有限标签，再对train RBP宏平均得到该序列的共识强度，缺失仍跳过；随后按同一余弦规则检索。该共识对所有目标RBP相同，是零样本对照，不声称任务特异预测。

两条轨分开训练、选参、评价和报告；不得把整RBP-OOD结果与序列ID90结果混为一个平均值。

## 3. validation-only选参与早停

- 神经模型固定AdamW、lr=`1e-3`、batch=`1024`、gradient clip=`1.0`；每个架构×轨在fold0/seed0仅搜索weight decay `{1e-4,1e-2,1}`。fold0/seed0搜索目录不生成test预测。
- 每epoch在validation All序列上逐实验计算Spearman，再宏平均。主早停/选参指标为该宏Spearman；常数真实值或预测产生NaN时该实验不进入相关均值，同时记录有效实验数。所有实验均无效、训练损失或预测非有限均阻断。
- `patience=20`、`min_delta=1e-4`，最多1000 epoch故障保险；到达1000 epoch视为失败而非accepted。恢复validation最佳checkpoint。正则并列时依次取validation宏MSE较小者、weight decay较小者；epoch并列保留最早最佳epoch。
- kNN在fold0仅以validation宏Spearman选择 `k∈{1,5,20}`；并列依次取validation宏MSE较小者、k较小者。某候选若在具有有限validation标签的位置产生无有限邻居标签的NaN，则该候选标记无效且不得冻结；只有观测位置全有限的候选参与比较，最终test在有限标签位置出现NaN仍阻断。kNN无随机训练，不伪做三seed重复。
- 每个轨×架构的fold0/seed0选择结果写入独立selection JSON并冻结。其后所有fold/seed只使用该配置；每个fold仍由自身validation早停选epoch。fold扩展前必须完成fold0三seed train/evaluate和独立重放门控。
- test只能由独立`evaluate`/`knn-evaluate`命令在checkpoint与selection冻结后读取，绝不参与正则、epoch、k、early stopping或模型选择。

## 4. 五折三seed矩阵与运行门

- 每个神经轨×架构：fold0–4 × seed0/1/2，共30个最终run；两轨两架构共60个最终run。seed只改变初始化与batch顺序，不改变冻结split。
- 第一阶段只运行fold0三seed（12个最终神经run）及两轨确定性kNN。全部ID、有限值、配置哈希、早停和指标重放通过后才允许fold1–4扩展。
- 嵌入、准备标签、选择、训练、评价、验收分别记录。正式模型训练与推理显式`cuda:0`，数值/设备探针失败即非零退出，禁止CPU静默降级。
- 外部执行使用独立nohup和GPU/RAM门控，不kill、不抢占既有RiNALMo、LoRA/full或其他任务；资源不足保持等待。代码在`/home`，大型标签、嵌入、checkpoint、预测和日志在`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/rnacompete_ray2013_model_baselines_v1/`。

## 5. 指标、汇总与验收

- 每个实验×fold×split×`All/SetA/SetB`在有限唯一RNA上报告Spearman、Pearson、MSE和n。Spearman/Pearson遇到常数真实值或预测记NaN并保留有效实验计数；MSE始终只在有限标签上计算。
- 序列轨按Overall/head/mid/tail分别宏平均，SetA/B另列；先fold内实验宏平均，再报告五折三seed分布，不按样本量加权。RBP-OOD分别报告实验宏平均与物种×基因宏平均，频率层不与序列轨混用。
- 每个run保存config、selection SHA256、split审计、逐epoch validation history、最佳checkpoint、validation/test带ID预测、逐实验指标和完成标记。独立验收从预测和冻结标签重放指标，并核验train/val/test角色及train恰为三个fold。
- fold0报告只称工程门控或初步结果；完整五折三seed前不写稳定外部泛化、SOTA、机制因果或“数据审计证明性能”的结论。
