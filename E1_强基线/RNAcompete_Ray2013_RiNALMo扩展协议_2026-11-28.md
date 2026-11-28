# RNAcompete Ray2013 RiNALMo 扩展协议附录 v1.0

日期：2026-11-28 ｜ 状态：**预注册附录；smoke/嵌入/矩阵未验收前不写性能** ｜ 母合同：`RNAcompete_Ray2013_模型基线预注册合同_v1.0_2026-11-26.md`、`RNAcompete_Ray2013_预注册数据合同_v1.0_2026-11-26.md`

## 1. 目的与范围

在已最终验收的 RNA-FM 冻结基线（70/70 eval）之外，补齐**跨编码器**对照：冻结 `multimolecule/rinalmo-giga`（revision `2a71f6f98fb41dd2e6542a5e131d3778111d1468`，1280 维）上的 regularized linear、浅 MLP 与 masked 余弦检索 kNN。只扩展编码器，不改变任务、划分、指标与统计口径。

## 2. 完全沿用的冻结项

- 数据版本 `rnacompete-ray2013-v1.0-id90-strict3train`，244 实验、219,990 唯一 RNA、序列 ID90 五折与整 RBP-OOD 双轨；`split_manifest.json` status=accepted，SHA 校验不变。
- 标签缓存：同一 `labels.unique_rna.npz` 生成规则（有限探针中位数、NaN 不填值不作负例），在新产物根独立重算并核对有限计数。
- 训练/选择：fold0 validation-only 冻结超参（weight_decay 网格、k∈{5,10,20}），masked MSE，AdamW lr=1e-3，batch 1024，patience20/min_delta1e-4，1000 epoch 故障保险；test 只由独立 evaluate 读取冻结 checkpoint。
- 指标：逐实验 Spearman/Pearson/MSE 后等权宏平均；常数真实值/预测的 NaN 相关按母合同排除并保留支持数；SetA/SetB 与基因宏分列。
- 三 seed（0,1,2）为统计重复；描述性 mean±SD，不做显著性或 SOTA 主张。

## 3. 本附录新增项

- 编码器：RiNALMo-giga 冻结全参数；mean-masked 池化（attention_mask=1 含特殊 token、排除 padding），autocast bf16，缓存 fp16；不截断（冻结序列最长 ≤40nt 量级，超 tokenizer 容量即报错）。
- 嵌入产物：`rinalmo.mean_masked.fp16.npy` + `rinalmo.ids`，ID 顺序必须与冻结 sequence_folds 完全一致；双 SHA + 逐 4096 行有限值检查。
- 产物根：`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/rnacompete_ray2013_rinalmo_baselines_v1/`（独立于已验收 RNA-FM 树，绝不写入后者）。
- 代码：`rnacompete_ray2013_rinalmo_embed.py`（新）、`rnacompete_ray2013_baselines.py` 最小参数化（`--encoder {rnafm,rinalmo}`、`--emb-dim`，默认 rnafm/640，encoder/dim 一致性强制校验）；`run_rnacompete_ray2013_rinalmo.sh` 独立 driver 与锁。
- 资源门：smoke 4GB、全量嵌入 9GB、头训练/评测 2GB，RAM ≥12GiB；与其他队列共存，不 kill 不抢占；CUDA 数值/反向/优化器探针强制，CPU 降级即失败。

## 4. 验收链

smoke（16 序列）→ 全量嵌入 manifest → 标签缓存 → 6 个 validation-only selection → fold0 双轨 3 seed + kNN → verify-fold0 → fold1–4 扩展 → verify-final（checkpoint CUDA 重放、selection 无 test 产物）→ summarize。任何一步未过不进入下一步；最终数字必须经独立聚合重算后才可写入报告。

## 5. 停止规则

不改冻结划分与 test；不为 RiNALMo 单独调 fold0 以外的选择协议；嵌入 ID/维度/有限值任一失败即停；与 RNA-FM 树产物零交叉。