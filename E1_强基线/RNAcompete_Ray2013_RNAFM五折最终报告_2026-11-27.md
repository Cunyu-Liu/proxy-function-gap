# RNAcompete Ray2013 冻结 RNA-FM 五折最终报告

日期：2026-11-27 ｜ 数据：`rnacompete-ray2013-v1.0-id90-strict3train` ｜ 状态：**五折三seed双轨完成并通过最终独立验收**

## 1. 证据范围与协议

本报告覆盖5个fold×3个seed×2种神经架构×2条轨，共60个神经run，以及5个fold×2条轨的10个确定性kNN评价。编码器为冻结`multimolecule/rnafm`（640维mean-masked），连续强度使用masked MSE；NaN不填值、不作负例。regularized linear、浅MLP和masked余弦检索均沿用fold0 validation冻结配置，test只用于最终评价。

- 序列ID90轨：244个实验，报告Overall/head/mid/tail及All/SetA/SetB。
- 整RBP-OOD轨：每折48–49个目标实验、40–42个物种×基因组；与序列OOD分开，使用可迁移共享头/检索共识。
- 每个神经配置15个run（5 fold×3 seed），kNN每配置5个run。表中为run间mean±SD；先逐实验计算Spearman/Pearson/MSE，再等权宏平均，不按RNA数加权。
- 常数真实值或预测导致相关指标NaN时按合同排除该相关均值并保留支持数；未观测标签不进入损失或指标。

## 2. 最终独立验收

`final_gate.json`确认70/70评价通过，60个神经checkpoint CUDA预测重放与保存预测一致，selection树无test产物。独立聚合脚本`code_rareseq/rnacompete_ray2013_final_summary.py`再次从逐run指标生成525条run级汇总和45条最终统计，并核验run支持数与RBP fold支持范围。

- Final gate SHA256：`83ec4cfef1c25c91839e8e51d40dc34272bb901f76d681d61ad6c199b41fe15b`
- Run-level汇总SHA256：`a276339a28b02a862a24c1edd48fed9cc3115c863a6b6f6e7fd3143acabba05c`
- 最终独立统计SHA256：`71a4b708120d8831acc7362da49a2a2f9159e4e486ad1cd82a4c610890b546f1`
- 并发任务JSONL中`verify-final`和`summarize`均有真实start/end及completed状态；summary产物路径4项。

## 3. 序列ID90五折三seed结果（All探针）

| 模型 | 切片 | Spearman | Pearson | MSE |
|---|---|---:|---:|---:|
| linear | Overall | 0.415823±0.001712 | 0.347198±0.004493 | 51.013646±12.885378 |
| linear | head | 0.449936±0.002849 | 0.370586±0.007654 | 48.897491±31.525911 |
| linear | mid | 0.367683±0.003192 | 0.312834±0.009830 | 72.685644±15.275969 |
| linear | tail | 0.430202±0.002864 | 0.358449±0.002265 | 30.968906±30.800392 |
| MLP | Overall | 0.457604±0.001636 | 0.411557±0.005651 | 49.174474±12.765944 |
| MLP | head | 0.489980±0.001928 | 0.436043±0.009409 | 46.756563±31.615986 |
| MLP | mid | 0.407665±0.002020 | 0.378362±0.011807 | 69.878898±15.062692 |
| MLP | tail | 0.475605±0.002162 | 0.420483±0.003787 | 30.430800±30.784656 |
| kNN | Overall | 0.342374±0.001178 | 0.255573±0.003251 | 52.719232±13.936291 |
| kNN | head | 0.378048±0.001336 | 0.281058±0.004462 | 50.370833±34.000584 |
| kNN | mid | 0.296765±0.001070 | 0.223840±0.008362 | 75.434995±15.973092 |
| kNN | tail | 0.352559±0.001378 | 0.261976±0.003714 | 31.842684±33.144329 |

在这组冻结基线内，浅MLP的Overall/tail平均Spearman和Pearson高于linear与kNN；MSE跨fold/seed变异很大，不能只用均值判断优劣，也不据此声称显著性或SOTA。

## 4. 频率tail的SetA/SetB

| 模型 | Set | Spearman | Pearson | MSE |
|---|---|---:|---:|---:|
| linear | SetA | 0.430213±0.003953 | 0.370000±0.005013 | 19.889321±6.635596 |
| linear | SetB | 0.430174±0.001969 | 0.367715±0.003767 | 41.943914±56.982367 |
| MLP | SetA | 0.475736±0.002976 | 0.434270±0.005058 | 19.350266±6.627804 |
| MLP | SetB | 0.475456±0.001640 | 0.431471±0.004767 | 41.406765±56.963780 |
| kNN | SetA | 0.352701±0.002082 | 0.269160±0.004511 | 20.838706±7.122253 |
| kNN | SetB | 0.352404±0.000910 | 0.269200±0.004183 | 42.742659±61.440412 |

SetA/SetB的Spearman接近，但MSE方差显著不同，尤其SetB；正式结论必须同时保留离散度，不能选择性报告单一集合或指标。

## 5. 整RBP-OOD五折结果（All探针）

实验宏共15个神经run或5个kNN run；基因宏为40–42个物种×基因组。

| 模型 | 实验宏Spearman | 实验宏Pearson | 实验宏MSE | 基因宏Spearman |
|---|---:|---:|---:|---:|
| linear | 0.098188±0.019112 | 0.092270±0.015223 | 53.597415±23.443392 | 0.088023±0.017107 |
| MLP | 0.111699±0.023439 | 0.107254±0.016644 | 53.509338±23.460374 | 0.100495±0.021166 |
| kNN | 0.185360±0.027500 | 0.143623±0.015548 | 53.486383±25.153494 | 0.165926±0.024592 |

kNN共识在该fold0冻结k=1的零样本RBP-OOD对照中平均相关更高，但它对所有目标RBP使用同一序列共识，不是任务特异预测；其MSE与其他模型接近且跨run变异很大，不能据此宣称统一优越或机制解释。

RBP-OOD SetA/SetB的完整实验/基因宏指标见`final_independent_summary.tsv`，未与序列频率tail混合。

## 6. 产物与结论边界

- `final_independent_summary.tsv`：45条最终聚合行，含Overall/head/mid/tail、All/SetA/SetB及RBP实验/基因宏。
- `final_independent_run_metrics.tsv.gz`：525条run级聚合记录。
- `final_independent_verification.json`：独立验收状态、计数、SHA和报告指针。
- `task_runs.jsonl`：selection/train/evaluate/kNN/verify/summary的真实并发任务记录。

本报告只支持该冻结Ray2013基准和既定对照的描述性五折结果，不构成SOTA、显著性、跨协议统一泛化或机制因果主张。过程报告见`E1_强基线/RNAcompete_Ray2013_RNAFM基线执行报告_2026-11-27.md`。
