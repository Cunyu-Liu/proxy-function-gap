# Schlusser HepG2 TE 一次性确认性跨终点 OOD 协议

## 冻结状态与授权边界

本协议在任何 HepG2 TE 分布、逐序列标签或模型结果检查之前冻结。Gate 0 的历史状态保持 `authorized_model_execution=false`、`outer_metrics_computed=false`，不被改写；本任务的新授权仅适用于 `schlusser2024-hepg2-te-confirmatory-v1` 这一新运行目录。输入是 Gate 0 已隔离的 **6,340** 条无标签 Learner，包含 6,309 个基因和 6,340 个转录本。Learner SHA256 为 `82a48e80053e7054d082278ce2c75bbf517633afb650634340ecbcb3b4354625`；Evaluator-only 保险库 SHA256 为 `b1b3056199141a96604d04af45d9255e4970a388f8a519aec6500dfad7799b40`，冻结阶段只核验文件哈希，不读取标签值。

这是 **Sample MRL proxy 向 HepG2 内源 mRNA 真实 TE 的跨终点部署失配评测**，不是 MRL 同终点复验。TE 是发表的 RNA-seq/Ribo-seq 派生测量终点，不等于直接蛋白产量；本轮不重算 FASTQ，也不声称前瞻湿实验或跨研究复制。

## 冻结模型与输出单位

两种 proxy 都只预测 MRL，逐成员使用各自 train-only scaler 逆变换到 measured-MRL 单位。不得把 MRL 与 TE 直接相减或称为跨终点绝对误差。

| 架构 | seed | checkpoint SHA256 | model hash |
|---|---:|---|---|
| Sample fresh CNN | 11 | `77b0539b72e5dfd5b893749107d9f24b03df04a6353badfd09ec0a313f2a9212` | `13eca7c6a3282f1a1f5c00a3f34cb50d8047d5a16bf146da50fb8f22722b4173` |
| Sample fresh CNN | 23 | `e92a9cf2815f8023fb9ec6617f1d680b223f47b767c564e71224997edec734b7` | `2f439c3836426aee7ed21e8b35a22d17673e04d78e78c85f58f74cd53c70b6b6` |
| Sample fresh CNN | 37 | `0a6647c4a6661e627555ffd0a055e59ab7b1d1eb7abb81bc2677774173cc55a6` | `967707e04bbb1fc0d8e210d0855817eb480afd0108e7a7eb47490c8924205a75` |
| 4-mer+GC/uAUG/poly-U MLP | 101 | `147a86970d37cd58f0b8810b661d18c4c8eb1d8ac2a3cdd7c91b54af0654fcba` | `b830b6a01126001968e437af28617a252a57529e2cf3342fb04abcf180942a47` |
| 4-mer+GC/uAUG/poly-U MLP | 202 | `9c1c1cbbff726b985a5a1bdbcf55b8f65458165c30bf8b9f634b35f048864a28` | `0519824f352de2153a034ba27b1b9089dc0dbe81cb9a30e3304ba60de7655b2f` |
| 4-mer+GC/uAUG/poly-U MLP | 303 | `0b09912e887ce490089645512d742628f4df28dc347b90886486b00f241214f3` | `204e25d62fec471b3d7e5fe12811ea85bee000dbb9e39ef04ce66676acbc7b35` |

4-mer 使用与 CNN 相同的 Sample 训练标签，仅是架构敏感性对照，不是独立模型验证。ensemble-v1 未逆变换产物和任何 outer 调参产物均不得使用。

## 候选、提案、cap 与基线

候选固定为全部 6,340 行，不按 TE、预测分数或结果进一步筛选。每个种子从全部行均匀、有放回抽取 16 个提案；cap `{1,2,4,8,16}` 使用同一提案序列的嵌套前缀。100 个配对算法种子为 `101,202,...,10100`，所有方法共用相同提案。

- **Random**：选择首个提案，不使用模型选择分数；因此同一种子跨 cap 的 Random 选择相同。
- **BoN-CNN**：在 cap 提案内最大化三成员 CNN MRL 均值；精确同分由最早提案胜出。
- **BoN-4mer**：在相同提案内最大化三成员 4-mer MLP MRL 均值；精确同分规则相同。
- **LCB-CNN**：固定 `mean - 1.0 * sample_std`，不调 lambda；作为不确定性基线。

cap16 BoN-CNN 是唯一主方法，Random 是唯一主参照；4-mer、LCB 和其他 cap 全部作为预先规定的次要结果完整报告。不得事后选择“最佳”cap、方法、种子或区间。cap1 时各模型方法与 Random 选择相同，配对差必须为 0。

每轮对全部候选实际执行 CNN 与 4-mer 三成员 **CUDA float32** 推理，禁止 CPU fallback，使用确定性算法、关闭 TF32。全池实际分数在方法与种子间共享；manifest 记录每架构 `3×6340=19020` 个成员 item-forward，选择记录另列 `3×cap` 的分配成本和 Random=0，不能把共享预计算隐去或重复冒充实际 forward。

## 主终点、配对效应与符号

真实 TE regret 定义为：

`regret = max(TE over frozen 6340 rows) - TE(selected row)`。

regret 非负且越低越好；样本内最大 TE 是固定池 oracle，不是总体转录组 oracle。主估计为 cap16 BoN-CNN 在 100 个算法种子上的平均真实 TE regret，以及共用提案的 `BoN-CNN - Random` 平均配对 regret 差。

- regret 差小于 0：有利于左方法；大于 0：有利于右方法。
- selected TE 差与 regret 差符号相反；TE 差大于 0 有利于左方法。
- 另完整报告 `BoN-4mer-Random`、`LCB-CNN-Random`、`BoN-CNN-BoN-4mer` 及左胜/平/右胜计数。
- 区间跨 0 时不声称方向性方法差异；正向、负向和零效应都保留。

描述性 Spearman MRL–TE 相关只在揭盲后汇总，不替代 regret，不参与配置选择。

## 基因/转录本簇 bootstrap

基因簇是主单位，转录本簇是敏感性单位；各 2,000 次，种子分别为 `20261124` 和 `20261125`，百分位 95% 区间。每次按簇有放回重采样，重建候选池，重新执行 100 个配对算法提案、所有方法/cap 的选择并重新计算样本 oracle 和配对效应。这同时反映候选池生物簇抽样与算法抽样，不把逐行当独立生物重复。

该区间仍只属于单一 HepG2 研究，不是独立研究间置信区间，也不能支持跨细胞、跨研究普遍结论。转录本在本冻结 Learner 中每行唯一，因此转录本分析接近行级敏感性，不冒充更强独立证据。

## 揭盲前 score-matching

对 cap16 的每个 BoN-CNN/BoN-4mer 选择，在**读取 TE 前**从全池寻找对应架构 MRL 均值绝对差最小的不同 row 且不同 UTR50；同分时依次以十进制行索引的 SHA256 和行索引确定性打破平局。匹配映射写入 Learner selection 后冻结。

Evaluator 只比较 selected/matched 的真实 TE 与 TE regret：TE 差 `selected-matched > 0` 有利于优化选择，regret 差 `selected-matched < 0` 有利于优化选择。禁止计算 MRL−TE“绝对误差”。簇 bootstrap 重跑选择并条件于这份冻结全池匹配映射计算区间；它不重选匹配规则。

## 一次性执行与三轮独立重算

1. `freeze` 在干净提交上写入配置、协议、代码 commit、模型、Learner 和保险库哈希； refuse overwrite。
2. 三轮分别从新进程语义加载模型并执行完整 CUDA 推理和无标签选择，输出独立目录。所有三轮 Learner 完成后，Evaluator 才首次读取保险库。
3. Evaluator 对三轮独立重算逐轮生成相同冻结协议的汇总；科学汇总必须逐字节语义哈希一致。三轮全部结束前不查看结果、不生成结果报告、不回调配置。
4. 最后自动渲染完整报告与交付哈希。任何结果后重新调参、挑选子集或隐藏负结果都属于协议违背，不得通过新文件名伪装为原确认性结果。

逐序列 TE 和 scored selections 仅保留在远端 Evaluator 区域；本地核心交付只同步配置、协议、聚合 summary、manifest、三轮一致性、日志和自动渲染报告，不同步保险库或逐序列标签。

## 必须显式保留的不可识别边界

- 单研究候选池不能识别跨研究普适性，簇 bootstrap 不创造独立研究。
- MRL 与 TE 的单位和生物含义不同；本实验检验部署失配，不证明 MRL 同终点性能或因果蛋白表达。
- 相同 UTR50 的多转录本若 TE 不同，固定 50 nt proxy 不可区分；报告重复组数与组内 TE range。
- 两架构共享训练数据，架构一致或不一致都不能独立认证生物功能。
- 发表 TE 表的纳排与 reads 派生过程原样继承；本轮不是 FASTQ 级复现。

## 执行命令

```bash
cd /home/proxy-function-gap
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=src:gate0:runtime/src
PY=/root/miniconda3/envs/lucaone_cu128/bin/python
$PY -m proxy_function_gap.schlusser_te_cli freeze --config configs/schlusser2024_hepg2_te_confirmatory_v1.json
$PY -m proxy_function_gap.schlusser_te_cli execute --config configs/schlusser2024_hepg2_te_confirmatory_v1.json
$PY -m proxy_function_gap.schlusser_te_cli verify --config configs/schlusser2024_hepg2_te_confirmatory_v1.json
```
