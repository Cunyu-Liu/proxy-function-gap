# RNAcompete Ray2013 RiNALMo 扩展执行进展

日期：2026-11-28 ｜ 状态：**嵌入与六选择完成，fold0 矩阵训练中；未过最终门，不写性能** ｜ 协议：`RNAcompete_Ray2013_RiNALMo扩展协议_2026-11-28.md`

## 时间线（真实 CUDA，资源门共存）

- 13:55 smoke（16 序列）通过：`RNACOMPETE_RINALMO_SMOKE_OK`，revision `2a71f6f98fb41dd2e6542a5e131d3778111d1468`。
- 13:55–14:03 全量嵌入 219,990 唯一 RNA（1280 维 mean-masked fp16）：`RNACOMPETE_RINALMO_EMBED_OK`，ID 顺序与冻结 sequence_folds 一致，双 SHA + 逐 4096 行有限值检查通过。
- 14:03–14:05 标签缓存独立重算验收（`LABEL_CACHE_OK`，有限计数与冻结实验审计一致）。
- 14:05–15:11 六个 validation-only selection 依次冻结：sequence/rbp_ood × linear/mlp/kNN（fold0/seed0，weight_decay 网格与 k∈{5,10,20}，test 未触碰）。
- 15:11 fold0 矩阵训练启动；**16:21:37 fold0 门通过（FOLD0_GATE_PASSED，verify-fold0 checkpoint CUDA 重放）**，自动进入 fold1–4 扩展（16:25 起）。

## 产物与日志

- 产物根：`/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/rnacompete_ray2013_rinalmo_baselines_v1/`（独立于已验收 RNA-FM 树，零交叉）。
- driver：`run_rnacompete_ray2013_rinalmo.sh`（锁 `logs/rnacompete_ray2013_rinalmo/driver.lock`）；日志 `logs/rnacompete_ray2013_rinalmo/`、driver 汇总 `logs/rnacompete_ray2013_rinalmo_driver.log`。
- 代码：`rnacompete_ray2013_rinalmo_embed.py`（新）、`rnacompete_ray2013_baselines.py` 参数化（`--encoder/--emb-dim`，默认 rnafm/640，合同测试通过）。

## 冻结选择配置（fold0/seed0 validation-only，test 未触碰）

| 轨 | 架构 | 选定配置 | val 宏 Spearman |
|---|---|---|---:|
| sequence | linear | weight_decay=1.0 | 0.4223 |
| sequence | MLP | weight_decay=1.0 | 0.4639 |
| sequence | kNN | k=20 | 0.1840 |
| rbp_ood | linear | weight_decay=1.0 | 0.1219 |
| rbp_ood | MLP | weight_decay=1e-4 | 0.1217 |
| rbp_ood | kNN | k=1 | 0.1592 |

注：val 指标为选择口径（fold0 validation 宏），不与 RNA-FM 已验收 test 数字横比；最终对照待全矩阵验收。

## 2026-11-28 18:44 fold2 训练 CUDA OOM 与重试修复

- `train_rbp_ood_linear_f2_s1` 在 `xtr_t.to(device)`（1.05GB）时 OOM：资源门 18:43:55 见 8746MiB 空闲，但 20 秒内外部进程（LoRA 9.2GB + 另一进程 8.6GB + codonflow ~10.9GB）占满至 162MiB。门与分配之间的竞态。
- **修复**（科学代码/RNG 不变）：driver `run_cuda` 增加 OOM 自动重试（最多 48 次，每次重过资源门 + sleep 120s），非 OOM 失败仍传播。`bash -n` 三遍通过；原脚本备份 `.bak_oom`。
- 19:09 重启 driver：fold0/1 已完成 run SKIP，verify_fold0 复核后从 fold2 断点续跑。GPU 空闲回升至 20GB。

## 待办

1. 逐 selection 配置与 val 指标摘录归档（下轮 SSH 窗口补取）。
2. fold0 门过后记录三 seed 双轨描述性结果；全矩阵 + verify-final + 独立聚合重算通过后另写最终报告，与 RNA-FM 70/70 同标准对照。
3. 过门前任何数字不得进入论文/PPT 结论页。