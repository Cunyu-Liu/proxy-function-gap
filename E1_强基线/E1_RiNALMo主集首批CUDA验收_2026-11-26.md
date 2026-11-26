# E1 RiNALMo 主集首批 CUDA 验收

日期：2026-11-26。当前结论：**smoke 验收通过，正式146实验嵌入与后续矩阵仍在远端队列中；未完成强基线矩阵，不作 SOTA 主张。**

## 已验证的真实产物

- 非 pilot tail 实验：`SUPV3L1_K562`，与正式结果隔离。
- RiNALMo-giga 参数冻结，最终层 mean-masked pooling，CUDA bf16 autocast，fp16 缓存。
- 正/负两个缓存共2301窗口，均为1280维；逐文件 IDs、行数、dtype 和有限值核验通过。
- 嵌入、BCE、kNN 均输出 `cuda:0` / RTX 5090 数值探针 `[2.5, -6.0]`。
- BCE 双轨均按 validation AUPRC 选择最佳 epoch，history、最佳 checkpoint 与停止 epoch 一致；未触发1000 epoch故障保险。
- kNN 的 k 仅由 tail-RBP validation AUPRC 选择，训练正样本按冻结 id90 cluster 去重。

## 独立审计结果

`results/rinalmo_main/smoke_audit.json` 记录：

- `n_experiments=1`、`n_modern_runs=1`、`n_knn_runs=1`；
- strict train 恰好三折，`fold=-1` 未进入 train；
- validation/test 原始 split 与预测键完全对齐；
- 所有评价分数有限；
- run JSONL 无畸形行。

静态验收也已通过：Python 编译、Shell 语法和 `E1_FOLD_CONTRACT_OK`。

## 正在运行的正式队列

- 驱动 PID：`4080653`。
- 状态：`/home/ProxyFunctionGap/logs/e1_rinalmo_main/driver.status`。
- 当前阶段：146实验主缓存；40个 pilot 实验仅在缓存一致性核验后复用，其余新生成。
- 主缓存完成后自动运行正式 fold0/seed0 BCE、fold0 kNN 和首批正式审计，再以两个 BCE worker 完成剩余14个 BCE run，随后完成其余4个 kNN run。
- 只有15个 BCE +5个 kNN 全部通过完整审计后才生成正式 `cv_summary`。

## 证据文件

- `results/rinalmo_main/smoke_audit.json`
- `results/rinalmo_main/smoke_run_manifest.tsv`
- `results/rinalmo_main/smoke_embedding_manifest.json`
- `results/rinalmo_main/run_status.json`

远端大产物位于 `/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_rinalmo_main_strict3train_p20_v1`，代码和完整日志位于 `/home/ProxyFunctionGap`。
