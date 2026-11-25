# PERSIST tie-breaking 与 bootstrap 差异审计 v1.1

日期：2026-11-26。范围为回顾性敏感性分析；远端 `/home/proxy-function-gap-persist-mrl` 只读，原 v1 预测、主结果和科学数据未改。**本轮仅 CPU；无训练、无模型推理。**

## 实际 tie 规则与代码哈希

v1 在 `persist_mrl.py:246-248` 使用 `(-prediction, Sequence ID, row position)` 升序，即预测最高并列时选**字典序最小 Sequence ID**，再选最低行号；不是 stable first。冻结协议 `primary.ties` 已预注册同一规则，**不存在 tie 规则文档缺口**。bootstrap 中 v1 的 ID 为 `Sequence ID#重抽样行序号`。远端目录无 `.git`，故不报告虚构 commit，实际 SHA-256 如下，均与 v1 `verification/code_manifest.json` 一致：

| 文件 | SHA-256 |
|---|---|
| `persist_mrl.py` | `94f667185a48abd035fc757c93d0fef6e455e4e1bc4d4bd981d0b56f999334da` |
| `independent_verify.py` | `12255cdfa548c580b343a84618095c16860be33d70233b0b8ff8b3a9aed9ab9d` |
| `frozen_protocol.json` | `1a764a4f516857255bcf4364ea74cca10ad8410fbfbc90b8aec86dbb281e3573` |
| `run_analysis.py` | `d7ae15b3c9c12645aeb6daf2bb1c614e3c255455f1927310be2cf2286d7ba9c4` |
| `finalize_artifacts.py` | `1261d3dacad5a66cee7f64e607b4212230cbe2f212a7477c72682a3d0cccc48d` |
| `tests/test_persist_mrl.py` | `0dbcc5d8a929da05dfb658f3b34f95d113eafc9135fecfc8f8ce99058a8fd4b0` |

200 个原始种子中共有 **9 个** cap16 最高预测精确并列；其中 **5 个**的 stable-first 与 v1 字典序选择不同：`20261147、20261254、20261289、20261290、20261324`。v1.1 在看结果方向前固定 alternate 规则：并列者按 `SHA256(UTF-8 原始 Sequence ID)` 十六进制升序，再按 Sequence ID、行号。bootstrap 仅重算这 5 个种子，其余 195 个及非选择列保持 v1；不按方向选择版本。

## v1 与 v1.1 并列结果

| 指标 | 原 v1 | 回顾性 v1.1 |
|---|---:|---:|
| 200 种子 BoN−Random MRL regret | +0.10472255084999998 | +0.10472255084999998 |
| 5 个受影响种子的平均 delta | +0.2830436019999999 | +0.2830436019999999 |
| 10000 次 UTR 簇 bootstrap 95% CI | [−0.1009639547475, 0.21270577331124993] | [−0.1009639547475, 0.21270577331124993] |
| 双侧符号 p | 0.24517548245175483 | 0.24517548245175483 |
| 显著 BoN 获益 | 否 | 否 |

五个原始种子的选择与效应均未改变。bootstrap 内有 79 个种子内选择变化、78 个重抽样的五种子子效应变化，单个数组单元最大绝对变化 `0.0011278031999999993`；但主区间和 p 值完全不变。**v1 仍是冻结主分析，v1.1 仅并列报告，结论方向未用于选版。**

## 独立 bootstrap 差异归因与验证器修复

差异不是 cluster 映射、RNG 或提案抽样变化：远端原环境 NumPy 2.2.6 / SciPy 1.15.3 的只读纯内存复算达到 **80000/80000 单元逐位一致**，本地独立实现的主效应、regret 和 coverage 也全部精确一致。

旧独立实现的 Spearman 差至多 `1.1102230246251565e-16`，来自 `scipy.spearmanr` 与 `rankdata + np.corrcoef` 的等价数值归约。Pearson/MAE 最大差 `5.808119329930861e-08` / `5.657390023117159e-08`，来自独立特征代码将“float32 计数后相除”改为“反复累加 `1.0/denominator`”，114 个特征单元不同、最大差 `2.9802322387695312e-08`，进而使平均预测最大差 `4.76837158203125e-07`；Pearson 还叠加了相关系数归约差。本地 NumPy 2.0.2 / SciPy 1.13.1 的跨版本连续归约另产生至多 `3.552713678800501e-15` 差异，文本 `csv/float` 与 pandas 的预测解析回读差至多 `8.881784197001252e-16`。

本地 `audit_bootstrap_difference.py` 是修正后的验证器：不再以统一 `<1e-6` 容差静默放行；选择列必须精确一致，远端原环境全数组必须逐位一致，同预测的本地连续归约单独受 `<5e-15` 限制，旧独立 MAE 余差必须满足实测预测误差上界。没有修改科学数组来迎合验证。

## 三轮真实核验与失败留痕

1. 单元与冻结输入核验：最终 `5 passed`。两次测试夹具失败保留于 `logs/round1_tests.log`、`logs/round1_tests_retry1.log`，通过日志为 `round1_tests_retry2.log`。
2. 全量实算与确定性重放：10000 次 bootstrap、差异归因均 PASS；重放前后六个结果文件 SHA-256 完全相同，见 `round2_*.log`。早期验证器三次失败和分析开发日志均保留。
3. 交付核验：检查原 v1 manifest、远端源码快照哈希、结果算术、报告和失败日志；结果由 `verification/round3_delivery.json` 记录。初次检查保留于 `round3_delivery.log`：发现原 v1 清单唯一的历史不一致项为 `delivery_verification.json`（记录 `12b1ac…`、当前 `8accde…`，均为 225 bytes）；本轮未修改该文件，所有冻结预测、主结果及科学输入哈希仍一致。初次 SSH 主机名失败也保留在 `setup_ssh_px2_failure.log`。

产物目录：`persist_mrl_external_v1/tie_audit_v1_1/`；关键结果为 `results/summary_v1_1.json`、`results/comparison_v1_vs_v1_1.json`、`results/affected_seeds.tsv`、`results/bootstrap_v1_1.npy` 和 `results/bootstrap_difference_audit.json`。逐文件交付哈希见 `artifact_manifest.json`。
