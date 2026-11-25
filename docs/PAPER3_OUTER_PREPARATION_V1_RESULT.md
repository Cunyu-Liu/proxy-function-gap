# Paper3 外测准备与门禁 v1：审计、预注册和三轮不揭盲演练

日期：2026-11-25
状态：**准备工作完成；Paper2 P0 calibration 门禁失败，Paper3 outer 继续硬锁，禁止揭盲。**

## 结论

- 已从 `/home/proxy-function-gap-paper3-v4`（源 HEAD `539cc7ef72d461f858ee89ffe8f1b26fbad4fd97`）建立独立副本 `/home/proxy-function-gap-paper3-outer`，移除副本中的共享 `.git` 指针；未修改 V4，未重跑 V4。
- V4 的六策略、development-only 超参选择、冻结 CNN/种子/求积和 CUDA 强制可复用；但 V4 在同一 calibration 池查询并计算 regret，不能直接作为 outer 的 Q/E 隔离设计。
- 新实现将 64 序列池按无标签 group 哈希固定拆为 32Q/32E，阻断跨 Q/E 及 development 的 exact/Hamming-1；Learner 冻结所有 E 选择后 Evaluator 才评分。
- 最终三轮 calibration-only 演练科学字段完全一致，SHA256 均为 `88262dfa43e0d8a0b9cc2ba21c8d3f4aad613664b5af8ced3260f0b135697906`；独立验证器逐条复算预算、账本、CUDA、Q/E、full-Q 和 measured-MRL regret。
- 最终 focused tests 三轮均为 **17 passed**。本工作包没有读取或评分永久 outer 标签，没有生成 outer query 清单、regret 或置信区间，没有修改 Paper2 文件。

## 审计判定

| 项目 | 判定 | 证据/边界 |
|---|---|---|
| random/max-variance/diversity/EI/top-proxy/regret-analytic | PASS | V4 aggregate 原始 `curve_records` 独立复算六策略齐全 |
| 表示与超参冻结 | PASS | 外测固定 kmer4，prior=0.1、noise=0.16 来自 V4 development-only 四折阻断 CV；不再 CV |
| V4 query/eval 隔离用于正式 outer | FAIL | `curve_case_v4` 的 query 与 evaluation 是同一池；不能原样外测 |
| v1 分离式 Q/E | PASS（演练） | 32Q/32E、group/exact/H1 隔离、Evaluator 后置评分，三轮通过 |
| 预算公平与 full-Q | PASS（演练） | 0/4/8/16/32，唯一账本；32 标签时八条曲线同一 E 选择 |
| CUDA | PASS | RTX5090，torch 2.7.1+cu128，拟合/推断/获取均为 cuda:0；CPU 仅用于明确标注的统计复核 |
| Paper2 P0 门禁 | FAIL / HARD LOCK | P0 已完成但 calibration 门禁失败，未评分 outer；Paper3 不得创建放行或启动正式入口 |
| 正式 outer | NOT RUN | 无 outer Learner、无 outer Evaluator、无 outer 指标 |

永久 outer 已被 Paper1/Paper2 历史研究线评分。即使本工作包未读取这些结果，后续也只能称 **Paper3 冻结的一次性历史回放**，不能称全课题从未使用的新独立盲测、独立研究复现或前瞻湿实验验证。

## 预注册科学决策

协议：`docs/PAPER3_OUTER_PROTOCOL_V1.md`；配置：`configs/paper3_outer_protocol_v1.json`。

- 主队列：`outer_designed_000`–`015` 全 16 池；次要队列：`outer_random_000`–`002` 单独报告。
- 唯一主比较：16 个唯一 Q 标签时 EI 减 random 的真实 measured-MRL top-1 regret 差；random 先在池内对三个冻结种子取均值。
- 主成功判据：16 个完整 pool 簇、10000 次、seed=20261125 的 percentile bootstrap 95% 区间上界 `<0`。
- 次要分析：六策略全预算曲线、0–16 AUC、逐池反例、zero/full-Q headroom、求积收敛和 random outer；不得替换主终点。
- 正式 outer 一旦开始，禁止按中途评分早停、换池、换策略、调参或补种子；主区间包含 0 即主假设不成立，负结果完整保留。

## 原始字段与统计复算（不是 outer 结果）

独立 CPU 脚本 `checks/audit_paper3_outer.py` 读取 V4 `curve_records`，先对 random 三种子取池内均值，再对固定五池取均值：

- 16 标签 EI=1.220083541558、random=3.497015922191，逐池配对差均值=-2.276932380633。
- 固定五池 calibration 簇 bootstrap 95% 区间 `[-4.036154690673, -0.517710070592]`。
- 这些数值与 V4 报告一致，但队列是 V2 已固定的 calibration 小队列，只用于预注册选择，不能当 outer 有效性证据。

新 32Q/32E 演练在 calibration_000 的 16 标签 EI−random 为 **+0.115090312963**，即本次拆分下 EI 略差；已保留该负结果，未据此调参或换池。演练只证明流程确定性和隔离，不作推断。

## 实现文件

远端独立副本新增：

- `src/proxy_function_gap/paper3_outer.py`：严格配置、无标签拆分、Learner/Evaluator、六策略分离式曲线、簇 bootstrap、calibration 演练。
- `src/proxy_function_gap/paper3_outer_cli.py`：只允许 calibration dry-run 的入口。
- `src/proxy_function_gap/paper3_outer_formal.py`：正式 outer 数据适配和 P0 硬门禁；未执行。
- `src/proxy_function_gap/paper3_outer_validate.py`：独立账本/预算/CUDA/regret/三轮验证。
- `tests/test_paper3_outer.py`：配置不可变、标签不影响拆分、H1 拒绝、vault、Evaluator、bootstrap、P0 失败硬锁和无标签 OuterPool 测试。
- `checks/audit_paper3_outer.py`：CPU-only 原始字段与聚合独立复算。
- `configs/paper3_outer_protocol_v1.json`、`docs/PAPER3_OUTER_PROTOCOL_V1.md`。

## 执行命令与结果

实际环境：`/root/miniconda3/envs/lucaone_cu128/bin/python`，`CUBLAS_WORKSPACE_CONFIG=:4096:8`，`PYTHONPATH=src:gate0:runtime/src`。

```bash
# 最终 focused tests（三轮分别留档）
python -m pytest -q tests/test_paper3_outer.py tests/test_paper3_headroom_v4.py
# 每轮：17 passed

# 最终三轮 calibration-only 演练
python -m proxy_function_gap.paper3_outer_cli \
  --config configs/paper3_outer_protocol_v1.json \
  --output /teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper3_outer_protocol_v1/dry_run_v3_round{1,2,3}

# 独立验证
python -m proxy_function_gap.paper3_outer_validate \
  dry_run_v3_round1 dry_run_v3_round2 dry_run_v3_round3 \
  --output validation_v3
# scientific_results_identical=true；outer_labels_read=false；outer_metrics_computed=false
```

远端产物根目录：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper3_outer_protocol_v1`。最终有效子集为 `dry_run_v3_round*`、`validation_v3/`、`tests_v3/`、`cpu_independent_audit_v3.json`、`source_snapshot_v3/`、`source_v3.sha256` 和 `protocol_config_sha256.txt`。

早期 `dry_run_round*` 因 output_dir 被误计入配置身份而未通过验证；修复后在全新目录重跑。`dry_run_v2_round*` 是正式 outer 适配层加入前的中间版本。所有历史目录均保留，只有 v3 作为最终交付。

## 后续唯一关键动作

**维持硬锁并登记 `BLOCKED_BY_PAPER2_P0`：不得创建放行 JSON，不得执行 `paper3_outer_formal.py`，不得读取或评分永久 outer。**

Paper2 P0 已完成且 calibration 门禁失败，未评分 outer；因此本工作包不存在合法的正式 outer 下一步。已实现的 P0 放行接口保持未调用，仅作为防御性代码存在，不能被解释为当前授权。若未来课题决策要求重开 Paper3 外测，必须先有新的科学门禁依据、新协议版本和新的独立外测资源，由主代理另行审查；不能复用本轮 outer 作为反复调参集。
