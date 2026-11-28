# 跨场景尾部可靠性关联检验（2026-11-28，V10 阶段 B）

状态：完成；三轮独立核验 PASS。纯 CPU 统计，基于冻结 artifacts，**未修改任何已查看 test**。回顾性、关联性，非因果。

## 1. 科学问题与定位到的缺口

V9 的核心主张"proxy 决策价值追踪高分尾部可靠性"此前只有 11 行**异质**尾部指标的目测散点（Figure20），无统一指标、无统计检验、无敏感性分析（执行提示词 P1 自指缺口）。本阶段把它推进为可证伪的定量关联。

## 2. 统一指标（B1）

对每个场景在**全局评价候选群**上用同一定义重算（`tail_audit_v2.py`，80 个输入文件哈希登记）：

- `k = max(3, ceil(0.10·N))`（top decile）；
- `tail_recall_decile = |预测top-k ∩ 真实top-k| / k`（尾部富集）；
- `tail_spearman_pred/true_decile` = 预测/真实 top-k 内 Spearman（尾部排序）；
- `global_spearman`、`top1pct_recall`。

正确性锚点：AAV `top1pct_recall = 0.1875`，与 V9 硬编码值精确一致；各场景 `global_spearman` 与 V1 一致。mRNA 四行复用冻结 `unified_scenario_metrics.json` 的全局 top-decile recall（同定义），其 within-tail Spearman 标 NA。TEM-1 从 raw + frozen_protocol 按 test split 重算。

## 3. 预注册关联检验（B2）

协议 `association_protocol_frozen.json`（先冻结后计算）。主假设 H-TAIL：tail_recall_decile 比 global_spearman 更强地正关联 relative_regret_reduction_pct。纳入 10 行（indel 无 reduction 排除）；200000 次场景级置换（seed 20261128）；10000 次场景 bootstrap 差值 CI（seed 20261129）。

### 主结果

| 量 | 值 |
| --- | ---: |
| n 场景 | 10 |
| ρ(tail_recall, reduction) | **0.8788** |
| 置换 p（tail） | **0.00076** |
| ρ(global_spearman, reduction) | 0.7697 |
| 置换 p（global） | 0.00627 |
| ρ_tail − ρ_global 均值 | 0.1026 |
| 差值 95% CI | [0.0000, 0.4616] |
| leave-one-out ρ_tail 范围 | 0.8333 – 0.9500 |

### 判定（诚实）

- **尾部富集与决策收益的跨场景关联成立**：ρ=0.879，p=0.0008，LOO 稳健（全部 >0.83）。这是主线的现象级统计支撑。
- **比较性主张未达门槛**：差值 CI 下界恰为 0，预注册成功规则（要求 CI 排除 0）`success_rule_met = false`。按失败规则如实保留：tail recall 数值上高于全局 ρ，但"显著优于全局相关"不成立为确认性结论，降级为描述性。不得事后改规则。

### 次要预测子（n=7，mRNA within-tail NA 排除）

| 预测子 | ρ |
| --- | ---: |
| tail_spearman_pred_decile | 0.4286 |
| tail_spearman_true_decile | 0.5714 |
| top1pct_recall | 0.8524 |

**机制含义**：尾部"富集"（recall，0.879/0.852）比尾部"精细排序"（within-tail Spearman，0.43–0.57）更强地预测固定池 BoN 收益。对"从池中选少数候选"的决策，proxy 高分尾是否**包含**真实精英比尾内精细排序更关键——与 T-cell 剩余损失在尾内排序、但跨场景收益主要由富集驱动的现象一致。

## 4. 三轮核验

| 轮 | 内容 | 结果 |
| --- | --- | --- |
| 1 | 独立 rank-Spearman 复算 ρ、协议/数据哈希、n/排除项 | PASS |
| 2 | 独立种子置换 p=0.00070（<0.01）、LOO 全 >0.8 | PASS |
| 3 | 成功规则逻辑、诚实 null 比较、AAV top1% 复现 | PASS |

## 5. 可写入论文的主张 / 禁止

允许：统一尾部富集与等预算决策收益存在经置换检验支持、LOO 稳健的跨场景正关联（回顾性、关联性）；数值上强于全局相关但差值 CI 含 0。
禁止：因果决定；普遍有效；前瞻验证；跨终点原始效应合并；把比较性 null 包装成显著优于全局。

## 6. 文件

- `source_data/cross_scenario_tail_audit_v2/{tail_audit_v2.json,tsv,input_hashes.json,association_protocol_frozen.json,run_association.py,association_results.json,verify_association.py,association_verification_round{1,2,3}.json}`
- 远端代码 `/home/proxy-function-gap-tail-audit-v2/tail_audit_v2.py`；远端 artifacts `/teams/.../cross_scenario_tail_audit_v2/`。