from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper2_risk_frozen_v1")


def load(name):
    return json.loads((OUT / name).read_text())


def f(value):
    return f"{value:.6f}"


def main():
    audit = load("existing_implementation_audit.json")
    frozen = load("frozen_protocol.json")
    calibration = frozen["calibration"]
    rows = []
    for item in frozen["development_grid"]:
        dev = item["dev"]
        chosen = "**是**" if item["quantile"] == frozen["selected_quantile"] else "否"
        rows.append(f"| {item['quantile']:.2f} | {f(item['threshold'])} | {dev['coverage']:.3f} | {f(dev['accepted_mean_regret'])} | {f(dev['regret_reduction'])} | [{f(dev['regret_reduction_95ci'][0])}, {f(dev['regret_reduction_95ci'][1])}] | {chosen} |")
    ci = calibration["regret_reduction_95ci"]
    text = f"""# Paper 2 ensemble-std 冻结阈值风险—覆盖核验 v1

## 结论

**calibration 门禁失败，未执行 outer 评分。** 当前冻结协议不支持“ensemble-std 拒选可降低真实 regret”的阳性主张。历史 outer 经验曲线仍是已查看 outer 的回顾性敏感性结果，不能替代本次冻结阈值验证，也不能据此回调阈值。

本工作包只进行冻结数组的 CPU 统计，没有新训练、模型推理或 CUDA 验证。

## 1. 旧实现审计

- 单位：{audit['unit_finding']}
- 聚合：{audit['aggregation_finding']}
- 分位点：{audit['threshold_finding']}
- outer 边界：{audit['outer_finding']}
- dev baseline 的 `outer_used_for_selection=false` 只覆盖 lambda/gamma 选择，不覆盖旧 risk–coverage 的 outer 经验分位点。

## 2. 预注册与数据适配

协议、分位数网格、主终点和门禁见 `preregistration.json` 与 `PROTOCOL.md`。dev 只在 `{{0.50, 0.75, 0.90}}` 中选一个分位数；calibration 不重选。主终点为全量 BoN 平均 regret 减接受决策平均 regret，正值有利于拒选。池簇 bootstrap 10000 次，seed=20261125。

两次准备阶段的实现修正均发生在 dev/calibration 结局汇总之前，完整保留在 `prefreeze_implementation_correction.json` 和失败日志中：

1. 冻结 NPZ 派生表与旧 dev CUDA 重算的所选候选 std 最大差为 `7.21e-7`，跨实现比较容差改为 `1e-6`；统计输入仍固定为 NPZ 派生表。
2. 旧 calibration selection 使用 Python `random.Random` 顺序提案，与 dev/outer 的 NumPy 提案不同，80 条中 66 条 top-1 不同。主分析统一从原始候选表按 outer 对齐的 NumPy 算法重建 dev/calibration，旧 calibration selection 不混入主分析。

这些修正没有改变分位数网格、选择规则、终点、bootstrap 或门禁。

## 3. dev 分位数选择

| 分位数 | 阈值 | 覆盖率 | 接受 regret | 降幅 | 95% 池簇 CI | 被选中 |
|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

dev 三个区间均跨 0。选择 q=0.50 是预注册点估计规则的结果，不表示 dev 已有显著收益。

## 4. calibration 单次门禁

- 冻结阈值：`{frozen['selected_threshold']}`。
- 80 条决策中接受 {calibration['accepted_decisions']} 条，覆盖 {calibration['accepted_pools']}/16 池，覆盖率 {calibration['coverage']:.3f}。
- 全量 BoN 平均 regret：{f(calibration['full_bon_mean_regret'])}。
- 接受决策平均 regret：{f(calibration['accepted_mean_regret'])}。
- 主终点降幅：{f(calibration['regret_reduction'])}，95% 池簇 CI `[{f(ci[0])}, {f(ci[1])}]`。
- 点估计方向为负且区间跨 0，故“点估计大于 0”和“区间下界大于 0”两项均失败；最小接受决策数与池数条件通过。

## 5. outer 与科学结论边界

`outer_scoring_authorized=false`。实际调用 `score-outer` 被硬锁拒绝，日志见 `outer_refused.log`；未生成 `outer_result.json` 或任何新 outer 决策 CSV。没有进行一次性 outer 评分，也没有在门禁失败后尝试其他阈值。

因此，本工作包的结论是：**在当前统一提案重建、所选候选级拒选规则和 calibration 留出上，冻结 ensemble-std 阈值未显示真实 regret 改善。** 这不否定所有可能的 pool-level 风险模型，也不把历史 outer 经验曲线升级成阳性证据。Paper 2 的 Q 池级阴性结果保持不变；如后续要研究新风险定义或新部署规则，必须建立新版本和独立评估角色，不能继续把本 outer 当未接触盲测。

## 6. 验证与复现

- Round 1：`verification_round1_raw.json`，原始候选/标签/提案逐条复算，dev 160 + calibration 80 条，冻结阈值与聚合复核。
- Round 2：`verification_round2_independent.json`，独立脚本重算 dev 全网格、calibration 和 10000 次区间；`pytest_pre_freeze_final.log` 为 7 passed。
- Round 3：`verification_round3_delivery.json`，teams、本地 source_data 和报告逐文件 SHA256 一致性。
- 运行配置和日志：`initialize.log`、`prepare_final.log`、`freeze.log`、`outer_refused.log`；早期失败日志亦保留。

关键命令：

```bash
/root/miniconda3/envs/lucaone_cu128/bin/python run_protocol.py initialize --code-path run_protocol.py
/root/miniconda3/envs/lucaone_cu128/bin/python run_protocol.py prepare
/root/miniconda3/envs/lucaone_cu128/bin/python run_protocol.py freeze
/root/miniconda3/envs/lucaone_cu128/bin/python verify_round1_raw.py
/root/miniconda3/envs/lucaone_cu128/bin/python verify_round2_independent.py
```

远端代码：`/home/proxy-function-gap-paper2-risk/paper2_risk_frozen_v1/`。冻结产物：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper2_risk_frozen_v1/`。本地同步：`deliverables/reports/source_data/paper2_risk_frozen_v1/`。

## 7. 后续唯一关键动作

**不要评分本版本 outer、不要调阈值。** 由主代理把 Paper 2 当前结论保留为 calibration 阴性，并决定是否需要以全新预注册版本和真正独立评估数据研究不同的风险定义；在此之前，主叙事不得把 ensemble-std 拒选写成已验证阳性发现。
"""
    (OUT / "PAPER2_RISK_FROZEN_V1.md").write_text(text)


if __name__ == "__main__":
    main()
