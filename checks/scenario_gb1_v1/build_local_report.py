#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def f(value):
    return f"{value:.6f}"


def fo(value):
    return "NA" if value is None else f(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    package = Path(args.package)
    aggregate = json.loads((package / "aggregate/test_aggregate.json").read_text())
    selection = json.loads((package / "aggregate/dev_selection.json").read_text())
    round1 = json.loads((package / "verification/round1_raw.json").read_text())
    round2 = json.loads((package / "verification/round2_independent.json").read_text())
    primary = aggregate["primary_delta_regret"]
    low, high = aggregate["primary_ci95"]
    if primary < 0:
        direction = "BoN 的 cap16 regret 低于 RandomN"
        if high < 0:
            interpretation = "在预设 55 个固定池上，cap16 改善且位置对簇 bootstrap 95% CI 不含 0；这只支持 GB1 回顾性固定池选择，不构成跨场景普遍复现。"
        else:
            interpretation = "cap16 点估计方向有利，但位置对簇 bootstrap 95% CI 含 0，不能声称稳健改善。"
    else:
        direction = "BoN 的 cap16 regret 未低于 RandomN"
        interpretation = "预设主终点未显示改善；该负结果保留，不能用点态指标或其他预算替代主终点作阳性结论。"
    rows = []
    for budget in (1, 4, 16, 64):
        item = aggregate["budgets"][str(budget)]
        ci = aggregate["bootstrap"][str(budget)]["ci95"]
        rows.append(f"| {budget} | {f(item['bon_regret'])} | {f(item['random_regret'])} | {f(item['delta_regret_bon_minus_random'])} | [{f(ci[0])}, {f(ci[1])}] | {f(item['hit_at_budget'])} | {f(item['top16_recall'])} | {f(item['top_decile_recall'])} |")
    dev_rows = []
    for item in selection["selection_summary"]:
        dev_rows.append(f"| {item['lambda']:g} | {f(item['mean_dev_cap16_delta_regret'])} | {f(item['mean_dev_cap16_bon_regret'])} | {f(item['mean_dev_spearman'])} | {item['max_iterations']} | {item['max_relative_residual']:.3e} |")
    seed_rows = []
    for item in aggregate["per_seed_budgets"]:
        seed_rows.append(f"| {item['seed']} | {f(item['budgets']['16']['bon_regret'])} | {f(item['budgets']['16']['delta_regret_bon_minus_random'])} |")
    point = aggregate["point_metrics_mean_over_seeds"]
    single = aggregate["single_pool_diagnostics"]
    text = f"""# GB1/IgG 位点阻断场景实验 v1

日期：2026-11-28
状态：完成固定训练、dev 选型、冻结预测与永久 test 单次评价。
科学边界：公开历史 DMS 的**回顾性揭盲**，不是新盲测、前瞻实验或湿实验验证。

## 0. 结论

**{direction}**：BoN−RandomN 的真实 `lnW` cap16 regret 差为 **{f(primary)}**，位置对簇 bootstrap 95% CI **[{f(low)}, {f(high)}]**。{interpretation}

本实验不预设普遍阳性，也没有把 GB1 单场景结果写成 decision-focused 结论已跨场景成立。所有终点均为 IgG-Fc 结合富集派生的 **`lnW`**，高分更优；它**不是 `Kd`**、直接结合自由能或稳定性测量。

## 1. 数据、隔离与固定池

- 来源：ProteinGym v1.3 `SPG1_STRSG_Olson_2014.csv`，SHA-256 `1050a28d1df6130c85b974d782a5aad4d96873f1e36328d9ccf2f57a966684cd`，536,962 条唯一变异。
- 位点按冻结 SHA-256 规则分为 train 39、dev 5、outer test 11；跨界双突变 245,049 条进入 `boundary_exclude`，不按标签再纳入。
- train/dev/test 分别为 268,145/3,705/20,063 条。位点互斥提供跨角色 exact/Hamming-1 为零的结构证书。
- test 的 55 个位置对各均匀无放回抽 256 个双突变，共 14,080 个池内变异；单突变不混入主池。
- `lnW` 下限记录 119,422 条（22.24%）保留；top-tail 指标使用真实标签并列阈值，不删除退化或低值记录。

## 2. 模型与只用 train/dev 的选型

模型为可审计 one-hot 加性 ridge：截距不惩罚，使用 20×19=380 个有向替换类型（野生型氨基酸→替换氨基酸）并跨位点共享效应，双突变预测为两个效应之和。该设计使被阻断的新位点可评分，但舍弃位点特异效应与上位性，是预先保留的模型局限。训练在 NVIDIA RTX 5090 上以 float64 CUDA 预条件共轭梯度求解；5 个固定初始化种子为 11/23/37/53/71，所有 35 个正则×种子组合均达到相对残差 ≤1e−10。

正则只按 dev 十个位置对池的 cap16 BoN−RandomN regret 选择，tie 依次由 dev Spearman 和较小正则决定。RandomN 是无放回均匀随机 B 项最大真实 `lnW` 的精确期望。

| lambda | dev cap16 Δregret | dev cap16 BoN regret | dev Spearman | 最大迭代 | 最大相对残差 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(dev_rows)}

冻结 lambda：**{selection['selected_lambda']:g}**。随后只对无标签 test 生成一次五种子预测；独立 evaluator 才读取 test 标签保险库，并一次性计算全部预算和辅助终点。

## 3. 永久 test 主终点与 top-tail 分解

Δregret = BoN regret − RandomN regret，负值有利于 proxy BoN。每池先在五个种子间聚合，再按位置对整簇 bootstrap 10,000 次。

| B | BoN regret | RandomN regret | Δregret | 簇 bootstrap 95% CI | hit@B | true top16 recall | true top-decile recall |
|---:|---:|---:|---:|:---|---:|---:|---:|
{chr(10).join(rows)}

cap16 单池诊断：{single['pools_with_negative_delta']}/55 池 Δ<0，{single['pools_with_positive_delta']}/55 池 Δ>0；最大单池绝对 Δ 占全部绝对 Δ 的 {single['max_absolute_share_of_total_absolute_delta']:.4%}。因此报告不隐藏反向池，也不把总体方向归于单一池。

| seed | cap16 BoN regret | cap16 Δregret |
|---:|---:|---:|
{chr(10).join(seed_rows)}

## 4. 点态与截断分解

| 指标 | 五种子均值 |
|:---|---:|
| MAE | {f(point['mae'])} |
| Spearman | {f(point['spearman'])} |
| censored MAE / Spearman | {f(point['censored_mae'])} / {fo(point['censored_spearman'])} |
| uncensored MAE / Spearman | {f(point['uncensored_mae'])} / {fo(point['uncensored_spearman'])} |
| 池内下限比例 | {point['censored_fraction']:.4%} |

点态误差与排序仅作辅助诊断，不用于替换 cap16 决策终点。下限处的 MAE/Spearman 也不被解释为精确结合差异。

## 5. 三轮核验与失败保留

1. **原始核验：{'PASS' if round1['pass'] else 'FAIL'}。** 独立标准库重读原始 CSV，逐条重建序列，复核角色、哈希、下限、bin 和无标签池重建；见 `verification/round1_raw.json`。
2. **独立算术核验：{'PASS' if round2['pass'] else 'FAIL'}。** 不导入生产模块，重算 55×5×4=1,100 个池×种子×预算 regret、聚合 MAE/Spearman、模型哈希与全部 CUDA 收敛记录；见 `verification/round2_independent.json`。
3. **交付核验：** 本地哈希、报告数值、四预算、辅助指标与脱敏边界由 `verification/round3_delivery.json` 记录；本机不同步原始 CSV、逐序列 test 标签、权重或预测数组。

远端初次 `px2` 别名不能解析；完整 SSH 可用。官方 ZIP 下载先遇到远端缺中间证书，本机直连又发生慢速超时；均保留日志，未把失败写成数据通过。最终以固定 SHA-256 和逐行审计为数据门禁。

## 6. 可复现路径

- 远端代码：`/home/proxy-function-gap-scenario-gb1`
- 远端完整 artifacts：`/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/scenario_gb1_v1`
- 本地脱敏聚合与代码：`deliverables/reports/source_data/scenario_gb1_v1/`
- 场景协议：`deliverables/reports/docs/SCENARIO_EXPANSION_PROTEIN_20261127.md`

结论仅适用于该公开 GB1 已测局部景观和固定 N=256 池，不能外推为新设计实验、优化轨迹动力学或 Kd 预测结论。
"""
    output = Path(args.output)
    output.write_text(text)
    files = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name not in {"delivery_manifest.json", "round3_delivery.json"}:
            relative = str(path.relative_to(package))
            files[relative] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"schema_version": "scenario_gb1_v1.local_delivery", "files": files, "report": {"bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, "raw_sequences_or_labels_included": False}
    (package / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
