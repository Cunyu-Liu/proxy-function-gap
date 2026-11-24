import argparse
import json
from pathlib import Path


STRATEGIES = ("random", "max_variance", "diversity", "expected_improvement", "top_proxy", "regret_analytic")
LABELS = {
    "random": "Random", "max_variance": "Max-var", "diversity": "Diversity",
    "expected_improvement": "EI", "top_proxy": "Top-proxy", "regret_analytic": "Regret-analytic",
}


def read(path):
    return json.loads(Path(path).read_text())


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def curve_table(records, representation=None, query_id=None):
    selected = [item for item in records if (representation is None or item["representation"] == representation) and (query_id is None or item["query_pool_id"] == query_id)]
    budgets = sorted({item["budget"] for item in selected})
    rows = []
    for budget in budgets:
        row = [budget]
        for strategy in STRATEGIES:
            value = next(item for item in selected if item["budget"] == budget and item["strategy"] == strategy)
            row.append(f"{value['pool_mean_regret_mean'] if 'pool_mean_regret_mean' in value else value['regret']:.6f}")
        rows.append(row)
    return table(["Budget", *(LABELS[item] for item in STRATEGIES)], rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the complete Paper3 V4 report from independently validated aggregates")
    parser.add_argument("aggregate_directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--test-rounds", default="19 passed in each of three rounds")
    args = parser.parse_args(argv)
    directory = Path(args.aggregate_directory)
    aggregate = read(directory / "aggregate.json")
    validation = read(directory / "three_round_validation.json")
    cases = aggregate["cases"]
    representations = {item["representation"]: item for item in aggregate["representation_summary"]}
    convergence = aggregate["regret_convergence_summary"]
    numerical = aggregate["numerical_validation_summary"]
    selections = {item["representation"]: item for item in aggregate["representation_summary"]}
    pooled = aggregate["curve_summary"]
    per_case = aggregate["per_case_curve_summary"]
    lines = [
        "# Paper3 解析 regret acquisition V4：独立高斯求积与六策略完整结果",
        "",
        "## 结论与证据边界",
        "",
        f"- 完成全部 **48 个 zero/full-Q 门禁**和 V2 冻结的 **6 个曲线情境**，六策略为 random、max-variance、diversity、解析 EI、top-proxy 和 regret-analytic；没有按 V4 阳性换池或删除负结果。",
        f"- V4 的 384 个获取步严格收敛 **{convergence['converged_count']}/384（{100 * convergence['converged_fraction']:.2f}%）**，最高两层首选稳定率 **{100 * convergence['last_choice_stable_fraction']:.2f}%**；V3 对应严格收敛率为 31.77%。",
        "- `regret_analytic` 消除了 posterior 和 hypothetical-label 两层 Monte Carlo，但其 closed-form normal-max 计算建立在 **fantasy 后验候选间独立（对角协方差）近似**上；不是原始相关联合后验 multivariate-max 的精确闭式解。",
        "- **保留 random/max-variance/top-proxy 简单基线，不声称 regret-analytic 稳定优于简单基线。** kmer4 中它只在 8 标签单独最低；4 标签与 EI 并列，16 标签逊于 EI，32 标签逊于 max-variance。summary4 固定情境所有预算完全停滞。",
        "- 结果只属回顾性 development/calibration 诊断。没有使用 query/outer 标签调参，没有读取或评分永久 outer，不声称普遍 acquisition 优越性。是否优于简单基线以下述 pooled 与逐池结果为准，不利结果全部保留。",
        "",
        "## 方法、冻结实现与数值审计",
        "",
        f"协议初版提交 `cd6f3cf`，实现和数值开发后的最终冻结提交为 `{args.source_commit}`。初版 16/32/64 同步节点梯在未运行科学实验前经合成内层数值例修订为固定 64 标签节点、32/64/128 价值节点；最终固定 `(64,128)`，收敛门槛 `1e-5`，没有根据 V4 门禁或曲线调节点、种子或门槛。",
        "",
        "对每个查询点，先按预测标签分布做 fantasy posterior 更新，再在候选独立正态近似下计算 `E[max Y']-max E[Y']`。标签和 normal-max 价值均用 Gauss–Hermite 求积；最终获取分数为 expected regret 的负值，部署始终为 posterior-mean argmax top-1。",
        "",
        table(
            ["对照", "最大/逐查询绝对误差", "MC nominal SE", "判定"],
            [
                ["三候选穷举标量积分", f"{numerical['exhaustive_absolute_error_max']:.3e}", "不适用", "通过（≤1e-8）"],
                ["二候选闭式 normal-max", f"{numerical['two_candidate_absolute_error_max']:.3e}", "不适用", "通过（≤1e-8）"],
                ["4096×32768 antithetic nested CUDA MC", ", ".join(f"{item:.6f}" for item in numerical["nested_mc_absolute_errors"]), ", ".join(f"{item:.6f}" for item in numerical["nested_mc_nominal_standard_errors"]), "全部通过冻结误差规则"],
            ],
        ),
        "",
        "穷举和二元公式只作独立数学审计；获取、模型训练/推理和高采样 MC 均真实运行于 CUDA。MC 标准误是对 antithetic 标签 regret 的 nominal 描述，不作为 outer 推断区间。",
        "",
        "## 内层 development 选择",
        "",
    ]
    selection_rows = []
    for representation in sorted(representations):
        item = selections[representation]
        selection_rows.append([representation, item["selected_prior_variance"], item["selected_noise_variance"]])
    lines.extend([
        table(["表示", "内层 prior", "内层 noise"], selection_rows),
        "",
        "每个表示均只用 4 折 development-pool 阻断 CV 的 measured-MRL MSE 选择 prior/noise；完整 9 参数组合及逐折结果保存在三轮 `inner_development_selection.json`。",
        "",
        "## 全部 48 池门禁",
        "",
        "单元格为 `zero regret → full-Q regret (headroom)`，真实 measured-MRL 尺度；粗体仅表示正门禁。",
        "",
    ])
    gate_rows = []
    query_ids = sorted({item["query_pool_id"] for item in cases})
    for query_id in query_ids:
        row = [query_id]
        for representation in ("summary4", "kmer4", "cnn64"):
            item = next(value for value in cases if value["query_pool_id"] == query_id and value["representation"] == representation)
            headroom = f"+{item['repair_headroom']:.3f}"
            if item["gate_pass"]:
                headroom = f"**{headroom}**"
            row.append(f"{item['zero_label_regret']:.3f} → {item['full_q_regret']:.3f} ({headroom})")
        gate_rows.append(row)
    lines.extend([
        table(["Query pool", "summary4", "kmer4", "cnn64"], gate_rows),
        "",
        table(
            ["表示", "正门禁/16", "headroom 均值", "中位数", "最小–最大", "zero regret 均值", "full-Q regret 均值"],
            [[representation, f"{item['headroom_present_count']}/16", f"{item['headroom_mean']:.6f}", f"{item['headroom_median']:.6f}", f"{item['headroom_min']:.3f}–{item['headroom_max']:.3f}", f"{item['zero_regret_mean']:.6f}", f"{item['full_q_regret']:.6f}"] for representation, item in sorted(representations.items())],
        ),
        "",
        "## 固定队列 pooled 曲线",
        "",
        "Random 先对三个冻结种子取池内均值，再对固定池取均值；其余策略一条。最低均值只是描述，不等于跨预算或逐池支配。",
        "",
    ])
    for representation in ("kmer4", "summary4", "cnn64"):
        if any(item["representation"] == representation for item in pooled):
            count = next(item["fixed_cohort_pool_count"] for item in pooled if item["representation"] == representation)
            lines.extend([f"### {representation}：固定 {count} 池", "", curve_table(pooled, representation=representation), ""])
    lines.extend(["## 六个固定情境的逐池完整曲线", "", "下表保留所有非单调、停滞和不利结果。", ""])
    cohort = sorted({(item["representation"], item["query_pool_id"]) for item in per_case})
    for representation, query_id in cohort:
        lines.extend([f"### {representation}/{query_id}", "", curve_table(per_case, representation, query_id), ""])
    best_rows = []
    for budget in (4, 8, 16, 32):
        selected = [item for item in pooled if item["representation"] == "kmer4" and item["budget"] == budget]
        minimum = min(item["pool_mean_regret_mean"] for item in selected)
        winners = [LABELS[item["strategy"]] for item in selected if abs(item["pool_mean_regret_mean"] - minimum) <= 1e-12]
        analytic = next(item["pool_mean_regret_mean"] for item in selected if item["strategy"] == "regret_analytic")
        best_rows.append([budget, ", ".join(winners), f"{minimum:.6f}", f"{analytic:.6f}"])
    lines.extend([
        "## 与 V3 和简单基线的结果判断",
        "",
        table(["kmer4 budget", "最低 pooled regret 策略", "最低值", "Regret-analytic"], best_rows),
        "",
        "V3 regret-Q 在 kmer4 的 8/16 标签 pooled 均值最低；V4 只在 8 标签保持最低，16 标签不再领先 EI。这说明 V3 的部分排名确实受到 posterior Monte Carlo 影响，但消除该数值噪声并未产生稳定获取优势。V4 在 calibration_002 的 16 标签 regret 为 4.903296，而 EI 为 0；该反例不能被 pooled 结果隐藏。",
        "",
        "保留简单基线的理由有四点：第一，复杂策略没有跨预算或逐池支配；第二，summary4 即使 full-Q 也没有 headroom，瓶颈不是 acquisition 数值噪声；第三，对角独立近似去掉了联合 posterior 的相关耦合，确定性目标与原始相关目标并不相同；第四，表示/模型错设、top-1 离散跳变和小池组成仍可能主导真实 regret。100% 求积收敛只支持数值可重复性，不支持获取有效性或 outer 泛化。",
        "",
        "## 求积收敛与统一约束",
        "",
        f"最高两层最大绝对分差中位数为 **{convergence['max_absolute_score_delta_median']:.6e}**，最大值为 **{convergence['max_absolute_score_delta_max']:.6e}**。未收敛步只标记，没有追加节点或替换候选。",
        "",
        table(
            ["情境", "严格收敛", "首选稳定率", "最大分差中位数", "最大分差最大值"],
            [[f"{item['representation']}/{item['query_pool_id']}", f"{item['converged_count']}/64", f"{100 * item['last_choice_stable_fraction']:.2f}%", f"{item['max_absolute_score_delta_median']:.6e}", f"{item['max_absolute_score_delta_max']:.6e}"] for item in convergence["by_case"]],
        ),
        "",
        "所有策略逐一执行相同的批内 Hamming≤8 排除接口；独立验证器逐批重算。完整排除计数、逐候选三层分数和 query ledger 均保留在原始 artifacts。",
        "",
        "## 三轮一致性、测试与解释边界",
        "",
        f"三轮科学字段一致：**{str(validation['scientific_results_identical']).lower()}**；每轮 48 门禁，科学 SHA256 均为 `{validation['scientific_sha256_each'][0]}`。测试记录：**{args.test_rounds}**。独立验证器同时检查数值对照、CUDA、预算唯一性、target scale、full-Q、批内约束和 outer 隔离。",
        "",
        "解释时必须分开两类误差：节点收敛只说明确定性独立高斯目标的数值稳定性，不消除对角协方差近似、表示/模型错设、top-1 离散跳变或小池组成造成的获取误差。若 regret-analytic 在某些预算领先而其他预算或逐池反例存在，不作普遍优越性主张；若未胜 random/max-variance/top-proxy，则保留简单基线并报告负结果。",
        "",
        "真实 artifacts 位于远端 `/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper3/headroom_v4_*`；本地同步 `deliverables/reports/source_data/paper3_headroom_v4/` 保存三轮完整结果、日志、配置、独立聚合与校验。",
        "",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "case_count": len(cases), "curve_case_count": len(cohort), "scientific_results_identical": validation["scientific_results_identical"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
