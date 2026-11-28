#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/scenario_mrna_non_tcell_v1")
CAPS = [1, 2, 4, 8, 16]
CONTEXTS = ["HepG2|TGFBR2", "HepG2|PDCD1", "K562|TGFBR2", "K562|PDCD1"]


def f(value, digits=4):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def ci(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, str):
        value = json.loads(value)
    return f"[{f(value[0], digits)}, {f(value[1], digits)}]"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |", "|" + "|".join(["---"] * len(headers)) + "|"] + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def effect_language(row):
    interval = row["ci95"]
    if interval is None:
        return "区间不可用"
    if row.get("holm_p") is not None and row["holm_p"] < 0.05 and interval[1] < 0:
        return "BoN 获益（探索性校正后）"
    if row.get("holm_p") is not None and row["holm_p"] < 0.05 and interval[0] > 0:
        return "Random(N) 获益（探索性校正后）"
    if interval[1] < 0:
        return "点估计/名义区间有利 BoN，校正后不显著"
    if interval[0] > 0:
        return "点估计/名义区间有利 Random(N)，校正后不显著"
    return "区间跨 0"


def main():
    protocol = json.loads((ROOT / "protocol" / "frozen_protocol.json").read_text())
    execution = json.loads((ROOT / "verification" / "round1_execution.json").read_text())
    summary = pd.read_csv(ROOT / "results" / "decision_summary.tsv", sep="\t")
    boot = pd.read_json(ROOT / "results" / "decision_bootstrap.json")
    boot_records = boot.to_dict("records")
    candidate = pd.read_csv(ROOT / "results" / "candidate_master.tsv", sep="\t")
    stability = pd.read_csv(ROOT / "results" / "stability_summary.tsv", sep="\t")
    baseline = pd.read_csv(ROOT / "results" / "strong_kozak_baseline.tsv", sep="\t")
    mrl_diagnostics = json.loads((ROOT / "results" / "mrl_frozen_diagnostics.json").read_text())
    budget = json.loads(Path(protocol["historical_inputs"]["budget_results"]).read_text())
    leppek = json.loads((ROOT / "results" / "leppek_historical_summary.json").read_text())
    schlusser = json.loads((ROOT / "results" / "schlusser_historical_summary.json").read_text())["summary"]
    isolation = json.loads((ROOT / "protocol" / "isolation_audit.json").read_text())
    independent_path = ROOT / "verification" / "round2_independent.json"
    independent = json.loads(independent_path.read_text()) if independent_path.exists() else None

    def s(pool, context, cap):
        selected = summary.loc[(summary["pool"] == pool) & (summary["context"] == context) & (summary["cap"] == cap)]
        if len(selected) != 1:
            raise ValueError((pool, context, cap, len(selected)))
        return selected.iloc[0]

    def b(pool, context, cap):
        selected = [x for x in boot_records if x["pool"] == pool and x["context"] == context and x["cap"] == cap]
        if len(selected) != 1:
            raise ValueError((pool, context, cap, len(selected)))
        return selected[0]

    indel_cap16 = [b("indel_kmer_48", f"primary|{context}", 16) for context in CONTEXTS]
    benefits = sum(x["ci95"][1] < 0 and x["holm_p"] < 0.05 for x in indel_cap16)
    harms = sum(x["ci95"][0] > 0 and x["holm_p"] < 0.05 for x in indel_cap16)
    stability_cap16 = b("stability_kmer_27", "primary|integrated_abundance", 16)
    stability_cap16_point = float(s("stability_kmer_27", "primary|integrated_abundance", 16)["bon_minus_random_n"])
    lines = []
    lines.append("# mRNA 非 T-cell 场景扩展：统一决策分析")
    lines.append("")
    lines.append("## 1. 证据定位与结论")
    lines.append("")
    lines.append("本包分析 Castillo-Hair 2024、Leppek 2022 与 Schlusser 2024 的历史公开数据，并复用 HEK293T N50/HepG2 MRL 的既有冻结预测和等预算结果。所有新计算均为**回顾性固定池分析**，不是新盲测、前瞻验证或湿实验；没有新训练，也没有新增 T-cell 分析。indel、稳定性与 TE 的结局只在各自终点内部计算 regret，绝不计算跨终点 MAE。")
    lines.append("")
    lines.append(f"- **48 UTR indel：** cap16 的四个 cell×target 情境中，校正后有利于 BoN 的情境 {benefits}/4，有利于 Random(N) 的情境 {harms}/4；完整方向和区间见表 3，不能概括为普遍阳性或普遍阴性。")
    lines.append(f"- **27 UTR 稳定性：** cap16 BoN−Random(N) integrated-abundance regret 差为 **{f(stability_cap16_point)}**，95% 区间 {ci(stability_cap16['ci95'])}，{effect_language(stability_cap16)}。这是固定 27 条构建体的结果，8 条 spike-in 只用于校准。")
    lines.append("- **MRL 同终点边界：** HEK293T 的高 cap 未显示持续改善；HepG2 在等预算 Random(N) 对照下部分 cap 有利，但完整区间和多重校正不支持单调获益叙事。Leppek 变长 UTR 的历史结果同样未显示显著获益。")
    lines.append(f"- **Schlusser HepG2 TE：** 仅整合历史一次结果。cap16 BoN-CNN−Random1 regret 差 **{f(schlusser['primary']['paired_comparison']['mean_regret_difference_left_minus_right'], 6)}**，基因簇 95% 区间 {ci(schlusser['primary']['paired_comparison']['gene_cluster_bootstrap_95_interval'], 6)}，没有观察到改善证据，也不能据此宣称 Random 普遍更好。本轮未重新揭盲或重称盲测。")
    lines.append("")
    lines.append("总体上，decision-focused 结论呈**情境和终点依赖性**：proxy 排名、绝对 MRL 误差和真实功能选择不能互相替代。")
    lines.append("")
    lines.append("## 2. 样本单位、计算设备与功效限制")
    lines.append("")
    n_rows = [
        ["HEK293T N50 MRL", "1,024 candidates=16 pools×64", "16 complete pools；单一研究/细胞", "历史冻结；算法种子不增加生物 n"],
        ["HepG2 defined-end MRL", "1,024 candidates=16 pools×64", "16 complete pools；单一研究/细胞", "历史冻结；非独立生物重复池"],
        ["Castillo indel", "48 UTR；192 construct-context summaries", "48 个独立 UTR，四情境共享", "剂量和重复不增加 n；小样本探索性区间"],
        ["Castillo 稳定性", "27 experimental+8 spike-in", "27 UTR；spike-in n=0 candidates", "六个时间点不是重复；稳定性无独立生物重复"],
        ["Leppek MRL", "59 constructs/49 UTR", "49 UTR clusters", "历史回顾性；单研究"],
        ["Schlusser HepG2 TE", "6,340 rows/6,309 genes", "6,309 gene clusters；单研究", "重尾；多转录本/相同 UTR50 不可识别"],
    ]
    lines.append(table(["场景", "分析 n", "独立/聚类单位", "主要功效限制"], n_rows))
    lines.append("")
    lines.append("新计算使用真实 **CPU float32** 冻结三成员 4-mer MLP 推理（48 条原生全长 UTR）及 CPU 统计；未训练，未调用 CNN，未使用 CUDA。HEK293T/HepG2 的 CNN 预测来自历史冻结 artifacts；本包没有重跑。indel 主分析复现作者 flagged 排除规则，K562 的 25 个 flagged 原始行明确排除并另做全行敏感性；其余缺失不插补。")
    lines.append("")
    lines.append("## 3. HEK293T/HepG2：冻结等预算 MRL 结果与诊断")
    lines.append("")
    mrl_rows = []
    for context in ("n50_hek293t_r1", "defined_hepg2_r1"):
        for cap in CAPS:
            source = next(x for x in budget["families"]["bon_minus_random_n"] if x["context"] == context and x["cap"] == cap)
            means = next(x for x in budget["context_summaries"] if x["context"] == context and x["cap"] == cap)
            mrl_rows.append([context, cap, f(means["bon_regret"]), f(means["random_n_regret"]), f(source["effect"]), ci(source["nominal_95_ci"]), f(source["holm_p"])])
    lines.append(table(["情境", "cap", "BoN regret", "Random(N) regret", "差值", "95% CI", "Holm p"], mrl_rows))
    lines.append("")
    diag_rows = []
    for row in mrl_diagnostics:
        diag_rows.append([row["context"], row["candidates"], f(row["mae"]), ci(row["mae_ci95"]), f(row["spearman"]), ci(row["spearman_ci95"]), f(row["top_decile_recall"]), f(row["top_tail_regret"])])
    lines.append(table(["情境", "n", "MAE", "MAE CI", "Spearman", "Spearman CI", "top-10% recall", "top-tail regret"], diag_rows))
    lines.append("")
    lines.append("top-tail 在每个 64 序列池内取 `k=ceil(0.1n)=7`，再对 16 池求均值；recall 是 proxy 与实测 top-k 的重合比例。该诊断是回顾性补充，不改写原冻结 regret 主结果。")
    lines.append("")
    lines.append("## 4. 48 UTR indel：主分析")
    lines.append("")
    lines.append("每个 cell×target×dose 内，以 Strong Kozak 两重复均值为参考，对该剂量下每个保留重复做比值，再跨剂量/重复取不加权均值和样本 SD。192/192 个均值和 SD 与作者汇总一致，最大绝对误差均为 `2.22e-16` 量级。BoN 与 Random(N) 使用同一嵌套提案前缀；regret 差小于 0 有利于 BoN。")
    lines.append("")
    indel_rows = []
    for context in CONTEXTS:
        for cap in CAPS:
            row = s("indel_kmer_48", f"primary|{context}", cap)
            br = b("indel_kmer_48", f"primary|{context}", cap)
            indel_rows.append([context, cap, f(row["bon_regret"]), f(row["random_n_regret"]), f(row["bon_minus_random_n"]), ci(br["ci95"]), f(br["bootstrap_p"]), f(br["holm_p"])])
    lines.append(table(["情境", "cap", "BoN regret", "Random(N) regret", "差值", "95% CI", "sign p", "Holm p"], indel_rows))
    lines.append("")
    lines.append("区间以完整 UTR 为重采样单位，携带全部四情境；每个重采样池沿用 200 个冻结提案前缀及 Random(N) 位置，并重新计算 BoN 选择和实测 oracle。主分析 10,000 次重采样、200 个算法种子；Holm 家族包含全部 20 个 context×cap 检验。它们是历史固定面板的探索性不确定性描述，不是前瞻功效保证。")
    lines.append("")
    lines.append("### 4.1 实测 MRL、公开预测 MRL 与 4-mer MLP 不可混用")
    lines.append("")
    prediction_rows = []
    for subset_name, subset in (("48 panel", candidate), ("27 stability subset", candidate.loc[candidate["integrated_abundance"].notna()])):
        measured = subset.loc[subset["measured_mrl"].notna()]
        for proxy, label in (("kmer_mlp", "4-mer MLP"), ("published_predicted_mrl", "作者 Predicted MRL")):
            prediction_rows.append([subset_name, label, len(measured), f(np.mean(np.abs(measured[proxy] - measured["measured_mrl"]))), f(stats.spearmanr(measured[proxy], measured["measured_mrl"]).statistic)])
    lines.append(table(["面板", "proxy", "实测 MRL n", "同终点 MAE", "同终点 Spearman"], prediction_rows))
    lines.append("")
    lines.append("48 条中仅 12 条有实测 MRL，45 条有作者预测 MRL，VAT1/LAMA5/Strong Kozak 两者均缺失；27 条稳定性子集中实测 MRL 仅 8 条。公开预测的 n45 是单独固定池，绝不把预测值回填到实测缺失。下表将实测 MRL 选择明确标为 oracle-only，不能解释为可部署策略。")
    lines.append("")
    proxy_rows = []
    for pool, proxy_name, cap in (("indel_kmer_48", "4-mer MLP n48", 16), ("indel_published_45", "作者预测 MRL n45", 16), ("indel_measured_12", "实测 MRL oracle n12", 8), ("indel_strict_h1", f"4-mer strict-H1 sensitivity n{isolation['conflict_free_utrs']}", 16)):
        for context in CONTEXTS:
            label = f"primary|{context}"
            row = s(pool, label, cap)
            proxy_rows.append([proxy_name, context, cap, f(row["bon_regret"]), f(row["random_n_regret"]), f(row["bon_minus_random_n"])])
    lines.append(table(["proxy/pool", "情境", "cap", "选择 regret", "Random(N) regret", "差值"], proxy_rows))
    lines.append("")
    lines.append(f"Sample 隔离审计发现 **{isolation['conflicted_utrs']}/48** 条 UTR 存在 native full UTR 或 N50 terminal-50 exact/H1 冲突；其余 {isolation['conflict_free_utrs']} 条进入 strict-H1 敏感性。完整逐序列角色在 `protocol/isolation_audit.json`。因此全 48 主池不能包装成训练谱系完全隔离的新外测。")
    lines.append("")
    lines.append("### 4.2 Strong Kozak 与敏感性")
    lines.append("")
    baseline_rows = []
    for context in CONTEXTS:
        label = f"primary|{context}"
        row = baseline.loc[(baseline["pool"] == "indel_kmer_48") & (baseline["context"] == label)].iloc[0]
        baseline_rows.append([context, f(row["selected_endpoint"]), f(row["regret"])])
    lines.append(table(["情境", "Strong Kozak normalized indel", "固定池 regret"], baseline_rows))
    lines.append("")
    sensitivity_rows = []
    for endpoint_type in ("primary", "all_flagged", "absolute_2pmol"):
        for context in CONTEXTS:
            row = s("indel_kmer_48", f"{endpoint_type}|{context}", 16)
            sensitivity_rows.append([endpoint_type, context, f(row["bon_regret"]), f(row["random_n_regret"]), f(row["bon_minus_random_n"])])
    lines.append(table(["终点", "情境", "BoN regret", "Random(N) regret", "差值"], sensitivity_rows))
    lines.append("")
    lines.append("绝对 2 pmol indel 与 Kozak 归一化 indel 的量纲不同，差值不能跨终点比较大小；此处只比较各自同终点内的选择方向。")
    lines.append("")
    lines.append("## 5. 27 实验序列稳定性固定池")
    lines.append("")
    lines.append("每个时间点只用 spikein_1/2/3 的 `[1,0.1,0.01]` 浓度拟合 `log10(concentration)~log10(UMI)`，随后换算实验序列并按 IVT/mean(IVT) 归一化；在 0.5、1、2、4、8、24 h 间线性插值并做梯形积分。没有拟合单指数半衰期。8 条 spike-in 不作为候选或独立样本。")
    lines.append("")
    stability_rows = []
    for cap in CAPS:
        row = s("stability_kmer_27", "primary|integrated_abundance", cap)
        br = b("stability_kmer_27", "primary|integrated_abundance", cap)
        stability_rows.append([cap, f(row["bon_regret"]), f(row["random_n_regret"]), f(row["bon_minus_random_n"]), ci(br["ci95"]), f(br["holm_p"])])
    lines.append(table(["cap", "BoN regret", "Random(N) regret", "差值", "95% CI", "Holm p"], stability_rows))
    lines.append("")
    cross_rows = []
    for pool in ("stability_kmer_27", "stability_published_27"):
        for context in CONTEXTS:
            label = f"cross_endpoint_indel|{context}"
            row = s(pool, label, 16)
            cross_rows.append([pool, context, f(row["bon_regret"]), f(row["random_n_regret"]), f(row["bon_minus_random_n"])])
    lines.append(table(["27 池 proxy", "indel 情境", "BoN indel regret", "Random(N) indel regret", "差值"], cross_rows))
    lines.append("")
    lines.append(f"27 条 integrated abundance 的范围为 {f(stability['integrated_abundance'].min())}–{f(stability['integrated_abundance'].max())}；free-to-total 仅作机制诊断。跨 indel 的结果是同一 27 UTR 的回顾性功能对照，不是稳定性模型外测，也不构成多目标可部署策略。")
    lines.append("")
    lines.append("## 6. Leppek 与 Schlusser：历史证据整合")
    lines.append("")
    lines.append(f"Leppek 59 constructs/49 UTR 的历史 BoN−Random1 MRL regret 差为 {f(leppek['primary']['bon_minus_random_regret'], 6)}，95% CI {ci(leppek['primary']['ci95'], 6)}；MAE={f(leppek['secondary']['mae']['value'])}，Spearman={f(leppek['secondary']['spearman']['value'])}。该比较不是本轮新 Random(N) 结果，不能与新固定池差值混排为同一协议。")
    lines.append("")
    pair = schlusser["primary"]["paired_comparison"]
    lines.append(f"Schlusser HepG2 TE 的 cap16 BoN-CNN regret={f(schlusser['primary']['configuration']['mean_regret'], 6)}，Random1 regret={f(schlusser['primary']['configuration']['mean_regret'] - pair['mean_regret_difference_left_minus_right'], 6)}；CNN MRL–TE Spearman={f(schlusser['endpoint_alignment_descriptive']['spearman_cnn_MRL_vs_TE'], 6)}。MAE 和 top-tail 本轮均不新增，以遵守只整合历史一次结果的边界。TE 极端右偏，绝对 regret 由池内 oracle 主导，必须优先看配对差及其跨零区间。")
    lines.append("")
    lines.append("## 7. 三轮核验与交付边界")
    lines.append("")
    lines.append(f"1. **原始/执行核验：** {execution['status']}。原始哈希、48/12/45 覆盖、192 个作者归一化汇总、27+8 映射、模型哈希、合成测试与 CPU 设备记录通过。首次 n12/cap16 门禁失败及初版协议均保留，修正发生在 decision 结果产生前。另有一次 SSH 停止后远端进程继续写入造成的并发混版，整批已隔离且不作为科学证据；正式结果由排他执行锁重新生成。")
    lines.append(f"2. **独立核验：** {independent['status'] if independent else 'PENDING'}。独立脚本不导入生产模块，重读 XLSX/CSV、用 NumPy 重写 MLP 前向、复算全部逐种子选择和完整 bootstrap。")
    lines.append("3. **交付核验：** 最终结果表、报告、源码快照、失败日志和哈希清单一并同步；本地另做哈希和报告数值断言。负结果、区间跨零及协议限制均保留。")
    lines.append("")
    lines.append("所有 oracle 都是相应固定池内的实测最优，不是全部可能 UTR、整个转录组或体内疗效最优。小样本区间和显著性不能提升独立研究数，也不能支持普遍临床结论。")
    lines.append("")
    (ROOT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
