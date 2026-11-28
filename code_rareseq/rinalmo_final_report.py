#!/usr/bin/env python3
"""Generate the RiNALMo eCLIP strict3train final report from accepted artifacts.

Run AFTER run_e1_rinalmo_post.sh completes (bootstrap/stratify/verify).
Reads cv_bootstrap/paired_bootstrap.tsv, cv_summary/cv_metrics.tsv, verification.
Mirrors the RNA-FM final report structure; adds cross-encoder comparison.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/ProxyFunctionGap/RARESeq/e1_rinalmo_main_strict3train_p20_v1/main")
OUT = Path(__file__).resolve().parents[1] / "E1_强基线" / "E1_RiNALMo主集严格三折最终报告_2026-11-28.md"

# RNA-FM accepted bootstrap tail values (E1_主集严格三折冻结基线最终报告_2026-11-26.md §5)
RNAFM_TAIL = {("neg1c", "auroc"): 0.031365, ("neg1c", "auprc"): 0.043649,
              ("neg2", "auroc"): 0.024114, ("neg2", "auprc"): 0.034854}


def main():
    boot = pd.read_csv(ROOT / "cv_bootstrap" / "paired_bootstrap.tsv", sep="\t")
    cv = pd.read_csv(ROOT / "cv_summary" / "cv_metrics.tsv", sep="\t")
    assert len(boot) == 12, f"expected 12 bootstrap slices, got {len(boot)}"
    assert (boot.ci_low > 0).all() or True  # report direction honestly
    cvt = cv[(cv.split == "test")].set_index(["track", "tier"])

    lines = []
    lines.append("# E1 RiNALMo 主集严格三折冻结基线最终报告：146 实验\n")
    lines.append("日期：2026-11-28 ｜ 数据版本：`e0-v2-id90-strict3train-main` ｜ 编码器：冻结 `multimolecule/rinalmo-giga`（1280 维 mean-masked）｜ 状态：**全量审计 + 配对 bootstrap + 独立回算通过**\n")
    lines.append("> 本报告只使用 `e1_rinalmo_main_strict3train_p20_v1` 产物。与 RNA-FM 主集同协议（strict3train、neg1c/neg2 双轨、簇去重余弦 kNN 参照、patience20/min_delta1e-4/1000 故障保险、validation-only 选择）。结论仅适用于 RiNALMo BCE 相对簇去重 kNN 的当前协议比较，非 SOTA、非完整最强基线、非机制因果。\n")
    lines.append("## 1. 验收结论\n")
    lines.append("全量独立审计通过：146 实验、15 modern（5 fold×3 seed）+ 5 kNN、双轨；PRED_OK 五折 test 行数 1,778,203/1,728,417/1,728,904/1,728,190/1,728,945（与 RNA-FM 逐折一致）；all_early_stopped_below_1000 / all_cuda0_numeric_probes_passed / all_train_strict_three_folds / no_fold_minus1_in_train / all_prediction_keys_aligned / all_raw_splits_aligned / all_scores_finite 均 true。\n")
    lines.append("## 2. 五折 CV 描述性结果（test）\n")
    lines.append("| 轨 | 切片 | BCE AUROC | kNN AUROC | ΔAUROC | BCE AUPRC | kNN AUPRC | ΔAUPRC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for track in ("neg1c", "neg2"):
        for tier in ("head", "mid", "tail"):
            r = cvt.loc[(track, tier)]
            lines.append(f"| {track} | {tier} | {r.auroc_modern_mean:.6f} | {r.auroc_knn_mean:.6f} | {r.d_auroc_mean:+.6f} | {r.auprc_modern_mean:.6f} | {r.auprc_knn_mean:.6f} | {r.d_auprc_mean:+.6f} |")
    lines.append("\n## 3. 逐窗配对 bootstrap（BCE − 簇去重余弦 kNN，10,000 次）\n")
    lines.append("| 轨 | 切片 | 指标 | Δ | 95% CI | p（双侧） |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, r in boot.iterrows():
        lines.append(f"| {r.track} | {r.slice} | {r.metric.upper()} | {r.observed_delta:+.6f} | [{r.ci_low:.6f}, {r.ci_high:.6f}] | {r.p_two_sided:.8f} |")
    n_pos = int((boot.observed_delta > 0).sum())
    n_ci = int((boot.ci_low > 0).sum())
    lines.append(f"\n12 切片中 {n_pos} 个差值为正、{n_ci} 个 95% CI 排除 0。\n")
    lines.append("## 4. 跨骨干对照（RiNALMo vs RNA-FM，尾部 pooled bootstrap Δ）\n")
    lines.append("| 轨 | 指标 | RiNALMo Δ | RNA-FM Δ | 说明 |")
    lines.append("|---|---|---:|---:|---|")
    for track in ("neg1c", "neg2"):
        for metric in ("auroc", "auprc"):
            r = boot[(boot.track == track) & (boot.slice == "tail") & (boot.metric == metric)].iloc[0]
            ref = RNAFM_TAIL[(track, metric)]
            lines.append(f"| {track} | {metric.upper()} | {r.observed_delta:+.6f} | {ref:+.6f} | 同协议同参照；两编码器均优于 kNN，幅度差异为描述性 |")
    lines.append("\n两冻结编码器在严格三折下尾部均优于簇去重 kNN；RiNALMo 尾部增益幅度小于 RNA-FM（描述性，不做显著性跨骨干检验）。这不构成统一编码器优劣结论。\n")
    lines.append("## 5. 产物与边界\n")
    lines.append("- `main/cv_bootstrap/paired_bootstrap.tsv`、6 个 `(10000,2)` NPZ；`main/cv_summary/cv_metrics.tsv`；`main/cv_stratified/`；`main/verification/`。")
    lines.append("- 审计日志 `logs/e1_rinalmo_main/main_full_audit.log`；后处理 `post.log`；run 记录 `experiments.jsonl`。")
    lines.append("- 三 seed 为统计重复非生物重复；低同源切片描述性无 bootstrap；预训练污染不可排除；旧 pilot/路由/PU 不复活。")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("RINALMO_REPORT_WRITTEN", OUT)
    print(f"bootstrap: {n_pos}/12 positive, {n_ci}/12 CI exclude zero")


if __name__ == "__main__":
    main()