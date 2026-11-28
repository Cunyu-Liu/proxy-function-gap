from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation

from build_benchmark_update_ppt import (
    INK, MUTED, NAVY, ORANGE, PALE_BLUE, PALE_ORANGE, PALE_TEAL, TEAL,
    add_footer, add_note, add_picture_fit, add_rect, add_text, add_title, sha256, slide_signature,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "生物序列长尾与OOD_基准证据更新_2026-11-28.pptx"
SOURCE = ROOT / "生物序列长尾与OOD_基准证据更新_2026-11-26.pptx"
FIG = ROOT / "figures"
AUDIT = ROOT / ".pptx_build/ppt_20261128_build_audit.json"


def clear_slide(slide) -> None:
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)


def build_mainline(slide) -> None:
    clear_slide(slide)
    add_title(slide, "科学主线 v2：一个悖论引出的测量合同", 31, 28)
    add_text(slide, "同一模型在不同评测里互有胜负——结论能活下来，取决于合同，而不是模型名字",
             0.55, 0.91, 12.23, 0.36, size=18, color=MUTED, name="Mainline subtitle")
    cards = [
        ("第一幕", "悖论与合同", "负例定义改变可学难度（Δ−0.0713）；修复 fold=-1 缺口后，尾部优势仍在。", PALE_TEAL, TEAL),
        ("第二幕", "排除捷径", "密度路由、PU 先验、共享残差接连未过门——负结果把问题逼回测量。", PALE_ORANGE, ORANGE),
        ("第三幕", "拒识幻象", "跨协议判别漂移；整 RBP 留出 23/24 指标冲突，无统一拒识赢家。", PALE_BLUE, NAVY),
        ("第四幕", "边界闭环", "蛋白 25/25 桶 BCE 占优；RNAcompete OOD 排序反转——条件化权衡。", PALE_TEAL, TEAL),
    ]
    for i, (number, heading, body, fill, accent) in enumerate(cards):
        x = 0.55 + i * 3.07
        add_rect(slide, x, 1.52, 2.72, 3.55, fill, line="DCE6ED", name=f"Mainline card {number}")
        add_text(slide, number, x + 0.20, 1.76, 1.20, 0.34, size=18, color=accent, bold=True,
                 name=f"Mainline number {number}")
        add_text(slide, heading, x + 0.20, 2.24, 2.30, 0.62, size=21, color=NAVY, bold=True,
                 valign=3, name=f"Mainline heading {number}")
        add_text(slide, body, x + 0.20, 3.08, 2.30, 1.54, size=16, color=INK,
                 name=f"Mainline body {number}", margin=0)
        if i < 3:
            add_text(slide, "→", x + 2.68, 2.91, 0.38, 0.50, size=25, color=ORANGE, bold=True,
                     align=2, valign=3, name=f"Mainline arrow {i + 1}", margin=0)
    add_note(slide, "每步只回答上一个实验真正排除不了的问题；M1 止损作为诚实负结果进入补充材料。",
             0.55, 5.48, 12.23, 0.72, 1, PALE_BLUE, NAVY)
    add_footer(
        slide,
        "证据等级：E1 M-test；E5 pilot fold0 三 seed 描述性；E6 5S-test；RNAcompete 五折三 seed 描述性；负结果 V-gate-legacy/P-test-legacy。",
        "叙事：论文证据/预印本科学主线与证据链_2026-11-28.md · 台账：_2026-11-25.md §6 主张矩阵",
    )


def build_fig_slide(slide, page, title, fig_name, notes, evidence, runs):
    clear_slide(slide)
    add_title(slide, title, page, 28)
    add_picture_fit(slide, FIG / fig_name, 0.45, 1.00, 9.25)
    fills = [PALE_BLUE, PALE_ORANGE, PALE_TEAL]
    accents = [NAVY, ORANGE, TEAL]
    positions = [(1.30, 1.50), (3.00, 1.50), (4.70, 1.50)]
    for i, (text, (y, h)) in enumerate(zip(notes, positions)):
        add_note(slide, text, 9.91, y, 2.87, h, i + 1, fills[i], accents[i])
    add_footer(slide, evidence, runs)


def build_status(slide) -> None:
    clear_slide(slide)
    add_title(slide, "执行状态｜四队列并行推进，未过门不写性能", 37, 28)
    rows = [
        ("RiNALMo eCLIP 主集", "20/20 run 训练完成；全量独立审计进行中", "审计→bootstrap→最终报告后写入", PALE_TEAL, TEAL),
        ("RNA-FM LoRA/full 微调", "fold0 rank4 真实 CUDA 训练；rank 仅 validation 选择", "fold0 三 seed→五折→final gate", PALE_BLUE, NAVY),
        ("CAFA3 ESM-C 时间留出", "内存门经实测修正（4.89GiB RSS）；fold0/id50/BPO 收敛中", "36 组合→独立审计", PALE_TEAL, TEAL),
        ("RNAcompete RiNALMo 扩展", "预注册附录冻结同协议；22 万 RNA 嵌入进行中", "selection→fold0 门→全矩阵", PALE_ORANGE, ORANGE),
    ]
    for i, (name, state, gate, fill, accent) in enumerate(rows):
        y = 1.42 + i * 1.06
        add_rect(slide, 0.55, y, 12.23, 0.92, fill, line="DCE6ED", name=f"Status row {i}")
        add_text(slide, name, 0.78, y + 0.10, 3.10, 0.72, size=17, color=accent, bold=True,
                 valign=3, name=f"Status name {i}")
        add_text(slide, state, 4.05, y + 0.10, 5.30, 0.72, size=15, color=INK,
                 valign=3, name=f"Status state {i}")
        add_text(slide, gate, 9.55, y + 0.10, 3.05, 0.72, size=14, color=MUTED,
                 valign=3, name=f"Status gate {i}")
    add_note(slide, "GPU 利用率 95%；CUDA 探针强制、CPU 降级即停；训练到收敛（patience20），不以步数截断。",
             0.55, 5.85, 12.23, 0.62, 1, PALE_BLUE, NAVY)
    add_footer(
        slide,
        "边界：以上均为进行中状态，不构成性能结论；已验收结论见前六页与各自最终报告。",
        "日志：/home/ProxyFunctionGap/logs/{e1_rinalmo_main,e1_finetune_main_driver,cafa3_esmc300m_p20,rnacompete_ray2013_rinalmo}",
    )


def main() -> None:
    prs = Presentation(str(SOURCE))
    slides = list(prs.slides)
    assert len(slides) == 37, f"unexpected slide count {len(slides)}"
    build_mainline(slides[30])
    build_fig_slide(
        slides[31], 32, "Fig. 1｜数据为什么需要合同：长尾、上限与跨系重合",
        "Fig1_data_audit_v2_20261128.png",
        ["RBP 位点 Gini=0.352，最大/最小 45.98×；20k 上限截断头部。",
         "跨细胞系同 RBP 位点 ±50nt 重合中位数约 27%——标签是条件特异的。",
         "E0 数据事实；重合为双方向合并计算，图内标注。"],
        "证据等级：E0 数据审计事实；不主张预训练无污染或全基因组无遗漏。",
        "数据：E0_数据审计/results/{positives_audit,rbp_distribution,dual_cellline_consistency}")
    build_fig_slide(
        slides[32], 33, "Fig. 2｜合同修复后，尾部优势幸存（12/12 CI 排除 0）",
        "Fig2_eclip_tail_forest_v2_20261128.png",
        ["尾部 pooled ΔAUROC/ΔAUPRC：neg1c +0.031365/+0.043649；neg2 +0.024114/+0.034854。",
         "10,000 次逐窗配对 bootstrap；双侧校正 p=0.0002。",
         "仅 frozen RNA-FM BCE vs 簇去重余弦 kNN；非 SOTA、非完整最强基线。"],
        "证据等级：M-test，strict3train 146 实验五折三 seed；三 seed 为统计重复。",
        "Runs: E1-paired-bootstrap-20261126-065044-ZDLg6Y · E1-main-independent-recalculation-20261126-105611-ZDLg6Y")
    build_fig_slide(
        slides[33], 34, "Fig. 4｜蛋白全桶 BCE 占优；跨协议判别随条件漂移",
        "Fig4_protein_crossprotocol_v2_20261128.png",
        ["GRIMM EC 五官方 split：25/25 频率桶 BCE>deep-kNN，逐桶 CI 排除 0。",
         "PAR-CLIP 9 RBP 三 seed：0.646192/0.607869/0.655701（宏 AUROC）。",
         "蛋白仅限当前协议；PAR-CLIP 描述性、三 seed 非生物重复。"],
        "证据等级：E6 5S-test；PAR-CLIP M-desc；旧 split_1 头尾反转系未收敛伪影，已撤回。",
        "Runs: E6-five-split-summary-20261125-080318-ZDLg6Y · E5-parclip-cross-protocol-contract-p20-20261125-{065228,065619,065953}-ZDLg6Y")
    build_fig_slide(
        slides[34], 35, "Fig. 5｜23/24 指标冲突：拒识没有统一赢家",
        "Fig5_ood_metric_conflicts_v2_20261128.png",
        ["deep-kNN AUROC-OOD 仅 0.507697/0.508286（近随机），FPR95 高达 0.947/0.946。",
         "AURC 最佳分属 energy（neg1c 0.155127）与 MSP（neg2 0.473079）。",
         "三指标必须联报；三 seed 描述性、无 OOD bootstrap。"],
        "证据等级：pilot fold0 test 三 seed 描述性宏平均；不支持可靠统一拒识。",
        "Run: E5-holdout-convergence-summary-20261126-112101-ZDLg6Y（72/72，独立三遍 verify complete）")
    build_fig_slide(
        slides[35], 36, "Fig. 6｜RNAcompete：序列尾部与整 RBP-OOD 排序反转",
        "Fig6_rnacompete_dual_track_v2_20261128.png",
        ["序列 tail Spearman：MLP 0.475605 > linear 0.430202 > kNN 0.352559。",
         "整 RBP-OOD 实验宏：kNN 0.185360 > MLP 0.111699 > linear 0.098188（绝对弱）。",
         "70/70 验收；任务依赖排序，非统一赢家；kNN OOD 为共享序列共识。"],
        "证据等级：RNAcompete Ray2013 frozen RNA-FM 五折三 seed 描述性；非显著性、SOTA。",
        "Runs: rnacompete-verify-final-…-4099546 · rnacompete-summarize-…-4101649（final gate SHA 83ec4cfe…）")
    build_status(slides[36])
    lowhom = prs.slides.add_slide(prs.slide_layouts[0])
    build_fig_slide(
        lowhom, 38, "补充图｜低同源切片：方向与主集一致（描述性）",
        "Fig3_low_homology_stratification_v2_20261128.png",
        ["neg1c：BCE/kNN AUROC 0.875332/0.830752（Δ+0.044580）；AUPRC 0.781584/0.697716（Δ+0.083869）。",
         "neg2：AUROC 0.790567/0.751782（Δ+0.038785）；AUPRC 0.653712/0.585196（Δ+0.068516）。",
         "测试正例 d_hom<0.60；三 seed 描述性、无 bootstrap；类别先验改变，不与闭集 AUPRC 横比。"],
        "证据等级：M-desc 低同源描述性分层；E1 最终报告 §6，独立回算已验收；不证明同源是因果来源。",
        "Run: E1-stratified-analysis-20261126-104609-ZDLg6Y · E1-main-independent-recalculation-20261126-105611-ZDLg6Y")
    conv = prs.slides.add_slide(prs.slide_layouts[0])
    build_fig_slide(
        conv, 39, "补充图｜收敛合同证据：4,380 任务全部早停、无故障保险",
        "Fig7_convergence_evidence_v2_20261128.png",
        ["15 run × 292 任务：stop = best + 20，best 范围 1–43，最大 stop 63。",
         "无一触及 1000 epoch 故障保险——训练到收敛，不以步数截断。",
         "已验收 E1 主集工程证据；非性能结论。"],
        "证据等级：已验收 strict3train 主集收敛合同证据（modern_results.tsv）。",
        "数据：figures/data_20261128/e1_convergence/ · 核验：verification_v2_20261128.json")
    prs.save(str(TARGET))
    audit = {"target": str(TARGET), "source_sha256": sha256(SOURCE),
             "target_sha256": sha256(TARGET), "slides": len(prs.slides._sldIdLst),
             "rebuilt": [30, 31, 32, 33, 34, 35, 36, 37],
             "figures": {n: sha256(FIG / n) for n in [
                 "Fig1_data_audit_v2_20261128.png", "Fig2_eclip_tail_forest_v2_20261128.png",
                 "Fig4_protein_crossprotocol_v2_20261128.png", "Fig5_ood_metric_conflicts_v2_20261128.png",
                 "Fig6_rnacompete_dual_track_v2_20261128.png"]}}
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print("PPT_20261128_BUILD_OK", json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()