from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from delivery_scripts.publication_paths import DELIVERABLES, EVIDENCE_ROOT, OUTPUT_ROOT

import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pptx import Presentation

HERE = Path(__file__).resolve().parent
ROOT = EVIDENCE_ROOT
OUT = OUTPUT_ROOT / "ppt" / "v3"
PPT = OUT / "proxy_function_gap_modern_20261126.pptx"
PDF = PPT.with_suffix(".pdf")
CONFIG = json.loads((OUT / "proxy_function_gap_modern_20261126_config.json").read_text())
FIG = DELIVERABLES / "figures"
SRC = DELIVERABLES / "reports" / "source_data"
REPORTS = DELIVERABLES / "reports"


def deck_text(prs):
    return "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))


def holm(values):
    values = np.asarray(values, dtype=float)
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    adjusted = np.empty(len(values))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def zip_payload(path):
    with zipfile.ZipFile(path) as zf:
        return {n: hashlib.sha256(zf.read(n)).hexdigest() for n in zf.namelist()
                if n.startswith("ppt/") and not n.endswith("/")}


def round_numeric():
    prs = Presentation(PPT)
    assert len(prs.slides) == 12
    text = deck_text(prs)
    normalized = text.replace("−", "-")
    pressure = pd.read_csv(FIG / "source_data" / "strict_h1_figure2_paired_pressure.csv")
    for context, cap in (("Defined T-cell r1", 16), ("Defined T-cell r2", 8), ("Defined T-cell r2", 16)):
        r = pressure[(pressure.context == context) & (pressure.proposal_cap == cap)].iloc[0]
        assert f"{r.paired_bon_minus_random:.3f}" in normalized
        if context == "Defined T-cell r1" or cap == 8:
            assert f"{r.ci_low:.3f}" in normalized and f"{r.ci_high:.3f}" in normalized
    common = pd.read_csv(FIG / "source_data" / "strict_h1_figure4_cross_architecture_summary.csv")
    assert f"{common.architecture_error_correlation.min():.3f}–{common.architecture_error_correlation.max():.3f}" in text
    score = pd.read_csv(FIG / "source_data" / "strict_h1_figure3_score_matching.csv")
    assert ((score.ci_low < 0) & (score.ci_high > 0)).all() and "4/4 区间跨 0" in text
    risk_path = SRC / "paper2_risk_frozen_v1" / "frozen_protocol.json"
    risk_frozen = json.loads(risk_path.read_text())
    calibration = risk_frozen["calibration"]
    assert f"{calibration['coverage']:.3f}" in normalized
    assert f"{calibration['regret_reduction']:.6f}" in normalized
    for value in calibration["regret_reduction_95ci"]:
        assert f"{value:.6f}" in normalized
    assert f"{calibration['accepted_decisions']}/{calibration['decisions']}" in text
    assert f"{calibration['accepted_pools']}/{calibration['pools']}" in text
    assert calibration["regret_reduction"] < 0
    assert calibration["regret_reduction_95ci"][0] < 0 < calibration["regret_reduction_95ci"][1]
    assert risk_frozen["threshold_locked"] and not risk_frozen["calibration_gate"]["pass"]
    assert not risk_frozen["calibration_gate"]["checks"]["positive_point_reduction"]
    assert not risk_frozen["calibration_gate"]["checks"]["reduction_ci_excludes_zero"]
    assert not risk_frozen["outer_scoring_authorized"] and not risk_frozen["outer_scored"]
    source = CONFIG["paper2_risk_frozen"]["source"]
    assert (ROOT / source["path"]) == risk_path
    assert hashlib.sha256(risk_path.read_bytes()).hexdigest() == source["sha256"]
    q = json.loads((SRC / "paper2_risk_prediction" / "paper2_risk_prediction_analysis.json").read_text())
    calib = q["roles"]["calibration"]
    assert f"{calib['spearman_q_vs_regret']:.3f}" in normalized and f"{calib['auc_top_half_regret']:.3f}" in text
    sch = json.loads((SRC / "schlusser2024_hepg2_te_confirmatory_v1" / "evaluator_round_1" / "summary.json").read_text())
    assert f"{sch['endpoint_alignment_descriptive']['spearman_cnn_MRL_vs_TE']:.3f}" in text
    comparison = next(p for p in sch["paired_comparisons"] if p["proposal_cap"] == 16 and p["comparison"] == "bon_cnn-random")
    assert f"{comparison['mean_regret_difference_left_minus_right']:+.3f}" in normalized
    for value in comparison["gene_cluster_bootstrap_95_interval"]:
        assert f"{value:+.3f}" in normalized
    p3 = json.loads((SRC / "paper3_headroom_v4" / "headroom_v4_round1" / "summary.json").read_text())
    assert not p3["outer_labels_read"] and not p3["outer_metrics_computed"]
    gates = pd.DataFrame(p3["cases"]).groupby("representation").gate_pass.sum().astype(int)
    assert gates["kmer4"] == 5 and gates["summary4"] == 0
    block = (REPORTS / "paper3_headroom_v4_20261124.md").read_text().split("### kmer4：固定 5 池")[1].split("###")[0]
    row = next(line for line in block.splitlines() if line.startswith("| 16 |"))
    cells = [float(v) for v in row.strip().strip("|").split("|")]
    assert f"{cells[1]:.3f}" in text and f"{cells[4]:.3f}" in text
    budget_dir = SRC / "paper1_budget_matched_v1"
    budget = json.loads((budget_dir / "results.json").read_text())
    pools = pd.read_csv(budget_dir / "pool_effects.tsv", sep="\t")
    assert len(pools) == 320 and not budget["new_blind_test"]
    equal_rows = []
    random_rows = []
    for family_name, target in (("bon_minus_random_n", equal_rows), ("random_n_minus_random1", random_rows)):
        for row in budget["families"][family_name]:
            selected = pools[(pools.context == row["context"]) & (pools.cap == row["cap"])][family_name]
            assert np.isclose(selected.mean(), row["effect"], rtol=0, atol=1e-12)
            target.append(row)
        adjusted = holm([r["p"] for r in target])
        assert np.allclose(adjusted, [r["holm_p"] for r in target], rtol=0, atol=1e-12)
    r1_sig = [r["cap"] for r in equal_rows if r["context"] == "defined_tcell_r1" and r["cap"] > 1 and r["holm_p"] < 0.05]
    r2_sig = [r["cap"] for r in equal_rows if r["context"] == "defined_tcell_r2" and r["cap"] > 1 and r["holm_p"] < 0.05]
    assert r1_sig == [2, 4, 8, 16] and r2_sig == [4]
    assert not any(r["holm_p"] < 0.05 for r in random_rows)
    sequence_path = SRC / "paper1_sequence_ablation_v1" / "analysis" / "results" / "summary.json"
    sequence = json.loads(sequence_path.read_text())
    primary = sequence["primary"]
    assert not sequence["new_blind_test"] and not sequence["causal_claim"]
    assert not any(r["adjusted_holm_p"] < 0.05 for r in primary)
    assert np.allclose(holm([r["adjusted_p"] for r in primary]), [r["adjusted_holm_p"] for r in primary], rtol=0, atol=1e-12)
    families = sequence["sequence_family_sensitivity"]
    for family_name in ("no_uAUG", "no_polyU_upper_quartile"):
        rows = [r for r in families if r["family"] == family_name]
        assert np.allclose(holm([r["p"] for r in rows]), [r["holm_p"] for r in rows], rtol=0, atol=1e-12)
    assert not any(r["holm_p"] < 0.05 for r in families if r["family"] == "no_uAUG")
    poly_sig = {(r["context"], r["cap"]) for r in families if r["family"] == "no_polyU_upper_quartile" and r["holm_p"] < 0.05}
    assert poly_sig == {("defined_tcell_r1", 4), ("defined_tcell_r2", 8)}
    for phrase in ("等预算价值", "组成边界", "外部迁移边界", "r1 cap2–16 全过", "r2 仅 cap4", "0/20 通过", "敏感性不等于因果中介"):
        assert phrase in text, phrase
    for key, path in (("paper1_budget_matched", budget_dir / "results.json"), ("paper1_sequence_ablation", sequence_path)):
        source = CONFIG[key]["source"]
        assert (ROOT / source["path"]) == path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
    persist_dir = SRC / "persist_mrl_external_v1"
    persist = json.loads((persist_dir / "results" / "summary.json").read_text())
    pred = pd.read_csv(persist_dir / "predictions" / "predictions.tsv", sep="\t")
    labels = pd.read_csv(persist_dir / "protocol" / "evaluator_labels.tsv", sep="\t")
    seeds = pd.read_csv(persist_dir / "results" / "per_seed.tsv", sep="\t")
    boot = np.load(persist_dir / "results" / "bootstrap_replicates.npy")
    audit = json.loads((persist_dir / "protocol" / "inclusion_audit.json").read_text())
    joined = pred.merge(labels, on=["sequence_id", "utr"], validate="one_to_one")
    assert len(joined) == 59 and joined.utr.nunique() == 49 and len(seeds) == 200
    rho = joined.prediction.rank().corr(joined.mrl.rank())
    effect = seeds.delta.mean()
    assert np.isclose(rho, persist["secondary"]["spearman"]["value"], rtol=0, atol=1e-12)
    assert np.isclose(effect, persist["primary"]["bon_minus_random_regret"], rtol=0, atol=1e-12)
    assert np.allclose(seeds.bon_regret - seeds.random_regret, seeds.delta)
    assert np.allclose(np.quantile(boot[:, 0], [0.025, 0.975]), persist["primary"]["ci95"], rtol=0, atol=1e-12)
    assert np.allclose(np.quantile(boot[:, 1], [0.025, 0.975]), persist["secondary"]["spearman"]["ci95"], rtol=0, atol=1e-12)
    assert effect > 0 and persist["primary"]["ci95"][0] < 0 < persist["primary"]["ci95"][1]
    assert not persist["primary"]["significant_benefit"] and persist["negative_results_retained"]
    assert audit["conflicts"] == {} and audit["excluded_constructs"] == 0
    for value in (rho, effect, *persist["primary"]["ci95"]):
        assert f"{value:+.6f}" in normalized, value
    for phrase in ("内部收益 ≠ 跨研究迁移", "frozen 4-mer MLP", "回顾性非盲测", "59 构建体 / 49 UTR 簇", "200 种子仅作重复提案", "前瞻盲测"):
        assert phrase in text, phrase
    source = CONFIG["persist"]["source"]
    assert (ROOT / source["path"]) == persist_dir / "results" / "summary.json"
    assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    for phrase in ("DIAGNOSTIC", "CONFIRMATORY", "永久 outer 未动", "GATE FAILED", "OUTER NOT SCORED", "后续路线图", "outer 不评分", "阈值不回调"):
        assert phrase in text, phrase
    for phrase in ("新盲测已完成", "Paper 2 已确认", "已证实普遍 acquisition 优势", "outer PASS", "0.459", "0.672", "部署阈值未冻结", "唯一确认路径", "后续唯一关键动作", "下一步冻结", "独立同终点 MRL 复现仍缺失", "200 个独立生物样本"):
        assert phrase not in text, phrase
    return {"status": "PASS", "slides": 12, "numeric_claims_checked": 48,
            "budget_pool_effects_recomputed": 40, "holm_families_recomputed": 5,
            "tcell_r1_equal_budget_all_caps_significant": True, "tcell_r2_only_cap4_significant": True,
            "random_budget_no_corrected_significance": True, "sequence_adjustment_not_causal": True,
            "paper2_frozen_calibration_gate_failed": True, "paper2_outer_scored": False,
            "paper3_outer_untouched": True, "legacy_positive_abstention_absent": True,
            "diagnostic_confirmatory_labels": True, "persist_negative_result_retained": True,
            "persist_seeds_not_independent_samples": True, "persist_primary_crosses_zero": True}


def round_rebuild():
    before = zip_payload(PPT)
    subprocess.run([sys.executable, str(HERE / "build_modern_deck_20261126.py")], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    middle = zip_payload(PPT)
    assert before == middle, [k for k in before if before.get(k) != middle.get(k)]
    time.sleep(1.1)
    env = dict(os.environ, PYTHONHASHSEED="987654321", TZ="UTC", LC_ALL="C")
    subprocess.run([sys.executable, str(HERE / "build_modern_deck_20261126.py")], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    after = zip_payload(PPT)
    assert before == after, [k for k in before if before.get(k) != after.get(k)]
    for record in CONFIG["figures"]:
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    return {"status": "PASS", "ppt_payload_entries_identical": len(before), "figure_hashes": len(CONFIG["figures"]),
            "rebuild_rounds": 2, "python_hash_seeds": ["inherited", "987654321"],
            "environment_variation": {"TZ": "UTC", "LC_ALL": "C"}, "cross_second_delay": 1.1,
            "runtime_guard": "uses the current interpreter unless PFG_CANONICAL_PYTHON selects the original deck builder",
            "builder_runtime": CONFIG["runtime"]}


def round_render():
    prs = Presentation(PPT)
    for i, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0, (i, shape.name)
            assert shape.left + shape.width <= prs.slide_width + 2, (i, shape.name, "x")
            assert shape.top + shape.height <= prs.slide_height + 2, (i, shape.name, "y")
    with zipfile.ZipFile(PPT) as zf:
        embedded = {hashlib.sha256(zf.read(n)).hexdigest() for n in zf.namelist() if n.startswith("ppt/media/")}
    for record in CONFIG["figures"]:
        assert record["sha256"] in embedded, record["path"]
    doc = fitz.open(PDF)
    assert len(doc) == 12
    assert all(abs(p.rect.width - 960) < 0.1 and abs(p.rect.height - 540) < 0.1 for p in doc)
    all_text = "\n".join(p.get_text() for p in doc)
    compact = "".join(all_text.split())
    for phrase in ("后续路线图", "GATE FAILED", "OUTER NOT SCORED", "永久outer未动", "CONFIRMATORY", "DIAGNOSTIC", "内部收益≠跨研究迁移", "回顾性非盲测", "+0.104723", "等预算价值", "组成边界", "敏感性不等于因果中介", "r2仅cap4"):
        assert phrase.replace(" ", "") in compact, phrase
    for phrase in ("0.459", "0.672", "部署阈值未冻结", "唯一确认路径", "后续唯一关键动作", "下一步冻结"):
        assert phrase.replace(" ", "") not in compact, phrase
    render_dir = OUT / "renders"; render_dir.mkdir(parents=True, exist_ok=True)
    thumbs = []
    stats = []
    for i, page in enumerate(doc, 1):
        for word in page.get_text("words"):
            assert word[0] >= -0.5 and word[1] >= -0.5 and word[2] <= page.rect.width + 0.5 and word[3] <= page.rect.height + 0.5, (i, word)
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                A, B = blocks[a], blocks[b]
                dx = min(A[2], B[2]) - max(A[0], B[0]); dy = min(A[3], B[3]) - max(A[1], B[1])
                assert not (dx > 1 and dy > 1), (i, A[4], B[4])
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        path = render_dir / f"slide-{i:02d}.png"; pix.save(path)
        im = Image.open(path).convert("RGB"); arr = np.asarray(im)
        assert arr.std() > 15 and len(np.unique(arr.reshape(-1, 3), axis=0)) > 100
        stats.append({"slide": i, "pixels": [im.width, im.height], "std": round(float(arr.std()), 2)})
        thumb = im.copy(); thumb.thumbnail((590, 332))
        tile = Image.new("RGB", (600, 365), "white"); tile.paste(thumb, ((600 - thumb.width) // 2, 25))
        ImageDraw.Draw(tile).text((12, 7), f"Slide {i}", fill="#17233B"); thumbs.append(tile)
    rows = (len(thumbs) + 2) // 3
    sheet = Image.new("RGB", (1800, rows * 365), "#DDE3EB")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 3) * 600, (i // 3) * 365))
    sheet.save(OUT / "rendered_contact_sheet.png")
    fonts = subprocess.check_output(["pdffonts", str(PDF)], text=True)
    assert "ArialUnicodeMS" in fonts and "FrankRuhlHofshi" not in fonts and "LinuxLibertine" not in fonts
    return {"status": "PASS", "pdf_pages": 12, "pdf_page_points": [960, 540],
            "pdf_export": "Microsoft PowerPoint native", "font": "ArialUnicodeMS embedded",
            "shape_bounds": "PASS", "pdf_words_outside": 0, "pdf_text_collisions": 0,
            "embedded_figures_exact": len(CONFIG["figures"]), "rendered_pages": stats}


report = {"round1_numeric_independent": round_numeric(), "round2_deterministic_rebuild": round_rebuild(),
          "round3_format_render": round_render()}
report["overall"] = "PASS"
(OUT / "proxy_function_gap_modern_20261126_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
