from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import fitz
from PIL import Image, ImageChops
from pptx import Presentation

from build_benchmark_update_ppt import slide_signature

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "生物序列长尾与OOD_基准证据更新_2026-11-26.pptx"
BASELINE = ROOT / ".pptx_build/pre_final_update.pptx"
PDF = ROOT / ".pptx_build/final_accepted_render/生物序列长尾与OOD_基准证据更新_2026-11-26.pdf"
OUT = ROOT / ".pptx_build/ppt_final_three_pass_verification.json"
CHANGED = (31, 32, 35, 36)
UNCHANGED = list(range(1, 31)) + [33, 34]


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def texts(prs: Presentation) -> dict[int, str]:
    return {n: "\n".join(s.text for s in prs.slides[n - 1].shapes if hasattr(s, "text")) for n in CHANGED}


def pass1_reports(prs: Presentation) -> dict:
    text = texts(prs)
    e1 = (ROOT / "E1_强基线/E1_主集严格三折冻结基线最终报告_2026-11-26.md").read_text(encoding="utf-8")
    e5 = (ROOT / "E5_OOD/E5_整RBP留出三seed双轨最终报告_2026-11-26.md").read_text(encoding="utf-8")
    figures = (ROOT / "论文证据/图表设计与数值核验_2026-11-25.md").read_text(encoding="utf-8")
    mainline = (ROOT / "论文证据/预印本科学主线与证据链_2026-11-25.md").read_text(encoding="utf-8")
    report_tokens = {
        "mainline": ["E1 strict3train 主集（146 实验）已最终验收", "E5 整 RBP 留出已最终验收 72/72 pair", "23/24 个 RBP×轨组有指标冲突"],
        "figures": ["当前 E1 strict3train 合同已修复并独立验收", "12 个 BCE−kNN 效应均为正且 95% CI 排除 0", "23/24 组存在指标赢家冲突，不选统一赢家"],
        "e1": ["+0.031365", "+0.024114", "+0.043649", "+0.034854", "0.00019998", "不支持“完整最强基线胜利”“SOTA”"],
        "e5": ["0.507697", "0.508286", "0.947266", "0.945827", "23 个存在指标赢家冲突", "不存在跨三项 OOD 指标一致的统一赢家"],
    }
    sources = {"mainline": mainline, "figures": figures, "e1": e1, "e5": e5}
    count = 0
    for source, tokens in report_tokens.items():
        for token in tokens:
            assert token in sources[source], (source, token)
            count += 1
    ppt_tokens = {
        31: ["E1 strict3train", "E5 72/72", "23/24", "非 SOTA"],
        32: ["146 实验", "fold=-1 不进入 CV train", "独立验收", "预训练语料无污染"],
        35: ["12/12", "95% CI", "10,000", "+0.031365", "+0.024114", "+0.043649", "+0.034854", "非 SOTA"],
        36: ["72/72", "216 行", "23/24", "0.507697", "0.508286", "0.947266", "0.945827", "无统一赢家"],
    }
    for number, tokens in ppt_tokens.items():
        for token in tokens:
            assert token in text[number], (number, token)
        assert "foldfix 重跑中" not in text[number]
        assert "E5-rbp-holdout-20261123" not in text[number]
    return {"status": "passed", "accepted_report_token_checks": count, "ppt_tokens": ppt_tokens,
            "obsolete_status_and_old_holdout_absent": True}


def pass2_raw_recalculation(prs: Presentation) -> dict:
    text = texts(prs)
    final_e1 = json.loads((ROOT / "E1_强基线/results/strict3train_p20_v2/verification/final_verification.json").read_text())
    assert final_e1["accepted"] is True and final_e1["old_npz_used"] is False
    protocol = final_e1["pass_2_protocol_alignment"]
    assert protocol["n_experiments"] == 146 and protocol["all_train_strict_three_folds"] is True
    assert protocol["no_fold_minus1_in_train"] is True and protocol["all_cuda0_numeric_probes_passed"] is True

    rows = read_tsv(ROOT / "figures/verification/data/fig2/paired_bootstrap.tsv")
    assert len(rows) == 12
    assert all(float(row["observed_delta"]) > 0 and float(row["ci_low"]) > 0 for row in rows)
    assert all(int(row["n_boot"]) == 10000 for row in rows)
    assert all(abs(float(row["p_two_sided"]) - 0.0001999800019998) < 1e-15 for row in rows)
    tail = {}
    for track in ("neg1c", "neg2"):
        for metric in ("auroc", "auprc"):
            row = next(row for row in rows if row["track"] == track and row["slice"] == "tail" and row["metric"] == metric)
            formatted = f"+{float(row['observed_delta']):.6f}"
            assert formatted in text[35]
            tail[f"{track}_{metric}"] = {"delta": formatted, "ci_low": f"{float(row['ci_low']):.6f}",
                                         "ci_high": f"{float(row['ci_high']):.6f}"}
    cv_rows = read_tsv(ROOT / "figures/verification/data/fig2/cv_test_by_fold.tsv")
    assert len(cv_rows) == 30
    assert sum(float(row["d_auroc"]) > 0 for row in cv_rows) == 30
    assert sum(float(row["d_auprc"]) > 0 for row in cv_rows) == 30

    final_e5 = json.loads((ROOT / "E5_OOD/results/holdout_convergence_final_summary/verification.json").read_text())
    assert final_e5["status"] == "complete" and final_e5["n_pairs_completed"] == 72
    assert final_e5["n_pairs_expected"] == 72 and final_e5["missing"] == []
    assert final_e5["cross_seed_window_npz_alignment"]["comparisons"] == 48
    raw = read_tsv(ROOT / "figures/verification/data/fig5/holdout_metrics_216.tsv")
    assert len(raw) == 216
    assert len({(row["holdout"], row["track"], row["seed"]) for row in raw}) == 72
    assert all(row["stop_reason"] == "validation_early_stop" for row in raw)
    assert all(int(row["epochs_run"]) - int(row["best_epoch"]) == 20 for row in raw)
    per_seed = defaultdict(list)
    per_group = defaultdict(list)
    for row in raw:
        for metric in ("auroc_ood", "fpr95", "aurc"):
            assert math.isfinite(float(row[metric]))
        key = (row["track"], int(row["seed"]), row["method"])
        per_seed[key].append(row)
        per_group[(row["holdout"], row["track"], row["method"])].append(row)
    assert all(len(values) == 12 for values in per_seed.values())
    macro = {}
    for track in ("neg1c", "neg2"):
        for method in ("msp", "energy", "deep_knn"):
            seed_means = {}
            for seed in range(3):
                values = per_seed[(track, seed, method)]
                seed_means[seed] = {metric: statistics.mean(float(row[metric]) for row in values)
                                    for metric in ("auroc_ood", "fpr95", "aurc")}
            macro[(track, method)] = {metric: statistics.mean(seed_means[seed][metric] for seed in range(3))
                                      for metric in ("auroc_ood", "fpr95", "aurc")}
    for track in ("neg1c", "neg2"):
        auroc = f"{macro[(track, 'deep_knn')]['auroc_ood']:.6f}"
        fpr95 = f"{macro[(track, 'deep_knn')]['fpr95']:.6f}"
        assert auroc in text[36] and fpr95 in text[36]
        assert macro[(track, "deep_knn")]["auroc_ood"] == max(macro[(track, method)]["auroc_ood"] for method in ("msp", "energy", "deep_knn"))
        assert macro[(track, "deep_knn")]["fpr95"] == min(macro[(track, method)]["fpr95"] for method in ("msp", "energy", "deep_knn"))
    assert min(macro[("neg1c", method)]["aurc"] for method in ("msp", "energy", "deep_knn")) == macro[("neg1c", "energy")]["aurc"]
    assert min(macro[("neg2", method)]["aurc"] for method in ("msp", "energy", "deep_knn")) == macro[("neg2", "msp")]["aurc"]

    conflicts = []
    for holdout, track in sorted({(row["holdout"], row["track"]) for row in raw}):
        means = {}
        for method in ("msp", "energy", "deep_knn"):
            values = per_group[(holdout, track, method)]
            assert len(values) == 3
            means[method] = {metric: statistics.mean(float(row[metric]) for row in values)
                             for metric in ("auroc_ood", "fpr95", "aurc")}
        winner_order = ("energy", "msp", "deep_knn")
        winners = {
            "auroc_ood": max(winner_order, key=lambda method: means[method]["auroc_ood"]),
            "fpr95": min(winner_order, key=lambda method: means[method]["fpr95"]),
            "aurc": min(winner_order, key=lambda method: means[method]["aurc"]),
        }
        if len(set(winners.values())) > 1:
            conflicts.append((holdout, track, winners))
    assert len(conflicts) == 23
    assert ("IGF2BP1", "neg1c") not in {(holdout, track) for holdout, track, _ in conflicts}
    return {"status": "passed", "e1_accepted_strict3train": True, "e1_12_bootstrap_rows": tail,
            "e1_60_cv_deltas_positive": True, "e5_72_pairs_216_rows": True,
            "e5_macro_deep_knn": {track: {metric: f"{value:.6f}" for metric, value in macro[(track, "deep_knn")].items()}
                                  for track in ("neg1c", "neg2")},
            "e5_winner_conflicts": len(conflicts), "e5_sole_no_conflict": "neg1c/IGF2BP1 (energy)",
            "msp_energy_exact_tie_canonical_representative": "energy"}


def pass3_pdf_layout_and_pixels(prs: Presentation) -> dict:
    document = fitz.open(PDF)
    assert len(document) == 36
    all_text = "\n".join(page.get_text() for page in document)
    assert "\ufffd" not in all_text and "\x00" not in all_text
    records = []
    for number in CHANGED:
        page = document[number - 1]
        page_normalized = normalize(page.get_text())
        spans = []
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        spans.append(span)
        assert all(span["bbox"][0] >= -0.1 and span["bbox"][1] >= -0.1 and
                   span["bbox"][2] <= page.rect.width + 0.1 and span["bbox"][3] <= page.rect.height + 0.1 for span in spans)
        used = set()
        boxes = {}
        minimum = math.inf
        for shape in prs.slides[number - 1].shapes:
            if not getattr(shape, "has_text_frame", False) or not shape.text.strip():
                continue
            expected = normalize(shape.text)
            assert expected in page_normalized, (number, shape.name, "text missing")
            x0, y0 = shape.left / 914400 * 72, shape.top / 914400 * 72
            x1, y1 = (shape.left + shape.width) / 914400 * 72, (shape.top + shape.height) / 914400 * 72
            matched = []
            for index, span in enumerate(spans):
                bbox = span["bbox"]
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                if x0 - 0.5 <= cx <= x1 + 0.5 and y0 - 0.5 <= cy <= y1 + 0.5:
                    matched.append((index, span))
            assert matched, (number, shape.name, "no spans")
            assert expected == normalize("".join(span["text"] for _, span in matched)), (number, shape.name, "text mismatch")
            for index, span in matched:
                used.add(index)
                bbox = span["bbox"]
                tolerance = 1.6
                assert bbox[0] >= x0 - tolerance and bbox[1] >= y0 - tolerance and bbox[2] <= x1 + tolerance and bbox[3] <= y1 + tolerance, (
                    number, shape.name, "overflow", bbox, (x0, y0, x1, y1), span["text"])
                if not shape.name.startswith("Footer"):
                    minimum = min(minimum, span["size"])
            boxes[shape.name] = (x0, y0, x1, y1)
        items = list(boxes.items())
        for i, (name_a, a) in enumerate(items):
            for name_b, b in items[i + 1:]:
                area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                assert area <= 0.5, (number, "text overlap", name_a, name_b)
        assert len(used) == len(spans), (number, "unmatched spans")
        assert minimum >= 17.95
        records.append({"slide": number, "text_shapes": len(boxes), "pdf_spans": len(spans),
                        "min_non_footer_font_pt": minimum, "no_overlap_or_overflow": True})
    for number in (32, 35, 36):
        pictures = [shape for shape in prs.slides[number - 1].shapes if int(shape.shape_type) == 13]
        assert len(pictures) == 1
        picture = pictures[0]
        assert all(value == 0 for value in (picture.crop_left, picture.crop_right, picture.crop_top, picture.crop_bottom))

    before = Presentation(BASELINE)
    for number in UNCHANGED:
        assert slide_signature(prs.slides[number - 1]) == slide_signature(before.slides[number - 1])
    changed_pixels = []
    for number in UNCHANGED:
        old = Image.open(ROOT / f".pptx_build/pre_final_update_render/slide-{number:02d}.png").convert("RGB")
        new = Image.open(ROOT / f".pptx_build/final_accepted_render/slide110-{number:02d}.png").convert("RGB")
        assert old.size == new.size
        if ImageChops.difference(old, new).getbbox() is not None:
            changed_pixels.append(number)
    assert not changed_pixels, changed_pixels
    return {"status": "passed", "pdf_pages": 36, "changed_slide_layout_checks": records,
            "all_pdf_text_no_replacement_or_nul": True, "main_figures_uncropped": True,
            "unchanged_slides_shape_text_geometry": UNCHANGED, "unchanged_slides_pixel_identical_110dpi": UNCHANGED,
            "visual_inspection": "Automated PDF text/geometry and pixel checks only; this subagent cannot receive image pixels."}


def main() -> None:
    prs = Presentation(PPTX)
    assert len(prs.slides) == 36
    result = {
        "status": "passed",
        "pptx": str(PPTX),
        "pdf": str(PDF),
        "pass1_accepted_reports": pass1_reports(prs),
        "pass2_raw_recalculation": pass2_raw_recalculation(prs),
        "pass3_pdf_layout_and_historical_pixels": pass3_pdf_layout_and_pixels(prs),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "passes": 3, "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
