from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path

import fitz
from PIL import Image, ImageChops
from pptx import Presentation

from build_benchmark_update_ppt import slide_signature

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "生物序列长尾与OOD_基准证据更新_2026-11-26.pptx"
BASELINE = ROOT / ".pptx_build/pre_rnacompete_update.pptx"
PDF = ROOT / ".pptx_build/final_37_render/生物序列长尾与OOD_基准证据更新_2026-11-26.pdf"
OUT = ROOT / ".pptx_build/ppt_rnacompete_three_pass_verification.json"
CHANGED = (31, 37)
UNCHANGED = list(range(1, 31)) + list(range(32, 37))


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def slide_text(prs: Presentation, number: int) -> str:
    return "\n".join(shape.text for shape in prs.slides[number - 1].shapes if hasattr(shape, "text"))


def pass1_reports(prs: Presentation) -> dict:
    report = (ROOT / "E1_强基线/RNAcompete_Ray2013_RNAFM五折最终报告_2026-11-27.md").read_text(encoding="utf-8")
    figure_report = (ROOT / "论文证据/图表设计与数值核验_2026-11-25.md").read_text(encoding="utf-8")
    report_tokens = ["五折三seed双轨完成并通过最终独立验收", "70/70评价通过", "0.475605±0.002162",
                     "0.185360±0.027500", "不能据此宣称统一优越或机制解释", "不构成SOTA、显著性"]
    figure_tokens = ["tail MLP−kNN Pearson=+0.158506", "RBP kNN−MLP Spearman=+0.073661",
                     "最高实验宏Spearman<0.20，不称可靠OOD"]
    for token in report_tokens:
        assert token in report, token
    for token in figure_tokens:
        assert token in figure_report, token
    required = {
        31: ["RNAcompete 五折", "五折三 seed", "MLP 尾部占优", "0.185"],
        37: ["序列尾部与整RBP-OOD的方法排序不同", "0.475605", "+0.123046", "Spearman=0.185360",
             "五折 × 三 seed", "70/70", "非 SOTA 或可靠 OOD"],
    }
    for number, tokens in required.items():
        text = slide_text(prs, number)
        assert all(token in text for token in tokens), (number, "missing")
    assert "AUROC=0.185" not in slide_text(prs, 37)
    return {"status": "passed", "final_report_tokens": report_tokens, "figure_report_tokens": figure_tokens,
            "ppt_tokens": required, "metric_0.185_is_spearman_not_auroc": True}


def pass2_raw_recalculation(prs: Presentation) -> dict:
    verification = json.loads((ROOT / "figures/data/rnacompete_ray2013_final/final_independent_verification.json").read_text())
    assert verification["status"] == "passed" and verification["n_evaluations"] == 70
    assert verification["n_neural_checkpoint_replays"] == 60 and verification["n_run_metric_rows"] == 525
    assert verification["n_summary_rows"] == 45 and verification["selection_test_used"] is False
    rows = read_tsv(ROOT / "figures/verification/data/fig6/run_metrics_525.tsv")
    assert len(rows) == 525
    for row in rows:
        assert int(row["fold"]) in range(5)
        for metric in ("spearman", "pearson", "mse"):
            assert math.isfinite(float(row[metric]))
    selected = {}
    for track, arch in (("sequence", "linear"), ("sequence", "mlp"), ("sequence", "knn"),
                        ("rbp_ood", "linear"), ("rbp_ood", "mlp"), ("rbp_ood", "knn")):
        tier = "tail" if track == "sequence" else "Overall"
        values = [row for row in rows if row["track"] == track and row["arch"] == arch and
                  row["probe_set"] == "All" and row["tier"] == tier]
        expected_runs = 5 if arch == "knn" else 15
        assert len(values) == expected_runs
        assert {int(row["fold"]) for row in values} == set(range(5))
        if arch != "knn":
            assert {int(float(row["seed"])) for row in values} == {0, 1, 2}
        selected[(track, arch)] = {
            "spearman": statistics.mean(float(row["spearman"]) for row in values),
            "pearson": statistics.mean(float(row["pearson"]) for row in values),
        }
    text = slide_text(prs, 37)
    tail_mlp = selected[("sequence", "mlp")]["spearman"]
    tail_knn = selected[("sequence", "knn")]["spearman"]
    tail_linear = selected[("sequence", "linear")]["spearman"]
    rbp_knn = selected[("rbp_ood", "knn")]["spearman"]
    rbp_mlp = selected[("rbp_ood", "mlp")]["spearman"]
    rbp_linear = selected[("rbp_ood", "linear")]["spearman"]
    assert f"{tail_mlp:.6f}" == "0.475605" and f"{tail_mlp:.6f}" in text
    assert tail_mlp > tail_linear and tail_mlp > tail_knn
    assert f"{tail_mlp - tail_knn:.6f}" == "0.123046" and f"+{tail_mlp - tail_knn:.6f}" in text
    assert f"{rbp_knn:.6f}" == "0.185360" and f"{rbp_knn:.6f}" in text
    assert rbp_knn > rbp_mlp > rbp_linear and rbp_knn < 0.20
    assert f"{rbp_knn - rbp_mlp:.6f}" == "0.073661"
    summary = read_tsv(ROOT / "figures/data/rnacompete_ray2013_final/final_independent_summary.tsv")
    assert len(summary) == 45
    return {"status": "passed", "run_rows": len(rows), "summary_rows": len(summary),
            "evaluations": 70, "neural_checkpoint_replays": 60,
            "tail_spearman": {"linear": f"{tail_linear:.6f}", "mlp": f"{tail_mlp:.6f}", "knn": f"{tail_knn:.6f}",
                              "mlp_minus_knn": f"{tail_mlp - tail_knn:.6f}"},
            "rbp_ood_experiment_macro_spearman": {"linear": f"{rbp_linear:.6f}", "mlp": f"{rbp_mlp:.6f}",
                                                  "knn": f"{rbp_knn:.6f}", "knn_minus_mlp": f"{rbp_knn - rbp_mlp:.6f}"},
            "descriptive_not_reliable_ood": True}


def pass3_layout_and_history(prs: Presentation) -> dict:
    document = fitz.open(PDF)
    assert len(document) == 37
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
            assert expected == normalize("".join(span["text"] for _, span in matched)), (number, shape.name, "mismatch")
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
    pictures = [shape for shape in prs.slides[36].shapes if int(shape.shape_type) == 13]
    assert len(pictures) == 1
    picture = pictures[0]
    assert all(value == 0 for value in (picture.crop_left, picture.crop_right, picture.crop_top, picture.crop_bottom))

    baseline = Presentation(BASELINE)
    for number in UNCHANGED:
        assert slide_signature(prs.slides[number - 1]) == slide_signature(baseline.slides[number - 1])
    changed_pixels = []
    for number in UNCHANGED:
        old = Image.open(ROOT / f".pptx_build/pre_rnacompete_render/slide-{number:02d}.png").convert("RGB")
        new = Image.open(ROOT / f".pptx_build/final_37_render/slide110-{number:02d}.png").convert("RGB")
        assert old.size == new.size
        if ImageChops.difference(old, new).getbbox() is not None:
            changed_pixels.append(number)
    assert not changed_pixels, changed_pixels
    return {"status": "passed", "pdf_pages": 37, "changed_slide_layout_checks": records,
            "all_pdf_text_no_replacement_or_nul": True, "fig6_uncropped": True,
            "unchanged_slide_shape_text_geometry": UNCHANGED, "unchanged_slides_pixel_identical_110dpi": UNCHANGED,
            "visual_inspection": "Automated PDF text/geometry and historical-pixel checks only; no image-pixel visual claim."}


def main() -> None:
    prs = Presentation(PPTX)
    assert len(prs.slides) == 37
    result = {"status": "passed", "pptx": str(PPTX), "pdf": str(PDF),
              "pass1_accepted_reports": pass1_reports(prs),
              "pass2_raw_recalculation": pass2_raw_recalculation(prs),
              "pass3_layout_and_historical_pixels": pass3_layout_and_history(prs)}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "passes": 3, "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
