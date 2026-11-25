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

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "生物序列长尾与OOD_基准证据更新_2026-11-26.pptx"
PDF = ROOT / ".pptx_build/final_render/生物序列长尾与OOD_基准证据更新_2026-11-26.pdf"
OUT = ROOT / ".pptx_build/ppt_three_pass_verification.json"
REPORTS = {
    "mainline": ROOT / "论文证据/预印本科学主线与证据链_2026-11-25.md",
    "figures": ROOT / "论文证据/图表设计与数值核验_2026-11-25.md",
    "e5": ROOT / "E5_OOD/PAR-CLIP跨协议评测报告_2026-11-25.md",
    "e6": ROOT / "E6_蛋白迁移/E6_五划分冻结基线报告_2026-11-25.md",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def slide_texts(prs: Presentation) -> dict[int, str]:
    return {
        number: "\n".join(shape.text for shape in prs.slides[number - 1].shapes if hasattr(shape, "text"))
        for number in range(31, 35)
    }


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def pass1_reports(prs: Presentation) -> dict:
    texts = slide_texts(prs)
    report_text = {key: path.read_text(encoding="utf-8") for key, path in REPORTS.items()}
    requirements = {
        31: {
            "mainline": ["fold=-1", "旧 E1 主集报告已撤回验收", "foldfix", "pseudo-OOD"],
        },
        32: {
            "mainline": ["21,420", "2,168,739", "2,147,319", "旧 E1 主集报告已撤回验收"],
            "figures": ["id90 连通分量整折是设计意图", "fold=-1 正例纳入每个 train 折"],
        },
        33: {
            "e5": ["0.646192±0.002641", "0.607869±0.002595", "0.655701±0.004951", "patience=20", "min_delta=1e-4"],
            "figures": ["9 RBP 外部描述性证据", "不同负例切片的差值不是模型提升或因果校正"],
        },
        34: {
            "e6": ["+0.270534", "+0.344729", "+0.313893", "+0.203189", "Test-2 只是 pseudo-OOD", "5/5 split 描述性优于 deep-kNN"],
            "figures": ["全部 20 个 split×频率桶单元格均为正", "95% CI 排除 0"],
        },
    }
    ppt_requirements = {
        31: ["fold=-1", "foldfix", "pseudo-OOD"],
        32: ["21,420", "2,168,739", "2,147,319", "foldfix"],
        33: ["0.646192", "0.607869", "0.655701", "p20", "外部描述性"],
        34: ["20/20", "+0.270534", "+0.344729", "+0.313893", "+0.203189", "pseudo-OOD v2"],
    }
    checked = []
    for number, sources in requirements.items():
        for source, tokens in sources.items():
            for token in tokens:
                assert token in report_text[source], (number, source, token)
                checked.append({"slide": number, "source": source, "token": token})
        for token in ppt_requirements[number]:
            assert token in texts[number], (number, token)
    forbidden = ["0.02515", "0.02789", "0.8901", "0.8188", "E5-rbp-holdout", "2/72", "M1a", "nnPU", "统一拒识赢家"]
    for number, text in texts.items():
        assert not any(token in text for token in forbidden), (number, "forbidden")
    assert len(prs.slides) == 34
    return {"status": "passed", "report_token_checks": len(checked), "ppt_numeric_and_boundary_checks": ppt_requirements,
            "old_E1_unfinished_E5_and_overclaim_tokens_absent": True}


def pass2_independent_recalculation(prs: Presentation) -> dict:
    texts = slide_texts(prs)
    per_seed = defaultdict(dict)
    raw_dir = ROOT / "figures/data/parclip_p20"
    for seed in range(3):
        rows = read_tsv(raw_dir / f"seed{seed}_cross_protocol_results.tsv")
        assert len(rows) == 27
        by_protocol = defaultdict(list)
        for row in rows:
            assert math.isfinite(float(row["auroc"]))
            assert int(row["stopped_epoch"]) - int(row["best_epoch"]) == 20
            by_protocol[row["eval_protocol"]].append(float(row["auroc"]))
        assert all(len(values) == 9 for values in by_protocol.values())
        for protocol, values in by_protocol.items():
            per_seed[protocol][seed] = statistics.mean(values)
    e5 = {}
    for protocol, seed_values in per_seed.items():
        values = [seed_values[seed] for seed in range(3)]
        formatted = f"{statistics.mean(values):.6f}"
        assert formatted in texts[33]
        e5[protocol] = {"mean": formatted, "sample_sd": f"{statistics.stdev(values):.6f}", "seed_values": values}
    assert e5["eCLIP-neg1c-fold0"]["sample_sd"] == "0.002641"
    assert e5["eCLIP-neg2-fold0"]["sample_sd"] == "0.002595"
    assert e5["PARCLIP-otherRBP-geneMatched-excl300"]["sample_sd"] == "0.004951"

    bucket_rows = read_tsv(ROOT / "E6_蛋白迁移/results/five_split_agg/five_split_buckets.tsv")
    assert len(bucket_rows) == 25
    frequency_rows = [row for row in bucket_rows if row["bucket"] != "all"]
    assert len(frequency_rows) == 20
    assert all(float(row["delta_macro_auprc"]) > 0 for row in frequency_rows)
    assert all(float(row["ci_low"]) > 0 for row in frequency_rows)
    assert all(int(row["n_bootstrap"]) == 10000 for row in bucket_rows)
    e6_buckets = {}
    for bucket in ("tail_lt10", "tail_10_29", "mid_30_99", "head_ge100"):
        values = [float(row["delta_macro_auprc"]) for row in bucket_rows if row["bucket"] == bucket]
        assert len(values) == 5
        mean = f"+{statistics.mean(values):.6f}"
        assert mean in texts[34]
        e6_buckets[bucket] = {"mean": mean, "sample_sd": f"{statistics.stdev(values):.6f}"}

    ood_rows = read_tsv(ROOT / "E6_蛋白迁移/results/five_split_agg/five_split_ood.tsv")
    assert len(ood_rows) == 15
    direction_counts = defaultdict(int)
    for split in range(1, 6):
        rows = {row["method"]: row for row in ood_rows if int(row["split"]) == split}
        for method in ("msp", "energy"):
            if float(rows[method]["auroc_ood"]) > float(rows["deep_knn"]["auroc_ood"]):
                direction_counts[(method, "auroc_ood")] += 1
            if float(rows[method]["fpr95"]) < float(rows["deep_knn"]["fpr95"]):
                direction_counts[(method, "fpr95")] += 1
            if float(rows[method]["aurc_ood_error"]) < float(rows["deep_knn"]["aurc_ood_error"]):
                direction_counts[(method, "aurc_ood_error")] += 1
        for row in rows.values():
            assert row["metric_definition"] == "e6-ood-v2:fpr95=id_at_ood_tpr95;aurc=ascending_ood_score"
            assert row["ood_recompute_run_id"] == "E6-ood-metric-recompute-v2-20261125-072246-ZDLg6Y"
    assert all(count == 5 for count in direction_counts.values())
    return {"status": "passed", "e5_macro_auroc_from_81_raw_rows": e5,
            "e6_20_frequency_cells_positive_and_ci_exclude_zero": True,
            "e6_bucket_means_from_25_raw_rows": e6_buckets,
            "e6_ood_v2_direction_counts": {f"{key[0]}:{key[1]}": value for key, value in direction_counts.items()}}


def pass3_rendered_pdf(prs: Presentation) -> dict:
    document = fitz.open(PDF)
    assert len(document) == 34
    all_pdf_text = "\n".join(page.get_text() for page in document)
    assert "\ufffd" not in all_pdf_text and "\x00" not in all_pdf_text
    records = []
    for number in range(31, 35):
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
                   span["bbox"][2] <= page.rect.width + 0.1 and span["bbox"][3] <= page.rect.height + 0.1
                   for span in spans)
        used_span_ids = set()
        shape_boxes = {}
        min_non_footer_size = math.inf
        for shape in prs.slides[number - 1].shapes:
            if not getattr(shape, "has_text_frame", False) or not shape.text.strip():
                continue
            expected = normalize(shape.text)
            assert expected in page_normalized, (number, shape.name, "text missing after rendering")
            x0 = shape.left / 914400 * 72
            y0 = shape.top / 914400 * 72
            x1 = (shape.left + shape.width) / 914400 * 72
            y1 = (shape.top + shape.height) / 914400 * 72
            matched = []
            for index, span in enumerate(spans):
                bbox = span["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                if x0 - 0.5 <= cx <= x1 + 0.5 and y0 - 0.5 <= cy <= y1 + 0.5:
                    matched.append((index, span))
            assert matched, (number, shape.name, "no PDF spans matched")
            matched_text = normalize("".join(span["text"] for _, span in matched))
            assert expected == matched_text, (number, shape.name, expected, matched_text)
            for index, span in matched:
                used_span_ids.add(index)
                bbox = span["bbox"]
                tolerance = 1.6
                assert bbox[0] >= x0 - tolerance and bbox[1] >= y0 - tolerance and bbox[2] <= x1 + tolerance and bbox[3] <= y1 + tolerance, (
                    number, shape.name, "PDF text outside PPT shape", bbox, (x0, y0, x1, y1), span["text"])
                if not shape.name.startswith("Footer"):
                    min_non_footer_size = min(min_non_footer_size, span["size"])
            shape_boxes[shape.name] = (x0, y0, x1, y1)
        box_items = list(shape_boxes.items())
        for i, (name_a, a) in enumerate(box_items):
            for name_b, b in box_items[i + 1:]:
                intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                assert intersection <= 0.5, (number, "text boxes overlap", name_a, name_b, intersection)
        assert len(used_span_ids) == len(spans), (number, "unmatched PDF spans")
        assert min_non_footer_size >= 17.95, (number, min_non_footer_size)
        records.append({"slide": number, "text_shapes_checked": len(shape_boxes), "pdf_spans_checked": len(spans),
                        "all_text_complete_and_inside_shape": True, "min_non_footer_pdf_font_pt": min_non_footer_size})
    for number in range(32, 35):
        pictures = [shape for shape in prs.slides[number - 1].shapes if int(shape.shape_type) == 13]
        assert len(pictures) == 1
        picture = pictures[0]
        assert all(value == 0 for value in (picture.crop_left, picture.crop_right, picture.crop_top, picture.crop_bottom))

    original_dir = ROOT / ".pptx_build/original_render"
    final_dir = ROOT / ".pptx_build/final_render_110"
    changed_history = []
    for number in range(1, 31):
        original = Image.open(original_dir / f"slide-{number:02d}.png").convert("RGB")
        final = Image.open(final_dir / f"slide-{number:02d}.png").convert("RGB")
        assert original.size == final.size
        if ImageChops.difference(original, final).getbbox() is not None:
            changed_history.append(number)
    assert not changed_history, changed_history
    return {"status": "passed", "pdf_pages": len(document), "new_slide_render_checks": records,
            "unicode_replacement_or_nul_characters": False, "main_figures_uncropped": True,
            "historical_slides_pixel_identical_at_110dpi": list(range(1, 31)),
            "visual_inspection": "Preview opened the four-slide contact sheet, but this subagent cannot receive image pixels; no human/visual claim is made."}


def main() -> None:
    prs = Presentation(PPTX)
    result = {
        "pptx": str(PPTX),
        "pdf": str(PDF),
        "pass1_ppt_against_accepted_reports": pass1_reports(prs),
        "pass2_independent_raw_table_recalculation": pass2_independent_recalculation(prs),
        "pass3_rendered_pdf_text_geometry_and_history": pass3_rendered_pdf(prs),
        "status": "passed",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUT), "passes": 3}, ensure_ascii=False))


if __name__ == "__main__":
    main()
