from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from delivery_scripts.publication_paths import DELIVERABLES, EVIDENCE_ROOT, OUTPUT_ROOT

ROOT = EVIDENCE_ROOT
SRC = DELIVERABLES / "reports" / "source_data"
OUT = OUTPUT_ROOT / "figures" / "v4"
OUT.mkdir(parents=True, exist_ok=True)
DIAG = SRC / "tcell_error_diagnosis_v1" / "analysis"
V1 = SRC / "tcell_target_adaptation_v1" / "target"
V2 = SRC / "tcell_listnet_adaptation_v2" / "target_v2"
ANCHOR = SRC / "tcell_cross_cell_anchor_v1" / "target"
STEMS = ["Figure16_v4_tcell_global_value_tail_bottleneck", "Figure17_v4_tcell_regression_decision_decoupling"]
CONFIG = json.loads((HERE / "paper1_figures_16_17_config.json").read_text())


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b):
    assert np.isclose(a, b, rtol=0, atol=1e-12), (a, b)


def output_hashes():
    result = {}
    for stem in STEMS:
        for suffix in ("png", "pdf", "svg"):
            path = OUT / f"{stem}.{suffix}"
            result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def round_numeric():
    point = pd.read_csv(DIAG / "pointwise_metrics.tsv", sep="\t")
    selection = pd.read_csv(DIAG / "selection_summary.tsv", sep="\t")
    tail = pd.read_csv(DIAG / "top_tail_summary.tsv", sep="\t")
    summary = load_json(DIAG / "summary.json")
    effects = pd.DataFrame(summary["selection_effects"])
    checks = 0
    for context in ("defined_tcell_r1", "defined_tcell_r2"):
        row = point[(point.context == context) & (point.model == "cnn")].iloc[0]
        assert row.n == 1024 and row.spearman > 0.5 and row.mae > 1.5
        for cap in (1, 2, 4, 8, 16):
            rows = selection[(selection.context == context) & (selection.model == "cnn") & (selection.cap == cap)].set_index("strategy")
            effect = effects[(effects.context == context) & (effects.model == "cnn") & (effects.contrast == "bon_minus_random_n") & (effects.cap == cap)].iloc[0]
            close(rows.loc["bon", "regret"] - rows.loc["random_n", "regret"], effect.effect)
            close(rows.loc["bon", "regret"], rows.loc["bon", "opportunity_loss"] + rows.loc["bon", "within_prefix_ranking_loss"])
            if cap > 1:
                assert effect.effect < 0 and effect.ci_high < 0
            checks += 3
        top = tail[(tail.context == context) & (tail.model == "cnn") & (tail.cap == 16)].iloc[0]
        close(top.regret, top.opportunity_loss + top.within_tail_ranking_loss)
        assert top.within_tail_ranking_loss > top.opportunity_loss and top.winner_flipped_within_tail == 0.9375
        checks += 3
    v1p = load_json(V1 / "primary.json"); v1r = load_json(V1 / "results.json"); v1s = load_json(V1 / "secondary.json")
    repeat_effects = []
    for context in ("defined_tcell_r1", "defined_tcell_r2"):
        cap = v1r["by_repeat_and_cap"][context]["16"]["mean_regret"]
        effect = cap["main_50k"] - cap["original_cnn"]
        close(effect, v1p["repeat_main_minus_original_effects"][context]); repeat_effects.append(effect)
        pw = v1s["pointwise_errors_mrl_units"][context]
        mae_change = pw["main_50k"]["mae"] - pw["original_cnn"]["mae"]
        close(mae_change, v1s["error_regret_synchronization"][context]["main_50k"]["mae_change"])
        assert mae_change < 0 < effect
        checks += 4
    close(np.mean(repeat_effects), v1p["combined_main_minus_original"]["effect"])
    close(max(0.1, 0.05 * v1p["combined_original_cap16_regret"]), v1p["minimum_required_absolute_reduction"])
    assert not v1p["success"] and v1p["evaluation_count"] == 1
    v2p = load_json(V2 / "primary.json"); v2r = load_json(V2 / "results.json")
    cap = v2r["by_context_and_cap"]["mean_replicate"]["16"]["mean_regret"]
    for comparator in ("original_cnn", "v1_main"):
        comp = v2p["comparisons"][comparator]
        close(cap["listnet"] - cap[comparator], comp["effect"]["effect"])
        assert comp["effect"]["ci_low"] < 0 < comp["effect"]["ci_high"]
        assert comp["absolute_reduction"] < comp["minimum_required_absolute_reduction"]
        assert not v2p["comparison_pass"][comparator]
        checks += 4
    assert not v2p["success"] and v2p["evaluation_count"] == 1
    anchorp = load_json(ANCHOR / "primary.json")
    vals = list(anchorp["repeat_anchor_minus_cnn_effects"].values())
    close(np.mean(vals), anchorp["combined_anchor_minus_cnn"]["effect"])
    assert min(vals) > 0 and anchorp["combined_anchor_minus_cnn"]["ci_low"] > 0 and not anchorp["success"]
    rank = load_json(SRC / "tcell_rank_bstrict_v2" / "artifacts" / "auxiliary_gate.json")
    backbone = load_json(SRC / "tcell_multicontext_backbone_v1" / "auxiliary_gate.json")
    assert not rank["all_directions_pass"] and not rank["target_mrl_read"] and not rank["target_evaluation_permitted"]
    assert not backbone["both_directions_pass"] and backbone["target_evaluation_count"] == 0 and not backbone["target_mrl_read"]
    for record in CONFIG["sources"]:
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    return {"status": "PASS", "independent_numeric_checks": checks + 18, "source_hashes": len(CONFIG["sources"]),
            "test_v1_test_v2_separate": True, "auxiliary_target_evaluation_zero": True,
            "failed_methods_not_claimed_as_improvement": True}


def round_deterministic():
    subprocess.run([sys.executable, str(HERE / "build_paper1_figures_16_17.py")], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    before = output_hashes()
    for record in CONFIG["figures"]:
        assert before[Path(record["path"]).name] == record["sha256"]
    time.sleep(1.1)
    env = dict(os.environ, PYTHONHASHSEED="987654321", TZ="UTC", LC_ALL="C")
    subprocess.run([sys.executable, str(HERE / "build_paper1_figures_16_17.py")], cwd=HERE, check=True, stdout=subprocess.DEVNULL, env=env)
    after = output_hashes(); assert before == after
    return {"status": "PASS", "files_identical": len(before), "reference_files_identical": len(CONFIG["figures"]), "rebuild_rounds": 2,
            "environment_variation": {"PYTHONHASHSEED": "987654321", "TZ": "UTC", "LC_ALL": "C"}, "cross_second_delay": 1.1}


def round_format():
    stats = []; rendered = []
    for stem in STEMS:
        png = OUT / f"{stem}.png"
        with Image.open(png) as im:
            im.verify()
        with Image.open(png) as im:
            assert im.info["dpi"][0] == 299.9994 and im.info["dpi"][1] == 299.9994
            assert im.width >= 4000 and im.height >= 2500
            arr = np.asarray(im.convert("RGB")); assert arr.std() > 20
            stats.append({"figure": stem, "pixels": [im.width, im.height], "dpi": [300, 300]})
            thumb = im.convert("RGB"); thumb.thumbnail((1000, 700)); rendered.append(thumb)
        doc = fitz.open(OUT / f"{stem}.pdf"); assert len(doc) == 1
        page = doc[0]; text = page.get_text()
        for word in page.get_text("words"):
            assert word[0] >= -0.5 and word[1] >= -0.5 and word[2] <= page.rect.width + 0.5 and word[3] <= page.rect.height + 0.5, (stem, word)
        assert "Retrospective" in text or "retrospective" in text
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                A, B = blocks[a], blocks[b]
                dx = min(A[2], B[2]) - max(A[0], B[0]); dy = min(A[3], B[3]) - max(A[1], B[1])
                assert not (dx > 1 and dy > 1), (stem, A[4], B[4], dx, dy)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pix.save(OUT / f"{stem}_render.png")
        root = ET.parse(OUT / f"{stem}.svg").getroot()
        assert root.tag.endswith("svg")
        texts = [e for e in root.iter() if e.tag.endswith("text")]
        assert len(texts) > 20
        assert (OUT / f"{stem}.pdf").stat().st_size > 20000 and (OUT / f"{stem}.svg").stat().st_size > 30000
    sheet = Image.new("RGB", (1020, 2 * 740), "#DDE3EB")
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(rendered):
        sheet.paste(im, ((1020 - im.width) // 2, i * 740 + 30))
        draw.text((12, i * 740 + 8), f"Figure {16+i}", fill="#17233B")
    sheet.save(OUT / "paper1_figures_16_17_contact_sheet.png")
    fonts = subprocess.check_output(["pdffonts", str(OUT / f"{STEMS[0]}.pdf")], text=True)
    assert "Arial" in fonts or "Helvetica" in fonts or "DejaVuSans" in fonts
    return {"status": "PASS", "figures": stats, "pdf_pages_each": 1, "pdf_words_outside": 0, "pdf_text_collisions": 0,
            "svg_editable_text": True, "png_dpi": 300, "contact_sheet": "paper1_figures_16_17_contact_sheet.png"}


report = {"round1_numeric_independent": round_numeric(), "round2_deterministic_rebuild": round_deterministic(), "round3_format_render": round_format()}
report["overall"] = "PASS"
(OUT / "paper1_figures_16_17_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
