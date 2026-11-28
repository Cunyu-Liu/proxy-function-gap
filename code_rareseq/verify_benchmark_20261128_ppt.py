from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "生物序列长尾与OOD_基准证据更新_2026-11-28.pptx"
SOURCE = ROOT / "生物序列长尾与OOD_基准证据更新_2026-11-26.pptx"
FIG = ROOT / "figures"
OUT = ROOT / ".pptx_build/ppt_20261128_three_pass_verification.json"

EXPECTED = {
    32: ["0.031365", "0.043649", "0.024114", "0.034854", "0.0002"],
    33: ["25/25", "0.646192", "0.607869", "0.655701"],
    34: ["0.507697", "0.508286", "0.155127", "0.473079", "23/24"],
    35: ["0.475605", "0.430202", "0.352559", "0.185360", "0.111699", "0.098188", "70/70"],
    37: ["0.875332", "0.044580", "0.083869", "0.790567", "0.038785", "0.068516"],
    38: ["4,380", "63"],
    30: ["0.0713"],
}
REPORT_VALUES = {
    "0.031365": 0.031365257056409676, "0.043649": 0.043648983651954916,
    "0.024114": 0.024114010919176396, "0.034854": 0.034854,
    "0.646192": 0.6461923914676485, "0.607869": 0.6078693548441341,
    "0.655701": 0.6557012966685122, "0.507697": 0.5076970864483988,
    "0.508286": 0.5082856970628693, "0.155127": 0.15512714314888562,
    "0.473079": 0.4730791298986947, "0.475605": 0.475605,
    "0.430202": 0.430202, "0.352559": 0.352559, "0.185360": 0.1853595262,
    "0.111699": 0.1116987234, "0.098188": 0.09818752856,
}


def slide_text(slide):
    return "\n".join(sh.text_frame.text for sh in slide.shapes if sh.has_text_frame)


def slide_images(slide):
    out = []
    for sh in slide.shapes:
        if sh.shape_type == 13:
            out.append(hashlib.sha256(sh.image.blob).hexdigest())
    return out


def main():
    tgt = Presentation(str(TARGET))
    src = Presentation(str(SOURCE))
    tsl, ssl = list(tgt.slides), list(src.slides)
    fails = []
    if len(tsl) != 39:
        fails.append(f"slide count {len(tsl)}")
    for i in range(30):
        if slide_text(tsl[i]) != slide_text(ssl[i]):
            fails.append(f"history slide {i} text changed")
    for idx, needles in EXPECTED.items():
        text = slide_text(tsl[idx])
        for n in needles:
            if n not in text:
                fails.append(f"slide {idx} missing {n}")
    for token, value in REPORT_VALUES.items():
        if abs(float(token) - value) > 5e-6:
            fails.append(f"report rounding mismatch {token} vs {value}")
    figmap = {31: "Fig1_data_audit_v2_20261128.png", 32: "Fig2_eclip_tail_forest_v2_20261128.png",
              33: "Fig4_protein_crossprotocol_v2_20261128.png", 34: "Fig5_ood_metric_conflicts_v2_20261128.png",
              35: "Fig6_rnacompete_dual_track_v2_20261128.png",
              37: "Fig3_low_homology_stratification_v2_20261128.png",
              38: "Fig7_convergence_evidence_v2_20261128.png"}
    for idx, name in figmap.items():
        want = hashlib.sha256((FIG / name).read_bytes()).hexdigest()
        got = slide_images(tsl[idx])
        if want not in got:
            fails.append(f"slide {idx} image sha mismatch for {name}")
    status = "passed" if not fails else "FAILED"
    result = {"status": status, "failures": fails, "slides": len(tsl),
              "history_unchanged": list(range(30)), "figure_sha_checked": list(figmap),
              "numeric_tokens_checked": len(REPORT_VALUES)}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("PPT_20261128_VERIFY", status, json.dumps(fails, ensure_ascii=False))
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()