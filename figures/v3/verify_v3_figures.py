from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from delivery_scripts.publication_paths import DELIVERABLES, EVIDENCE_ROOT, OUTPUT_ROOT

import fitz
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = EVIDENCE_ROOT
OUT = OUTPUT_ROOT / "figures" / "v3"
DATA = DELIVERABLES / "figures" / "source_data"
SRC = DELIVERABLES / "reports" / "source_data"
REPORTS = DELIVERABLES / "reports"
CONFIG = json.loads((OUT / "v3_build_config.json").read_text())
E = CONFIG["evidence"]
SOURCES = {
    "pressure": DATA / "strict_h1_figure2_paired_pressure.csv",
    "common": DATA / "strict_h1_figure4_cross_architecture_summary.csv",
    "score": DATA / "strict_h1_figure3_score_matching.csv",
    "risk_frozen": SRC / "paper2_risk_frozen_v1" / "frozen_protocol.json",
    "paper3": SRC / "paper3_headroom_v4" / "headroom_v4_round1" / "summary.json",
    "curve": REPORTS / "paper3_headroom_v4_20261124.md",
    "schlusser": SRC / "schlusser2024_hepg2_te_confirmatory_v1" / "evaluator_round_1" / "summary.json",
    "leppek": SRC / "leppek2022_cds_mechanism_audit_v5" / "rank_summary.tsv",
    "multiplicity": SRC / "paper1_strict_h1_v3" / "analysis" / "strict_h1_multiplicity_sensitivity.json",
    "persist_summary": SRC / "persist_mrl_external_v1" / "results" / "summary.json",
    "persist_predictions": SRC / "persist_mrl_external_v1" / "predictions" / "predictions.tsv",
    "persist_labels": SRC / "persist_mrl_external_v1" / "protocol" / "evaluator_labels.tsv",
    "persist_seeds": SRC / "persist_mrl_external_v1" / "results" / "per_seed.tsv",
    "persist_bootstrap": SRC / "persist_mrl_external_v1" / "results" / "bootstrap_replicates.npy",
    "persist_audit": SRC / "persist_mrl_external_v1" / "protocol" / "inclusion_audit.json",
    "persist_raw": SRC / "persist_mrl_external_v1" / "raw" / "GSE173083_Table_S1_-_Attributes_for_pooled_233_sequences.xlsx",
}


def close(actual, expected):
    assert np.isclose(actual, expected, rtol=0, atol=1e-12), (actual, expected)


def round_numeric():
    for record in CONFIG["sources"]:
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
    pressure = pd.read_csv(SOURCES["pressure"])
    common = pd.read_csv(SOURCES["common"])
    score = pd.read_csv(SOURCES["score"])
    risk_frozen = json.loads(SOURCES["risk_frozen"].read_text())
    p3 = json.loads(SOURCES["paper3"].read_text())
    sch = json.loads(SOURCES["schlusser"].read_text())
    lep = pd.read_csv(SOURCES["leppek"], sep="\t")
    mult = json.loads(SOURCES["multiplicity"].read_text())
    assert np.allclose(pressure.bon_mean_regret - pressure.random_mean_regret, pressure.paired_bon_minus_random)
    t = pressure[(pressure.context == "Defined T-cell r1") & (pressure.proposal_cap == 16)].iloc[0]
    close(t.paired_bon_minus_random, E["tcell_cap16"])
    adjusted = [r for r in mult["rows"] if r["context"] in {"defined_tcell_r1", "defined_tcell_r2"} and r["proposal_cap"] >= 2]
    assert len(adjusted) == 8 and all(r["bonferroni_20_comparisons_99_75_percent_interval"][1] < 0 for r in adjusted)
    assert not mult["source"]["new_blind_test"]
    close(common.architecture_error_correlation.min(), E["correlation_min"])
    close(common.architecture_error_correlation.max(), E["correlation_max"])
    assert len(score) == 4 and ((score.ci_low < 0) & (score.ci_high > 0)).all()
    calibration = risk_frozen["calibration"]
    close(calibration["coverage"], E["risk_coverage"])
    close(calibration["regret_reduction"], E["risk_reduction"])
    for actual, expected in zip(calibration["regret_reduction_95ci"], E["risk_ci"]):
        close(actual, expected)
    close(calibration["full_bon_mean_regret"] - calibration["accepted_mean_regret"], calibration["regret_reduction"])
    assert calibration["accepted_decisions"] == 34 and calibration["decisions"] == 80
    assert calibration["accepted_pools"] == 15 and calibration["pools"] == 16
    assert calibration["regret_reduction"] < 0
    assert calibration["regret_reduction_95ci"][0] < 0 < calibration["regret_reduction_95ci"][1]
    assert risk_frozen["threshold_locked"] and not risk_frozen["calibration_gate"]["pass"]
    assert not risk_frozen["calibration_gate"]["checks"]["positive_point_reduction"]
    assert not risk_frozen["calibration_gate"]["checks"]["reduction_ci_excludes_zero"]
    assert not risk_frozen["outer_scoring_authorized"] and not risk_frozen["outer_scored"]
    gates = pd.DataFrame(p3["cases"]).groupby("representation").gate_pass.sum().astype(int).to_dict()
    assert gates == E["gates"] and not p3["outer_labels_read"] and not p3["outer_metrics_computed"]
    close(sch["endpoint_alignment_descriptive"]["spearman_cnn_MRL_vs_TE"], E["schlusser_rho"])
    paired16 = [p for p in sch["paired_comparisons"] if p["proposal_cap"] == 16]
    assert len(paired16) == 4 and all(p["gene_cluster_bootstrap_95_interval"][0] < 0 < p["gene_cluster_bootstrap_95_interval"][1] for p in paired16)
    rank = lep[(lep.source_family == "cds_24") & (lep.left_endpoint == "cds_in_solution_half_life") & (lep.right_endpoint == "nluc_fluc_6h")].iloc[0]
    assert int(rank.discordant_pairs) == 171 and int(rank.n_unordered_pairs) == 276
    close(rank.discordant_pairs / rank.n_unordered_pairs, E["leppek_reversal"])
    block = SOURCES["curve"].read_text().split("### kmer4：固定 5 池")[1].split("###")[0]
    rows = [[float(v) for v in line.strip().strip("|").split("|")] for line in block.splitlines() if line.startswith("| 16 |")]
    assert len(rows) == 1
    close(rows[0][1], E["random16"]); close(rows[0][4], E["ei16"])
    persist = json.loads(SOURCES["persist_summary"].read_text())
    pred = pd.read_csv(SOURCES["persist_predictions"], sep="\t")
    labels = pd.read_csv(SOURCES["persist_labels"], sep="\t")
    seeds = pd.read_csv(SOURCES["persist_seeds"], sep="\t")
    boot = np.load(SOURCES["persist_bootstrap"])
    audit = json.loads(SOURCES["persist_audit"].read_text())
    raw = pd.read_excel(SOURCES["persist_raw"], sheet_name="Attributes for pooled 233 seque")
    group = raw[raw["Group"] == "UTR (5') variants"]
    assert len(raw) == 233 and len(group) == 59 and group["Sequence 5 UTR"].nunique() == 49
    joined = pred.merge(labels, on=["sequence_id", "utr"], validate="one_to_one")
    assert len(joined) == 59 and joined.utr.nunique() == 49 and len(seeds) == 200
    close(joined.prediction.rank().corr(joined.mrl.rank()), E["persist"]["spearman"]["value"])
    close(seeds.random_regret.mean(), E["persist"]["random_top1_regret"]["value"])
    close(seeds.bon_regret.mean(), E["persist"]["bon_top1_regret"]["value"])
    close(seeds.delta.mean(), E["persist"]["primary"]["bon_minus_random_regret"])
    assert np.allclose(seeds.bon_regret - seeds.random_regret, seeds.delta)
    for col, key in ((1, "spearman"), (4, "random_top1_regret"), (5, "bon_top1_regret")):
        interval = np.quantile(boot[:, col], [0.025, 0.975])
        for actual, expected in zip(interval, E["persist"][key]["ci95"]):
            close(actual, expected)
    for actual, expected in zip(np.quantile(boot[:, 0], [0.025, 0.975]), E["persist"]["primary"]["ci95"]):
        close(actual, expected)
    sign_p = 2 * min((np.sum(boot[:, 0] <= 0) + 1) / (len(boot) + 1),
                     (np.sum(boot[:, 0] >= 0) + 1) / (len(boot) + 1))
    close(sign_p, E["persist"]["primary"]["p_two_sided_sign"])
    assert E["persist"]["primary"]["bon_minus_random_regret"] > 0
    assert E["persist"]["primary"]["ci95"][0] < 0 < E["persist"]["primary"]["ci95"][1]
    assert not persist["primary"]["significant_benefit"] and persist["negative_results_retained"]
    assert audit["conflicts"] == {} and audit["excluded_constructs"] == 0 and audit["excluded_utr_clusters"] == 0
    assert persist["algorithm_seeds"] == 200 and persist["bootstrap_replicates"] == 10000
    return {"status": "PASS", "source_hashes": len(CONFIG["sources"]), "tcell_bonferroni_intervals": 8,
            "score_matching_cross_zero": 4, "schlusser_cross_zero": 4, "leppek_reversal": "171/276",
            "paper2_frozen_calibration_gate_failed": True, "paper2_outer_scored": False,
            "paper3_outer_untouched": True, "persist_constructs": 59, "persist_utr_clusters": 49,
            "persist_algorithm_seeds_descriptive": 200, "persist_cluster_bootstrap_replicates": 10000,
            "persist_primary_crosses_zero": True, "persist_significant_benefit": False}


def round_deterministic():
    names = CONFIG["figures"]
    paths = [OUT / f"{name}.{ext}" for name in names for ext in ("png", "pdf", "svg")]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    subprocess.run([sys.executable, str(HERE / "build_v3_figures.py")], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    middle = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before == middle, [k for k in before if before[k] != middle[k]]
    time.sleep(1.1)
    env = dict(os.environ, PYTHONHASHSEED="987654321")
    subprocess.run([sys.executable, str(HERE / "build_v3_figures.py")], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before == after, [k for k in before if before[k] != after[k]]
    return {"status": "PASS", "byte_identical_rebuilds": len(paths), "rebuild_rounds": 2,
            "python_hash_seeds": ["inherited", "987654321"], "cross_second_delay": 1.1,
            "runtime_guard": "uses the current interpreter unless PFG_CANONICAL_PYTHON selects the original renderer",
            "renderer_runtime": CONFIG["runtime"]}


def round_format():
    result = {}
    required = {
        "Figure11_v3_evidence_chain": ["GATE FAILED", "MIXED EVIDENCE", "coverage 0.425", "Δ regret reduction −0.005420", "95% CI [−0.509271, +0.628623] crosses 0.", "outer scoring not authorized", "1.220", "62.0%"],
        "Figure12_v3_data_governance_ood": ["H2 cross-Sample pairs", "NO BENEFIT", "PERSIST", "outer labels unread", "exact + Hamming-1"],
        "Figure13_v3_persist_external": ["Spearman ρ = −0.008447", "[−0.326701, +0.315406]", "+0.104723", "[−0.100964, +0.212706]", "59 constructs · 49 UTR clusters", "200 algorithm seeds · cap16", "exact / equal-length H1 = 0", "RETROSPECTIVE · NON-BLIND", "not independent biological samples", "No BoN benefit"],
    }
    forbidden = {
        "Figure11_v3_evidence_chain": ["0.459", "0.672", "THRESHOLD NOT FROZEN", "Freeze a threshold", "abstain when blind risk is high"],
        "Figure12_v3_data_governance_ood": ["No independent same-endpoint", "OPEN GAP"],
        "Figure13_v3_persist_external": ["significant BoN benefit", "200 biological samples", "prospective confirmation", "new blind test"],
    }
    for name, phrases in required.items():
        png = OUT / f"{name}.png"
        with Image.open(png) as im:
            im.verify()
        with Image.open(png) as im:
            dpi = im.info.get("dpi", (0, 0))
            assert all(abs(v - 300) < 0.02 for v in dpi), dpi
            assert im.width >= 3000 and im.height >= 1500
            pixels = [im.width, im.height]
        doc = fitz.open(OUT / f"{name}.pdf")
        assert len(doc) == 1
        page = doc[0]
        text = page.get_text()
        for phrase in phrases:
            assert phrase in text, (name, phrase)
        for phrase in forbidden[name]:
            assert phrase not in text, (name, "forbidden", phrase)
        assert all(w[0] >= -0.5 and w[1] >= -0.5 and w[2] <= page.rect.width + 0.5 and w[3] <= page.rect.height + 0.5 for w in page.get_text("words"))
        assert not page.get_images(), "PDF should remain vector"
        root = ET.parse(OUT / f"{name}.svg").getroot()
        tags = [el.tag.rsplit("}", 1)[-1] for el in root.iter()]
        assert "text" in tags and "image" not in tags
        result[name] = {"png_px": pixels, "png_dpi": [round(v, 3) for v in dpi], "pdf_pages": 1,
                        "pdf_vector": True, "svg_editable_text": True, "page_bounds": "PASS"}
    return {"status": "PASS", "figures": result}


report = {"round1_numeric_independent": round_numeric(), "round2_deterministic_render": round_deterministic(),
          "round3_format_render": round_format()}
report["overall"] = "PASS"
(OUT / "v3_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
