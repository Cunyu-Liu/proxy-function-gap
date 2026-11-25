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
SRC = DELIVERABLES / "reports" / "source_data"
BUDGET = SRC / "paper1_budget_matched_v1"
SEQUENCE = SRC / "paper1_sequence_ablation_v1"
RESULTS = SEQUENCE / "analysis" / "results"
CONFIG_PATH = OUT / "v3_figures_14_15_build_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text())


def close(actual, expected):
    assert np.isclose(actual, expected, rtol=0, atol=1e-12), (actual, expected)


def holm(values):
    values = np.asarray(values, dtype=float)
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    result = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        result[index] = running
    return result


def hashed_seed(base, label, context, cap):
    digest = hashlib.sha256(f"{base}|{label}|{context}|{cap}".encode()).hexdigest()
    return int(digest[:8], 16)


def tail_p(draws):
    low = (1 + np.sum(draws <= 0)) / (len(draws) + 1)
    high = (1 + np.sum(draws >= 0)) / (len(draws) + 1)
    return min(1.0, 2 * min(low, high))


def round_numeric():
    for record in CONFIG["sources"]:
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
    budget = json.loads((BUDGET / "results.json").read_text())
    budget_pools = pd.read_csv(BUDGET / "pool_effects.tsv", sep="\t")
    family_table = pd.read_csv(BUDGET / "family_effects.tsv", sep="\t")
    assert len(budget_pools) == 320 and len(family_table) == 60
    assert not budget["new_blind_test"] and budget["status"] == "PASS"
    bootstrap_checks = 0
    for family_name, rows in budget["families"].items():
        assert len(rows) == 20
        for row in rows:
            selected = budget_pools[(budget_pools.context == row["context"]) & (budget_pools.cap == row["cap"])]
            assert len(selected) == 16
            values = selected[family_name].to_numpy(float)
            close(values.mean(), row["effect"])
            if np.all(values == 0):
                close(row["p"], 1)
                assert row["nominal_95_ci"] == [0.0, 0.0]
            else:
                rng = np.random.Generator(np.random.PCG64(row["bootstrap_seed"]))
                indices = rng.integers(0, 16, size=(10000, 16))
                draws = values[indices].mean(axis=1)
                assert np.allclose(np.percentile(draws, [2.5, 97.5]), row["nominal_95_ci"], rtol=0, atol=1e-12)
                assert np.allclose(np.percentile(draws, [0.125, 99.875]), row["bonferroni_20_ci"], rtol=0, atol=1e-12)
                close(tail_p(draws), row["p"])
                bootstrap_checks += 1
            original = budget["families"]["bon_minus_random1"]
            if family_name != "bon_minus_random1":
                old = next(r for r in original if r["context"] == row["context"] and r["cap"] == row["cap"])
                old_values = budget_pools[(budget_pools.context == row["context"]) & (budget_pools.cap == row["cap"])]
                assert np.allclose(old_values.bon_minus_random1,
                                   old_values.bon_minus_random_n + old_values.random_n_minus_random1,
                                   rtol=0, atol=1e-12)
                close(old["effect"], row["effect"] + next(
                    r["effect"] for r in budget["families"]["random_n_minus_random1"]
                    if family_name == "bon_minus_random_n" and r["context"] == row["context"] and r["cap"] == row["cap"]
                )) if family_name == "bon_minus_random_n" else None
        corrected = holm([r["p"] for r in rows])
        assert np.allclose(corrected, [r["holm_p"] for r in rows], rtol=0, atol=1e-12)
        assert np.allclose(np.minimum(1, 20 * np.array([r["p"] for r in rows])),
                           [r["bonferroni_p"] for r in rows], rtol=0, atol=1e-12)
    equal = budget["families"]["bon_minus_random_n"]
    random_rows = budget["families"]["random_n_minus_random1"]
    r1 = [r for r in equal if r["context"] == "defined_tcell_r1" and r["cap"] > 1]
    r2 = [r for r in equal if r["context"] == "defined_tcell_r2" and r["cap"] > 1]
    assert all(r["holm_p"] < 0.05 for r in r1)
    assert [r["cap"] for r in r2 if r["holm_p"] < 0.05] == [4]
    assert not any(r["holm_p"] < 0.05 for r in random_rows)
    assert not any(r["holm_p"] < 0.05 for r in equal if r["context"] == "n50_hek293t_r1")
    assert [r["cap"] for r in equal if r["context"] == "defined_hepg2_r1" and r["holm_p"] < 0.05] == [2]

    frozen = json.loads((SEQUENCE / "frozen" / "frozen_analysis.json").read_text())
    primary = pd.read_csv(RESULTS / "primary_paired.tsv", sep="\t")
    sequence_pools = pd.read_csv(RESULTS / "pool_effects.tsv", sep="\t")
    families = pd.read_csv(RESULTS / "sequence_family_sensitivity.tsv", sep="\t")
    summary = json.loads((RESULTS / "summary.json").read_text())
    assert len(primary) == 20 and len(sequence_pools) == 320 and len(families) == 40
    assert frozen["bootstrap"]["replicates"] == 10000 and frozen["poly_u_q75_nearest_rank"] == 3
    assert not summary["new_blind_test"] and not summary["causal_claim"]
    unadjusted_bootstrap_checks = 0
    for _, target in primary.iterrows():
        selected = sequence_pools[(sequence_pools.context == target.context) & (sequence_pools.cap == target.cap)]
        unadjusted = selected.mean_delta_regret.to_numpy(float)
        close(unadjusted.mean(), target.unadjusted_effect)
        if target.cap == 1:
            assert np.all(unadjusted == 0)
            continue
        rng = np.random.RandomState(hashed_seed(20261125, "unadjusted", target.context, target.cap))
        draws = np.array([unadjusted[rng.randint(0, 16, 16)].mean() for _ in range(10000)])
        assert np.allclose(np.percentile(draws, [2.5, 97.5]),
                           [target.unadjusted_ci_low, target.unadjusted_ci_high], rtol=0, atol=1e-12)
        close(tail_p(draws), target.unadjusted_p)
        unadjusted_bootstrap_checks += 1
    for prefix in ("unadjusted", "adjusted"):
        assert np.allclose(holm(primary[prefix + "_p"]), primary[prefix + "_holm_p"], rtol=0, atol=1e-12)
        assert np.allclose(np.minimum(1, 20 * primary[prefix + "_p"]),
                           primary[prefix + "_bonferroni_p"], rtol=0, atol=1e-12)
    assert not (primary.adjusted_holm_p < 0.05).any()
    nominal_adjusted = primary[(primary.context == "defined_tcell_r1") & primary.cap.isin([4, 8])]
    assert (nominal_adjusted.adjusted_ci_high < 0).all()
    assert primary[(primary.context == "defined_tcell_r2") & (primary.cap == 4)].adjusted_effect.iloc[0] > 0
    hepg2_reversal = primary[(primary.context == "defined_hepg2_r1") & (primary.cap == 4)].iloc[0]
    assert hepg2_reversal.adjusted_effect > 0 and hepg2_reversal.adjusted_holm_p > 0.05
    for family_name, group in families.groupby("family", sort=False):
        assert len(group) == 20
        assert np.allclose(holm(group.p), group.holm_p, rtol=0, atol=1e-12)
        assert np.allclose(np.minimum(1, 20 * group.p), group.bonferroni_p, rtol=0, atol=1e-12)
    assert not (families[families.family == "no_uAUG"].holm_p < 0.05).any()
    poly = families[(families.family == "no_polyU_upper_quartile") & (families.holm_p < 0.05)]
    assert set(zip(poly.context, poly.cap)) == {("defined_tcell_r1", 4), ("defined_tcell_r2", 8)}
    borderline = families[(families.family == "no_polyU_upper_quartile") &
                          (families.context == "defined_tcell_r1") & (families.cap == 8)].iloc[0]
    close(borderline.holm_p, 0.061193880611938804)
    upstream = json.loads((SEQUENCE / "verification" / "round2_independent.json").read_text())
    assert upstream["pass"] and not upstream["mismatches"] and upstream["comparisons"] == 620
    remote_log = (DELIVERABLES / "figures" / "v3" / "remote_sequence_independent_20261126.log").read_text().splitlines()
    remote = json.loads(next(line for line in reversed(remote_log) if line.startswith("{")))
    assert remote["pass"] and not remote["mismatches"] and remote["comparisons"] == 620
    assert remote["maximum_absolute_difference"] < 1e-12
    hash_lines = (DELIVERABLES / "figures" / "v3" / "remote_sequence_source_hashes_20261126.log").read_text().splitlines()
    remote_hashes = {line.split()[0] for line in hash_lines if line and not line.startswith("ssh:")}
    local_verifier_hash = hashlib.sha256((SEQUENCE / "source_snapshot" / "verify_independent.py").read_bytes()).hexdigest()
    assert local_verifier_hash in remote_hashes
    for path in (RESULTS / "primary_paired.tsv", RESULTS / "sequence_family_sensitivity.tsv"):
        assert hashlib.sha256(path.read_bytes()).hexdigest() in remote_hashes
    failed_local = DELIVERABLES / "figures" / "v3" / "verify_14_15_20261126_round1.log"
    assert failed_local.exists() and "-0.09272572668702395" in failed_local.read_text()
    return {
        "status": "PASS", "source_hashes": len(CONFIG["sources"]),
        "budget_pool_effects": 320, "budget_bootstrap_rows_recomputed": bootstrap_checks,
        "budget_holm_families": 3, "sequence_pool_effects": 320,
        "sequence_unadjusted_bootstrap_rows_recomputed": unadjusted_bootstrap_checks,
        "sequence_holm_families": 4, "family_holm_families": 2,
        "upstream_sequence_raw_independent_comparisons": upstream["comparisons"],
        "remote_px2_seed_level_independent_comparisons": remote["comparisons"],
        "remote_px2_maximum_absolute_difference": remote["maximum_absolute_difference"],
        "remote_local_verifier_and_result_hashes_match": True,
        "local_pool_mean_adjustment_limitation_retained": str(failed_local.name),
        "tcell_r1_all_cap2_16_holm": True, "tcell_r2_only_cap4_holm": True,
        "random_n_minus_random1_significant": 0, "adjusted_significant": 0,
        "no_uaug_significant": 0, "no_polyu_significant": 2,
        "sign_reversals_retained": True,
    }


def round_deterministic():
    paths = [OUT / f"{name}.{ext}" for name in CONFIG["figures"] for ext in ("png", "pdf", "svg")]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    subprocess.run([sys.executable, str(HERE / "build_v3_figures_14_15.py")], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    middle = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before == middle, [k for k in before if before[k] != middle[k]]
    time.sleep(1.1)
    env = dict(os.environ, PYTHONHASHSEED="987654321", TZ="UTC", LC_ALL="C")
    subprocess.run([sys.executable, str(HERE / "build_v3_figures_14_15.py")], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before == after, [k for k in before if before[k] != after[k]]
    return {"status": "PASS", "byte_identical_rebuilds": len(paths), "rebuild_rounds": 2,
            "python_hash_seeds": ["inherited", "987654321"],
            "environment_variation": {"TZ": "UTC", "LC_ALL": "C"}, "cross_second_delay": 1.1,
            "renderer_runtime": CONFIG["runtime"]}


def round_format():
    required = {
        "Figure14_v3_equal_budget_value": [
            "Equal-budget ranking value", "BoN(N) − Random(N)", "Random(N) − Random(1)",
            "4/4 caps > 1", "Only cap 4", "0/20 comparisons survive", "Not a new blind test",
        ],
        "Figure15_v3_sequence_composition_sensitivity": [
            "Sequence composition delimits", "0/20 adjusted comparisons survive Holm",
            "T-cell r2 cap 4 reverses sign", "+0.879", "uAUG exclusion: 0/20",
            "only T-cell r1 cap 4 and r2 cap 8 survive", "SENSITIVITY ≠ MEDIATION",
            "not causal mediation", "no new blind test",
        ],
    }
    forbidden = {
        "Figure14_v3_equal_budget_value": ["all four contexts", "new blind validation", "T-cell r2 all caps"],
        "Figure15_v3_sequence_composition_sensitivity": ["causally mediates", "causal proof", "independent of sequence composition"],
    }
    result = {}
    for name, phrases in required.items():
        png = OUT / f"{name}.png"
        with Image.open(png) as im:
            im.verify()
        with Image.open(png) as im:
            dpi = im.info.get("dpi", (0, 0))
            assert all(abs(v - 300) < 0.02 for v in dpi), dpi
            assert im.width >= 4000 and im.height >= 2400
            pixels = [im.width, im.height]
        doc = fitz.open(OUT / f"{name}.pdf")
        assert len(doc) == 1
        page = doc[0]
        text = page.get_text()
        for phrase in phrases:
            assert phrase in text, (name, phrase)
        for phrase in forbidden[name]:
            assert phrase not in text, (name, phrase)
        assert all(w[0] >= -0.5 and w[1] >= -0.5 and w[2] <= page.rect.width + 0.5 and
                   w[3] <= page.rect.height + 0.5 for w in page.get_text("words"))
        assert not page.get_images(), "PDF should remain vector"
        root = ET.parse(OUT / f"{name}.svg").getroot()
        tags = [element.tag.rsplit("}", 1)[-1] for element in root.iter()]
        assert "text" in tags and "image" not in tags
        result[name] = {"png_px": pixels, "png_dpi": [round(v, 3) for v in dpi], "pdf_pages": 1,
                        "pdf_vector": True, "svg_editable_text": True, "page_bounds": "PASS"}
    return {"status": "PASS", "figures": result}


report = {"round1_numeric_independent": round_numeric(),
          "round2_deterministic_cross_environment": round_deterministic(),
          "round3_format_render": round_format()}
report["overall"] = "PASS"
(OUT / "v3_figures_14_15_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
