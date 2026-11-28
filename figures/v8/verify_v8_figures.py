#!/usr/bin/env python3
"""Three-round verification of the V10 (v8) figure set."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
MAIN = ["Figure1_v8_equal_budget_value", "Figure2_v8_tail_bottleneck",
        "Figure3_v8_regression_decision_dissociation", "Figure4_v8_upts_falsification",
        "Figure5_v8_cross_scenario_forest", "Figure6_v8_unified_tail_diagnostic"]
ED = ["ED_Figure1_v8_label_distributions"]
ALL = MAIN + ED


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hashes():
    return {f"{b}.{s}": sha(HERE / f"{b}.{s}") for b in ALL for s in ("png", "pdf", "svg")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[1, 2, 3], required=True)
    args = ap.parse_args()
    checks = {}
    if args.round == 1:
        before = hashes()
        subprocess.run([PY, str(HERE / "build_v8_figures.py")], check=True, capture_output=True)
        subprocess.run([PY, str(HERE / "build_v8_extended.py")], check=True, capture_output=True)
        after = hashes()
        checks["deterministic_rebuild_byte_identical"] = before == after
        checks["all_formats_present"] = all((HERE / f"{b}.{s}").exists() for b in ALL for s in ("png", "pdf", "svg"))
        checks["figure_count"] = len(ALL) == 7
    elif args.round == 2:
        data = json.loads((HERE / "figures_v8_data.json").read_text())
        checks["tcell_r1_cap16_effect"] = abs(data["tcell_equal_budget"]["defined_tcell_r1"][-1]["effect"] - (-1.7057596554698837)) < 1e-9
        checks["association_rho_tail"] = abs(data["association"]["rho_tail_recall"] - 0.8787878787878788) < 1e-9
        checks["association_p_tail"] = data["association"]["permutation_p_tail"] < 0.001
        checks["upts_gate_failed"] = data["upts"]["all_directions_pass"] is False
        checks["anchor_harmful"] = data["anchor"]["combined"]["effect"] > 0
        checks["audit_rows_11"] = len(data["tail_audit_v2"]) == 11
    elif args.round == 3:
        for b in ALL:
            checks[f"{b}_pdf_nonempty"] = (HERE / f"{b}.pdf").stat().st_size > 5000
            checks[f"{b}_svg_has_text"] = "text" in (HERE / f"{b}.svg").read_text()[:200000]
        cfg = json.loads((HERE / "v8_build_config.json").read_text())
        checks["config_retrospective"] = cfg["retrospective"] is True
    status = "PASS" if all(v is True for v in checks.values() if isinstance(v, bool)) else "FAIL"
    out = {"round": args.round, "status": status, "checks": checks}
    (HERE / f"v8_verification_round{args.round}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"round": args.round, "status": status, "n_checks": len(checks)}))


if __name__ == "__main__":
    main()