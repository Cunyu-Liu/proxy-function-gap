from __future__ import annotations

import json
from pathlib import Path

from paper2_risk import sha256_file, write_json

TEAMS = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap")
OUT = TEAMS / "paper2_risk_frozen_v1"
SOURCE = Path("/home/proxy-function-gap-paper2/src/proxy_function_gap/paper2_baselines.py")


def main():
    source = SOURCE.read_text()
    required = [
        'risk = float(std[selected] if method != "distance" else current_distance[selected])',
        'ordered = sorted(current, key=lambda row: row["risk"])',
        'for coverage in np.linspace(0.1, 1.0, 10):',
        '"threshold_source": "retrospective empirical coverage; not a deployment threshold"',
    ]
    if any(fragment not in source for fragment in required):
        raise ValueError("existing implementation no longer matches the audited source")
    frozen = json.loads((TEAMS / "paper2_baselines_v1/dev_frozen.json").read_text())
    summaries = {}
    for role in ("outer_designed", "outer_random"):
        path = TEAMS / "paper2_baselines_v1" / f"{role}_summary.json"
        payload = json.loads(path.read_text())
        bon = [row for row in payload["risk_coverage"] if row["method"] == "bon"]
        expected_decisions = 5 * payload["pools"]
        if bon[-1]["accepted"] != expected_decisions:
            raise ValueError("unexpected old risk-coverage denominator")
        summaries[role] = {
            "source": {"path": str(path), "sha256": sha256_file(path)},
            "pools": payload["pools"],
            "seeds_per_pool": 5,
            "full_coverage_rows": expected_decisions,
            "bon_rows": bon,
            "evidence_status": payload["evidence_status"],
        }
    result = {
        "schema_version": "paper2-risk-frozen-v1-audit",
        "source_code": {"path": str(SOURCE), "sha256": sha256_file(SOURCE)},
        "unit_finding": "Selected-decision level: one BoN top-1 selected candidate for each pool x seed. It is not pool-level aggregation and does not filter every candidate within a pool.",
        "risk_finding": "For BoN, risk is the ensemble standard deviation at selected_index.",
        "aggregation_finding": "Pool and seed rows are flattened before global sorting; there is no per-pool risk summary or per-pool aggregation before the curve.",
        "threshold_finding": "Each empirical coverage uses order statistics from the same evaluated role. No numeric deployable threshold is frozen or stored.",
        "outer_finding": "The old outer curves therefore used outer empirical quantiles. The dev_frozen outer_used_for_selection=false statement applies to baseline lambda/gamma selection, not to these risk-coverage quantiles.",
        "old_outer_use": "Retrospective sensitivity only; inadmissible as an untouched blind deployment-threshold validation.",
        "baseline_hyperparameters": {
            "outer_used_for_selection": frozen["outer_used_for_selection"],
            "calibration_used_for_selection": frozen["calibration_used_for_selection"],
            "selection_rule": frozen["selection_rule"],
        },
        "summaries": summaries,
        "new_protocol_correction": "Select and freeze one numeric threshold on dev, gate once on calibration, then apply unchanged to outer pool x seed decisions.",
    }
    write_json(OUT / "existing_implementation_audit.json", result)
    print(json.dumps({key: result[key] for key in ("unit_finding", "threshold_finding", "outer_finding")}, indent=2))


if __name__ == "__main__":
    main()
