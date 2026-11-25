from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper2_risk import sha256_file, write_json
from run_protocol import OUT, learner_comparison, learner_decisions, read_csv


def close(left, right):
    if left is None or right is None:
        return left == right
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(left, right, rtol=0, atol=1e-12))
    return left == right


def main():
    report = {"schema_version": "paper2-risk-frozen-v1-verification-round1", "pass": True, "roles": {}}
    for role in ("dev", "calibration"):
        expected, sources = learner_decisions(role)
        actual = read_csv(OUT / f"{role}_decisions.csv")
        if len(expected) != len(actual):
            raise ValueError("decision count differs")
        mismatches = []
        for left, right in zip(expected, actual):
            for key in left:
                if key == "source_sha256":
                    continue
                if not close(left[key], right[key]):
                    mismatches.append({"pool_id": left["pool_id"], "seed": left["seed"], "field": key})
        if mismatches:
            raise ValueError(mismatches[:10])
        risks = np.asarray([row["risk"] for row in actual])
        regrets = np.asarray([row["regret"] for row in actual])
        role_report = {
            "decisions": len(actual),
            "pools": len(set(row["pool_id"] for row in actual)),
            "raw_field_mismatches": len(mismatches),
            "risk_sum": float(risks.sum()),
            "regret_sum": float(regrets.sum()),
            "regret_mean": float(regrets.mean()),
            "source_files": len(sources),
        }
        if role == "calibration":
            mismatch_count = 0
            for row in actual:
                historical = learner_comparison(role, row["pool_id"])[row["seed"]]
                mismatch_count += historical["selected_index"] != row["selected_index"]
            if mismatch_count != 66:
                raise ValueError("historical calibration proposal-generator audit changed")
            role_report["historical_random_random_selection_mismatches"] = mismatch_count
            role_report["historical_selection_used_in_primary"] = False
        report["roles"][role] = role_report
    frozen = json.loads((OUT / "frozen_protocol.json").read_text())
    dev = read_csv(OUT / "dev_decisions.csv")
    threshold = float(np.quantile([row["risk"] for row in dev], 0.5, method="linear"))
    if not close(threshold, frozen["selected_threshold"]):
        raise ValueError("frozen threshold differs from raw dev quantile")
    if frozen["outer_scoring_authorized"] or (OUT / "outer_result.json").exists():
        raise ValueError("outer boundary violated")
    report["frozen_threshold_raw_recomputation"] = threshold
    report["outer_scored"] = False
    write_json(OUT / "verification_round1_raw.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
