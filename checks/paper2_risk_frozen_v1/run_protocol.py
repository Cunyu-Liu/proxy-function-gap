from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from paper2_risk import (
    BOOTSTRAP_SEED,
    MIN_ACCEPTED_DECISIONS,
    MIN_ACCEPTED_POOLS,
    N_BOOT,
    QUANTILES,
    SEEDS,
    calibration_gate,
    finite_row,
    select_quantile,
    sha256_file,
    summarize,
    write_json,
)

TEAMS = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap")
OUT = TEAMS / "paper2_risk_frozen_v1"
POOLS = TEAMS / "pool_validation/production_v3"
BASELINES = TEAMS / "paper2_baselines_v1"
SOURCE_POOL_MANIFEST = TEAMS / "data/derived/fresh_proxy_fixed_pools_v2/build_a/pool_manifest.json"
COUNTS = {"dev": 32, "calibration": 16, "outer_designed": 16, "outer_random": 3}
FIELDS = [
    "role", "pool_id", "seed", "selected_index", "selected_id", "risk", "proxy_mean",
    "selected_measured_rl", "pool_best_measured_rl", "regret", "source_file", "source_sha256",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in FIELDS} for row in rows])


def read_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["selected_index"] = int(row["selected_index"])
        for key in ("risk", "proxy_mean", "selected_measured_rl", "pool_best_measured_rl", "regret"):
            row[key] = float(row[key])
        finite_row(row)
    return rows


def evidence(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def verify_evidence(item):
    path = Path(item["path"]).resolve()
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"source hash mismatch: {path}")
    return path


def tsv_rows(path):
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def learner_decisions(role):
    if role not in ("dev", "calibration"):
        raise PermissionError("learner extraction is restricted to dev/calibration")
    result = []
    sources = []
    for index in range(COUNTS[role]):
        pool_id = f"{role}_{index:03d}"
        manifest_path = POOLS / pool_id / "pool_manifest.json"
        manifest = read_json(manifest_path)
        if manifest["pool_id"] != pool_id or manifest["pool_role"] != role or manifest["synthetic"] is not False:
            raise ValueError(f"invalid learner pool identity: {pool_id}")
        candidate_path = verify_evidence(manifest["candidate_file"])
        label_path = verify_evidence(manifest["label_source"])
        prediction_manifest = verify_evidence(manifest["prediction_manifest"])
        prediction = read_json(prediction_manifest)
        if prediction["prediction_units"] != "measured_rl" or prediction["inverse_target_transform"] is not True:
            raise ValueError("predictions are not in measured-MRL units")
        if prediction["outer_labels_read"] or prediction["outer_metrics_computed"]:
            raise PermissionError("outer-contaminated prediction manifest")
        candidates = tsv_rows(candidate_path)
        if len(candidates) != manifest["candidate_count"]:
            raise ValueError("candidate count mismatch")
        labels = {row["row_id"]: float(row["measured_rl"]) for row in tsv_rows(label_path)}
        measured = np.asarray([labels[row["row_id"]] for row in candidates], dtype=np.float64)
        means = np.asarray([float(row["proxy_mean"]) for row in candidates], dtype=np.float64)
        stds = np.asarray([float(row["proxy_std"]) for row in candidates], dtype=np.float64)
        proposal = np.asarray(manifest["proposal_probs"], dtype=np.float64)
        best = float(measured.max())
        comparison = learner_comparison(role, pool_id) if role == "dev" else None
        for seed in SEEDS:
            proposals = np.random.default_rng(seed).choice(len(candidates), size=16, replace=True, p=proposal).tolist()
            selected = sorted(set(proposals), key=lambda item: (-means[item], item))[0]
            row = {
                "role": role,
                "pool_id": pool_id,
                "seed": seed,
                "selected_index": selected,
                "selected_id": candidates[selected]["sequence_id"],
                "risk": float(stds[selected]),
                "proxy_mean": float(means[selected]),
                "selected_measured_rl": float(measured[selected]),
                "pool_best_measured_rl": best,
                "regret": float(best - measured[selected]),
                "source_file": str(manifest_path),
                "source_sha256": sha256_file(manifest_path),
            }
            finite_row(row)
            if comparison is not None:
                compare_learner_record(row, proposals, comparison[seed])
            result.append(row)
        sources.extend([evidence(manifest_path), evidence(candidate_path), evidence(label_path)])
    return result, sources


def learner_comparison(role, pool_id):
    if role == "dev":
        payload = read_json(BASELINES / "dev/evaluation" / f"{pool_id}.json")
        rows = [row for row in payload["records"] if row["method"] == "bon" and row["proposal_cap"] == 16 and row["permutation_id"] == 0 and row["seed"] in SEEDS]
    else:
        payload = read_json(TEAMS / "selection/calibration_v3" / f"{pool_id}.json")
        rows = [row for row in payload["records"] if row["strategy"] == "bon" and row["alpha"] == 0 and row["proposal_cap"] == 16 and row["seed"] in SEEDS]
    if len(rows) != len(SEEDS):
        raise ValueError(f"comparison record count mismatch: {pool_id}")
    return {row["seed"]: row for row in rows}


def compare_learner_record(row, proposals, other):
    if row["selected_id"] != other["selected_id"] or row["selected_index"] != other["selected_index"]:
        raise ValueError(f"selected candidate mismatch: {row['pool_id']}/{row['seed']}")
    if not np.isclose(row["regret"], other["regret"], rtol=0, atol=1e-12):
        raise ValueError(f"regret mismatch: {row['pool_id']}/{row['seed']}")
    if "risk" in other and not np.isclose(row["risk"], other["risk"], rtol=0, atol=1e-6):
        raise ValueError(f"risk mismatch: {row['pool_id']}/{row['seed']}")
    if isinstance(other.get("proposals"), list) and proposals != other["proposals"]:
        raise ValueError(f"proposal mismatch: {row['pool_id']}/{row['seed']}")


def outer_decisions(role):
    if role not in ("outer_designed", "outer_random"):
        raise PermissionError("unknown outer role")
    result = []
    sources = []
    for index in range(COUNTS[role]):
        pool_id = f"{role}_{index:03d}"
        evaluation_path = BASELINES / role / "evaluation" / f"{pool_id}.json"
        selection_path = BASELINES / role / "selection" / f"{pool_id}.json"
        evaluation = read_json(evaluation_path)
        selection = read_json(selection_path)
        if evaluation["selection_sha256"] != sha256_file(selection_path) or not evaluation["evaluator_labels_used"]:
            raise ValueError(f"outer evaluation lineage mismatch: {pool_id}")
        selected_records = [row for row in selection["records"] if row["method"] == "bon" and row["proposal_cap"] == 16 and row["permutation_id"] == 0 and row["seed"] in SEEDS]
        evaluated_records = [row for row in evaluation["records"] if row["method"] == "bon" and row["proposal_cap"] == 16 and row["permutation_id"] == 0 and row["seed"] in SEEDS]
        if len(selected_records) != 5 or len(evaluated_records) != 5:
            raise ValueError(f"outer decision count mismatch: {pool_id}")
        for selected, evaluated in zip(selected_records, evaluated_records):
            if any(selected[key] != evaluated[key] for key in ("pool_id", "seed", "selected_id", "selected_index", "risk")):
                raise ValueError(f"outer selection/evaluation mismatch: {pool_id}")
            row = {
                "role": role,
                "pool_id": pool_id,
                "seed": evaluated["seed"],
                "selected_index": evaluated["selected_index"],
                "selected_id": evaluated["selected_id"],
                "risk": evaluated["risk"],
                "proxy_mean": None,
                "selected_measured_rl": evaluated["selected_measured_rl"],
                "pool_best_measured_rl": evaluated["selected_measured_rl"] + evaluated["regret"],
                "regret": evaluated["regret"],
                "source_file": str(evaluation_path),
                "source_sha256": sha256_file(evaluation_path),
            }
            result.append(row)
        sources.extend([evidence(evaluation_path), evidence(selection_path)])
    return result, sources


def initialize(code_path):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "preregistration.json"
    if path.exists():
        raise FileExistsError(path)
    protocol = {
        "schema_version": "paper2-risk-frozen-v1",
        "created_utc": now(),
        "compute": "CPU-only statistics on frozen arrays; no training, model inference or CUDA validation",
        "decision_unit": "One BoN cap16 top-1 selected candidate per pool and paired seed; not pool-level risk and not within-pool all-candidate filtering",
        "risk": "ensemble_std of the BoN-selected candidate; accept when risk <= threshold, otherwise abstain without replacement",
        "roles": {"development": "select one quantile", "calibration": "single gate evaluation", "outer_designed": "primary one-time frozen scoring", "outer_random": "secondary one-time frozen scoring in the same invocation"},
        "seeds": list(SEEDS),
        "quantile_grid": list(QUANTILES),
        "quantile_definition": "numpy linear quantile of development selected-decision risks only",
        "development_selection_rule": "maximum full-BoN mean regret minus accepted mean regret; ties by higher development coverage, then lower quantile",
        "primary_endpoint": "full-BoN mean regret minus accepted mean regret; positive favors ensemble-std abstention",
        "uncertainty": {"method": "pool-cluster percentile bootstrap", "rounds": N_BOOT, "seed": BOOTSTRAP_SEED, "interval": [2.5, 97.5]},
        "calibration_gate": {"minimum_accepted_decisions": MIN_ACCEPTED_DECISIONS, "minimum_accepted_pools": MIN_ACCEPTED_POOLS, "point_reduction": "> 0", "reduction_95ci_lower": "> 0"},
        "outer_policy": "Score only if the calibration gate passes; score both outer roles once; never revise threshold or endpoints afterward",
        "historical_outer_boundary": "Outer labels and empirical risk-coverage curves were previously viewed. Any new outer score is retrospective frozen-threshold confirmation, not an untouched blind test.",
        "code_sha256": sha256_file(code_path),
        "outer_labels_read": False,
        "outer_metrics_computed": False,
    }
    write_json(path, protocol)
    print(json.dumps({"initialized": str(path), "sha256": sha256_file(path)}, indent=2))


def prepare():
    protocol = read_json(OUT / "preregistration.json")
    if protocol["outer_metrics_computed"]:
        raise PermissionError("invalid preregistration")
    manifest = {"roles": {}}
    for role in ("dev", "calibration"):
        rows, sources = learner_decisions(role)
        path = OUT / f"{role}_decisions.csv"
        write_csv(path, rows)
        manifest["roles"][role] = {"decisions": len(rows), "pools": COUNTS[role], "file": evidence(path), "sources": sources}
    manifest["outer_labels_read"] = False
    manifest["outer_metrics_computed"] = False
    write_json(OUT / "learner_source_manifest.json", manifest)
    print(json.dumps({role: value["decisions"] for role, value in manifest["roles"].items()}, indent=2))


def freeze():
    protocol = read_json(OUT / "preregistration.json")
    dev = read_csv(OUT / "dev_decisions.csv")
    calibration = read_csv(OUT / "calibration_decisions.csv")
    selected, grid = select_quantile(dev)
    calibration_summary = summarize(calibration, selected["threshold"])
    gate = calibration_gate(calibration_summary)
    frozen = {
        "schema_version": "paper2-risk-frozen-v1",
        "frozen_utc": now(),
        "preregistration_sha256": sha256_file(OUT / "preregistration.json"),
        "decision_source_manifest_sha256": sha256_file(OUT / "learner_source_manifest.json"),
        "implementation": {
            "run_protocol_sha256": sha256_file(Path(__file__)),
            "paper2_risk_sha256": sha256_file(Path(__file__).with_name("paper2_risk.py")),
            "prefreeze_correction_sha256": sha256_file(OUT / "prefreeze_implementation_correction.json"),
        },
        "selected_quantile": selected["quantile"],
        "selected_threshold": selected["threshold"],
        "selection_rule": protocol["development_selection_rule"],
        "development_grid": grid,
        "calibration": calibration_summary,
        "calibration_gate": gate,
        "outer_scoring_authorized": gate["pass"],
        "outer_scored": False,
        "threshold_locked": True,
        "primary_endpoint": protocol["primary_endpoint"],
        "evidence_boundary": protocol["historical_outer_boundary"],
    }
    write_json(OUT / "frozen_protocol.json", frozen)
    print(json.dumps({"selected_quantile": selected["quantile"], "threshold": selected["threshold"], "calibration": calibration_summary, "gate": gate}, indent=2))


def score_outer():
    output = OUT / "outer_result.json"
    if output.exists():
        raise FileExistsError("outer was already scored; repeat scoring is forbidden")
    frozen = read_json(OUT / "frozen_protocol.json")
    if not frozen["threshold_locked"] or not frozen["outer_scoring_authorized"] or not frozen["calibration_gate"]["pass"]:
        raise PermissionError("Paper2 calibration gate has not authorized outer scoring")
    result = {
        "schema_version": "paper2-risk-frozen-v1",
        "scored_utc": now(),
        "frozen_protocol_sha256": sha256_file(OUT / "frozen_protocol.json"),
        "selected_quantile": frozen["selected_quantile"],
        "selected_threshold": frozen["selected_threshold"],
        "primary_role": "outer_designed",
        "roles": {},
        "evidence_status": "retrospective_previously_viewed_outer_frozen_threshold_not_unseen_blind_test",
        "threshold_locked_after_scoring": True,
        "new_training_or_inference": False,
        "compute": "CPU-only scoring of frozen evaluated decisions",
    }
    for role in ("outer_designed", "outer_random"):
        rows, sources = outer_decisions(role)
        path = OUT / f"{role}_decisions.csv"
        write_csv(path, rows)
        summary = summarize(rows, frozen["selected_threshold"])
        summary["success"] = summary["regret_reduction"] is not None and summary["regret_reduction"] > 0 and summary["regret_reduction_95ci"][0] > 0
        result["roles"][role] = {"summary": summary, "decision_file": evidence(path), "sources": sources}
    write_json(output, result)
    print(json.dumps({role: value["summary"] for role, value in result["roles"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("initialize", "prepare", "freeze", "score-outer"))
    parser.add_argument("--code-path", type=Path)
    args = parser.parse_args()
    if args.command == "initialize":
        if args.code_path is None:
            raise ValueError("initialize requires --code-path")
        initialize(args.code_path)
    elif args.command == "prepare":
        prepare()
    elif args.command == "freeze":
        freeze()
    else:
        score_outer()


if __name__ == "__main__":
    main()
