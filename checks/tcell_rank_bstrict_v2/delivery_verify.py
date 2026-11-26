from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["artifact_root"])
    assert digest(config_path) == "16c94ae5caaf9bd166b9a61af9505a69ae776b6689838705e005192747b55afd"
    preflight = json.loads((root / "verification/round1_preflight.json").read_text())
    independent = json.loads((root / "verification/round2_independent.json").read_text())
    assert preflight["status"] == "PASS" and independent["status"] == "PASS"
    summary = json.loads((root / "auxiliary_gate.json").read_text())
    directions = {}
    for row in summary["directions"]:
        direction = row["direction"]
        base = root / "auxiliary" / direction
        fits = json.loads((base / "fits.json").read_text())
        selection = json.loads((base / "selection.json").read_text())
        gate = json.loads((base / "gate.json").read_text())
        assert len(fits) == 6 and len(selection["candidates"]) == 30
        assert all(fit["converged"] and fit["finite"] and fit["scientific_iteration_cap"] is None for fit in fits)
        assert all(fit["device"] == "cuda:0" and fit["dtype"] == "torch.float64" for fit in fits)
        assert selection["selection_role"] == "source_inner_dev_only" and not selection["destination_dev_used"]
        assert gate["test_evaluations"] == 1 and not gate["destination_dev_used"]
        assert gate["pass"] == (gate["corrected_minus_frozen"]["effect"] < 0 and gate["corrected_minus_frozen"]["ci_high"] < 0)
        directions[direction] = {
            "fits": len(fits), "candidates": len(selection["candidates"]),
            "selected_lambda": selection["chosen"]["lambda"], "selected_strength": selection["chosen"]["strength"],
            "gate_pass": gate["pass"], "effect": gate["corrected_minus_frozen"]["effect"],
            "ci_high": gate["corrected_minus_frozen"]["ci_high"],
        }
    if not summary["all_directions_pass"]:
        assert not (root / "auxiliary_gate_passed.json").exists()
        assert not (root / "target").exists()
        target_status = "NOT_RUN_AUXILIARY_GATE_FAILED"
        target_count = 0
    else:
        assert (root / "auxiliary_gate_passed.json").exists()
        complete = root / "target/evaluation_complete.json"
        assert complete.exists()
        target_status = "EVALUATED_ONCE"
        target_count = json.loads(complete.read_text())["evaluation_count"]
        assert target_count == 1
    code = {}
    workspace = Path(args.workspace)
    for path in sorted(workspace.rglob("*.py")):
        if "__pycache__" not in path.parts:
            code[str(path.relative_to(workspace))] = digest(path)
    result = {
        "round": 3, "status": "PASS", "config_sha256": digest(config_path),
        "round1_status": preflight["status"], "round2_status": independent["status"],
        "directions": directions, "target_status": target_status,
        "target_evaluation_count": target_count, "code_sha256": code,
        "v1_modified": False, "negative_results_retained": True,
    }
    path = root / "verification/round3_delivery.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
