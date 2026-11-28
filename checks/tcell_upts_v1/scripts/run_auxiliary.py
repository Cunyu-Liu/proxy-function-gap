from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from upts import cap16_regret, load_json, member_std, save_json, sha256, spearman, top16_by_score, upts_score


def verify_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"hash mismatch for {path}: {actual} != {expected}")


def run_direction(d, config, hashes, out_root):
    source, dest = d["source"], d["destination"]
    direction = f"{source}_to_{dest}"
    aux = Path(config["auxiliary"]["arrays_root"])
    ctx_dir = aux / dest
    dir_dir = aux / direction

    files = {name: ctx_dir / name for name in ("labels.npy", "cnn_predictions.npy", "member_predictions.npy", "split.npy")}
    files.update({name: dir_dir / name for name in ("dev_pools.npy", "dev_proposals.npy", "test_pools.npy", "test_proposals.npy", "selection.json")})
    for name, path in files.items():
        verify_hash(path, hashes[str(path)])

    y = np.load(files["labels.npy"], allow_pickle=False)
    p = np.load(files["cnn_predictions.npy"], allow_pickle=False)
    members = np.load(files["member_predictions.npy"], allow_pickle=False)
    split = np.load(files["split.npy"], allow_pickle=False)
    if members.ndim == 2 and members.shape[0] == 3 and members.shape[1] == len(p):
        members = members.T
    if members.ndim != 2 or members.shape[1] != 3 or members.shape[0] != len(p):
        raise SystemExit(f"unexpected member_predictions shape {members.shape}")
    mean_diff = float(np.max(np.abs(members.astype(np.float64).mean(axis=1) - p.astype(np.float64))))
    if mean_diff > config["auxiliary"]["member_mean_check_max_abs_diff"]:
        raise SystemExit(f"member mean differs from frozen cnn_predictions: {mean_diff}")
    std = member_std(members)
    if not np.all(np.isfinite(std)) or np.any(std < 0):
        raise SystemExit("nonfinite or negative member std")

    residual_selection = load_json(files["selection.json"])
    pools = {role: np.load(dir_dir / f"{role}_pools.npy", allow_pickle=False) for role in ("dev", "test")}
    proposals = {role: np.load(dir_dir / f"{role}_proposals.npy", allow_pickle=False) for role in ("dev", "test")}

    p64 = p.astype(np.float64)
    baseline = {}
    for role in ("dev", "test"):
        result = cap16_regret(p64, y, pools[role], proposals[role])
        expected = float(residual_selection["baseline"][role]["cap16_mean_regret"])
        if result["mean_regret"] != expected:
            raise SystemExit(f"kappa=0 identity failed for {direction} {role}: {result['mean_regret']} != {expected}")
        baseline[role] = result["mean_regret"]

    candidates = []
    for kappa in config["method"]["kappa_grid"]:
        score = upts_score(p64, std, kappa)
        dev = cap16_regret(score, y, pools["dev"], proposals["dev"])
        candidates.append({"kappa": kappa, "dev_cap16_mean_regret": dev["mean_regret"]})
    tolerance = config["auxiliary"]["tie_tolerance"]
    best = min(row["dev_cap16_mean_regret"] for row in candidates)
    tier = [row for row in candidates if abs(row["dev_cap16_mean_regret"] - best) <= tolerance]
    chosen = sorted(tier, key=lambda row: row["kappa"])[0]

    chosen_score = upts_score(p64, std, chosen["kappa"])
    test = cap16_regret(chosen_score, y, pools["test"], proposals["test"])
    improvement = baseline["test"] - test["mean_regret"]
    gate_pass = bool(test["mean_regret"] < baseline["test"])

    diag_std, diag_err = [], []
    for pool in pools["dev"]:
        idx = top16_by_score(p64[pool])
        sel = pool[idx]
        diag_std.append(std[sel])
        diag_err.append(p64[sel] - y[sel].astype(np.float64))
    diag_std = np.concatenate(diag_std)
    diag_err = np.concatenate(diag_err)
    diagnostics = {
        "surface": "destination dev pools, per-pool top16 by frozen proxy_mean, index tie-break",
        "n": int(len(diag_std)),
        "spearman_std_vs_signed_error": spearman(diag_std, diag_err),
        "spearman_std_vs_abs_error": spearman(diag_std, np.abs(diag_err)),
        "mean_std_top16": float(diag_std.mean()),
        "used_for_selection": False,
    }

    selection = {
        "direction": direction,
        "source": source,
        "destination": dest,
        "member_mean_max_abs_diff": mean_diff,
        "kappa0_identity": {"dev": baseline["dev"], "test": baseline["test"], "exact_match_residual_v1": True},
        "baseline": {"dev": {"cap16_mean_regret": baseline["dev"]}, "test": {"cap16_mean_regret": baseline["test"]}},
        "candidates": candidates,
        "tie_rule": "smaller_kappa within tie_tolerance 1e-12",
        "chosen": chosen,
        "test": {
            "cap16_mean_regret": test["mean_regret"],
            "cap16_regret_improvement": improvement,
        },
        "gate_pass": gate_pass,
        "diagnostics": diagnostics,
        "target_mrl_read": False,
        "pools_sha256": {role: hashes[str(dir_dir / f"{role}_pools.npy")] for role in pools},
        "proposals_sha256": {role: hashes[str(dir_dir / f"{role}_proposals.npy")] for role in proposals},
    }
    out = out_root / "auxiliary" / direction
    save_json(out / "selection.json", selection)
    save_json(out / "gate.json", {
        "direction": direction,
        "pass": gate_pass,
        "reason": "PASS" if gate_pass else "destination test cap16 regret did not strictly improve",
        "target_evaluation_permitted": False,
    })
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    out_root = Path(config["artifact_root"])
    hashes = load_json(out_root / "input_hashes.json")
    results = [run_direction(d, config, hashes, out_root) for d in config["auxiliary"]["directions"]]
    passed = all(row["gate_pass"] for row in results)
    summary = {
        "all_directions_pass": passed,
        "directions": [
            {"direction": row["direction"], "gate_pass": row["gate_pass"], "chosen": row["chosen"], "test": row["test"], "diagnostics": row["diagnostics"]}
            for row in results
        ],
        "target_mrl_read": False,
        "target_evaluation_permitted": False,
        "compute": "CPU-only; no training/inference",
    }
    save_json(out_root / "auxiliary_gate.json", summary)
    print(summary)
    if not passed:
        raise SystemExit(2)
    save_json(out_root / "auxiliary_gate_passed.json", {
        "pass": True,
        "selection_sha256": {row["direction"]: sha256(out_root / "auxiliary" / row["direction"] / "selection.json") for row in results},
    })


if __name__ == "__main__":
    main()