from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from upts import load_json, save_json, sha256


def independent_regret(p, y, std, pools, proposals, kappa):
    total = 0.0
    count = 0
    per_pool_seed = np.empty((len(pools), proposals.shape[1]), dtype=np.float64)
    for pi in range(len(pools)):
        pool = pools[pi]
        oracle = max(float(y[j]) for j in pool)
        for si in range(proposals.shape[1]):
            best_pos = None
            best_score = None
            for pos in proposals[pi, si]:
                j = int(pool[int(pos)])
                score = float(p[j]) - kappa * float(std[j])
                if best_score is None or score > best_score:
                    best_score = score
                    best_pos = j
            regret = oracle - float(y[best_pos])
            per_pool_seed[pi, si] = regret
            total += regret
            count += 1
    return total / count, per_pool_seed


def rebuild_proposals(destination, role, count, seeds, cap=16, pool_size=64):
    out = np.empty((count, len(seeds), cap), dtype=np.int64)
    for pi in range(count):
        for si, seed in enumerate(seeds):
            rng = random.Random(f"{destination}|{role}|{pi}|{seed}")
            out[pi, si] = [min(pool_size - 1, int(rng.random() * pool_size)) for _ in range(cap)]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    root = Path(config["artifact_root"])
    aux = Path(config["auxiliary"]["arrays_root"])
    seeds = config["auxiliary"]["seeds"]
    checks = []
    for d in config["auxiliary"]["directions"]:
        dest = d["destination"]
        direction = f"{d['source']}_to_{dest}"
        selection = load_json(root / "auxiliary" / direction / "selection.json")
        y = np.load(aux / dest / "labels.npy", allow_pickle=False)
        p = np.load(aux / dest / "cnn_predictions.npy", allow_pickle=False).astype(np.float64)
        members = np.load(aux / dest / "member_predictions.npy", allow_pickle=False)
        if members.shape[0] == 3:
            members = members.T
        std = members.astype(np.float64).std(axis=1, ddof=1)
        row = {"direction": direction}
        for role in ("dev", "test"):
            pools = np.load(aux / direction / f"{role}_pools.npy", allow_pickle=False)
            proposals = np.load(aux / direction / f"{role}_proposals.npy", allow_pickle=False)
            rebuilt = rebuild_proposals(dest, role, len(pools), seeds)
            assert np.array_equal(rebuilt, proposals), f"proposal rebuild mismatch {direction} {role}"
            kappa0, _ = independent_regret(p, y, std, pools, proposals, 0.0)
            base = selection["baseline"][role]["cap16_mean_regret"]
            assert abs(kappa0 - base) < 1e-9, (kappa0, base)
            row[f"{role}_kappa0_max_abs_diff"] = abs(kappa0 - base)
            if role == "test":
                kappa = selection["chosen"]["kappa"]
                mean, _ = independent_regret(p, y, std, pools, proposals, kappa)
                reported = selection["test"]["cap16_mean_regret"]
                assert abs(mean - reported) < 1e-9, (mean, reported)
                row["test_chosen_kappa"] = kappa
                row["test_chosen_max_abs_diff"] = abs(mean - reported)
                row["test_gate_pass_recomputed"] = bool(mean < base)
                assert row["test_gate_pass_recomputed"] == selection["gate_pass"]
        checks.append(row)
    try:
        from scipy.stats import spearmanr
        diag_checks = []
        for d in config["auxiliary"]["directions"]:
            dest = d["destination"]
            direction = f"{d['source']}_to_{dest}"
            selection = load_json(root / "auxiliary" / direction / "selection.json")
            y = np.load(aux / dest / "labels.npy", allow_pickle=False).astype(np.float64)
            p = np.load(aux / dest / "cnn_predictions.npy", allow_pickle=False).astype(np.float64)
            members = np.load(aux / dest / "member_predictions.npy", allow_pickle=False)
            if members.shape[0] == 3:
                members = members.T
            std = members.astype(np.float64).std(axis=1, ddof=1)
            pools = np.load(aux / direction / "dev_pools.npy", allow_pickle=False)
            ss, ee = [], []
            for pool in pools:
                order = sorted(range(len(pool)), key=lambda i: (-p[pool[i]], i))[:16]
                for i in order:
                    j = pool[i]
                    ss.append(std[j])
                    ee.append(p[j] - y[j])
            ss = np.asarray(ss)
            ee = np.asarray(ee)
            r_signed = float(spearmanr(ss, ee).statistic)
            r_abs = float(spearmanr(ss, np.abs(ee)).statistic)
            ref = selection["diagnostics"]
            assert abs(r_signed - ref["spearman_std_vs_signed_error"]) < 1e-9
            assert abs(r_abs - ref["spearman_std_vs_abs_error"]) < 1e-9
            diag_checks.append({"direction": direction, "spearman_signed": r_signed, "spearman_abs": r_abs, "match": True})
        diagnostics_scipy = True
    except ImportError:
        diag_checks = []
        diagnostics_scipy = False
    report = {
        "round": 2,
        "status": "PASS",
        "method": "independent pure-python loop regret + proposal rebuild from frozen formula; scipy spearman if available",
        "checks": checks,
        "diagnostics_scipy_available": diagnostics_scipy,
        "diagnostics_checks": diag_checks,
        "selection_sha256": {
            row["direction"]: sha256(root / "auxiliary" / row["direction"] / "selection.json") for row in checks
        },
    }
    save_json(root / "verification" / "round2_independent.json", report)
    print({"round": 2, "status": "PASS", "directions": len(checks), "scipy": diagnostics_scipy})


if __name__ == "__main__":
    main()