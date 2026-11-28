#!/usr/bin/env python3
"""Three-round independent verification of the cross-scenario tail association."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_independent(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[1, 2, 3], required=True)
    args = ap.parse_args()
    rows = json.loads((HERE / "tail_audit_v2.json").read_text())["rows"]
    res = json.loads((HERE / "association_results.json").read_text())
    proto = json.loads((HERE / "association_protocol_frozen.json").read_text())
    inc = [r for r in rows if r["relative_regret_reduction_pct"] is not None]
    y = [r["relative_regret_reduction_pct"] for r in inc]
    tail = [r["tail_recall_decile"] for r in inc]
    glob = [r["global_spearman"] for r in inc]
    checks = {}

    rt = spearman_independent(tail, y)
    rg = spearman_independent(glob, y)
    checks["rho_tail_independent_matches"] = abs(rt - res["rho_tail_recall"]) < 1e-9
    checks["rho_global_independent_matches"] = abs(rg - res["rho_global_spearman"]) < 1e-9
    checks["n_scenarios"] = len(inc) == res["n_scenarios"] == 10
    checks["indel_excluded"] = "Castillo indel" in res["excluded"]
    checks["protocol_hash_matches"] = res["protocol_sha256"] == sha(HERE / "association_protocol_frozen.json")
    checks["data_hash_matches"] = res["data_sha256"] == sha(HERE / "tail_audit_v2.json")

    if args.round == 2:
        rng = np.random.default_rng(proto["primary_test"]["seed"] + 1)
        obs = rt
        count = 0
        nperm = 50000
        for _ in range(nperm):
            yp = rng.permutation(np.asarray(y, float))
            if spearman_independent(tail, yp) >= obs:
                count += 1
        p2 = (count + 1) / (nperm + 1)
        checks["permutation_p_independent_seed_lt_0.01"] = p2 < 0.01
        checks["permutation_p_independent_value"] = p2
        loo_min = min(l["rho_tail"] for l in res["leave_one_out"])
        checks["loo_robust_all_gt_0.8"] = loo_min > 0.8
    if args.round == 3:
        checks["success_rule_logic"] = res["success_rule_met"] == bool(
            res["rho_tail_recall"] > res["rho_global_spearman"]
            and res["permutation_p_tail"] < 0.05
            and res["difference_rho_tail_minus_global"]["ci95"][0] > 0)
        checks["honest_null_comparative"] = (not res["success_rule_met"]) and res["difference_rho_tail_minus_global"]["ci95"][0] <= 0
        checks["aav_top1pct_reproduces_v1"] = abs(next(r["top1pct_recall"] for r in rows if r["scenario"] == "AAV9 production") - 0.1875) < 1e-9

    status = "PASS" if all(v is True for k, v in checks.items() if isinstance(v, bool)) else "FAIL"
    out = {"round": args.round, "status": status, "checks": checks}
    (HERE / f"association_verification_round{args.round}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()