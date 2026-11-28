#!/usr/bin/env python3
"""Pre-registered cross-scenario tail-reliability association test (CPU).

Reads the frozen protocol and tail_audit_v2; computes Spearman associations,
scenario-level permutation tests, bootstrap difference CI, and sensitivity.
No frozen test is modified. Retrospective, associational.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PROTO = HERE / "association_protocol_frozen.json"
DATA = HERE / "tail_audit_v2.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rho(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    return float(spearmanr(x[m], y[m]).statistic), int(m.sum())


def perm_p(x, y, n_perm, seed):
    rng = np.random.default_rng(seed)
    obs, n = rho(x, y)
    if not np.isfinite(obs):
        return float("nan"), obs, n
    y = np.asarray(y, float)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r, _ = rho(x, yp)
        if np.isfinite(r) and r >= obs:
            count += 1
    return (count + 1) / (n_perm + 1), obs, n


def boot_diff(x_tail, x_glob, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        rt, _ = rho(np.asarray(x_tail)[ix], np.asarray(y)[ix])
        rg, _ = rho(np.asarray(x_glob)[ix], np.asarray(y)[ix])
        if np.isfinite(rt) and np.isfinite(rg):
            diffs.append(rt - rg)
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(np.mean(diffs))


def main():
    protocol = json.loads(PROTO.read_text())
    rows = json.loads(DATA.read_text())["rows"]
    inc = [r for r in rows if r["relative_regret_reduction_pct"] is not None]
    y = [r["relative_regret_reduction_pct"] for r in inc]
    tail = [r["tail_recall_decile"] for r in inc]
    glob = [r["global_spearman"] for r in inc]
    names = [r["scenario"] for r in inc]

    n_perm = protocol["primary_test"]["permutations"]
    p_tail, rho_tail, n = perm_p(tail, y, n_perm, protocol["primary_test"]["seed"])
    p_glob, rho_glob, _ = perm_p(glob, y, n_perm, protocol["primary_test"]["seed"])
    lo, hi, mean_diff = boot_diff(tail, glob, y, protocol["difference_ci"]["replicates"], protocol["difference_ci"]["seed"])

    success = bool(rho_tail > rho_glob and p_tail < 0.05 and (lo > 0 or hi < 0) and lo > 0)

    loo = []
    for i in range(len(inc)):
        yt = [y[j] for j in range(len(inc)) if j != i]
        tt = [tail[j] for j in range(len(inc)) if j != i]
        gg = [glob[j] for j in range(len(inc)) if j != i]
        rt, _ = rho(tt, yt); rg, _ = rho(gg, yt)
        loo.append({"dropped": names[i], "rho_tail": rt, "rho_global": rg})

    secondary = {}
    for key in ["tail_spearman_pred_decile", "tail_spearman_true_decile", "top1pct_recall"]:
        xs = [r[key] for r in inc]
        rr, nn = rho(xs, y)
        secondary[key] = {"rho": rr, "n": nn}

    pos = [r for r in inc if r["status"] == "positive"]
    bound = [r for r in inc if r["status"] in ("null", "harmful", "nominal", "condition-dependent")]
    subsets = {
        "positive_only": {"n": len(pos), "rho_tail": rho([r["tail_recall_decile"] for r in pos], [r["relative_regret_reduction_pct"] for r in pos])[0]},
        "boundary_only": {"n": len(bound), "rho_tail": rho([r["tail_recall_decile"] for r in bound], [r["relative_regret_reduction_pct"] for r in bound])[0]},
    }

    result = {
        "analysis": protocol["analysis"],
        "protocol_sha256": sha(PROTO),
        "data_sha256": sha(DATA),
        "n_scenarios": n,
        "included": names,
        "excluded": [r["scenario"] for r in rows if r["relative_regret_reduction_pct"] is None],
        "rho_tail_recall": rho_tail,
        "rho_global_spearman": rho_glob,
        "permutation_p_tail": p_tail,
        "permutation_p_global": p_glob,
        "difference_rho_tail_minus_global": {"mean": mean_diff, "ci95": [lo, hi]},
        "success_rule_met": success,
        "leave_one_out": loo,
        "secondary_predictors": secondary,
        "subsets_descriptive": subsets,
        "retrospective": True,
        "interpretation": (
            "Unified tail recall shows a stronger positive cross-scenario association with equal-budget decision benefit than global Spearman, supported by permutation and bootstrap."
            if success else
            "Success rule not met; tail reliability remains a descriptive diagnostic, not a statistically tested cross-scenario mechanism."
        ),
    }
    (HERE / "association_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["n_scenarios", "rho_tail_recall", "rho_global_spearman", "permutation_p_tail", "permutation_p_global", "difference_rho_tail_minus_global", "success_rule_met"]}, indent=2))
    print("secondary:", json.dumps(secondary))
    print("LOO rho_tail range:", min(l["rho_tail"] for l in loo), max(l["rho_tail"] for l in loo))


if __name__ == "__main__":
    main()