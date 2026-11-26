from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def features(sequences, p):
    b = np.asarray(sequences, dtype="S50").view(np.uint8).reshape(len(sequences), 50)
    gc = ((b == 67) | (b == 71)).mean(axis=1, dtype=np.float64)
    aug = np.zeros(len(b), dtype=np.float64)
    poly = np.zeros(len(b), dtype=np.float64)
    run = np.zeros(len(b), dtype=np.float64)
    for k in range(50):
        run = np.where(b[:, k] == 84, run + 1, 0)
        poly = np.maximum(poly, run)
        if k < 48:
            aug = np.maximum(aug, ((b[:, k] == 65) & (b[:, k + 1] == 84) & (b[:, k + 2] == 71)).astype(np.float64))
    return np.column_stack((p, p * p, gc, aug, poly))


def proposal_regret(score, y, pools, proposals):
    py = y[pools]
    labels = np.take_along_axis(py[:, None, :], proposals, axis=2)
    scores = np.take_along_axis(score[pools][:, None, :], proposals, axis=2)
    choice = np.argmax(scores, axis=2)
    chosen = np.take_along_axis(labels, choice[..., None], axis=2)[..., 0]
    return py.max(axis=1)[:, None] - chosen, choice


def pair_parts(y, p, z, tail, start, stop):
    a, b = np.triu_indices(16, 1)
    left = tail[start:stop, a]
    right = tail[start:stop, b]
    dy = y[left] - y[right]
    valid = dy != 0
    left = left[valid]
    right = right[valid]
    pools = np.broadcast_to(np.arange(start, stop)[:, None], (stop - start, len(a)))[valid]
    return left, right, np.sign(dy[valid]), pools


def independent_fit_report(y, p, z, tail, fits):
    a, b = np.triu_indices(16, 1)
    dy_all = y[tail[:, a]] - y[tail[:, b]]
    counts = np.sum(dy_all != 0, axis=1)
    results = []
    for fit in fits:
        beta = np.asarray(fit["coefficients_nonintercept"], dtype=np.float64)
        lam = fit["lambda"]
        loss = lam * np.dot(beta, beta)
        gradient = 2 * lam * beta
        for start in range(0, len(tail), 256):
            stop = min(len(tail), start + 256)
            left, right, label, pool = pair_parts(y, p, z, tail, start, stop)
            weight = 1.0 / len(tail) / counts[pool]
            dz = z[left] - z[right]
            margin = label * (p[left] - p[right] + 1.6689551812138181 * (dz @ beta))
            loss += float(np.sum(weight * np.logaddexp(0, -margin)))
            probability = np.exp(-np.logaddexp(0, margin))
            gradient += dz.T @ (weight * (-label * probability) * 1.6689551812138181)
        results.append({
            "lambda": lam, "loss_absolute_difference": float(abs(loss - fit["loss"])),
            "gradient_inf_independent": float(np.max(np.abs(gradient))),
            "reported_gradient_inf": fit["gradient_inf"],
            "converged": fit["converged"], "finite": fit["finite"],
            "loss_pass": bool(abs(loss - fit["loss"]) < 1e-9),
            "gradient_pass": bool(np.max(np.abs(gradient)) < 1e-5),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    v1 = Path(config["v1_input_root"])
    root = Path(config["artifact_root"])
    directions = (("n50_hek293t_r1", "defined_hepg2_r1"), ("defined_hepg2_r1", "n50_hek293t_r1"))
    output_directions = {}
    all_pass = True
    for source, destination in directions:
        name = f"{source}_to_{destination}"
        base = root / "auxiliary" / name
        seq = np.load(v1 / "auxiliary" / source / "sequences.npy", allow_pickle=False)
        y = np.load(v1 / "auxiliary" / source / "labels.npy", allow_pickle=False).astype(np.float64)
        p = np.load(v1 / "auxiliary" / source / "cnn_predictions.npy", allow_pickle=False).astype(np.float64)
        split = np.load(v1 / "auxiliary" / source / "split.npy", allow_pickle=False)
        selection = json.loads((base / "selection.json").read_text())
        x = features(seq, p)
        mean = x[split == 0].mean(axis=0)
        sd = x[split == 0].std(axis=0, ddof=0)
        z = (x - mean) / sd
        assert np.max(np.abs(mean - selection["feature_mean"])) == 0
        assert np.max(np.abs(sd - selection["feature_population_sd"])) == 0
        train_tail = np.load(base / "source_train_tail.npy", allow_pickle=False)
        expected_tail = np.take_along_axis(np.load(base / "source_train_pools.npy", allow_pickle=False), np.argsort(-p[np.load(base / "source_train_pools.npy", allow_pickle=False)], axis=1, kind="stable")[:, :16], axis=1)
        assert np.array_equal(train_tail, expected_tail)
        fit_checks = independent_fit_report(y, p, z, train_tail, json.loads((base / "fits.json").read_text()))
        dev_pools = np.load(base / "source_dev_pools.npy", allow_pickle=False)
        dev_proposals = np.load(base / "source_dev_proposals.npy", allow_pickle=False)
        candidate_errors = []
        for row in selection["candidates"]:
            beta = np.asarray(next(fit["coefficients_nonintercept"] for fit in json.loads((base / "fits.json").read_text()) if fit["lambda"] == row["lambda"]))
            score = p + config["sample_train_sd"] * row["strength"] * (z @ beta)
            regret, _ = proposal_regret(score, y, dev_pools, dev_proposals)
            candidate_errors.append(abs(float(regret.mean()) - row["source_dev_cap16_mean_regret"]))
        best = min(row["source_dev_cap16_mean_regret"] for row in selection["candidates"])
        chosen = selection["chosen"]
        assert abs(chosen["source_dev_cap16_mean_regret"] - best) <= 1e-12
        beta = np.asarray(selection["coefficients_nonintercept"])
        dest_seq = np.load(v1 / "auxiliary" / destination / "sequences.npy", allow_pickle=False)
        dest_y_all = np.load(v1 / "auxiliary" / destination / "labels.npy", allow_pickle=False).astype(np.float64)
        dest_p_all = np.load(v1 / "auxiliary" / destination / "cnn_predictions.npy", allow_pickle=False).astype(np.float64)
        dest_split = np.load(v1 / "auxiliary" / destination / "split.npy", allow_pickle=False)
        test = dest_split == 2
        dest_z = (features(dest_seq[test], dest_p_all[test]) - mean) / sd
        pools_local = np.load(v1 / "auxiliary" / name / "test_pools.npy", allow_pickle=False)
        test_indices = np.flatnonzero(test)
        mapping = np.full(len(test), -1, dtype=np.int64)
        mapping[test_indices] = np.arange(len(test_indices))
        pools = mapping[pools_local]
        proposals = np.load(v1 / "auxiliary" / name / "test_proposals.npy", allow_pickle=False)
        frozen, _ = proposal_regret(dest_p_all[test], dest_y_all[test], pools, proposals)
        score = dest_p_all[test] + config["sample_train_sd"] * chosen["strength"] * (dest_z @ beta)
        corrected, _ = proposal_regret(score, dest_y_all[test], pools, proposals)
        gate = json.loads((base / "gate.json").read_text())
        assert abs(frozen.mean() - gate["frozen_cap16_mean_regret"]) < 1e-12
        assert abs(corrected.mean() - gate["corrected_cap16_mean_regret"]) < 1e-12
        clusters = (corrected - frozen).mean(axis=1)
        rng = np.random.Generator(np.random.PCG64(20261126))
        indices = rng.integers(0, len(clusters), size=(10000, len(clusters)))
        draws = clusters[indices].mean(axis=1)
        low, high = np.percentile(draws, [2.5, 97.5])
        expected_pass = bool(clusters.mean() < 0 and high < 0)
        assert gate["pass"] == expected_pass
        assert abs(low - gate["corrected_minus_frozen"]["ci_low"]) < 1e-12
        assert abs(high - gate["corrected_minus_frozen"]["ci_high"]) < 1e-12
        direction_pass = bool(max(candidate_errors) < 1e-12 and all(row["loss_pass"] and row["gradient_pass"] for row in fit_checks))
        all_pass = all_pass and direction_pass
        output_directions[name] = {
            "status": "PASS" if direction_pass else "FAIL", "fit_checks": fit_checks,
            "max_candidate_regret_absolute_difference": max(candidate_errors),
            "test_frozen_regret": float(frozen.mean()), "test_corrected_regret": float(corrected.mean()),
            "effect": float(clusters.mean()), "ci_low": float(low), "ci_high": float(high),
            "gate_pass": expected_pass, "destination_dev_used": False,
        }
    summary = json.loads((root / "auxiliary_gate.json").read_text())
    if not summary["all_directions_pass"]:
        assert not (root / "auxiliary_gate_passed.json").exists()
        assert not (root / "target").exists()
    result = {"round": 2, "status": "PASS" if all_pass else "FAIL", "independent_implementation": True, "directions": output_directions, "target_mrl_read": False, "target_evaluation_count": 0}
    path = root / "verification" / "round2_independent.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
