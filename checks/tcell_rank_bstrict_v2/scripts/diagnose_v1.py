from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def selection_metrics(p, y, pools, proposals, score):
    pool_y = y[pools]
    proposed_y = np.take_along_axis(pool_y[:, None, :], proposals, axis=2)
    base_local = np.argmax(np.take_along_axis(p[pools][:, None, :], proposals, axis=2), axis=2)
    corrected_local = np.argmax(np.take_along_axis(score[pools][:, None, :], proposals, axis=2), axis=2)
    base_y = np.take_along_axis(proposed_y, base_local[..., None], axis=2)[..., 0]
    corrected_y = np.take_along_axis(proposed_y, corrected_local[..., None], axis=2)[..., 0]
    changed = base_local != corrected_local
    delta = corrected_y - base_y
    oracle = pool_y.max(axis=1)[:, None]
    return {
        "frozen_regret": float(np.mean(oracle - base_y)),
        "corrected_regret": float(np.mean(oracle - corrected_y)),
        "selection_turnover": float(changed.mean()),
        "changed": int(changed.sum()),
        "changed_improved": int(np.sum(changed & (delta > 0))),
        "changed_worsened": int(np.sum(changed & (delta < 0))),
        "changed_same_mrl": int(np.sum(changed & (delta == 0))),
        "changed_mean_mrl_delta": float(delta[changed].mean()) if changed.any() else 0.0,
        "all_mean_mrl_delta": float(delta.mean()),
    }


def tail_metrics(p, y, pools, score):
    pair_i, pair_j = np.triu_indices(16, 1)
    totals = {
        "pairs": 0, "label_ties": 0, "frozen_correct": 0, "corrected_correct": 0,
        "correct_to_wrong": 0, "wrong_to_correct": 0, "correct_to_tied": 0,
        "wrong_to_tied": 0, "tied_to_correct": 0, "tied_to_wrong": 0,
    }
    for start in range(0, len(pools), 512):
        block = pools[start:start + 512]
        order = np.argsort(-p[block], axis=1, kind="stable")[:, :16]
        tail_y = np.take_along_axis(y[block], order, axis=1)
        tail_p = np.take_along_axis(p[block], order, axis=1)
        tail_score = np.take_along_axis(score[block], order, axis=1)
        dy = tail_y[:, pair_i] - tail_y[:, pair_j]
        dp = tail_p[:, pair_i] - tail_p[:, pair_j]
        dc = tail_score[:, pair_i] - tail_score[:, pair_j]
        valid = dy != 0
        frozen_product = dp * dy
        corrected_product = dc * dy
        frozen_correct = frozen_product > 0
        corrected_correct = corrected_product > 0
        frozen_wrong = frozen_product < 0
        corrected_wrong = corrected_product < 0
        frozen_tied = frozen_product == 0
        corrected_tied = corrected_product == 0
        totals["pairs"] += int(valid.sum())
        totals["label_ties"] += int((~valid).sum())
        totals["frozen_correct"] += int(np.sum(valid & frozen_correct))
        totals["corrected_correct"] += int(np.sum(valid & corrected_correct))
        totals["correct_to_wrong"] += int(np.sum(valid & frozen_correct & corrected_wrong))
        totals["wrong_to_correct"] += int(np.sum(valid & frozen_wrong & corrected_correct))
        totals["correct_to_tied"] += int(np.sum(valid & frozen_correct & corrected_tied))
        totals["wrong_to_tied"] += int(np.sum(valid & frozen_wrong & corrected_tied))
        totals["tied_to_correct"] += int(np.sum(valid & frozen_tied & corrected_correct))
        totals["tied_to_wrong"] += int(np.sum(valid & frozen_tied & corrected_wrong))
    pairs = totals["pairs"]
    totals.update({
        "frozen_pair_accuracy_excluding_label_ties": totals["frozen_correct"] / pairs,
        "corrected_pair_accuracy_excluding_label_ties": totals["corrected_correct"] / pairs,
        "correct_to_wrong_fraction": totals["correct_to_wrong"] / pairs,
        "wrong_to_correct_fraction": totals["wrong_to_correct"] / pairs,
        "net_wrong_to_correct_minus_correct_to_wrong_fraction": (totals["wrong_to_correct"] - totals["correct_to_wrong"]) / pairs,
        "tail_definition": "frozen CNN full-pool top16; stable score order; all 120 unordered pairs; exact MRL ties excluded from pair accuracy",
    })
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.v1_root)
    config = json.loads((root / "export/configs/bstrict_frozen.json").read_text())
    sample_sd = config["sample_train_sd"]
    results = []
    for direction in ("n50_hek293t_r1_to_defined_hepg2_r1", "defined_hepg2_r1_to_n50_hek293t_r1"):
        path = root / "auxiliary" / direction
        selection = json.loads((path / "selection.json").read_text())
        destination = selection["destination"]
        data = root / "auxiliary" / destination
        p = np.load(data / "cnn_predictions.npy", allow_pickle=False)
        y = np.load(data / "labels.npy", allow_pickle=False)
        seq = np.load(data / "sequences.npy", allow_pickle=False)
        bases = seq.view(np.uint8).reshape(len(seq), 50)
        gc = ((bases == ord("C")) | (bases == ord("G"))).mean(axis=1)
        uaug = np.zeros(len(seq), dtype=np.float64)
        longest = np.zeros(len(seq), dtype=np.float64)
        run = np.zeros(len(seq), dtype=np.float64)
        for position in range(50):
            run = np.where(bases[:, position] == ord("T"), run + 1, 0)
            longest = np.maximum(longest, run)
            if position <= 47:
                uaug = np.maximum(uaug, ((bases[:, position] == ord("A")) & (bases[:, position + 1] == ord("T")) & (bases[:, position + 2] == ord("G"))).astype(np.float64))
        x = np.column_stack((p, p * p, gc, uaug, longest, np.ones(len(p))))
        beta = np.asarray(selection["coefficients"], dtype=np.float64)
        strength = selection["chosen"]["strength"]
        corrected = p + sample_sd * strength * (x @ beta)
        roles = {}
        for role in ("dev", "test"):
            pools = np.load(path / f"{role}_pools.npy", allow_pickle=False)
            proposals = np.load(path / f"{role}_proposals.npy", allow_pickle=False)
            roles[role] = {
                "selection": selection_metrics(p, y, pools, proposals, corrected),
                "frozen_top16_pairs": tail_metrics(p, y, pools, corrected),
            }
        results.append({
            "direction": direction,
            "chosen_lambda": selection["chosen"]["lambda"],
            "chosen_strength": strength,
            "coefficients_in_order_p_p2_gc_uAUG_polyU_intercept": beta.tolist(),
            "dev_to_test_regret_change": roles["test"]["selection"]["corrected_regret"] - roles["test"]["selection"]["frozen_regret"] - (roles["dev"]["selection"]["corrected_regret"] - roles["dev"]["selection"]["frozen_regret"]),
            "roles": roles,
        })
    output = {
        "schema_version": "1.0",
        "scope": "v1 auxiliary arrays only; no T-cell MRL, target scoring, model search, or hyperparameter selection",
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
