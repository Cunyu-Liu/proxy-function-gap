from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def features(seq, p):
    bytes_ = seq.view(np.uint8).reshape(len(seq), 50)
    gc = ((bytes_ == ord("G")) | (bytes_ == ord("C"))).mean(axis=1)
    ua = np.zeros(len(seq), dtype=float)
    poly = np.zeros(len(seq), dtype=int)
    current = np.zeros(len(seq), dtype=int)
    for j in range(50):
        current = np.where(bytes_[:, j] == ord("T"), current + 1, 0)
        poly = np.maximum(poly, current)
        if j < 48:
            ua = np.maximum(ua, (bytes_[:, j] == ord("A")) & (bytes_[:, j + 1] == ord("T")) & (bytes_[:, j + 2] == ord("G")))
    return np.column_stack([p, p ** 2, gc, ua, poly, np.ones(len(p))])


def independent_fit(x, y, lam):
    device = torch.device("cuda:0")
    tx = torch.tensor(x, dtype=torch.float64, device=device)
    ty = torch.tensor(y, dtype=torch.float64, device=device)
    penalty = torch.ones(x.shape[1], dtype=torch.float64, device=device)
    penalty[-1] = 0
    a = tx.T @ tx / len(x) + lam * torch.diag(penalty)
    b = tx.T @ ty / len(x)
    condition = float(torch.linalg.cond(a).item())
    beta = torch.linalg.solve(a, b).cpu().numpy()
    torch.cuda.synchronize(device)
    assert np.isfinite(beta).all() and np.isfinite(condition) and condition <= 1e12
    return beta, condition


def regret(score, y, pools, proposals):
    selected = []
    oracle = []
    for pool_indices, pool_proposals in zip(pools, proposals):
        oracle.append(float(y[pool_indices].max()))
        for proposal in pool_proposals:
            choices = pool_indices[proposal]
            winner = choices[0]
            for index in choices[1:]:
                if score[index] > score[winner]:
                    winner = index
            selected.append(y[winner])
    return float(np.mean(np.asarray(oracle)[:, None] - np.asarray(selected).reshape(len(pools), -1)))


def verify_auxiliary(config, root, independent=False):
    checks = 0
    details = {}
    for source, destination in (("n50_hek293t_r1", "defined_hepg2_r1"), ("defined_hepg2_r1", "n50_hek293t_r1")):
        direction = f"{source}_to_{destination}"
        out = root / "auxiliary" / direction
        selected = read_json(out / "selection.json")
        source_root = root / "auxiliary" / source
        destination_root = root / "auxiliary" / destination
        source_split = np.load(source_root / "split.npy")
        source_y = np.load(source_root / "labels.npy")
        source_p = np.load(source_root / "cnn_predictions.npy")
        source_seq = np.load(source_root / "sequences.npy")
        destination_split = np.load(destination_root / "split.npy")
        destination_y = np.load(destination_root / "labels.npy")
        destination_p = np.load(destination_root / "cnn_predictions.npy")
        destination_seq = np.load(destination_root / "sequences.npy")
        train = source_split == 0
        residual = source_y[train] - source_p[train]
        standardized = (residual - residual.mean()) / residual.std()
        if independent:
            beta, condition = independent_fit(features(source_seq[train], source_p[train]), standardized, selected["chosen"]["lambda"])
            assert np.allclose(beta, selected["coefficients"], rtol=1e-8, atol=1e-8)
            checks += 2
        else:
            beta = np.asarray(selected["coefficients"])
            condition = None
        dx = features(destination_seq, destination_p)
        pools = {role: np.load(out / f"{role}_pools.npy") for role in ("dev", "test")}
        proposals = {role: np.load(out / f"{role}_proposals.npy") for role in ("dev", "test")}
        best = None
        for candidate in selected["candidates"]:
            lam = candidate["lambda"]
            if lam == selected["chosen"]["lambda"]:
                coefficients = beta
            else:
                fit = next(row for row in read_json(out / "ridge_fits.json") if row["lambda"] == lam)
                coefficients = np.asarray(fit["coefficients"])
            score = destination_p + config["sample_train_sd"] * candidate["strength"] * (dx @ coefficients)
            dev_regret = regret(score, destination_y, pools["dev"], proposals["dev"])
            dev_mse = float(np.mean((score[destination_split == 1] - destination_y[destination_split == 1]) ** 2))
            assert abs(dev_regret - candidate["dev_cap16_mean_regret"]) < 1e-10
            assert abs(dev_mse - candidate["dev_mse"]) < 1e-10
            key = (dev_regret, dev_mse, -lam, candidate["strength"])
            if best is None or key < best:
                best = key
            checks += 2
        chosen_score = destination_p + config["sample_train_sd"] * selected["chosen"]["strength"] * (dx @ beta)
        test_regret = regret(chosen_score, destination_y, pools["test"], proposals["test"])
        baseline_regret = regret(destination_p, destination_y, pools["test"], proposals["test"])
        chosen_key = (selected["chosen"]["dev_cap16_mean_regret"], selected["chosen"]["dev_mse"], -selected["chosen"]["lambda"], selected["chosen"]["strength"])
        assert best == chosen_key
        assert abs(test_regret - selected["test"]["cap16_mean_regret"]) < 1e-10
        assert abs(baseline_regret - selected["baseline"]["test"]["cap16_mean_regret"]) < 1e-10
        assert (test_regret < baseline_regret) == selected["gate_pass"]
        details[direction] = {"independent": independent, "selected_beta_max_abs_difference": 0.0 if not independent else float(np.max(np.abs(beta - np.asarray(selected["coefficients"])))), "condition_number": condition, "test_regret": test_regret, "baseline_regret": baseline_regret, "candidate_metric_checks": 2 * len(selected["candidates"])}
        checks += 3
    return {"status": "PASS", "checks": checks, "directions": details, "target_mrl_read": False}


def verify_raw(config, root):
    preflight = read_json(root / "raw_preflight.json")
    assert preflight["raw_hashes_pass"] and not preflight["target_mrl_read"]
    checks = 1
    details = {}
    for context in ("n50_hek293t_r1", "defined_hepg2_r1"):
        out = root / "auxiliary" / context
        prepare = read_json(out / "prepare_manifest.json")
        split = read_json(out / "split_manifest.json")
        assert prepare["retained_rows"] == prepare["retained_unique"] == split["rows"]
        assert prepare["inference"]["actual_cuda"] and not prepare["inference"]["cpu_fallback"]
        assert not prepare["target_mrl_read"]
        if context == "n50_hek293t_r1":
            assert all(row["exact"] == 0 and row["h1_only"] == 0 for row in prepare["excluded_by_target"].values())
        else:
            assert prepare["excluded_by_target"]["defined_tcell_r1"]["exact"] == 316495
            assert prepare["excluded_by_target"]["defined_tcell_r2"]["exact"] == 319264
        for name, expected in prepare["files"].items():
            assert sha(out / name) == expected
            checks += 1
        seq = np.load(out / "sequences.npy")
        y = np.load(out / "labels.npy")
        p = np.load(out / "cnn_predictions.npy")
        members = np.load(out / "member_predictions.npy")
        split_values = np.load(out / "split.npy")
        assert np.isfinite(y).all() and np.isfinite(p).all()
        assert np.array_equal(p, members.mean(axis=0))
        assert set(np.unique(split_values)) == {0, 1, 2}
        assert sum(split["roles"][role]["rows"] for role in ("train", "dev", "test")) == len(seq)
        assert split["exact_h1_component_cross_role"] == 0
        details[context] = {"retained": len(seq), "excluded": prepare["excluded"], "roles": split["roles"], "components": split["components"]}
        checks += 8
    return {"status": "PASS", "checks": checks, "contexts": details, "target_mrl_read": False}


def verify_target(config, root):
    summary = read_json(root / "target" / "summary.json")
    checks = 0
    repeat_effects = []
    combined_pool_effects = []
    for context in config["target"]["contexts"]:
        selection = read_json(root / "target" / context / "selection.json")
        evaluation = read_json(root / "target" / context / "evaluation.json")
        labels = {}
        with gzip.open(Path(config["pool_root"]) / context / "evaluator" / "labels.tsv.gz", "rt", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                labels[row["sequence_id"]] = float(row["measured_rl"])
        score_pack = np.load(root / "target" / context / "scores.npz")
        assert not any("utr50" in key for key in selection)
        assert all(row["regret"] is None for row in selection["records"])
        evaluated = {(row["model"], row["pool_index"], row["proposal_cap"], row["seed"]): row for row in evaluation["records"]}
        assert len(evaluated) == len(selection["records"])
        pool_max = {}
        for pool_index in range(16):
            rows = [row for row in evaluation["records"] if row["pool_index"] == pool_index and row["model"] == "frozen_cnn" and row["proposal_cap"] == 1 and row["seed"] == 101]
            learner_path = Path(config["pool_root"]) / context / "learner" / f"castillo_{context}_{pool_index:03d}.jsonl.gz"
            with gzip.open(learner_path, "rt") as handle:
                learners = [json.loads(line) for line in handle]
            pool_max[pool_index] = max(labels[row["sequence_id"]] for row in learners)
        for original in selection["records"]:
            key = (original["model"], original["pool_index"], original["proposal_cap"], original["seed"])
            actual = evaluated[key]
            score = score_pack[original["model"]][original["pool_index"] * 64:original["pool_index"] * 64 + 64]
            expected_proposal = []
            rng = random.Random(original["seed"])
            for _ in range(original["proposal_cap"]):
                expected_proposal.append(min(63, int(rng.random() * 64)))
            assert expected_proposal == original["proposal_indices"]
            winner = expected_proposal[0]
            for index in expected_proposal[1:]:
                if score[index] > score[winner]:
                    winner = index
            expected_regret = pool_max[original["pool_index"]] - labels[original["selected_id"]]
            assert winner == original["selected_index"] == actual["selected_index"]
            assert abs(expected_regret - actual["regret"]) < 1e-12
            checks += 4
        for cap in config["target"]["caps"]:
            for model in config["target"]["models"]:
                values = [row["regret"] for row in evaluation["records"] if row["proposal_cap"] == cap and row["model"] == model]
                expected = next(row for row in summary["results"] if row.get("context") == context and row.get("cap") == cap and row.get("model") == model and "mean_regret" in row)
                assert abs(float(np.mean(values)) - expected["mean_regret"]) < 1e-12
                checks += 1
            for model in ("corrected_cnn", "affine_only"):
                pool_effects = []
                for pool_index in range(16):
                    frozen = [row["regret"] for row in evaluation["records"] if row["proposal_cap"] == cap and row["pool_index"] == pool_index and row["model"] == "frozen_cnn"]
                    changed = [row["regret"] for row in evaluation["records"] if row["proposal_cap"] == cap and row["pool_index"] == pool_index and row["model"] == model]
                    assert len(frozen) == len(changed) == 5
                    pool_effects.append(float(np.mean(changed) - np.mean(frozen)))
                actual = next(row for row in summary["results"] if row.get("context") == context and row.get("cap") == cap and row.get("model") == model and "contrast" in row)
                assert abs(float(np.mean(pool_effects)) - actual["effect"]) < 1e-12
                assert np.allclose(pool_effects, actual["pool_effects"], rtol=0, atol=1e-12)
                if model == "corrected_cnn" and cap == 16:
                    repeat_effects.append(actual)
                    combined_pool_effects.extend(pool_effects)
                checks += 3
    assert all(row["effect"] < 0 for row in repeat_effects) == summary["primary"]["both_repeats_improved"]
    combined_effect = float(np.mean(combined_pool_effects))
    assert abs(combined_effect - summary["primary"]["combined_corrected_minus_frozen"]["effect"]) < 1e-12
    expected_threshold = max(0.1, 0.05 * summary["primary"]["combined_frozen_mean_regret"])
    assert expected_threshold == summary["primary"]["minimum_absolute_improvement"]
    expected_success = all(row["effect"] < 0 for row in repeat_effects) and summary["primary"]["combined_corrected_minus_frozen"]["ci_high"] < 0 and -combined_effect >= expected_threshold
    assert expected_success == summary["primary"]["main_success"]
    return {"status": "PASS", "checks": checks + 4, "combined_effect": combined_effect, "main_success_independently_recomputed": expected_success}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("round", choices=("round1", "round2", "round3"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    root = Path(config["artifact_root"])
    if args.round == "round1":
        result = {"raw": verify_raw(config, root), "auxiliary": verify_auxiliary(config, root, independent=False)}
    else:
        result = {"raw": verify_raw(config, root), "auxiliary_independent": verify_auxiliary(config, root, independent=True)}
        if (root / "auxiliary_gate_passed.json").exists():
            result["target_independent"] = verify_target(config, root)
        else:
            result["target"] = {"status": "NOT_RUN_AUXILIARY_GATE_FAILED", "target_mrl_read": False, "target_evaluation_count": 0}
        if args.round == "round3":
            result["delivery"] = {"summary_sha256": sha(root / "target" / "summary.json") if (root / "target" / "summary.json").exists() else None, "sequence_label_exported": False, "status": "PASS"}
    write(root / "verification" / f"{args.round}.json", result)


if __name__ == "__main__":
    main()
