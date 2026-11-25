import csv
import hashlib
import json
from pathlib import Path

import numpy as np

AUDIT_DIR = Path(__file__).resolve().parent
WORKSPACE = AUDIT_DIR.parent.parent
V1_ROOT = WORKSPACE / "deliverables" / "reports" / "source_data" / "persist_mrl_external_v1"
RESULTS = AUDIT_DIR / "results"
SEEDS = list(range(20261125, 20261325))
AFFECTED_SEEDS = [20261147, 20261254, 20261289, 20261290, 20261324]
INPUT_HASHES = {
    "predictions/predictions.tsv": "61919494970ff3e2a9cb65ce6f3a3f6679b2ecb8e3242ef1f604a253b3e5d6c1",
    "protocol/evaluator_labels.tsv": "96645ba2d93b758ec2fb590a668ae84b78d1b4559ac1486b7bc12f4e6ac4991c",
    "protocol/eligible_sequences.tsv": "c3d15c9c908023a85122650f48b26b9b1b354750f8d7667f9b7638ca9a4ea1cc",
    "results/per_seed.tsv": "26698e4e39b80f3d3a282a1ee9ac2279b460d224d63274ac9a32e0f50594739b",
    "results/summary.json": "6408d1efde788b1a96a8caab1888c02680a8567c58cb86e0406a65d4728ac32f",
    "results/bootstrap_replicates.npy": "f5e0d69aa1a85aea73d31aad54e2bbe60bbff32722e19292b062aa3a53433baa",
    "verification/round2_independent_bootstrap.npy": "271ff72f70df3537eadf32795929d80a45aa7685f8aeb4cff7adb07d45f61fe9",
    "artifact_manifest.json": "2fcb11b83301bdec14b6338ec78439249c52b3ceb867d9fbd12f93c9e69db549",
}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_id(sequence_id):
    return hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()


def write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_tsv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def order_for_seed(n, seed):
    return np.random.default_rng(seed).permutation(n)


def choose(order, scores, ids, cap=None, rule="sequence_id", hash_ids=None):
    candidates = order if cap is None else order[:cap]
    if rule == "sequence_id":
        return sorted(candidates, key=lambda i: (-scores[i], ids[i], int(i)))[0]
    if rule == "sha256_sequence_id":
        if hash_ids is None:
            hash_ids = ids
        return sorted(candidates, key=lambda i: (-scores[i], sha256_id(hash_ids[i]), ids[i], int(i)))[0]
    if rule == "stable_first":
        return sorted(candidates, key=lambda i: -scores[i])[0]
    raise ValueError(f"Unknown tie rule: {rule}")


def run_proposals(labels, scores, ids, seeds, seed_offset=None, rule="sequence_id", hash_ids=None):
    records = []
    for seed in seeds:
        actual_seed = seed if seed_offset is None else [seed_offset[0], seed_offset[1], seed]
        order = order_for_seed(len(labels), actual_seed)
        oracle = choose(order, labels, ids, rule="sequence_id")
        random_index = int(order[0])
        bon_index = choose(order, scores, ids, 16, rule, hash_ids)
        oracle_value = labels[oracle]
        random_regret = oracle_value - labels[random_index]
        bon_regret = oracle_value - labels[bon_index]
        records.append((random_regret, bon_regret, bon_regret - random_regret,
                        float(labels[random_index] == oracle_value),
                        float(labels[bon_index] == oracle_value)))
    return np.asarray(records)


def load_frozen_inputs():
    for name, expected in INPUT_HASHES.items():
        actual = sha256_file(V1_ROOT / name)
        if actual != expected:
            raise RuntimeError(f"Frozen v1 input changed: {name}: {actual} != {expected}")
    labels_rows = read_tsv(V1_ROOT / "protocol" / "evaluator_labels.tsv")
    prediction_rows = read_tsv(V1_ROOT / "predictions" / "predictions.tsv")
    predictions = {(row["sequence_id"], row["utr"]): float(row["prediction"]) for row in prediction_rows}
    if len(predictions) != len(labels_rows):
        raise ValueError("Prediction-label keys are not one-to-one")
    ids = [row["sequence_id"] for row in labels_rows]
    utrs = [row["utr"] for row in labels_rows]
    labels = np.asarray([float(row["mrl"]) for row in labels_rows], dtype=float)
    scores = np.asarray([predictions[(row["sequence_id"], row["utr"])] for row in labels_rows], dtype=float)
    return ids, utrs, labels, scores


def per_seed_frame(seeds, values):
    return [{
        "algorithm_seed": str(seed),
        "random_regret": format(values[i, 0], ".17g"),
        "bon_regret": format(values[i, 1], ".17g"),
        "delta": format(values[i, 2], ".17g"),
        "random_oracle_coverage": format(values[i, 3], ".17g"),
        "bon_oracle_coverage": format(values[i, 4], ".17g"),
    } for i, seed in enumerate(seeds)]


def write_tsv(rows, path, fieldnames):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_primary(base, bootstrap):
    delta = bootstrap[:, 0]
    ci = [float(x) for x in np.percentile(delta, [2.5, 97.5])]
    p_value = min(1.0, 2 * min(((delta >= 0).sum() + 1) / 10001,
                               ((delta <= 0).sum() + 1) / 10001))
    point = float(base[:, 2].mean())
    return {
        "bon_minus_random_regret": point,
        "ci95": ci,
        "p_two_sided_sign": float(p_value),
        "significant_benefit": bool(ci[1] < 0 and p_value < 0.05),
        "direction": "BoN favored" if point < 0 else "Random favored" if point > 0 else "Tie",
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    ids, utrs, labels, scores = load_frozen_inputs()
    base_v1 = run_proposals(labels, scores, ids, SEEDS)
    frozen_per_seed = read_tsv(V1_ROOT / "results" / "per_seed.tsv")
    frozen_base = np.asarray([[float(row[key]) for key in
                               ["random_regret", "bon_regret", "delta", "random_oracle_coverage", "bon_oracle_coverage"]]
                              for row in frozen_per_seed])
    if not np.array_equal(base_v1, frozen_base):
        raise AssertionError("Recomputed v1 per-seed values differ from frozen per_seed.tsv")

    tie_rows = []
    for seed in SEEDS:
        order = order_for_seed(len(ids), seed)
        candidates = order[:16]
        maximum = scores[candidates].max()
        tied = [int(i) for i in candidates if scores[i] == maximum]
        if len(tied) < 2:
            continue
        selected = {rule: choose(order, scores, ids, 16, rule)
                    for rule in ["sequence_id", "stable_first", "sha256_sequence_id"]}
        tie_rows.append({
            "algorithm_seed": str(seed),
            "prediction": format(maximum, ".17g"),
            "tied_sequence_ids": ";".join(ids[i] for i in tied),
            "stable_first_sequence_id": ids[selected["stable_first"]],
            "v1_sequence_id": ids[selected["sequence_id"]],
            "v1_1_sha256_sequence_id": ids[selected["sha256_sequence_id"]],
            "stable_first_differs_from_v1": str(selected["stable_first"] != selected["sequence_id"]).lower(),
            "sha256_differs_from_v1": str(selected["sha256_sequence_id"] != selected["sequence_id"]).lower(),
        })
    observed_affected = [int(row["algorithm_seed"]) for row in tie_rows if row["stable_first_differs_from_v1"] == "true"]
    if observed_affected != AFFECTED_SEEDS:
        raise AssertionError(f"Unexpected order-sensitive seeds: {observed_affected}")
    if len(tie_rows) != 9:
        raise AssertionError(f"Expected 9 exact-tie seeds, observed {len(tie_rows)}")

    base_v1_1 = run_proposals(labels, scores, ids, SEEDS, rule="sha256_sequence_id")
    changed_outside_scope = [seed for i, seed in enumerate(SEEDS)
                             if seed not in AFFECTED_SEEDS and not np.array_equal(base_v1[i], base_v1_1[i])]
    if changed_outside_scope:
        raise AssertionError(f"Changes outside five prespecified seeds: {changed_outside_scope}")

    affected_rows = []
    for seed in AFFECTED_SEEDS:
        i = SEEDS.index(seed)
        order = order_for_seed(len(ids), seed)
        tied = [int(j) for j in order[:16] if scores[j] == scores[order[:16]].max()]
        v1_index = choose(order, scores, ids, 16, "sequence_id")
        alt_index = choose(order, scores, ids, 16, "sha256_sequence_id")
        affected_rows.append({
            "algorithm_seed": str(seed),
            "tied_sequence_ids": ";".join(ids[j] for j in tied),
            "v1_selected_sequence_id": ids[v1_index],
            "v1_1_selected_sequence_id": ids[alt_index],
            "selection_changed": str(v1_index != alt_index).lower(),
            "v1_delta": format(base_v1[i, 2], ".17g"),
            "v1_1_delta": format(base_v1_1[i, 2], ".17g"),
            "delta_v1_1_minus_v1": format(base_v1_1[i, 2] - base_v1[i, 2], ".17g"),
        })

    original_bootstrap = np.load(V1_ROOT / "results" / "bootstrap_replicates.npy")
    bootstrap_v1_1 = original_bootstrap.copy()
    clusters = sorted(set(utrs))
    cluster_rows = [np.asarray([i for i, value in enumerate(utrs) if value == cluster], dtype=int)
                    for cluster in clusters]
    rng = np.random.default_rng(20261126)
    fraction = len(AFFECTED_SEEDS) / len(SEEDS)
    selection_changes = 0
    tied_seed_replicates = 0
    changed_effect_replicates = 0
    for b in range(10000):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([cluster_rows[i] for i in sampled])
        b_labels = labels[indices]
        b_scores = scores[indices]
        b_ids = [f"{ids[i]}#{rank}" for rank, i in enumerate(indices)]
        original_ids = [ids[i] for i in indices]
        original_five = run_proposals(b_labels, b_scores, b_ids, AFFECTED_SEEDS,
                                      (20261127, b), "sequence_id", original_ids)
        alternate_five = run_proposals(b_labels, b_scores, b_ids, AFFECTED_SEEDS,
                                       (20261127, b), "sha256_sequence_id", original_ids)
        for seed in AFFECTED_SEEDS:
            order = order_for_seed(len(indices), [20261127, b, seed])
            maximum = b_scores[order[:16]].max()
            tied = np.flatnonzero(b_scores[order[:16]] == maximum)
            if len(tied) > 1:
                tied_seed_replicates += 1
            original_index = choose(order, b_scores, b_ids, 16, "sequence_id", original_ids)
            alternate_index = choose(order, b_scores, b_ids, 16, "sha256_sequence_id", original_ids)
            selection_changes += int(original_index != alternate_index)
        changed_effect_replicates += int(original_five[:, 2].mean() != alternate_five[:, 2].mean())
        bootstrap_v1_1[b, 0] += fraction * (alternate_five[:, 2].mean() - original_five[:, 2].mean())
        bootstrap_v1_1[b, 5] += fraction * (alternate_five[:, 1].mean() - original_five[:, 1].mean())
        bootstrap_v1_1[b, 7] += fraction * (alternate_five[:, 4].mean() - original_five[:, 4].mean())

    original_summary = json.loads((V1_ROOT / "results" / "summary.json").read_text())
    primary_v1_1 = summarize_primary(base_v1_1, bootstrap_v1_1)
    secondary_v1_1 = json.loads(json.dumps(original_summary["secondary"]))
    secondary_v1_1["bon_top1_regret"] = {
        "value": float(base_v1_1[:, 1].mean()),
        "ci95": [float(x) for x in np.percentile(bootstrap_v1_1[:, 5], [2.5, 97.5])],
    }
    secondary_v1_1["bon_oracle_coverage"] = {
        "value": float(base_v1_1[:, 4].mean()),
        "ci95": [float(x) for x in np.percentile(bootstrap_v1_1[:, 7], [2.5, 97.5])],
    }
    affected_indices = [SEEDS.index(seed) for seed in AFFECTED_SEEDS]
    summary = {
        "analysis_id": "persist_mrl_external_v1_1_tie_sensitivity",
        "parent_analysis_id": "persist_mrl_external_v1",
        "retrospective_sensitivity_only": True,
        "version_selection_by_direction": False,
        "tie_rule": "Among exactly maximum predictions, ascending SHA256(UTF-8 original Sequence ID), then Sequence ID, then row position",
        "scope": "All 200 base seeds audited; bootstrap updates only the five base seeds where stable-first differs from frozen v1; the other 195 seeds and all non-selection columns remain v1",
        "algorithm_seeds": 200,
        "exact_tie_seed_count": len(tie_rows),
        "order_sensitive_seed_count": len(AFFECTED_SEEDS),
        "order_sensitive_seeds": AFFECTED_SEEDS,
        "bootstrap_replicates": 10000,
        "bootstrap_tied_seed_replicates_in_scope": tied_seed_replicates,
        "bootstrap_seed_selection_changes": selection_changes,
        "bootstrap_changed_effect_replicates": changed_effect_replicates,
        "primary": primary_v1_1,
        "secondary": secondary_v1_1,
        "affected_five_mean_delta": {
            "v1": float(base_v1[affected_indices, 2].mean()),
            "v1_1": float(base_v1_1[affected_indices, 2].mean()),
        },
        "statistical_compute": "CPU only; no model training or inference",
    }
    comparison = {
        "v1": original_summary["primary"],
        "v1_1": primary_v1_1,
        "v1_1_minus_v1_point": primary_v1_1["bon_minus_random_regret"] - original_summary["primary"]["bon_minus_random_regret"],
        "per_seed_changed_count": int(np.sum(base_v1[:, 2] != base_v1_1[:, 2])),
        "bootstrap_changed_element_count": int(np.sum(original_bootstrap != bootstrap_v1_1)),
        "bootstrap_max_absolute_change": float(np.max(np.abs(original_bootstrap - bootstrap_v1_1))),
        "reporting_policy": "Retain v1 and v1.1 side by side; no direction-based version selection; v1 remains primary",
        "frozen_v1_inputs_unchanged": True,
    }

    write_tsv(per_seed_frame(SEEDS, base_v1_1), RESULTS / "per_seed_v1_1.tsv",
              ["algorithm_seed", "random_regret", "bon_regret", "delta", "random_oracle_coverage", "bon_oracle_coverage"])
    write_tsv(tie_rows, RESULTS / "tie_seeds.tsv", list(tie_rows[0]))
    write_tsv(affected_rows, RESULTS / "affected_seeds.tsv", list(affected_rows[0]))
    np.save(RESULTS / "bootstrap_v1_1.npy", bootstrap_v1_1)
    write_json(summary, RESULTS / "summary_v1_1.json")
    write_json(comparison, RESULTS / "comparison_v1_vs_v1_1.json")
    for name, expected in INPUT_HASHES.items():
        if sha256_file(V1_ROOT / name) != expected:
            raise AssertionError(f"v1 input mutated during analysis: {name}")
    print(json.dumps({"summary": summary, "comparison": comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
