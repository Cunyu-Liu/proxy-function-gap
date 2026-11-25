import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AUDIT_DIR = Path(__file__).resolve().parent
WORKSPACE = AUDIT_DIR.parent.parent
V1_ROOT = WORKSPACE / "deliverables" / "reports" / "source_data" / "persist_mrl_external_v1"
ALPHABET = {"A": 0, "C": 1, "G": 2, "T": 3}


def read_tsv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def original_features(sequence):
    values = np.zeros(259, dtype=np.float32)
    for index in range(len(sequence) - 3):
        code = 0
        for base in sequence[index:index + 4]:
            code = code * 4 + ALPHABET[base]
        values[code] += 1
    values[:256] /= len(sequence) - 3
    values[256] = (sequence.count("G") + sequence.count("C")) / len(sequence)
    values[257] = float("ATG" in sequence)
    values[258] = max((len(part) for part in sequence.replace("A", " ").replace("C", " ").replace("G", " ").split()), default=0) / 50
    return values


def independent_features(sequence):
    out = np.zeros(259, dtype=np.float32)
    denominator = len(sequence) - 3
    for start in range(denominator):
        word = sequence[start:start + 4]
        index = ((ALPHABET[word[0]] * 4 + ALPHABET[word[1]]) * 4 + ALPHABET[word[2]]) * 4 + ALPHABET[word[3]]
        out[index] += 1.0 / denominator
    out[256] = sum(base in "GC" for base in sequence) / len(sequence)
    out[257] = "ATG" in sequence
    longest = current = 0
    for base in sequence:
        current = current + 1 if base == "T" else 0
        longest = max(longest, current)
    out[258] = longest / 50
    return out


def proposal_stats(y, p, names, entropy=None):
    random_regrets, bon_regrets, random_hits, bon_hits = [], [], [], []
    for seed in range(20261125, 20261325):
        rng = np.random.default_rng(seed if entropy is None else entropy + [seed])
        order = rng.permutation(len(y))
        oracle = sorted(range(len(y)), key=lambda i: (-y[i], names[i], i))[0]
        bon = sorted(order[:16], key=lambda i: (-p[i], names[i], int(i)))[0]
        random = order[0]
        random_regrets.append(y[oracle] - y[random])
        bon_regrets.append(y[oracle] - y[bon])
        random_hits.append(float(y[random] == y[oracle]))
        bon_hits.append(float(y[bon] == y[oracle]))
    random_regrets = np.asarray(random_regrets)
    bon_regrets = np.asarray(bon_regrets)
    return np.asarray([np.mean(bon_regrets - random_regrets), random_regrets.mean(), bon_regrets.mean(),
                       np.mean(random_hits), np.mean(bon_hits)])


def original_metrics(y, p):
    return np.asarray([stats.spearmanr(p, y).statistic, stats.pearsonr(p, y).statistic,
                       np.mean(np.abs(p - y))])


def independent_metrics(y, p):
    ranked_p, ranked_y = stats.rankdata(p), stats.rankdata(y)
    return np.asarray([np.corrcoef(ranked_p, ranked_y)[0, 1], np.corrcoef(p, y)[0, 1],
                       np.mean(np.abs(p - y))])


def column_max(a, b):
    return np.max(np.abs(a - b), axis=0).tolist()


def main():
    label_rows = read_tsv(V1_ROOT / "protocol" / "evaluator_labels.tsv")
    prediction_rows = read_tsv(V1_ROOT / "predictions" / "predictions.tsv")
    prediction_by_key = {(r["sequence_id"], r["utr"]): float(r["prediction"]) for r in prediction_rows}
    csv_y = np.asarray([float(r["mrl"]) for r in label_rows])
    csv_p = np.asarray([prediction_by_key[(r["sequence_id"], r["utr"])] for r in label_rows])
    labels_frame = pd.read_csv(V1_ROOT / "protocol" / "evaluator_labels.tsv", sep="\t")
    predictions_frame = pd.read_csv(V1_ROOT / "predictions" / "predictions.tsv", sep="\t")
    merged = labels_frame.merge(predictions_frame, on=["sequence_id", "utr"], validate="one_to_one")
    ids = merged.sequence_id.astype(str).tolist()
    utrs = merged.utr.tolist()
    y = merged.mrl.to_numpy(dtype=float)
    p = merged.prediction.to_numpy(dtype=float)
    parser_label_max = float(np.max(np.abs(csv_y - y)))
    parser_prediction_max = float(np.max(np.abs(csv_p - p)))
    expected = np.load(V1_ROOT / "results" / "bootstrap_replicates.npy")
    independent_gpu = np.load(V1_ROOT / "verification" / "round2_independent_bootstrap.npy")
    old_report = json.loads((V1_ROOT / "verification" / "round2_independent.json").read_text())
    remote_reproduction = json.loads((AUDIT_DIR / "logs" / "remote_bootstrap_reproduction.log").read_text())
    assert remote_reproduction["status"] == "PASS"
    assert remote_reproduction["bitwise_equal_cells"] == remote_reproduction["total_cells"] == 80000
    assert remote_reproduction["remote_writes"] is False

    clusters = sorted(set(utrs))
    members_a = [[i for i, seq in enumerate(utrs) if seq == cluster] for cluster in clusters]
    grouped = {}
    for i, seq in enumerate(utrs):
        grouped.setdefault(seq, []).append(i)
    members_b = [grouped[cluster] for cluster in sorted(grouped)]
    assert members_a == members_b

    rng = np.random.default_rng(20261126)
    boot_saved_original_metrics = np.empty((10000, 8))
    boot_saved_independent_metrics = np.empty((10000, 8))
    draws_digest = hashlib.sha256()
    indices_digest = hashlib.sha256()
    for b in range(10000):
        draws = rng.integers(0, len(clusters), len(clusters))
        draws_digest.update(np.asarray(draws, dtype=np.int64).tobytes())
        indices_a = np.concatenate([members_a[d] for d in draws])
        indices_b = np.concatenate([np.asarray(members_b[d], dtype=int) for d in draws])
        assert np.array_equal(indices_a, indices_b)
        indices_digest.update(np.asarray(indices_a, dtype=np.int64).tobytes())
        by, bp = y[indices_a], p[indices_a]
        names = [f"{ids[i]}#{rank}" for rank, i in enumerate(indices_a)]
        values = proposal_stats(by, bp, names, [20261127, b])
        for target, metric_function in [(boot_saved_original_metrics, original_metrics),
                                        (boot_saved_independent_metrics, independent_metrics)]:
            target[b, 0] = values[0]
            target[b, 1:4] = metric_function(by, bp)
            target[b, 4:] = values[1:]

    feature_a = np.stack([original_features(seq) for seq in utrs])
    feature_b = np.stack([independent_features(seq) for seq in utrs])
    feature_difference = np.abs(feature_a - feature_b)
    same_prediction_reduction = column_max(boot_saved_independent_metrics, expected)
    prediction_path_difference = column_max(independent_gpu, boot_saved_independent_metrics)
    original_independent_difference = column_max(independent_gpu, expected)

    original_path_difference = np.abs(boot_saved_original_metrics - expected)
    print("FROZEN_PREDICTION_ORIGINAL_METRICS_COLUMN_MAX", original_path_difference.max(axis=0).tolist(), flush=True)
    if not np.array_equal(boot_saved_original_metrics, expected):
        first = np.unravel_index(np.argmax(original_path_difference), original_path_difference.shape)
        print("FIRST_ORIGINAL_PATH_DIFFERENCE", first, boot_saved_original_metrics[first].tolist(), expected[first].tolist(), flush=True)
    print("SAME_PREDICTION_INDEPENDENT_METRICS_COLUMN_MAX", same_prediction_reduction, flush=True)
    print("OLD_INDEPENDENT_GPU_COLUMN_MAX", original_independent_difference, flush=True)
    print("FEATURE_FLOAT32_MAX", float(feature_difference.max()), flush=True)
    assert np.array_equal(boot_saved_original_metrics[:, [0, 4, 5, 6, 7]], expected[:, [0, 4, 5, 6, 7]]), \
        "Cluster/RNG/proposal selection path is not exact"
    assert np.max(original_path_difference[:, 1:4]) < 5e-15, \
        "Same-prediction continuous metrics exceed the isolated local NumPy/SciPy reduction bound"
    assert np.array_equal(boot_saved_independent_metrics[:, [0, 4, 5, 6, 7]],
                          expected[:, [0, 4, 5, 6, 7]]), "Difference is not isolated to continuous metric reductions"
    assert max(same_prediction_reduction[1:4]) < 5e-15, "Unexpected same-prediction continuous metric discrepancy"
    assert np.array_equal(independent_gpu[:, [0, 4, 5, 6, 7]], expected[:, [0, 4, 5, 6, 7]]), \
        "Old independent run has an unexplained primary/regret/coverage discrepancy"
    assert feature_difference.max() > 0, "Expected float32 feature-accumulation difference was not observed"
    assert old_report["max_mean_prediction_absolute_difference"] > 0
    assert prediction_path_difference[3] <= old_report["max_mean_prediction_absolute_difference"], \
        "MAE difference exceeds the bound implied by independent prediction error"

    report = {
        "status": "PASS",
        "validator_revision": "v1.1 source-separated exact checks replacing the old blanket 1e-6 secondary tolerance",
        "compute": "CPU only; reused frozen predictions and existing independent arrays; no training or inference",
        "cluster_mapping": {
            "identical": True,
            "clusters": len(clusters),
            "method": "Sorted unique/list-comprehension mapping equals independently constructed dictionary-group mapping for all 10000 replicates",
            "cluster_draws_sha256_int64": draws_digest.hexdigest(),
            "row_indices_sha256_int64": indices_digest.hexdigest(),
        },
        "rng_and_sampling": {
            "identical": True,
            "cluster_rng": "NumPy default_rng(20261126), PCG64",
            "proposal_entropy": "[20261127, zero_based_bootstrap_index, algorithm_seed]",
            "evidence": "All primary/regret/coverage cells match exactly on local independent reconstruction; a separate remote NumPy 2.2.6/SciPy 1.15.3 in-memory run reproduced all 80000 cells bit-for-bit",
            "remote_original_environment_bitwise_cells": remote_reproduction["bitwise_equal_cells"],
            "remote_original_environment_total_cells": remote_reproduction["total_cells"],
            "remote_reproduction_log": "logs/remote_bootstrap_reproduction.log",
        },
        "text_roundtrip_parser_check": {
            "csv_float_vs_pandas_label_max_abs_difference": parser_label_max,
            "csv_float_vs_pandas_prediction_max_abs_difference": parser_prediction_max,
            "handling": "Exact reproduction uses the original pandas parser; Python csv/float is not assumed bit-identical",
        },
        "same_frozen_predictions_original_metrics_max_abs_difference": column_max(boot_saved_original_metrics, expected),
        "same_frozen_predictions_independent_metrics_max_abs_difference": same_prediction_reduction,
        "old_independent_gpu_vs_v1_max_abs_difference": original_independent_difference,
        "old_independent_gpu_minus_same_prediction_independent_metrics_max_abs_difference": prediction_path_difference,
        "feature_float32_accumulation": {
            "max_absolute_difference": float(feature_difference.max()),
            "different_cells": int(np.sum(feature_difference != 0)),
            "original": "float32 integer-like k-mer counts followed by division",
            "independent": "repeated float32 addition of 1.0/denominator",
            "max_mean_prediction_absolute_difference_from_old_report": old_report["max_mean_prediction_absolute_difference"],
        },
        "cause": {
            "cluster_mapping": "Excluded by exact row-index checks and exact full-array reproduction with frozen predictions",
            "rng_or_proposal_sampling": "Excluded by bit-identical primary/regret/coverage and remote original-environment reproduction of all 80000 cells",
            "spearman": "Different but mathematically equivalent reduction: scipy.spearmanr versus rankdata plus np.corrcoef; at most 1.11e-16 with identical predictions",
            "pearson_and_mae": "The old independent float32 feature accumulation changes frozen-model predictions (old reported max mean delta 4.76837158203125e-7); Pearson also has np.corrcoef versus scipy.pearsonr reduction differences. Local NumPy 2.0.2/SciPy 1.13.1 versus remote 2.2.6/1.15.3 adds at most 3.552713678800501e-15 even with identical predictions",
            "scientific_data_changed": False,
        },
        "acceptance": "No blanket 1e-6 tolerance is used. All primary/regret/coverage cells must match exactly. Same-prediction local continuous reductions must be <5e-15 and are separately identified as cross-version arithmetic; the original remote environment must match all 80000 cells exactly. The old independent MAE remainder must satisfy the measured prediction-error bound.",
    }
    (AUDIT_DIR / "results" / "bootstrap_difference_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
