import hashlib
import json
from pathlib import Path

import numpy as np

from .production_selection import FreshMemberScorer
from .trainer.checkpoint import atomic_json_write
from .trainer.data import sha256_file


def sequence_features(sequence):
    return {
        "gc": (sequence.count("G") + sequence.count("C")) / len(sequence),
        "uAUG": "ATG" in sequence,
        "longest_poly_u": max((len(part) for part in sequence.replace("A", " ").replace("C", " ").replace("G", " ").split()), default=0),
    }


def score_pool(pool, inference_config, output):
    members = [FreshMemberScorer(member, pool, inference_config) for member in inference_config.members]
    rows = []
    for index, sequence_id in enumerate(pool.sequence_ids):
        values = np.asarray([member.score_index(index) for member in members])
        rows.append({
            "sequence_id": sequence_id,
            "utr50": pool.utr50[index],
            "proxy_mean": float(values.mean()),
            "proxy_std": float(values.std(ddof=1)),
            "proxy_range": float(values.max() - values.min()),
            "support": float(pool.support[index]),
            **sequence_features(pool.utr50[index]),
        })
    metadata = {"pool_id": pool.pool_id, "actual_cuda_inference": True, "member_gpu": [member.gpu for member in members], "member_item_forwards": 3 * len(pool), "outer_labels_read": False}
    atomic_json_write({"metadata": metadata, "records": rows}, output)
    return {"pool_id": pool.pool_id, "records": len(rows), "outer_labels_read": False}


def _labels(pool_dir):
    import csv
    import gzip
    with gzip.open(Path(pool_dir) / "evaluator" / "labels.tsv.gz", "rt", newline="") as handle:
        return {row["sequence_id"]: float(row["measured_rl"]) for row in csv.DictReader(handle, delimiter="\t")}


def _bootstrap(values):
    if not values:
        return None
    values = np.asarray(values)
    rng = np.random.default_rng(20261124)
    samples = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(10000)]
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def evaluate_mechanism(pool_dir, pool_id, scores_path, selection_path, output):
    scores_payload = json.loads(Path(scores_path).read_text())
    if scores_payload["metadata"]["outer_labels_read"] is not False or scores_payload["metadata"]["actual_cuda_inference"] is not True:
        raise ValueError("invalid mechanism score metadata")
    scores = {row["sequence_id"]: row for row in scores_payload["records"]}
    labels = _labels(pool_dir)
    for row in scores.values():
        row["error"] = row["proxy_mean"] - labels[row["sequence_id"]]
        row["abs_error"] = abs(row["error"])
    selection = json.loads(Path(selection_path).read_text())
    pairs = []
    for row in selection["records"]:
        if row["strategy"] != "bon" or row["proposal_cap"] != 16:
            continue
        selected = scores[row["selected_id"]]
        alternatives = [candidate for candidate in scores.values() if candidate["sequence_id"] != selected["sequence_id"]]
        matched = min(alternatives, key=lambda candidate: (abs(candidate["proxy_mean"] - selected["proxy_mean"]), hashlib.sha256(candidate["sequence_id"].encode()).hexdigest()))
        pairs.append({"optimized_error": selected["error"], "matched_error": matched["error"], "optimized_abs_error": selected["abs_error"], "matched_abs_error": matched["abs_error"], "proxy_gap": abs(selected["proxy_mean"] - matched["proxy_mean"])})
    spreads = np.asarray([row["proxy_std"] for row in scores.values()])
    errors = np.asarray([row["abs_error"] for row in scores.values()])
    spread_cut = float(np.quantile(spreads, 0.25))
    error_cut = float(np.quantile(errors, 0.75))
    low = spreads <= spread_cut
    high = errors >= error_cut
    family = {}
    for name, mask in {
        "uAUG": np.asarray([row["uAUG"] for row in scores.values()]),
        "high_gc_q4": np.asarray([row["gc"] >= np.quantile([x["gc"] for x in scores.values()], 0.75) for row in scores.values()]),
        "long_poly_u_q4": np.asarray([row["longest_poly_u"] >= np.quantile([x["longest_poly_u"] for x in scores.values()], 0.75) for row in scores.values()]),
        "low_support_q1": np.asarray([row["support"] <= np.quantile([x["support"] for x in scores.values()], 0.25) for row in scores.values()]),
        "low_spread_q1": low,
    }.items():
        family[name] = {"count": int(mask.sum()), "mean_error": float(errors[mask].mean()) if mask.any() else None, "mean_bias": float(np.asarray([row["error"] for row in scores.values()])[mask].mean()) if mask.any() else None}
    report = {
        "pool_id": pool_id,
        "candidate_count": len(scores),
        "score_matched": {
            "pairs": len(pairs),
            "mean_proxy_gap": float(np.mean([row["proxy_gap"] for row in pairs])),
            "optimized_mean_bias": float(np.mean([row["optimized_error"] for row in pairs])),
            "matched_mean_bias": float(np.mean([row["matched_error"] for row in pairs])),
            "optimized_mean_abs_error": float(np.mean([row["optimized_abs_error"] for row in pairs])),
            "matched_mean_abs_error": float(np.mean([row["matched_abs_error"] for row in pairs])),
            "abs_error_difference": float(np.mean([row["optimized_abs_error"] - row["matched_abs_error"] for row in pairs])),
        },
        "common_proxy": {"low_spread_cut": spread_cut, "high_error_cut": error_cut, "low_spread_high_error_count": int((low & high).sum()), "low_spread_high_error_rate": float((low & high).mean()), "high_error_rate": float(high.mean()), "low_spread_mean_error": float(errors[low].mean()), "other_spread_mean_error": float(errors[~low].mean())},
        "sequence_families": family,
        "labels_returned_to_learner": False,
    }
    atomic_json_write(report, output)
    return report
