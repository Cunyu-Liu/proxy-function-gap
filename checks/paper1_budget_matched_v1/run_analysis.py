#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "frozen_config.json"
FAMILY_NAMES = ("bon_minus_random1", "bon_minus_random_n", "random_n_minus_random1")


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_nested_prefix(pool_size: int, cap: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    probability = 1.0 / pool_size
    prefix = []
    for _ in range(cap):
        threshold = rng.random()
        cumulative = 0.0
        selected = pool_size - 1
        for index in range(pool_size):
            cumulative += probability
            if threshold < cumulative:
                selected = index
                break
        prefix.append(selected)
    return prefix


def choose_bon_position(prefix: list[int], scores: dict[int, float]) -> int:
    best_position = 0
    best_score = scores[prefix[0]]
    for position in range(1, len(prefix)):
        score = scores[prefix[position]]
        if score > best_score:
            best_position = position
            best_score = score
    return best_position


def random_n_position(entropy: int, spawn_key: tuple[int, int, int, int], cap: int) -> int:
    sequence = np.random.SeedSequence(entropy=entropy, spawn_key=spawn_key)
    rng = np.random.Generator(np.random.PCG64(sequence))
    return int(rng.integers(0, cap))


def decompose(bon: float, random_n: float, random1: float) -> tuple[float, float, float]:
    return bon - random1, bon - random_n, random_n - random1


def bootstrap_p(samples: np.ndarray) -> float:
    lower = (1 + int(np.count_nonzero(samples <= 0))) / (len(samples) + 1)
    upper = (1 + int(np.count_nonzero(samples >= 0))) / (len(samples) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def bootstrap_stats(values: np.ndarray, seed: int, replicates: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    effect = float(values.mean())
    if np.all(values == 0):
        return {
            "effect": effect,
            "nominal_95_ci": [0.0, 0.0],
            "bonferroni_20_ci": [0.0, 0.0],
            "p": 1.0,
        }
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "effect": effect,
        "nominal_95_ci": np.percentile(samples, [2.5, 97.5]).tolist(),
        "bonferroni_20_ci": np.percentile(samples, [0.125, 99.875]).tolist(),
        "p": bootstrap_p(samples),
    }


def add_multiplicity(rows: list[dict]) -> None:
    if len(rows) != 20:
        raise ValueError(f"all-20 family required, received {len(rows)}")
    order = sorted(range(len(rows)), key=lambda index: (rows[index]["p"], index))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["p"]))
        rows[index]["holm_p"] = running
    for row in rows:
        row["bonferroni_p"] = min(1.0, len(rows) * row["p"])


def tsv_text(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def build(config: dict) -> tuple[dict, list[dict], list[dict], dict]:
    pool_root = Path(config["pool_root"])
    source_root = Path(config["source_root"])
    contexts = config["contexts"]
    caps = config["caps"]
    seeds = config["seeds"]
    original_summary_path = source_root / "strict_h1_paired_summary.json"
    original_summary = read_json(original_summary_path)
    if original_summary["new_blind_test"] is not False:
        raise ValueError("frozen source evidence boundary mismatch")
    source_hashes = {str(original_summary_path): sha256(original_summary_path)}
    paired_records = []
    source_checks = 0
    cap1_matches = 0
    pool_groups = {}
    for context_index, context in enumerate(contexts):
        for pool_index in range(config["pool_count"]):
            pool_id = f"castillo_{context}_{pool_index:03d}"
            paths = {
                "learner": pool_root / context / "learner" / f"{pool_id}.jsonl.gz",
                "labels": pool_root / context / "evaluator" / "labels.tsv.gz",
                "scores": source_root / context / "cnn_scores" / f"{pool_id}.json",
                "selection": source_root / context / "selection" / f"{pool_id}.json",
                "evaluation": source_root / context / "evaluation" / f"{pool_id}.json",
            }
            for path in paths.values():
                source_hashes[str(path)] = sha256(path)
            with gzip.open(paths["learner"], "rt") as handle:
                learner = [json.loads(line) for line in handle if line.strip()]
            if len(learner) != config["pool_size"]:
                raise ValueError(f"pool size mismatch: {pool_id}")
            if any(row["pool_id"] != pool_id or row["q_value"] != 1.0 / config["pool_size"] for row in learner):
                raise ValueError(f"pool identity or uniform proposal mismatch: {pool_id}")
            sequence_ids = [row["sequence_id"] for row in learner]
            if len(set(sequence_ids)) != len(sequence_ids):
                raise ValueError(f"duplicate candidate ID: {pool_id}")
            with gzip.open(paths["labels"], "rt", newline="") as handle:
                labels = {row["sequence_id"]: float(row["measured_rl"]) for row in csv.DictReader(handle, delimiter="\t")}
            pool_max = max(labels[sequence_id] for sequence_id in sequence_ids)
            score_payload = read_json(paths["scores"])
            score_rows = score_payload["records"]
            if [row["sequence_id"] for row in score_rows] != sequence_ids:
                raise ValueError(f"prediction order mismatch: {pool_id}")
            scores = {index: float(row["proxy_mean"]) for index, row in enumerate(score_rows)}
            if score_payload["metadata"]["outer_labels_read"] is not False:
                raise ValueError(f"learner label boundary mismatch: {pool_id}")
            selection = read_json(paths["selection"])
            evaluation = read_json(paths["evaluation"])
            if evaluation["selection_config_sha256"] != selection["selection_config_sha256"]:
                raise ValueError(f"selection/evaluation hash mismatch: {pool_id}")
            if evaluation["labels_returned_to_learner"] is not False:
                raise ValueError(f"evaluator label boundary mismatch: {pool_id}")
            evaluated = {
                (row["strategy"], row["proposal_cap"], row["seed"]): row
                for row in evaluation["records"]
            }
            expected_keys = {(strategy, cap, seed) for strategy in ("random", "bon") for cap in caps for seed in seeds}
            if set(evaluated) != expected_keys or len(evaluation["records"]) != len(expected_keys):
                raise ValueError(f"evaluation grid mismatch: {pool_id}")
            prefixes = {seed: draw_nested_prefix(config["pool_size"], max(caps), seed) for seed in seeds}
            for cap_index, cap in enumerate(caps):
                group = []
                for seed_index, seed in enumerate(seeds):
                    prefix = prefixes[seed][:cap]
                    original_random = evaluated[("random", cap, seed)]
                    original_bon = evaluated[("bon", cap, seed)]
                    if original_random["selected_index"] != prefix[0]:
                        raise ValueError(f"Random prefix mismatch: {pool_id}/{cap}/{seed}")
                    bon_position = choose_bon_position(prefix, scores)
                    bon_index = prefix[bon_position]
                    if original_bon["selected_index"] != bon_index:
                        raise ValueError(f"BoN prefix/tie mismatch: {pool_id}/{cap}/{seed}")
                    random1_index = prefix[0]
                    random1_regret = float(pool_max - labels[sequence_ids[random1_index]])
                    bon_regret = float(pool_max - labels[sequence_ids[bon_index]])
                    if random1_regret != original_random["regret"] or bon_regret != original_bon["regret"]:
                        raise ValueError(f"frozen regret mismatch: {pool_id}/{cap}/{seed}")
                    if original_random["selected_id"] != sequence_ids[random1_index] or original_bon["selected_id"] != sequence_ids[bon_index]:
                        raise ValueError(f"frozen selected ID mismatch: {pool_id}/{cap}/{seed}")
                    source_checks += 2
                    position = random_n_position(
                        config["random_n"]["entropy"],
                        (context_index, pool_index, seed_index, cap_index),
                        cap,
                    )
                    random_n_index = prefix[position]
                    random_n_regret = float(pool_max - labels[sequence_ids[random_n_index]])
                    original, ranking, budget = decompose(bon_regret, random_n_regret, random1_regret)
                    if cap == 1:
                        if random_n_index != random1_index or random_n_regret != random1_regret:
                            raise ValueError(f"cap1 structural identity failed: {pool_id}/{seed}")
                        cap1_matches += 1
                    row = {
                        "context": context,
                        "pool_id": pool_id,
                        "pool_index": pool_index,
                        "seed": seed,
                        "seed_index": seed_index,
                        "cap": cap,
                        "cap_index": cap_index,
                        "random1_selected_index": random1_index,
                        "random1_selected_id": sequence_ids[random1_index],
                        "random1_regret": random1_regret,
                        "bon_selected_index": bon_index,
                        "bon_selected_id": sequence_ids[bon_index],
                        "bon_regret": bon_regret,
                        "random_n_position": position,
                        "random_n_selected_index": random_n_index,
                        "random_n_selected_id": sequence_ids[random_n_index],
                        "random_n_regret": random_n_regret,
                        "bon_minus_random1": original,
                        "bon_minus_random_n": ranking,
                        "random_n_minus_random1": budget,
                        "prefix_sha256": hashlib.sha256(",".join(map(str, prefix)).encode()).hexdigest(),
                    }
                    paired_records.append(row)
                    group.append(row)
                pool_groups[(context_index, pool_index, cap_index)] = group
    if source_checks != 3200 or len(paired_records) != 1600 or cap1_matches != 320:
        raise ValueError("raw grid count mismatch")
    pool_rows = []
    for (context_index, pool_index, cap_index), group in sorted(pool_groups.items()):
        row = {
            "context": contexts[context_index],
            "pool_id": group[0]["pool_id"],
            "cap": caps[cap_index],
        }
        for field in ("random1_regret", "bon_regret", "random_n_regret", *FAMILY_NAMES):
            row[field] = float(np.mean([item[field] for item in group]))
        original, ranking, budget = decompose(row["bon_regret"], row["random_n_regret"], row["random1_regret"])
        for field, value in zip(FAMILY_NAMES, (original, ranking, budget)):
            if not np.isclose(row[field], value, rtol=0, atol=1e-12):
                raise ValueError("pool decomposition identity failed")
        pool_rows.append(row)
    context_summaries = []
    families = {name: [] for name in FAMILY_NAMES}
    for context_index, context in enumerate(contexts):
        source_pressure = original_summary["contexts"][context]["pressure"]
        for cap_index, cap in enumerate(caps):
            selected = [row for row in pool_rows if row["context"] == context and row["cap"] == cap]
            if len(selected) != config["pool_count"]:
                raise ValueError("pool summary count mismatch")
            summary = {"context": context, "cap": cap, "outer_pools": len(selected)}
            for field in ("random1_regret", "bon_regret", "random_n_regret", *FAMILY_NAMES):
                summary[field] = float(np.mean([row[field] for row in selected]))
            context_summaries.append(summary)
            source_row = next(row for row in source_pressure["rows"] if row["proposal_cap"] == cap)
            source_pool = {row["pool_id"]: row for row in source_pressure["per_pool"] if row["proposal_cap"] == cap}
            for row in selected:
                if not np.isclose(row["bon_minus_random1"], source_pool[row["pool_id"]]["paired_bon_minus_random"], rtol=0, atol=1e-12):
                    raise ValueError("original per-pool effect mismatch")
            for family_index, family in enumerate(FAMILY_NAMES):
                values = np.asarray([row[family] for row in selected], dtype=np.float64)
                if family == "bon_minus_random1":
                    seed = config["bootstrap"]["original_base_seed"] + 100 * context_index + cap_index
                else:
                    offset = 0 if family == "bon_minus_random_n" else config["bootstrap"]["candidate_budget_offset"]
                    seed = config["bootstrap"]["new_base_seed"] + offset + 100 * context_index + cap_index
                stats = bootstrap_stats(values, seed, config["bootstrap"]["replicates"])
                row = {
                    "context": context,
                    "cap": cap,
                    "outer_pools": len(values),
                    "family": family,
                    **stats,
                    "bootstrap_seed": seed,
                }
                if family == "bon_minus_random1":
                    if not np.isclose(stats["effect"], source_row["paired_bon_minus_random"], rtol=0, atol=1e-12):
                        raise ValueError("original aggregate effect mismatch")
                    if not np.allclose(stats["nominal_95_ci"], source_row["paired_cluster_bootstrap_95_percent_interval"], rtol=0, atol=1e-12):
                        raise ValueError("original bootstrap interval mismatch")
                families[family].append(row)
    for rows in families.values():
        add_multiplicity(rows)
    results = {
        "schema_version": "1.0",
        "analysis": config["analysis"],
        "status": "PASS",
        "protocol": config["protocol"],
        "evidence_boundary": "Retrospective sensitivity on previously inspected outer contexts; not a new blind test.",
        "new_blind_test": False,
        "execution": config["execution"],
        "grid": {"contexts": len(contexts), "pools_per_context": config["pool_count"], "seeds": seeds, "caps": caps},
        "random_n": config["random_n"],
        "bootstrap": config["bootstrap"],
        "effect_signs": {
            "bon_minus_random1": "BoN regret minus original one-candidate Random regret; negative favors BoN",
            "bon_minus_random_n": "BoN(N) regret minus same-prefix Random(N) regret; negative favors proxy ranking at equal budget",
            "random_n_minus_random1": "Random(N) regret minus Random(1) regret; negative favors the Random(N) realization",
        },
        "multiplicity": "Separate complete all-20 Holm and Bonferroni families, each including cap1 structural zeros with p=1",
        "context_summaries": context_summaries,
        "families": families,
        "source_sha256": source_hashes,
    }
    audit = {
        "status": "PASS",
        "original_random_bon_selection_id_index_regret_checks": source_checks,
        "original_grid_records": 3200,
        "original_paired_rows": 1600,
        "new_random_n_records": len(paired_records),
        "cap1_random_n_id_regret_matches": cap1_matches,
        "pool_seed_cap_decomposition_checks": len(paired_records),
        "pool_rows": len(pool_rows),
        "family_rows": {name: len(rows) for name, rows in families.items()},
        "contains_sequence_strings_in_remote_audit": False,
        "remote_audit_contains_selected_ids_and_regret": True,
        "local_export_contains_sequence_level_labels": False,
        "cpu_only": True,
        "model_training_or_inference": False,
    }
    return results, paired_records, pool_rows, audit


PAIR_FIELDS = [
    "context", "pool_id", "pool_index", "seed", "seed_index", "cap", "cap_index",
    "random1_selected_index", "random1_selected_id", "random1_regret",
    "bon_selected_index", "bon_selected_id", "bon_regret",
    "random_n_position", "random_n_selected_index", "random_n_selected_id", "random_n_regret",
    "bon_minus_random1", "bon_minus_random_n", "random_n_minus_random1", "prefix_sha256",
]
POOL_FIELDS = ["context", "pool_id", "cap", "random1_regret", "bon_regret", "random_n_regret", *FAMILY_NAMES]


def encoded_outputs(config: dict) -> dict[str, str]:
    results, paired, pools, audit = build(config)
    return {
        "analysis/results.json": json.dumps(results, indent=2) + "\n",
        "analysis/paired_records.tsv": tsv_text(paired, PAIR_FIELDS),
        "analysis/pool_effects.tsv": tsv_text(pools, POOL_FIELDS),
        "analysis/selection_audit.json": json.dumps(audit, indent=2) + "\n",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    outputs = encoded_outputs(config)
    root = Path(config["output_root"])
    for relative, encoded in outputs.items():
        path = root / relative
        if args.check:
            if path.read_text() != encoded:
                raise ValueError(f"output is not reproducible: {path}")
        else:
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded)
    print(json.dumps({"status": "PASS", "check": args.check, "outputs": len(outputs), "cpu_only": True}))


if __name__ == "__main__":
    main()
