#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def close(actual, expected, label):
    if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"{label}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"{label}: {actual} != {expected}")


def frozen_prefix(seed):
    generator = random.Random(seed)
    answer = []
    for draw in range(16):
        u = generator.random()
        index = 63
        for candidate in range(64):
            if u < (candidate + 1) / 64:
                index = candidate
                break
        answer.append(index)
    return answer


def independent_position(context, pool, seed, cap, cap_size):
    child = np.random.SeedSequence([20261125], spawn_key=(context, pool, seed, cap))
    return int(np.random.Generator(np.random.PCG64(child)).integers(cap_size))


def interval_and_p(values, seed):
    if not any(values):
        return 0.0, [0.0, 0.0], [0.0, 0.0], 1.0
    generator = np.random.Generator(np.random.PCG64(seed))
    cluster_values = np.asarray(values, dtype=np.float64)
    indices = generator.integers(0, 16, size=(10000, 16))
    estimates = cluster_values[indices].mean(axis=1)
    low_tail = (1 + np.sum(estimates <= 0)) / 10001
    high_tail = (1 + np.sum(estimates >= 0)) / 10001
    return (
        sum(values) / 16,
        list(np.percentile(estimates, [2.5, 97.5])),
        list(np.percentile(estimates, [0.125, 99.875])),
        min(1.0, 2 * min(low_tail, high_tail)),
    )


def corrected(rows):
    ordered = sorted(rows, key=lambda row: row["p"])
    previous = 0.0
    for index, row in enumerate(ordered):
        previous = max(previous, min(1.0, (20 - index) * row["p"]))
        row["expected_holm"] = previous
    for row in rows:
        row["expected_bonferroni"] = min(1.0, 20 * row["p"])


def verify(config):
    pool_root = Path(config["pool_root"])
    source_root = Path(config["source_root"])
    output_root = Path(config["output_root"])
    with (output_root / "analysis/paired_records.tsv").open(newline="") as handle:
        delivered_pairs = list(csv.DictReader(handle, delimiter="\t"))
    pair_lookup = {(row["context"], row["pool_id"], int(row["seed"]), int(row["cap"])): row for row in delivered_pairs}
    if len(pair_lookup) != 1600 or len(delivered_pairs) != 1600:
        raise AssertionError("delivered pair identities are not unique")
    values = defaultdict(list)
    mean_values = defaultdict(list)
    raw_selection_checks = 0
    cap1_checks = 0
    for ci, context in enumerate(config["contexts"]):
        with gzip.open(pool_root / context / "evaluator/labels.tsv.gz", "rt", newline="") as handle:
            outcome = {row["sequence_id"]: float(row["measured_rl"]) for row in csv.DictReader(handle, delimiter="\t")}
        for pi in range(16):
            stem = f"castillo_{context}_{pi:03d}"
            with gzip.open(pool_root / context / "learner" / f"{stem}.jsonl.gz", "rt") as handle:
                candidates = [json.loads(line) for line in handle if line.strip()]
            identifiers = [row["sequence_id"] for row in candidates]
            best_outcome = max(outcome[identifier] for identifier in identifiers)
            proxy = {
                row["sequence_id"]: float(row["proxy_mean"])
                for row in json.loads((source_root / context / "cnn_scores" / f"{stem}.json").read_text())["records"]
            }
            original = json.loads((source_root / context / "evaluation" / f"{stem}.json").read_text())["records"]
            frozen = {(row["strategy"], row["proposal_cap"], row["seed"]): row for row in original}
            for si, original_seed in enumerate(config["seeds"]):
                proposals = frozen_prefix(original_seed)
                for ki, cap in enumerate(config["caps"]):
                    current = proposals[:cap]
                    best_identifier = max(current, key=lambda index: (proxy[identifiers[index]], -current.index(index)))
                    best_identifier = identifiers[best_identifier]
                    random1_identifier = identifiers[current[0]]
                    random_n_identifier = identifiers[current[independent_position(ci, pi, si, ki, cap)]]
                    regrets = {
                        "bon": best_outcome - outcome[best_identifier],
                        "random1": best_outcome - outcome[random1_identifier],
                        "random_n": best_outcome - outcome[random_n_identifier],
                    }
                    for strategy, identifier in (("bon", best_identifier), ("random", random1_identifier)):
                        old = frozen[(strategy, cap, original_seed)]
                        if old["selected_id"] != identifier:
                            raise AssertionError(f"independent selected ID mismatch: {stem}/{strategy}/{cap}/{original_seed}")
                        close(old["regret"], regrets["bon" if strategy == "bon" else "random1"], "old regret")
                        raw_selection_checks += 1
                    delivered = pair_lookup[(context, stem, original_seed, cap)]
                    for field, identifier in (("bon_selected_id", best_identifier), ("random1_selected_id", random1_identifier), ("random_n_selected_id", random_n_identifier)):
                        if delivered[field] != identifier:
                            raise AssertionError(f"delivered selected ID mismatch: {field}/{stem}/{cap}/{original_seed}")
                    for strategy in ("bon", "random1", "random_n"):
                        close(float(delivered[f"{strategy}_regret"]), regrets[strategy], f"delivered {strategy} regret")
                    effects = (
                        regrets["bon"] - regrets["random1"],
                        regrets["bon"] - regrets["random_n"],
                        regrets["random_n"] - regrets["random1"],
                    )
                    close(effects[0], effects[1] + effects[2], "seed decomposition")
                    for name, value in zip(config["families"], effects):
                        values[(name, ci, ki, pi)].append(value)
                        close(float(delivered[name]), value, f"delivered {name}")
                    for strategy in ("bon", "random1", "random_n"):
                        mean_values[(strategy, ci, ki, pi)].append(regrets[strategy])
                    if cap == 1:
                        if random_n_identifier != random1_identifier:
                            raise AssertionError("cap1 identity")
                        cap1_checks += 1
    if raw_selection_checks != 3200 or cap1_checks != 320:
        raise AssertionError("independent raw check count")
    with (output_root / "analysis/pool_effects.tsv").open(newline="") as handle:
        delivered_pools = list(csv.DictReader(handle, delimiter="\t"))
    if len(delivered_pools) != 320:
        raise AssertionError("pool output count")
    for row in delivered_pools:
        ci = config["contexts"].index(row["context"])
        ki = config["caps"].index(int(row["cap"]))
        pi = int(row["pool_id"][-3:])
        for name in config["families"]:
            close(float(row[name]), sum(values[(name, ci, ki, pi)]) / 5, f"pool {name}")
        for strategy in ("bon", "random1", "random_n"):
            close(float(row[f"{strategy}_regret"]), sum(mean_values[(strategy, ci, ki, pi)]) / 5, f"pool {strategy} mean")
    expected = json.loads((output_root / "analysis/results.json").read_text())
    for ci, context in enumerate(config["contexts"]):
        for ki, cap in enumerate(config["caps"]):
            context_row = next(row for row in expected["context_summaries"] if row["context"] == context and row["cap"] == cap)
            for name in config["families"]:
                seed = 20261124 + 100 * ci + ki if name == "bon_minus_random1" else 20261125 + (10000 if name == "random_n_minus_random1" else 0) + 100 * ci + ki
                pool_effects = [sum(values[(name, ci, ki, pi)]) / 5 for pi in range(16)]
                effect, nominal, bonferroni, p_value = interval_and_p(pool_effects, seed)
                row = next(row for row in expected["families"][name] if row["context"] == context and row["cap"] == cap)
                close(row["effect"], effect, f"independent {name} effect")
                for actual, wanted in zip(row["nominal_95_ci"], nominal):
                    close(actual, wanted, f"independent {name} nominal")
                for actual, wanted in zip(row["bonferroni_20_ci"], bonferroni):
                    close(actual, wanted, f"independent {name} Bonferroni CI")
                close(row["p"], p_value, f"independent {name} p")
                close(row["bootstrap_seed"], seed, "bootstrap seed")
                close(context_row[name], effect, f"context summary {name}")
            for strategy in ("bon", "random1", "random_n"):
                aggregate = sum(sum(mean_values[(strategy, ci, ki, pi)]) / 5 for pi in range(16)) / 16
                close(context_row[f"{strategy}_regret"], aggregate, f"context mean {strategy}")
    for name in config["families"]:
        rows = expected["families"][name]
        if len(rows) != 20:
            raise AssertionError("family size")
        corrected(rows)
        for row in rows:
            close(row["holm_p"], row.pop("expected_holm"), "Holm")
            close(row["bonferroni_p"], row.pop("expected_bonferroni"), "Bonferroni p")
            if row["cap"] == 1:
                close(row["effect"], 0, "cap1 zero")
                close(row["p"], 1, "cap1 p")
    return {
        "round": 2,
        "status": "PASS",
        "independent_implementation": True,
        "producer_module_imported": False,
        "original_selected_id_regret_checks": raw_selection_checks,
        "random_n_selected_id_regret_checks": len(delivered_pairs),
        "cap1_identity_checks": cap1_checks,
        "pool_effect_checks": len(delivered_pools),
        "all20_families_verified": len(config["families"]),
        "bootstrap_replicates_per_nonzero_row": 10000,
        "cpu_only": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "frozen_config.json")
    parser.add_argument("--round", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = verify(config)
    result["round"] = args.round
    output = Path(config["output_root"]) / "validation" / f"independent_round{args.round}.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
