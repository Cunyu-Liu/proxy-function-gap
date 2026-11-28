#!/usr/bin/env python3
import argparse
import collections
import csv
import hashlib
import json
import math
import random
from pathlib import Path

from gb1_common import parse_mutant, sha256_file, write_json

EXPECTED_COUNTS = {
    "train": {"single": 741, "double": 267404, "total": 268145},
    "dev": {"single": 95, "double": 3610, "total": 3705},
    "test": {"single": 209, "double": 19854, "total": 20063},
    "boundary_exclude": {"single": 0, "double": 245049, "total": 245049},
}


def write_rows(path, rows, labeled):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mutant", "DMS_score"] if labeled else ["mutant"])
        for mutant, label in rows:
            writer.writerow([mutant, format(label, ".17g")] if labeled else [mutant])


def build_pools(rows, positions, seed, pool_size):
    rng = random.Random(seed)
    by_pair = collections.defaultdict(list)
    for mutant, _ in rows:
        tokens = parse_mutant(mutant)
        if len(tokens) == 2:
            pair = tuple(sorted(token[1] for token in tokens))
            by_pair[pair].append(mutant)
    expected_pairs = {tuple(sorted((a, b))) for i, a in enumerate(positions) for b in positions[i + 1:]}
    if set(by_pair) != expected_pairs:
        raise RuntimeError("Position-pair coverage mismatch")
    pools = []
    for pair in sorted(by_pair):
        candidates = sorted(by_pair[pair])
        if len(candidates) not in (360, 361):
            raise RuntimeError(f"Unexpected pair coverage {pair}: {len(candidates)}")
        selected = sorted(rng.sample(candidates, pool_size))
        pools.append({"pool_id": f"GB1_{pair[0]}_{pair[1]}", "position_pair": list(pair), "candidate_count": len(candidates), "mutants": selected})
    return pools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    raw = Path(args.raw)
    output = Path(args.output)
    if sha256_file(raw) != config["source_sha256"] or raw.stat().st_size != config["source_bytes"]:
        raise RuntimeError("Raw input hash or size mismatch")
    ordered_positions = sorted(range(228, 283), key=lambda p: hashlib.sha256(f"{config['position_salt']}{p}".encode()).hexdigest())
    test_positions = sorted(ordered_positions[:11])
    dev_positions = sorted(ordered_positions[11:16])
    train_positions = sorted(ordered_positions[16:])
    if test_positions != config["expected_test_positions"] or dev_positions != config["expected_dev_positions"]:
        raise RuntimeError("Frozen position assignment mismatch")
    role_for_position = {p: "test" for p in test_positions}
    role_for_position.update({p: "dev" for p in dev_positions})
    role_for_position.update({p: "train" for p in train_positions})
    rows_by_role = {role: [] for role in EXPECTED_COUNTS}
    seen = set()
    counts = {role: collections.Counter() for role in EXPECTED_COUNTS}
    bin_counts = collections.Counter()
    reference = None
    floor_count = 0
    nonfinite = 0
    total = 0
    with raw.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"}
        if not required.issubset(reader.fieldnames):
            raise RuntimeError(f"Missing columns: {required - set(reader.fieldnames)}")
        for row in reader:
            total += 1
            mutant = row["mutant"]
            if mutant in seen:
                raise RuntimeError(f"Duplicate mutant: {mutant}")
            seen.add(mutant)
            tokens = parse_mutant(mutant)
            sequence = row["mutated_sequence"]
            if len(sequence) != 448:
                raise RuntimeError("Unexpected sequence length")
            if reference is None:
                reference = list(sequence)
                for wild, position, _ in tokens:
                    reference[position - 1] = wild
                reference = "".join(reference)
            reconstructed = list(reference)
            for wild, position, replacement in tokens:
                if reference[position - 1] != wild or replacement == wild:
                    raise RuntimeError(f"Invalid mutation against reference: {mutant}")
                reconstructed[position - 1] = replacement
            if "".join(reconstructed) != sequence:
                raise RuntimeError(f"Sequence reconstruction failure: {mutant}")
            label = float(row["DMS_score"])
            if not math.isfinite(label):
                nonfinite += 1
                raise RuntimeError(f"Nonfinite label: {mutant}")
            floor_count += int(label == config["lnw_floor"])
            bin_counts[row["DMS_score_bin"]] += 1
            positions = {token[1] for token in tokens}
            roles = {role_for_position[p] for p in positions}
            role = roles.pop() if len(roles) == 1 else "boundary_exclude"
            order_name = "single" if len(tokens) == 1 else "double" if len(tokens) == 2 else "invalid"
            if order_name == "invalid" or not positions.issubset(set(range(228, 283))):
                raise RuntimeError(f"Unexpected mutation order or position: {mutant}")
            rows_by_role[role].append((mutant, label))
            counts[role][order_name] += 1
            counts[role]["total"] += 1
    normalized_counts = {role: {key: counts[role][key] for key in ("single", "double", "total")} for role in EXPECTED_COUNTS}
    if total != 536962 or normalized_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Role counts mismatch: {counts}")
    if nonfinite or floor_count != 119422 or bin_counts != {"0": 155261, "1": 381701}:
        raise RuntimeError("Label distribution mismatch")
    for role, rows in rows_by_role.items():
        rows_by_role[role] = sorted(rows)
    write_rows(output / "roles/train.csv", rows_by_role["train"], True)
    write_rows(output / "roles/dev.csv", rows_by_role["dev"], True)
    write_rows(output / "roles/test_unlabeled.csv", rows_by_role["test"], False)
    write_rows(output / "roles/boundary_exclude.csv", rows_by_role["boundary_exclude"], False)
    write_rows(output / "evaluator/test_labels.csv", rows_by_role["test"], True)
    dev_pools = build_pools(rows_by_role["dev"], dev_positions, config["pool_seed"], config["pool_size"])
    test_pools = build_pools(rows_by_role["test"], test_positions, config["pool_seed"], config["pool_size"])
    if len(dev_pools) != 10 or len(test_pools) != 55:
        raise RuntimeError("Pool count mismatch")
    write_json(output / "pools/dev_pools.json", dev_pools)
    write_json(output / "evaluator/test_pools.json", test_pools)
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema_version": config["schema_version"],
        "assay": config["assay"],
        "label": "lnW",
        "label_semantics": config["label_semantics"],
        "raw": {"path": str(raw), "bytes": raw.stat().st_size, "sha256": sha256_file(raw)},
        "reference_sequence_length": len(reference),
        "reference_sequence_sha256": hashlib.sha256(reference.encode()).hexdigest(),
        "positions": {"train": train_positions, "dev": dev_positions, "test": test_positions},
        "counts": EXPECTED_COUNTS,
        "total_rows": total,
        "unique_mutants": len(seen),
        "sequence_reconstruction_failures": 0,
        "nonfinite_labels": nonfinite,
        "lnw_floor_count": floor_count,
        "lnw_floor_fraction": floor_count / total,
        "dms_score_bin_counts": dict(bin_counts),
        "cross_role_exact_duplicates": 0,
        "cross_role_hamming1": 0,
        "hamming1_certificate": "Train/dev/test mutation-position sets are pairwise disjoint; any cross-role pair differs at every mutated position, so Hamming-1 is impossible.",
        "dev_pools": {"count": len(dev_pools), "size": config["pool_size"]},
        "test_pools": {"count": len(test_pools), "size": config["pool_size"], "unique_variants": sum(len(p["mutants"]) for p in test_pools)},
        "files": files,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("total_rows", "counts", "lnw_floor_count", "dev_pools", "test_pools")}, indent=2))


if __name__ == "__main__":
    main()
