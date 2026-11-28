#!/usr/bin/env python3
import argparse
import collections
import csv
import hashlib
import json
import random
import re
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse(mutant):
    return [(match.group(1), int(match.group(2)), match.group(3)) for match in (re.fullmatch(r"([A-Z])([0-9]+)([A-Z])", token) for token in mutant.split(":"))]


def ids(path):
    with Path(path).open(newline="") as handle:
        return sorted(row["mutant"] for row in csv.DictReader(handle))


def rebuild_pools(rows, positions, seed):
    grouped = collections.defaultdict(list)
    for mutant in rows:
        tokens = parse(mutant)
        if len(tokens) == 2:
            grouped[tuple(sorted((tokens[0][1], tokens[1][1])))].append(mutant)
    rng = random.Random(seed)
    pools = []
    for pair in sorted(grouped):
        pools.append({"pool_id": f"GB1_{pair[0]}_{pair[1]}", "position_pair": list(pair), "candidate_count": len(grouped[pair]), "mutants": sorted(rng.sample(sorted(grouped[pair]), 256))})
    return pools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    checks = {}
    checks["raw_sha256"] = file_hash(args.raw) == config["source_sha256"]
    checks["raw_bytes"] = Path(args.raw).stat().st_size == config["source_bytes"]
    ordered = sorted(range(228, 283), key=lambda position: hashlib.sha256(f"protein-scenario-v1:{position}".encode()).hexdigest())
    test_positions, dev_positions = sorted(ordered[:11]), sorted(ordered[11:16])
    checks["positions"] = test_positions == config["expected_test_positions"] and dev_positions == config["expected_dev_positions"]
    role_map = {position: "test" for position in test_positions}
    role_map.update({position: "dev" for position in dev_positions})
    role_map.update({position: "train" for position in ordered[16:]})
    role_ids = collections.defaultdict(list)
    counts = collections.defaultdict(collections.Counter)
    total = 0
    floor_count = 0
    bins = collections.Counter()
    reference = None
    with Path(args.raw).open(newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            mutant = row["mutant"]
            tokens = parse(mutant)
            if reference is None:
                reference = list(row["mutated_sequence"])
                for wild, position, _ in tokens:
                    reference[position - 1] = wild
            rebuilt = list(reference)
            for wild, position, replacement in tokens:
                assert reference[position - 1] == wild
                rebuilt[position - 1] = replacement
            assert "".join(rebuilt) == row["mutated_sequence"]
            label = float(row["DMS_score"])
            floor_count += int(label == config["lnw_floor"])
            bins[row["DMS_score_bin"]] += 1
            roles = {role_map[token[1]] for token in tokens}
            role = roles.pop() if len(roles) == 1 else "boundary_exclude"
            role_ids[role].append(mutant)
            counts[role]["single" if len(tokens) == 1 else "double"] += 1
            counts[role]["total"] += 1
    checks["all_sequences_reconstructed"] = True
    checks["row_count"] = total == 536962
    checks["floor_count"] = floor_count == 119422
    checks["bin_counts"] = bins == {"0": 155261, "1": 381701}
    normalized_counts = {role: {key: counts[role][key] for key in ("single", "double", "total")} for role in ("train", "dev", "test", "boundary_exclude")}
    checks["role_counts"] = normalized_counts == manifest["counts"]
    checks["train_ids"] = ids(data / "roles/train.csv") == sorted(role_ids["train"])
    checks["dev_ids"] = ids(data / "roles/dev.csv") == sorted(role_ids["dev"])
    checks["test_unlabeled_ids"] = ids(data / "roles/test_unlabeled.csv") == sorted(role_ids["test"])
    checks["boundary_ids"] = ids(data / "roles/boundary_exclude.csv") == sorted(role_ids["boundary_exclude"])
    checks["no_cross_role_exact_duplicates"] = len(set(role_ids["train"]) & set(role_ids["dev"]) | set(role_ids["train"]) & set(role_ids["test"]) | set(role_ids["dev"]) & set(role_ids["test"])) == 0
    checks["position_disjoint_hamming1_certificate"] = len(set(ordered[:16]) & set(ordered[16:])) == 0 and len(set(test_positions) & set(dev_positions)) == 0
    dev_rebuilt = rebuild_pools(role_ids["dev"], dev_positions, config["pool_seed"])
    test_rebuilt = rebuild_pools(role_ids["test"], test_positions, config["pool_seed"])
    checks["dev_pools_label_independent_rebuild"] = dev_rebuilt == json.loads((data / "pools/dev_pools.json").read_text())
    checks["test_pools_label_independent_rebuild"] = test_rebuilt == json.loads((data / "evaluator/test_pools.json").read_text())
    checks["pool_partition"] = len(test_rebuilt) == 55 and len({mutant for pool in test_rebuilt for mutant in pool["mutants"]}) == 14080
    file_checks = {}
    for relative, record in manifest["files"].items():
        path = data / relative
        file_checks[relative] = path.stat().st_size == record["bytes"] and file_hash(path) == record["sha256"]
    checks["manifest_file_hashes"] = all(file_checks.values())
    report = {"round": 1, "scope": "raw source, roles, sequences, label-independent pools and hashes; no model metrics", "checks": checks, "file_checks": file_checks, "counts": manifest["counts"], "floor_count": floor_count, "pass": all(checks.values())}
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": report["pass"], "checks": checks}, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
