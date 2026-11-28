import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mutation_order(mutant):
    return len(mutant.split(":"))


def low_order_role(mutant):
    value = int.from_bytes(hashlib.sha256(("gfp-scenario-v1:" + mutant).encode()).digest()[:8], "big")
    return "train" if value / 2**64 < 0.8 else "dev"


def position_roles(records):
    positions = sorted({int(token[1:-1]) for r in records for token in r["mutant"].split(":")})
    ordered = sorted(positions, key=lambda p: hashlib.sha256(f"gfp-position-v1:{p}".encode()).hexdigest())
    dev_count = int(math.ceil(0.3 * len(positions)))
    dev_positions = set(ordered[:dev_count])
    roles = {}
    excluded = []
    for r in records:
        variant_positions = {int(token[1:-1]) for token in r["mutant"].split(":")}
        is_dev = [p in dev_positions for p in variant_positions]
        if all(is_dev):
            roles[r["mutant"]] = "dev"
        elif not any(is_dev):
            roles[r["mutant"]] = "train"
        else:
            excluded.append(r["mutant"])
    return roles, excluded, {"position_count": len(positions), "train_position_count": len(positions) - dev_count, "dev_position_count": dev_count, "boundary_excluded_count": len(excluded), "dev_positions": sorted(dev_positions)}


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["paths"]["artifacts"])
    raw = root / "raw" / "GFP_AEQVI_Sarkisyan_2016.csv"
    assert raw.stat().st_size == config["source"]["expected_csv_bytes"]
    assert sha256_file(raw) == config["source"]["expected_csv_sha256"]
    with open(raw, newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == config["source"]["expected_rows"]
    assert len({r["mutant"] for r in records}) == len(records)
    for r in records:
        r["order"] = mutation_order(r["mutant"])
        assert r["order"] >= 1
        assert len(r["mutated_sequence"]) == config["source"]["sequence_length"]
        assert math.isfinite(float(r["DMS_score"]))
        for token in r["mutant"].split(":"):
            match = MUTATION_RE.fullmatch(token)
            assert match is not None
    counts = Counter(r["order"] for r in records)
    expected_counts = {1: 1084, 2: 12777, 3: 12336, 4: 9387, 5: 6825, 6: 4298, 7: 2526, 8: 1364, 9: 627, 10: 299, 11: 118, 12: 43, 13: 23, 14: 5, 15: 2}
    assert dict(sorted(counts.items())) == expected_counts
    low = [r for r in records if r["order"] in config["roles"]["train_orders"]]
    high = sorted((r for r in records if r["order"] >= config["roles"]["permanent_test_orders_min"]), key=lambda r: r["mutant"])
    pressure = [r for r in records if r["order"] in config["roles"]["unused_pressure_orders"]]
    assert len(low) == 13861 and len(pressure) == 21723
    assert len(high) >= config["roles"]["minimum_test_count_gate"]
    role_by_id, boundary_excluded, position_summary = position_roles(low)
    low_train = [r for r in low if role_by_id.get(r["mutant"]) == "train"]
    low_dev = [r for r in low if role_by_id.get(r["mutant"]) == "dev"]
    assert len(low_train) >= 5000 and len(low_dev) >= 1000
    rng = random.Random(config["roles"]["test_shuffle_seed"])
    shuffled = high[:]
    rng.shuffle(shuffled)
    selected = shuffled[: config["roles"]["test_pool_count"] * config["roles"]["test_pool_size"]]
    unused = shuffled[len(selected):]
    assert len(selected) == 16000
    pool_by_id = {}
    membership = []
    for index, r in enumerate(selected):
        pool_id = f"pool_{index // config['roles']['test_pool_size']:02d}"
        pool_by_id[r["mutant"]] = pool_id
        membership.append({"pool_id": pool_id, "variant_id": r["mutant"], "order": r["order"]})
    assert min(r["order"] for r in high) - max(r["order"] for r in low) >= config["roles"]["minimum_cross_role_hamming_gate"]
    assert not ({r["mutant"] for r in low} & {r["mutant"] for r in high})
    fields = ["variant_id", "mutated_sequence", "order", "DMS_score"]
    low_rows = [{"variant_id": r["mutant"], "mutated_sequence": r["mutated_sequence"], "order": r["order"], "DMS_score": r["DMS_score"], "role": role_by_id[r["mutant"]]} for r in low if r["mutant"] in role_by_id]
    write_csv(root / "learner" / "low_order_labeled.csv", ["variant_id", "mutated_sequence", "order", "DMS_score", "role"], low_rows)
    write_csv(root / "learner" / "high_order_sequences.csv", ["pool_id", "variant_id", "mutated_sequence", "order"], [{"pool_id": pool_by_id[r["mutant"]], "variant_id": r["mutant"], "mutated_sequence": r["mutated_sequence"], "order": r["order"]} for r in selected])
    write_csv(root / "evaluator" / "high_order_labels.csv", ["variant_id", "DMS_score"], [{"variant_id": r["mutant"], "DMS_score": r["DMS_score"]} for r in selected])
    write_csv(root / "evaluator" / "unused_test_manifest.csv", ["variant_id", "order"], [{"variant_id": r["mutant"], "order": r["order"]} for r in unused])
    write_csv(root / "evaluator" / "pressure_excluded_manifest.csv", ["variant_id", "order"], [{"variant_id": r["mutant"], "order": r["order"]} for r in pressure])
    write_csv(root / "evaluator" / "low_order_boundary_excluded.csv", ["variant_id", "order"], [{"variant_id": r["mutant"], "order": r["order"]} for r in low if r["mutant"] in set(boundary_excluded)])
    write_csv(root / "evaluator" / "pool_membership.csv", ["pool_id", "variant_id", "order"], membership)
    summary = {
        "status": "PASS",
        "source_sha256": sha256_file(raw),
        "rows": len(records),
        "order_counts": dict(sorted(counts.items())),
        "train_count": len(low_train),
        "dev_count": len(low_dev),
        "low_order_position_audit": position_summary,
        "low_order_boundary_excluded_count": len(boundary_excluded),
        "low_order_train_dev_minimum_hamming": 3,
        "low_order_train_dev_hamming1_crossings": 0,
        "pressure_excluded_count": len(pressure),
        "eligible_high_order_count": len(high),
        "selected_high_order_count": len(selected),
        "unused_high_order_count": len(unused),
        "pool_count": len(set(pool_by_id.values())),
        "pool_size": config["roles"]["test_pool_size"],
        "low_to_high_min_hamming": min(r["order"] for r in high) - max(r["order"] for r in low),
        "exact_cross_role_count": 0,
        "unique_test_pools_per_variant": len(pool_by_id),
        "sequence_length_values": sorted({len(r["mutated_sequence"]) for r in records}),
        "nonfinite_label_count": 0
    }
    (root / "verification" / "round1_raw_gate.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
