import argparse
import json
import math
from collections import Counter
from pathlib import Path

from proxy_function_gap.paper2_baselines import CAPS, METHODS, PAIRED_SEEDS, TOP_K, load_role_pool
from proxy_function_gap.paper2_com import COMConfig, verify_com_run
from proxy_function_gap.trainer.data import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--com-config", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.round not in (1, 2, 3):
        raise ValueError("verification round must be 1, 2 or 3")
    config = json.loads(Path(args.baseline_config).read_text())
    com = verify_com_run(COMConfig.from_json(args.com_config))
    selected = json.loads(Path(args.frozen_config).read_text())["selected"]
    pools = []
    for index in range(args.count):
        pool_id = f"{args.role}_{index:03d}"
        path = Path(args.evaluation_dir) / f"{pool_id}.json"
        payload = json.loads(path.read_text())
        if not payload["actual_cuda_inference"] or payload["labels_returned_to_learner"]:
            raise ValueError(f"invalid CUDA/evaluator boundary for {pool_id}")
        if sha256_file(Path(args.evaluation_dir).parent / "selection" / f"{pool_id}.json") != payload["selection_sha256"]:
            raise ValueError(f"selection hash mismatch for {pool_id}")
        records = payload["records"]
        expected = len(METHODS) * len(PAIRED_SEEDS) * (len(CAPS) + 32)
        if len(records) != expected:
            raise ValueError(f"record count mismatch for {pool_id}: {len(records)} != {expected}")
        pool = load_role_pool(config, args.role, index)
        if payload["outer_metrics_computed"]:
            import csv
            import gzip
            source = Path(config["source_pool_manifest"])
            vault = json.loads((source.parent / "evaluator" / "vault_references.json").read_text())
            labels = {}
            with gzip.open(vault[args.role]["path"], "rt", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if row.get("measured_rl"):
                        labels.setdefault(row["sequence_id"], []).append(float(row["measured_rl"]))
            measured = {key: sorted(values)[len(values) // 2] if len(values) % 2 else sum(sorted(values)[len(values) // 2 - 1:len(values) // 2 + 1]) / 2 for key, values in labels.items()}
            values = [measured[value] for value in pool.sequence_ids]
        else:
            values = list(pool.measured)
        best = max(values)
        keys = Counter()
        for record in records:
            if record["method"] not in METHODS or record["proposal_cap"] not in CAPS or record["seed"] not in PAIRED_SEEDS:
                raise ValueError(f"unexpected record identity in {pool_id}")
            setting = selected[record["method"]]
            if record["lambda"] != setting["lambda"] or record["gamma"] != setting["gamma"]:
                raise ValueError(f"non-frozen hyperparameters in {pool_id}")
            if (record["permutation_id"] > 0) != (record["proposal_cap"] == 16 and record["permutation_id"] > 0):
                raise ValueError("impossible permutation identity")
            keys[(record["method"], record["seed"], record["proposal_cap"], record["permutation_id"])] += 1
            if not math.isclose(record["regret"], best - values[record["selected_index"]], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"regret mismatch in {pool_id}")
            for k in TOP_K:
                metric = record["top_k"][str(k)]
                selected_values = [values[item] for item in record["top_indices"][str(k)]]
                if not math.isclose(metric["regret_at_k"], best - max(selected_values), rel_tol=0, abs_tol=1e-12) or not math.isclose(metric["mean_measured_rl"], sum(selected_values) / len(selected_values), rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"top-k mismatch in {pool_id}")
        if set(keys.values()) != {1}:
            raise ValueError(f"duplicate records in {pool_id}")
        for method in METHODS:
            for seed in PAIRED_SEEDS:
                for cap in CAPS:
                    permutations = range(33) if cap == 16 else (0,)
                    if any(keys[(method, seed, cap, permutation)] != 1 for permutation in permutations):
                        raise ValueError(f"missing permutation/cap records in {pool_id}")
        pools.append({"pool_id": pool_id, "records": len(records), "sha256": sha256_file(path), "pass": True})
    result = {"pass": True, "verification_round": args.round, "com_cuda_reload": com, "baseline_pools": pools, "records": sum(row["records"] for row in pools), "frozen_config_sha256": sha256_file(args.frozen_config), "outer_labels_returned_to_learner": False}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pass": True, "round": args.round, "pools": len(pools), "records": result["records"]}))


if __name__ == "__main__":
    main()
