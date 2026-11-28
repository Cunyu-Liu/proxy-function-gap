#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--dev-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = Path(args.raw_dir)
    dev = Path(args.dev_dir)
    protocol = json.loads(Path(args.protocol).read_text())
    checks = {}

    manifest = json.loads((raw / "input_manifest.json").read_text())
    hash_results = {}
    for item in manifest["files"]:
        path = Path(item["path"])
        actual = sha256(path)
        hash_results[path.name] = actual == item["sha256"]
        if actual != item["sha256"]:
            raise AssertionError(path)
    checks["all_seven_input_hashes_match"] = all(hash_results.values()) and len(hash_results) == 7

    frames = {}
    for name, filename in protocol["data"]["conditions"].items():
        frame = pd.read_csv(raw / filename)
        assert len(frame) == 4997 and frame.hgvs_pro.nunique() == 4997
        frame["score"] = pd.to_numeric(frame.score, errors="coerce")
        frames[name] = frame.set_index("hgvs_pro").score
    unselected = pd.read_csv(raw / protocol["data"]["unselected_file"])
    assert len(unselected) == 4997 and unselected.hgvs_pro.nunique() == 4997
    ids = set(frames["amp10"].index)
    assert all(set(x.index) == ids for x in frames.values()) and set(unselected.hgvs_pro) == ids
    checks["4997_aligned_unique_variants_per_condition"] = True
    expected_missing = {"amp10": 133, "amp39": 0, "amp156": 0, "amp625": 0, "amp2500": 0, "cefotaxime0.15": 19}
    assert {name: int(score.isna().sum()) for name, score in frames.items()} == expected_missing
    checks["missingness_matches_raw_audit"] = True

    all_frame = pd.DataFrame(frames)
    positions = sorted(set(int(re.search(r"p\.[A-Za-z*]{3}(\d+)", x).group(1)) for x in ids))
    split_seed = protocol["split"]["seed"]
    ordered = sorted(positions, key=lambda p: hashlib.sha256(f"scenario_tem1_v1|split|{split_seed}|{p}".encode()).hexdigest())
    n_train = int(len(ordered) * 0.6)
    n_dev = int(len(ordered) * 0.2)
    split = {p: "train" for p in ordered[:n_train]}
    split.update({p: "dev" for p in ordered[n_train:n_train + n_dev]})
    split.update({p: "test" for p in ordered[n_train + n_dev:]})
    assert len(set(ordered)) == len(positions) == 263
    checks["position_split_263_unique_and_disjoint"] = True

    membership = pd.read_csv(dev / "pool_membership.tsv.gz", sep="\t")
    positions_in_pools = membership.hgvs_pro.map(lambda x: int(re.search(r"p\.[A-Za-z*]{3}(\d+)", x).group(1)))
    assert (positions_in_pools == membership.position).all()
    assert all(split[p] == "dev" for p in membership.position.unique())
    sizes = membership.groupby("pool_id").agg(N=("N", "first"), rows=("hgvs_pro", "size"), unique=("hgvs_pro", "nunique"), positions=("position", "nunique"))
    assert (sizes.rows == sizes.N).all() and (sizes.unique == sizes.N).all() and (sizes.positions == 1).all()
    checks["fixed_pool_exact_budget_without_replacement"] = True
    counts = membership.groupby(["position", "N"]).replicate.nunique()
    assert (counts == 20).all()
    checks["twenty_replicates_per_eligible_position_and_N"] = True

    complete = all_frame.notna().all(axis=1)
    position_by_id = pd.Series(all_frame.index, index=all_frame.index).map(lambda x: int(re.search(r"p\.[A-Za-z*]{3}(\d+)", x).group(1)))
    for (position, budget, replicate), group in membership.groupby(["position", "N", "replicate"]):
        variants = sorted(all_frame.index[(position_by_id == position) & complete])
        rng = np.random.default_rng(seed(f"scenario_tem1_v1|pool|20261128|{position}|{budget}|{replicate}"))
        expected = sorted(rng.choice(variants, size=budget, replace=False).tolist())
        assert sorted(group.hgvs_pro) == expected
    checks["independent_pool_rng_replay_exact"] = True

    selections = pd.read_csv(dev / "selections.tsv.gz", sep="\t")
    assert len(selections) == membership.pool_id.nunique() * 25
    assert (selections.groupby(["pool_id", "source", "target"]).size() == 1).all()
    assert (selections.bon_regret >= -1e-12).all() and (selections.random_n_regret >= -1e-12).all()
    assert np.max(np.abs(selections.delta_bon_minus_random - (selections.bon_regret - selections.random_n_regret))) < 1e-12
    checks["direction_counts_and_regret_arithmetic"] = True

    fixture = pd.DataFrame({"proxy": [0.1, 0.2, 0.9, 0.3], "target": [1.0, 2.0, 0.0, 1.5]})
    proxy_pick = fixture.proxy.idxmax()
    oracle = fixture.target.max()
    bon = oracle - fixture.loc[proxy_pick, "target"]
    random_n = oracle - fixture.target.mean()
    assert bon == 2.0 and random_n == 0.875 and bon - random_n == 1.125
    checks["adverse_bon_arithmetic_fixture"] = True

    result = {"round": 1, "status": "PASS", "checks": checks, "n_checks": len(checks), "input_hashes": hash_results, "test_labels_read": False}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
