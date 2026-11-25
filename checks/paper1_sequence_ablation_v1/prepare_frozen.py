#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONTEXTS = ["n50_hek293t_r1", "defined_hepg2_r1", "defined_tcell_r1", "defined_tcell_r2"]
POOL_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/data/derived/castillo_fixed_pools_v3_strict_h1_retry1")
ANALYSIS_ROOT = Path("/teams/HoulabData_1788502323/liucunyu/proxy_function_gap/paper1_v3_strict_h1_retry1")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = []
    poly_u = []
    context_counts = {}
    for context in CONTEXTS:
        context_counts[context] = {"pools": 0, "candidates": 0}
        labels = POOL_ROOT / context / "evaluator" / "labels.tsv.gz"
        inputs.append({"path": str(labels), "sha256": sha256(labels), "bytes": labels.stat().st_size})
        for pool_index in range(16):
            stem = f"castillo_{context}_{pool_index:03d}"
            paths = [
                ANALYSIS_ROOT / context / "cnn_scores" / f"{stem}.json",
                ANALYSIS_ROOT / context / "selection" / f"{stem}.json",
                ANALYSIS_ROOT / context / "evaluation" / f"{stem}.json",
            ]
            for path in paths:
                inputs.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
            records = json.loads(paths[0].read_text())["records"]
            assert len(records) == 64
            assert len({r["sequence_id"] for r in records}) == 64
            poly_u.extend(int(r["longest_poly_u"]) for r in records)
            context_counts[context]["pools"] += 1
            context_counts[context]["candidates"] += len(records)
    poly_u.sort()
    cutoff = int(poly_u[math.ceil(0.75 * len(poly_u)) - 1])
    config = {
        "schema_version": "1.0",
        "analysis": "paper1_sequence_ablation_v1",
        "status": "retrospective/exploratory mechanism sensitivity",
        "new_blind_test": False,
        "causal_claim": False,
        "contexts": CONTEXTS,
        "pool_root": str(POOL_ROOT),
        "analysis_root": str(ANALYSIS_ROOT),
        "context_counts": context_counts,
        "candidate_count": len(poly_u),
        "poly_u_q75_nearest_rank": cutoff,
        "poly_u_upper_quartile_rule": "longest_poly_u >= cutoff",
        "poly_u_upper_quartile_candidate_count": sum(x >= cutoff for x in poly_u),
        "threshold_source": "all strict-H1 candidate sequence features only; no MRL/selection/regret/mechanism result",
        "bootstrap": {"replicates": 10000, "seed": 20261125, "cluster": "pool", "percentiles": [2.5, 97.5]},
        "features": ["gc", "uAUG", "longest_poly_u", "support"],
        "caps": [1, 2, 4, 8, 16],
        "seeds": [101, 202, 303, 404, 505],
        "protocol_sha256": sha256(Path(args.protocol)),
        "raw_input_manifest": inputs,
    }
    (output / "frozen_analysis.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "raw_input_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print(json.dumps({"output": str(output), "candidate_count": len(poly_u), "poly_u_q75": cutoff, "upper_q_count": config["poly_u_upper_quartile_candidate_count"], "raw_files": len(inputs), "protocol_sha256": config["protocol_sha256"]}))


if __name__ == "__main__":
    main()
