#!/usr/bin/env python3
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def poly_u(sequence):
    return max((len(run) for run in sequence.split("A") for run in run.split("C") for run in run.split("G") if run and set(run) == {"T"}), default=0)


def longest_t(sequence):
    best = current = 0
    for base in sequence:
        current = current + 1 if base == "T" else 0
        best = max(best, current)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen_path = Path(args.frozen)
    config = json.loads(frozen_path.read_text())
    errors = []
    checks = 0
    for record in config["raw_input_manifest"]:
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            errors.append(f"input hash/size mismatch: {path}")
        checks += 1
    pool_root = Path(config["pool_root"])
    analysis_root = Path(config["analysis_root"])
    candidate_total = pair_total = 0
    for context in config["contexts"]:
        with gzip.open(pool_root / context / "evaluator" / "labels.tsv.gz", "rt") as handle:
            label_rows = list(csv.DictReader(handle, delimiter="\t"))
        labels = {r["sequence_id"]: float(r["measured_rl"]) for r in label_rows}
        if len(labels) != 1024:
            errors.append(f"{context}: expected 1024 unique labels, got {len(labels)}")
        for pool_index in range(16):
            stem = f"castillo_{context}_{pool_index:03d}"
            score_records = json.loads((analysis_root / context / "cnn_scores" / f"{stem}.json").read_text())["records"]
            selection_records = json.loads((analysis_root / context / "selection" / f"{stem}.json").read_text())["records"]
            evaluation_records = json.loads((analysis_root / context / "evaluation" / f"{stem}.json").read_text())["records"]
            if len(score_records) != 64 or len(selection_records) != 50 or len(evaluation_records) != 50:
                errors.append(f"{stem}: wrong row count")
            candidate_map = {r["sequence_id"]: r for r in score_records}
            pool_maximum = max(labels[r["sequence_id"]] for r in score_records)
            for candidate in score_records:
                sequence = candidate["utr50"]
                if len(sequence) != 50 or set(sequence) - set("ACGT"):
                    errors.append(f"{candidate['sequence_id']}: invalid UTR50")
                expected_gc = (sequence.count("G") + sequence.count("C")) / 50
                if abs(candidate["gc"] - expected_gc) > 1e-12:
                    errors.append(f"{candidate['sequence_id']}: GC mismatch")
                if bool(candidate["uAUG"]) != ("ATG" in sequence):
                    errors.append(f"{candidate['sequence_id']}: uAUG mismatch")
                if int(candidate["longest_poly_u"]) != longest_t(sequence):
                    errors.append(f"{candidate['sequence_id']}: poly-U mismatch")
                if not isinstance(candidate["support"], (float, int)) or not isinstance(candidate["proxy_mean"], (float, int)):
                    errors.append(f"{candidate['sequence_id']}: support/proxy not numeric")
                if candidate["sequence_id"] not in labels or labels[candidate["sequence_id"]] < 0:
                    errors.append(f"{candidate['sequence_id']}: missing/invalid measured MRL")
                checks += 8
            selection_lookup = {(r["proposal_cap"], r["seed"], r["strategy"]): r for r in selection_records}
            evaluation_lookup = {(r["proposal_cap"], r["seed"], r["strategy"]): r for r in evaluation_records}
            if set(selection_lookup) != set(evaluation_lookup):
                errors.append(f"{stem}: selection/evaluation key mismatch")
            for key, selected in selection_lookup.items():
                evaluated = evaluation_lookup[key]
                if selected["selected_id"] != evaluated["selected_id"] or selected["selected_id"] not in candidate_map:
                    errors.append(f"{stem}/{key}: selected ID mismatch")
                expected_regret = pool_maximum - labels[selected["selected_id"]]
                if abs(expected_regret - evaluated["regret"]) > 1e-9:
                    errors.append(f"{stem}/{key}: regret mismatch")
                if key[0] == 1 and selection_lookup[(1, key[1], "random")]["selected_id"] != selection_lookup[(1, key[1], "bon")]["selected_id"]:
                    errors.append(f"{stem}/{key}: cap1 BoN != Random")
                checks += 4
            candidate_total += len(score_records)
            pair_total += len(evaluation_records) // 2
    report = {
        "round": 1,
        "pass": not errors,
        "checks": checks,
        "raw_files": len(config["raw_input_manifest"]),
        "candidates": candidate_total,
        "bon_random_pairs": pair_total,
        "uAUG_definition": "ATG substring in UTR50",
        "poly_u_definition": "longest contiguous T run; T=U",
        "regret_definition": "pool maximum measured MRL minus selected measured MRL",
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
