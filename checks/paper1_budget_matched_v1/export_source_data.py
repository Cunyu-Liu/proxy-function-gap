#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_tsv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "export"
    output.mkdir(exist_ok=False)
    results = json.loads((args.root / "analysis/results.json").read_text())
    write_tsv(
        output / "context_summary.tsv",
        results["context_summaries"],
        ["context", "cap", "outer_pools", "random1_regret", "bon_regret", "random_n_regret", "bon_minus_random1", "bon_minus_random_n", "random_n_minus_random1"],
    )
    family_rows = [row for family in results["families"].values() for row in family]
    write_tsv(
        output / "family_effects.tsv",
        family_rows,
        ["family", "context", "cap", "outer_pools", "effect", "nominal_95_ci", "bonferroni_20_ci", "p", "holm_p", "bonferroni_p", "bootstrap_seed"],
    )
    for name in ("results.json", "pool_effects.tsv", "selection_audit.json"):
        shutil.copyfile(args.root / "analysis" / name, output / name)
    print(json.dumps({"status": "PASS", "export": str(output), "sequence_level_labels": False, "selected_ids": False}))


if __name__ == "__main__":
    main()
