#!/usr/bin/env python3
"""Derive artifact-only evidence for CAFA3 runs completed before task logging."""
import argparse
import json
import os
from pathlib import Path

from cafa3_tasklog import artifact, sha256_file


def derived_record(path, embedding):
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed_artifacts = []
    for seed in summary["bce"]:
        number = seed["seed"]
        seed_artifacts.append({"seed": number, "best_epoch": seed["best_epoch"], "tau": seed["tau"],
                               "fmax_threshold": seed["fmax_threshold"], "smin_threshold": seed["smin_threshold"],
                               "artifacts": [artifact(path / f"bce_seed{number}.pt"),
                                             artifact(path / f"validation_thresholds_seed{number}.tsv")]})
    return {"record_type": "derived_from_artifacts", "run_id": f"derived-cafa3-fold{summary['fold']}-{summary['track']}-{summary['ontology']}",
            "fold": summary["fold"], "track": summary["track"], "ontology": summary["ontology"],
            "code_sha256": summary["code_sha256"], "data_sha256": summary["data_manifest_sha256"],
            "cuda": {"device": embedding["cuda_device"], "torch_version": embedding["torch_version"],
                     "evidence": "derived_from_embedding_manifest"},
            "start_utc": None, "end_utc": None, "status": "derived_from_artifacts",
            "selection": {"tau": summary["selected_tau"], "t1_selection": summary["t1_selection"]},
            "artifacts": {"combo": [artifact(path / ".complete")], "seeds": seed_artifacts,
                          "knn": [artifact(path / "validation_thresholds_knn.tsv")],
                          "summary": [artifact(summary_path), artifact(path / "metrics.tsv.gz"),
                                      artifact(path / "per_label_metrics.tsv.gz"), artifact(path / "validation_history.tsv")]},
            "derived_by_code_sha256": sha256_file(os.path.abspath(__file__))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    embedding = json.loads((root / "embeddings/embedding_manifest.json").read_text(encoding="utf-8"))
    records = [derived_record(path, embedding) for path in sorted(root.glob("runs/fold*/*/*"))
               if (path / ".complete").exists()]
    if not records:
        raise ValueError("No completed CAFA3 artifacts found")
    out = Path(args.out)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, out)
    print(json.dumps({"derived_from_artifacts": len(records), "out": str(out)}))


if __name__ == "__main__":
    main()
