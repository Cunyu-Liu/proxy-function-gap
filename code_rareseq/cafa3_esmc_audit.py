#!/usr/bin/env python3
"""Independent completeness and selection-contract audit for CAFA3 baseline runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_checkpoint(history, min_delta=1e-4):
    best_loss, best_epoch = np.inf, None
    for row in history.sort_values("epoch").itertuples():
        if row.validation_bce < best_loss - min_delta:
            best_loss, best_epoch = row.validation_bce, row.epoch
    return best_loss, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root, data = Path(args.run_root), Path(args.data_dir)
    manifest = json.loads((data / "frozen_manifest.json").read_text())
    if manifest["summary_sha256"] != sha256_file(data / "cleaning_summary.json"):
        raise ValueError("Frozen CAFA3 summary SHA mismatch")
    embedding = json.loads((root / "embeddings/embedding_manifest.json").read_text())
    expected_rows = {"t0": 66841, "t1": 3328}
    for split in embedding["splits"]:
        name = split["split"]
        if split["shape"] != [expected_rows[name], 960] or split["sha256"] != sha256_file(root / f"embeddings/{name}.npy"):
            raise ValueError(f"Embedding audit failed: {name}")
        array = np.load(root / f"embeddings/{name}.npy", mmap_mode="r")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite embedding: {name}")
    checked = []
    for fold in range(3):
        for track in ("main", "id30", "id50", "id90"):
            for ontology in ("MFO", "BPO", "CCO"):
                path = root / f"runs/fold{fold}/{track}/{ontology}"
                if not (path / ".complete").exists():
                    if args.allow_partial:
                        continue
                    raise ValueError(f"Incomplete run: {path}")
                summary = json.loads((path / "summary.json").read_text())
                if summary["seeds"] != [0, 1, 2] or summary["deep_knn"]["seed"] != "deterministic":
                    raise ValueError(f"Seed contract failed: {path}")
                tau = pd.read_csv(path / "tau_validation.tsv", sep="\t")
                expected_tau = float(tau.sort_values("tau").iloc[tau.validation_tail100_macro_auprc.idxmax()].tau)
                if summary["selected_tau"] != expected_tau:
                    raise ValueError(f"Tau was not selected on validation: {path}")
                for seed in summary["bce"]:
                    history = pd.read_csv(path / "validation_history.tsv", sep="\t")
                    history = history[history.seed == seed["seed"]]
                    best_loss, best_epoch = validation_checkpoint(history)
                    if (len(history) > 1000 or not np.isclose(best_loss, seed["validation_bce"], rtol=0, atol=1e-12) or
                            best_epoch != seed["best_epoch"]):
                        raise ValueError(f"Epoch/validation history mismatch: {path} seed {seed['seed']}")
                    grid = pd.read_csv(path / f"validation_thresholds_seed{seed['seed']}.tsv", sep="\t")
                    if len(grid) != 100 or not np.allclose(grid.threshold, np.round(np.arange(.01, 1.001, .01), 2)):
                        raise ValueError(f"Threshold grid mismatch: {path} seed {seed['seed']}")
                    fmax = float(grid.sort_values("threshold").iloc[grid.fmax_at_val_threshold.idxmax()].threshold)
                    smin = float(grid.sort_values("threshold").iloc[grid.smin_at_val_threshold.idxmin()].threshold)
                    if seed["fmax_threshold"] != fmax or seed["smin_threshold"] != smin:
                        raise ValueError(f"Validation threshold mismatch: {path} seed {seed['seed']}")
                metrics = pd.read_csv(path / "metrics.tsv.gz", sep="\t")
                expected_models = {("frozen_bce_la", "0"), ("frozen_bce_la", "1"), ("frozen_bce_la", "2"),
                                   ("deep_knn", "deterministic")}
                if set(zip(metrics.model, metrics.seed.astype(str))) != expected_models:
                    raise ValueError(f"Model matrix incomplete: {path}")
                t1 = metrics[(metrics.split == "t1") & (metrics.slice_type == "role") & (metrics.frequency_bucket == "overall")]
                if set(t1.role) != {"NK", "LK"} or not np.isfinite(t1[["fmax_at_val_threshold", "smin_at_val_threshold"]]).all().all():
                    raise ValueError(f"Invalid t1 role metrics: {path}")
                if ontology == "BPO":
                    nk = t1[(t1.role == "NK") & (t1.model == "deep_knn")]
                    if not ((nk.official_role_proteins == 849) & (nk.n_proteins == 848)).all():
                        raise ValueError("BPO NK 848/849 coverage contract failed")
                per_label = pd.read_csv(path / "per_label_metrics.tsv.gz", sep="\t", low_memory=False)
                for (model, seed, split, role), group in per_label.groupby(["model", "seed", "split", "role"]):
                    observed = group.auprc.mean()
                    selected = metrics[(metrics.model == model) & (metrics.seed.astype(str) == str(seed)) &
                                       (metrics.split == split) & (metrics.role == role) &
                                       (metrics.slice_type.isin(["all", "role"])) & (metrics.frequency_bucket == "overall")]
                    if len(selected) != 1 or not np.isclose(observed, selected.iloc[0].macro_auprc, equal_nan=True):
                        raise ValueError(f"Macro AUPRC recomputation failed: {path} {model} {seed} {split} {role}")
                if "combo_run_id" in summary:
                    records = [json.loads(line) for line in (root / "tasks.jsonl").read_text().splitlines()]
                    combo_id = summary["combo_run_id"]
                    terminal = [record for record in records if record["status"] == "completed" and
                                (record["run_id"] == combo_id or record.get("parent_run_id") == combo_id)]
                    counts = {kind: sum(record["task_type"] == kind for record in terminal)
                              for kind in ("combo", "seed", "knn", "summary")}
                    if counts != {"combo": 1, "seed": 3, "knn": 1, "summary": 1}:
                        raise ValueError(f"Task JSONL matrix mismatch: {path} {counts}")
                    for record in terminal:
                        if (not record["start_utc"] or not record["end_utc"] or not record["cuda"]["available"] or
                                record["data_sha256"] != summary["data_manifest_sha256"]):
                            raise ValueError(f"Task JSONL evidence mismatch: {path} {record['run_id']}")
                        for item in record["artifacts"]:
                            if item["sha256"] != sha256_file(item["path"]):
                                raise ValueError(f"Task artifact SHA mismatch: {item['path']}")
                else:
                    derived = [json.loads(line) for line in (root / "derived_from_artifacts.jsonl").read_text().splitlines()]
                    key = f"derived-cafa3-fold{fold}-{track}-{ontology}"
                    record = next((item for item in derived if item["run_id"] == key), None)
                    if (record is None or record["status"] != "derived_from_artifacts" or
                            record["start_utc"] is not None or record["end_utc"] is not None):
                        raise ValueError(f"Derived artifact evidence mismatch: {path}")
                checked.append({"fold": fold, "track": track, "ontology": ontology,
                                "summary_sha256": sha256_file(path / "summary.json")})
    if not checked:
        raise ValueError("No complete runs to audit")
    report = {"status": "passed", "partial": args.allow_partial, "runs": len(checked), "bce_seed_runs": len(checked) * 3,
              "deep_knn_runs": len(checked), "embeddings": expected_rows,
              "t0_proxy_boundary": "held-out fold also selects parameters; not independent test or all-homology isolation",
              "t1": "final temporal OOD only; validation-selected tau and thresholds", "checked": checked}
    (root / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "runs": report["runs"], "bce_seed_runs": report["bce_seed_runs"]}))


if __name__ == "__main__":
    main()
