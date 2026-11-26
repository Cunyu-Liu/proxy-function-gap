#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd

from e1_rinalmo_frozen_baselines import (TaskLoader, knn_scores, metrics, parse_clstr,
                                         prediction_rows, require_cuda, split_audit_rows)
from e1_locked_logger import E1RunLogger as RunLogger
from log_run import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--neg-dir", required=True)
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--positives-audit", required=True)
    parser.add_argument("--split-clstr", required=True)
    parser.add_argument("--exps-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="rinalmo-giga")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--tracks", default="neg1c,neg2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ks", default="5,10,20")
    parser.add_argument("--log", required=True)
    parser.add_argument("--data-version", default="e0-v2-id90-strict3train-main")
    parser.set_defaults(command="knn")
    args = parser.parse_args()
    if args.model != "rinalmo-giga" or len([int(x) for x in args.folds.split(",")]) != 1:
        raise ValueError("one RiNALMo fold per run is required")
    os.makedirs(args.out, exist_ok=True)
    config = vars(args).copy()
    config["code_sha256"] = sha256_file(os.path.abspath(__file__))
    logger = RunLogger(args.log, stage="E1", kind="frozen-knn-rinalmo-lowmem")
    logger.start(config=config, seed=args.seed, data_version=args.data_version,
                 inputs=[args.split_clstr, args.exps_file], code_version=config["code_sha256"])
    try:
        import torch
        device, device_info = require_cuda()
        loader = TaskLoader(args)
        cluster = parse_clstr(args.split_clstr)
        ks = [int(x) for x in args.ks.split(",")]
        validation_rows, audit_rows = [], []
        cached = {}
        tasks = list(loader.tasks())
        for index, task in enumerate(tasks, 1):
            arrays = loader.arrays(task)
            audit_rows.append(split_audit_rows(task))
            (train_ids, xtr, ytr), (_, xva, yva), (_, xte, yte) = arrays
            seen, keep = set(), []
            for row, (window_id, label) in enumerate(zip(train_ids, ytr)):
                if label == 1:
                    cluster_id = cluster.get(window_id, f"missing:{window_id}")
                    if cluster_id in seen:
                        continue
                    seen.add(cluster_id)
                keep.append(row)
            keep = np.asarray(keep, dtype=np.int64)
            xtr, ytr = xtr[keep], ytr[keep]
            val_scores = knn_scores(xtr, ytr, xva, ks, device)
            test_scores = knn_scores(xtr, ytr, xte, ks, device)
            key = (task["experiment"], task["fold"], task["track"])
            cached[key] = (val_scores, test_scores, arrays[1][0], yva, arrays[2][0], yte, len(ytr))
            for k in ks:
                auroc, auprc = metrics(yva, val_scores[k])
                validation_rows.append({**{name: task[name] for name in
                                           ("experiment", "rbp", "tier", "fold", "track")},
                                        "k": k, "split": "val", "n_train": len(ytr), "n_eval": len(yva),
                                        "auroc": auroc, "auprc": auprc})
            print(f"KNN {index}/{len(tasks)} {task['experiment']} f{task['fold']} "
                  f"{task['track']} train={len(ytr)}", flush=True)
            del arrays, xtr, ytr, xva, xte
            torch.cuda.empty_cache()
        validation = pd.DataFrame(validation_rows)
        tail = validation[validation.tier == "tail"]
        if tail.empty:
            raise ValueError("tail-RBP validation is empty")
        selected = tail.groupby(["track", "k"]).auprc.mean().groupby("track").idxmax()
        best_k = {track: int(k) for track, k in selected}
        print("SELECTED_K", best_k, flush=True)
        result_rows, test_predictions, val_predictions = [], [], []
        for task in tasks:
            key = (task["experiment"], task["fold"], task["track"])
            val_scores, test_scores, val_ids, yva, test_ids, yte, _ = cached[key]
            k = best_k[task["track"]]
            for split, scores, ids, labels in (("val", val_scores[k], val_ids, yva),
                                               ("test", test_scores[k], test_ids, yte)):
                auroc, auprc = metrics(labels, scores)
                result_rows.append({**{name: task[name] for name in
                                       ("experiment", "rbp", "tier", "fold", "track")},
                                    "k": k, "split": split, "n_eval": len(labels),
                                    "auroc": auroc, "auprc": auprc})
                target = val_predictions if split == "val" else test_predictions
                target.append(prediction_rows(task, split, ids, labels, scores))
        validation.to_csv(os.path.join(args.out, "knn_validation.tsv"), sep="\t", index=False)
        pd.DataFrame(audit_rows).to_csv(os.path.join(args.out, "split_audit.tsv"), sep="\t", index=False)
        results = pd.DataFrame(result_rows)
        results.to_csv(os.path.join(args.out, "knn_results.tsv"), sep="\t", index=False)
        results.groupby(["track", "split", "tier"])[["auroc", "auprc"]].mean().round(6).to_csv(
            os.path.join(args.out, "knn_summary.tsv"), sep="\t")
        pd.concat(test_predictions, ignore_index=True).to_csv(
            os.path.join(args.out, "knn_test_predictions.tsv.gz"), sep="\t", index=False, compression="gzip")
        pd.concat(val_predictions, ignore_index=True).to_csv(
            os.path.join(args.out, "knn_val_predictions.tsv.gz"), sep="\t", index=False, compression="gzip")
        config.update({"selected_k_by_track": best_k, "selection": "tail-RBP mean validation AUPRC",
                       "dedup": "one training positive per frozen id90 cluster; negatives unchanged",
                       "cuda": device_info, "strict_train_folds": True,
                       "memory_contract": "per-task embeddings released; only scores and IDs retained"})
        with open(os.path.join(args.out, "knn_config.json"), "w") as handle:
            json.dump(config, handle, indent=2)
        test = results[results.split == "test"]
        result = {"n_tasks": len(results) // 2, "best_k": best_k, "cuda": device_info,
                  "test_macro": {track: {"auroc": float(group.auroc.mean()),
                                         "auprc": float(group.auprc.mean())}
                                 for track, group in test.groupby("track")}}
        logger.finish(metrics=result, outputs=[args.out])
        print("E1_DONE", json.dumps(result), flush=True)
    except Exception as exc:
        logger.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
