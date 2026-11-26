#!/usr/bin/env python3
"""Independently reload compressed checkpoints and reproduce validation without reading test."""
import argparse
import gc
import json
import os
from types import SimpleNamespace

import numpy as np

from e1_finetune import (SequenceLoader, atomic_json, atomic_marker, atomic_npz, build_model,
                         evaluate_scores, file_sha256, load_checkpoint_compressed, load_tokenizer,
                         metrics, require_cuda, restore_state, task_dir)
from e1_locked_logger import E1RunLogger as RunLogger
from log_run import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    if any(os.path.exists(os.path.join(args.out, name)) for name in
           ("finetune_results.tsv", "finetune_test_predictions.tsv.gz", ".eval_complete")):
        raise RuntimeError("Validation-only recheck refuses a run with test artifacts")
    train_config = json.load(open(os.path.join(args.out, "finetune_config.json")))
    if not os.path.isfile(os.path.join(args.out, ".train_complete")):
        raise RuntimeError("Training incomplete")
    train_args = SimpleNamespace(**train_config["args"])
    lg = RunLogger(args.log, stage="E1", kind=f"finetune-validate-{train_args.mode}")
    lg.start(config=vars(args), seed=train_args.seed, data_version=train_args.data_version,
             inputs=[os.path.join(args.out, "finetune_config.json")],
             code_version=sha256_file(os.path.abspath(__file__)))
    try:
        device, device_info = require_cuda()
        loader = SequenceLoader(train_args)
        tasks = list(loader.tasks())
        tokenizer, fast_encode, fast_ok = load_tokenizer(train_args.model)
        if fast_ok != train_config["fast_tokenizer_verified"]:
            raise RuntimeError("Tokenizer verification changed")
        root = os.path.join(args.out, "independent_validation_tasks")
        os.makedirs(root, exist_ok=True)
        rows = []
        for i, task in enumerate(tasks, 1):
            stem = f"{task['experiment']}.f{task['fold']}.{task['track']}"
            directory = os.path.join(root, stem)
            os.makedirs(directory, exist_ok=True)
            metadata_path = os.path.join(directory, "metadata.json")
            if not os.path.isfile(metadata_path):
                checkpoint = load_checkpoint_compressed(task_dir(args.out, task))
                model, head = build_model(train_args, train_args.model, device)
                restore_state(model, head, checkpoint)
                arrays = loader.arrays(task, include_test=False)
                score = evaluate_scores(model, head, tokenizer, fast_encode, arrays[1][1],
                                        train_args.batch, device)
                auroc, auprc = metrics(arrays[1][2], score)
                train_metadata = json.load(open(os.path.join(task_dir(args.out, task), "metadata.json")))
                if not np.allclose([auroc, auprc], [train_metadata["val_auroc"], train_metadata["val_auprc"]],
                                   atol=1e-12):
                    raise RuntimeError("Independent checkpoint validation reproduction failed")
                manifest = json.load(open(os.path.join(task_dir(args.out, task), "checkpoint_manifest.json")))
                prediction_path = os.path.join(directory, "validation_predictions.npz")
                atomic_npz(prediction_path, window_id=np.asarray(arrays[1][0]),
                           label=arrays[1][2], score=score)
                atomic_json(metadata_path, {"experiment": task["experiment"], "fold": task["fold"],
                                            "track": task["track"], "auroc": auroc, "auprc": auprc,
                                            "checkpoint_raw_sha256": manifest["raw_sha256"],
                                            "checkpoint_compressed_sha256": manifest["compressed_sha256"],
                                            "validation_predictions_sha256": file_sha256(prediction_path),
                                            "evaluation_device": device_info,
                                            "test_used": False})
                del model, head, checkpoint
                gc.collect()
            rows.append(json.load(open(metadata_path)))
            print(f"INDEPENDENT_VALIDATION {i}/{len(tasks)} {task['experiment']} {task['track']}", flush=True)
        atomic_json(os.path.join(args.out, "independent_validation.json"),
                    {"n_tasks": len(rows), "test_used": False, "cuda": device_info,
                     "tasks": rows})
        atomic_marker(os.path.join(args.out, ".independent_validation_complete"))
        lg.finish(metrics={"n_tasks": len(rows), "validation_reproduced": True, "test_used": False},
                  outputs=[args.out])
        print("E1_FINETUNE_INDEPENDENT_VALIDATION_DONE", len(rows), flush=True)
    except Exception as exc:
        lg.finish(status="failed", notes=f"{type(exc).__name__}: {exc}", outputs=[args.out])
        raise


if __name__ == "__main__":
    main()
